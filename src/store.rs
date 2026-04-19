//! Memory-mapped page store and page allocator.
//!
//! [`PageStore`] owns the mmap'd file and exposes raw page slices plus
//! [`TrieNode`]-level read/write helpers.  It does **not** implement any trie
//! logic — that lives in a higher layer.
//!
//! # Thread safety
//!
//! `PageStore` is `Send` but not `Sync`.  The intended usage is to wrap it in
//! a `Mutex<PageStore>` at the trie level (not `RwLock`) because mmap reads are
//! not safe to interleave with a concurrent `grow()` that remaps the file.  A
//! single mutex serializes both reads and writes, which is correct and avoids
//! the complexity of coordinating readers across a remap.
//!
//! # Page allocation
//!
//! A free-page singly-linked list is stored in the file header.  When the list
//! is exhausted, [`PageStore::alloc_page`] calls [`PageStore::grow`], which
//! extends the file by [`GROW_BATCH`] pages, remaps, and pushes all but the
//! first new page onto the free list.  This amortizes the `set_len` + remap
//! cost across many subsequent allocations.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::sync::Mutex;

use memmap2::MmapMut;

use crate::error::{Result, TrieError};
use crate::node::{
    FILE_MAGIC, FileHeader, GROW_BATCH, NULL_PAGE, PAGE_SIZE, TrieNode, read_next_free,
    write_free_page,
};

// ---------------------------------------------------------------------------
// PageStore
// ---------------------------------------------------------------------------

/// Manages the memory-mapped file backing the trie.
///
/// Wrap in `Mutex<PageStore>` at the trie level for thread-safe access.
pub struct PageStore {
    file: File,
    mmap: MmapMut,
    /// Cached total page count (= `header().num_pages`); avoids round-trips
    /// through the mmap header for every bounds check.
    pub total_pages: u64,
    /// Single mutex protecting all mutations.  See module-level doc for why
    /// this is a `Mutex` rather than an `RwLock`.
    pub write_lock: Mutex<()>,
}

impl PageStore {
    /// Opens the trie file at `path`, creating it if it does not exist.
    ///
    /// - **New file**: writes a file header, initializes the root node at
    ///   page 1, and pre-allocates `GROW_BATCH` pages.
    /// - **Existing file**: validates magic and version from the header.
    pub fn open(path: &Path) -> Result<Self> {
        let file_exists = path.exists() && std::fs::metadata(path)?.len() > 0;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(!file_exists)
            .open(path)?;

        if file_exists {
            Self::open_existing(file)
        } else {
            Self::create_new(file)
        }
    }

    fn create_new(file: File) -> Result<Self> {
        // Allocate GROW_BATCH pages upfront: page 0 = header, page 1 = root,
        // pages 2..GROW_BATCH = free list.
        let initial_pages = GROW_BATCH;
        file.set_len(initial_pages * PAGE_SIZE as u64)?;

        // SAFETY: We just created/truncated the file and hold exclusive access.
        let mut mmap = unsafe { MmapMut::map_mut(&file) }?;

        // Build and write the file header.
        let mut hdr = FileHeader::new_empty();
        hdr.root_page = 1;
        hdr.num_pages = initial_pages;
        // Free list starts at page 2; pages 2..initial_pages form the chain.
        hdr.free_list_head = if initial_pages > 2 { 2 } else { NULL_PAGE };
        mmap[0..PAGE_SIZE].copy_from_slice(&hdr.to_page());

        // Initialize root node (page 1) as an empty interior node.
        let root_bytes = TrieNode::new().to_page()?;
        mmap[PAGE_SIZE..2 * PAGE_SIZE].copy_from_slice(&root_bytes);

        // Chain free pages: 2 → 3 → … → (initial_pages−1) → NULL.
        for i in 2..initial_pages {
            let next = if i + 1 < initial_pages {
                i + 1
            } else {
                NULL_PAGE
            };
            let start = i as usize * PAGE_SIZE;
            write_free_page(&mut mmap[start..start + PAGE_SIZE], next);
        }

        mmap.flush()?;

        Ok(PageStore {
            file,
            mmap,
            total_pages: initial_pages,
            write_lock: Mutex::new(()),
        })
    }

    fn open_existing(file: File) -> Result<Self> {
        // SAFETY: We own the file and hold it open for the mapping's lifetime.
        let mmap = unsafe { MmapMut::map_mut(&file) }?;

        if mmap.len() < PAGE_SIZE {
            return Err(TrieError::Corruption(
                "file too small for header page".into(),
            ));
        }

        // Validate magic before trusting any other fields.
        let raw_magic = u32::from_le_bytes(mmap[0..4].try_into().unwrap());
        if raw_magic != FILE_MAGIC {
            return Err(TrieError::Corruption(format!(
                "bad file magic: {raw_magic:#010x}"
            )));
        }

        let hdr_bytes: &[u8; PAGE_SIZE] = mmap[0..PAGE_SIZE].try_into().unwrap();
        let hdr = FileHeader::from_page(hdr_bytes)?;
        let total_pages = hdr.num_pages;

        Ok(PageStore {
            file,
            mmap,
            total_pages,
            write_lock: Mutex::new(()),
        })
    }

    // -----------------------------------------------------------------------
    // Raw page access
    // -----------------------------------------------------------------------

    /// Returns the raw byte slice for page `idx`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `idx >= total_pages`.
    #[inline]
    pub fn page(&self, idx: u64) -> &[u8] {
        debug_assert!(idx < self.total_pages, "page {idx} out of range");
        let start = idx as usize * PAGE_SIZE;
        &self.mmap[start..start + PAGE_SIZE]
    }

    /// Returns the mutable raw byte slice for page `idx`.
    ///
    /// # Panics
    ///
    /// Panics in debug builds if `idx >= total_pages`.
    #[inline]
    pub fn page_mut(&mut self, idx: u64) -> &mut [u8] {
        debug_assert!(idx < self.total_pages, "page {idx} out of range");
        let start = idx as usize * PAGE_SIZE;
        &mut self.mmap[start..start + PAGE_SIZE]
    }

    // -----------------------------------------------------------------------
    // Header access
    // -----------------------------------------------------------------------

    /// Deserializes and returns the file header, verifying its CRC.
    pub fn header(&self) -> Result<FileHeader> {
        let bytes: &[u8; PAGE_SIZE] = self.mmap[0..PAGE_SIZE].try_into().unwrap();
        FileHeader::from_page(bytes)
    }

    pub fn write_header(&mut self, hdr: &FileHeader) -> Result<()> {
        self.mmap[0..PAGE_SIZE].copy_from_slice(&hdr.to_page());
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Page allocator
    // -----------------------------------------------------------------------

    /// Allocates one page, returning its index.
    ///
    /// Pops from the free list if available; otherwise calls [`Self::grow`].
    /// The returned page retains whatever bytes were previously written to it —
    /// callers **must** initialize the page (e.g. via [`Self::write_node`])
    /// before treating it as a valid live node.
    pub fn alloc_page(&mut self) -> Result<u64> {
        let hdr = self.header()?;
        if hdr.free_list_head != NULL_PAGE {
            let idx = hdr.free_list_head;
            // Read next_free before mutating the header (borrow ends here).
            let next_free = read_next_free(self.page(idx));
            let mut new_hdr = hdr;
            new_hdr.free_list_head = next_free;
            self.write_header(&new_hdr)?;
            Ok(idx)
        } else {
            self.grow()
        }
    }

    /// Returns page `idx` to the free-page pool.
    ///
    /// The page is overwritten with a freed-page sentinel; the header's
    /// `free_list_head` is updated to point to `idx`.
    pub fn free_page(&mut self, idx: u64) -> Result<()> {
        debug_assert_ne!(idx, NULL_PAGE, "cannot free the header page");
        debug_assert_ne!(idx, 1, "cannot free the root page via free_page");
        debug_assert!(idx < self.total_pages, "page {idx} out of range");

        let old_head = self.header()?.free_list_head;

        {
            let start = idx as usize * PAGE_SIZE;
            write_free_page(&mut self.mmap[start..start + PAGE_SIZE], old_head);
        }

        let mut hdr = self.header()?;
        hdr.free_list_head = idx;
        self.write_header(&hdr)
    }

    // -----------------------------------------------------------------------
    // TrieNode helpers
    // -----------------------------------------------------------------------

    /// Deserializes a [`TrieNode`] from page `idx`, verifying magic and CRC32.
    pub fn read_node(&self, idx: u64) -> Result<TrieNode> {
        let bytes: &[u8; PAGE_SIZE] = self.page(idx).try_into().unwrap();
        TrieNode::from_page(bytes)
    }

    /// Serializes `node` and writes it to page `idx`, embedding the CRC32.
    pub fn write_node(&mut self, idx: u64, node: &TrieNode) -> Result<()> {
        let page_bytes = node.to_page()?;
        let start = idx as usize * PAGE_SIZE;
        self.mmap[start..start + PAGE_SIZE].copy_from_slice(&page_bytes);
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Flush
    // -----------------------------------------------------------------------

    /// Flushes all dirty mmap pages to the underlying file.
    pub fn flush(&self) -> Result<()> {
        self.mmap.flush().map_err(TrieError::from)
    }

    // -----------------------------------------------------------------------
    // File growth
    // -----------------------------------------------------------------------

    /// Extends the file by [`GROW_BATCH`] pages, remaps, and returns the first
    /// new page index (the one handed to `alloc_page`'s caller).
    ///
    /// All remaining new pages (`first_new + 1 .. new_total`) are chained onto
    /// the front of the free list so subsequent allocations come at zero cost.
    ///
    /// The `map_anon(1)` swap keeps `self.mmap` non-dangling while the file is
    /// being extended — the same technique used in `mappedbptree`.
    fn grow(&mut self) -> Result<u64> {
        let first_new = self.total_pages;
        let new_total = first_new + GROW_BATCH;
        let new_size = new_total * PAGE_SIZE as u64;

        // Swap the live mapping out before touching the file size.
        // `map_anon(1)` is a 1-byte anonymous mapping — valid but harmless.
        let old_mmap = std::mem::replace(&mut self.mmap, MmapMut::map_anon(1)?);
        drop(old_mmap); // fully release before the file is extended

        self.file.set_len(new_size)?;

        // SAFETY: File has been extended; we own it exclusively (ensured by
        // the write_lock at the trie level and &mut self here).
        self.mmap = unsafe { MmapMut::map_mut(&self.file) }?;
        self.total_pages = new_total;

        // Read the current free-list head from the (now re-mapped) header.
        let mut hdr = self.header()?;
        let old_free_head = hdr.free_list_head;

        // Chain new pages [first_new+1 .. new_total) into a singly-linked list
        // and prepend that list to the existing free list.
        // Forward pass: page i → page i+1, except the last page → old_free_head.
        // The resulting chain: first_new+1 → first_new+2 → … → new_total-1 → old_free_head.
        for i in first_new + 1..new_total {
            let next = if i + 1 < new_total {
                i + 1
            } else {
                old_free_head
            };
            let start = i as usize * PAGE_SIZE;
            write_free_page(&mut self.mmap[start..start + PAGE_SIZE], next);
        }

        hdr.free_list_head = if new_total > first_new + 1 {
            first_new + 1
        } else {
            old_free_head
        };
        hdr.num_pages = new_total;
        self.write_header(&hdr)?;
        self.mmap.flush()?;

        Ok(first_new)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::node::{ChildSlot, FLAG_HAS_VALUE, MAX_CHILDREN, MAX_SEG_LEN};

    fn make_child(segment: &[u8], child_page: u64) -> ChildSlot {
        let mut seg_bytes = [0u8; MAX_SEG_LEN];
        seg_bytes[..segment.len()].copy_from_slice(segment);
        ChildSlot {
            seg_len: segment.len() as u16,
            seg_bytes,
            child_page,
        }
    }

    #[test]
    fn create_and_reopen() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        {
            let store = PageStore::open(&path).unwrap();
            // After create: header at page 0, root at page 1, GROW_BATCH pages total.
            assert_eq!(store.total_pages, GROW_BATCH);

            // Root page should be an empty interior node.
            let root = store.read_node(1).unwrap();
            assert_eq!(root.children.len(), 0);
            assert!(!root.is_terminal());
        }

        // Reopen and confirm the header validates correctly.
        let store = PageStore::open(&path).unwrap();
        assert_eq!(store.total_pages, GROW_BATCH);
        let root = store.read_node(1).unwrap();
        assert_eq!(root.children.len(), 0);
    }

    #[test]
    fn write_and_read_node() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut store = PageStore::open(tmp.path()).unwrap();

        let mut node = TrieNode::new();
        node.children.push(make_child(b"seg_a", 2));
        node.children.push(make_child(b"seg_b", 3));
        store.write_node(1, &node).unwrap();

        let decoded = store.read_node(1).unwrap();
        assert_eq!(decoded.children.len(), 2);
        assert_eq!(decoded.children[0].segment(), b"seg_a");
        assert_eq!(decoded.children[1].segment(), b"seg_b");
    }

    #[test]
    fn alloc_page_uses_free_list() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut store = PageStore::open(tmp.path()).unwrap();

        // Pages 2..GROW_BATCH are on the free list after create_new.
        let p1 = store.alloc_page().unwrap();
        let p2 = store.alloc_page().unwrap();
        assert_ne!(p1, p2);
        assert!(p1 >= 2);
        assert!(p2 >= 2);

        // Write nodes to the allocated pages.
        store.write_node(p1, &TrieNode::new()).unwrap();
        store.write_node(p2, &TrieNode::new_terminal()).unwrap();

        let n1 = store.read_node(p1).unwrap();
        let n2 = store.read_node(p2).unwrap();
        assert!(!n1.is_terminal());
        assert!(n2.is_terminal());
    }

    #[test]
    fn free_and_reallocate() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut store = PageStore::open(tmp.path()).unwrap();

        let idx = store.alloc_page().unwrap();
        store.write_node(idx, &TrieNode::new()).unwrap();

        store.free_page(idx).unwrap();

        // The very next alloc should hand back the same page (LIFO free list).
        let reused = store.alloc_page().unwrap();
        assert_eq!(reused, idx);
    }

    #[test]
    fn grow_on_exhausted_free_list() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut store = PageStore::open(tmp.path()).unwrap();

        // Drain the entire free list (GROW_BATCH − 2 pages: 2..GROW_BATCH).
        let free_count = GROW_BATCH - 2;
        let mut pages = Vec::new();
        for _ in 0..free_count {
            pages.push(store.alloc_page().unwrap());
        }
        // Free list is now empty; next alloc must trigger grow().
        let extra = store.alloc_page().unwrap();
        assert!(extra >= GROW_BATCH); // came from the grown region

        // Total pages should have grown by GROW_BATCH.
        assert_eq!(store.total_pages, 2 * GROW_BATCH);

        // Write and read back the new page.
        store.write_node(extra, &TrieNode::new_terminal()).unwrap();
        assert!(store.read_node(extra).unwrap().is_terminal());
    }

    #[test]
    fn persist_across_reopen() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let written_page;
        {
            let mut store = PageStore::open(&path).unwrap();

            let mut root = TrieNode::new();
            root.children.push(make_child(b"alpha", 2));
            root.children.push(make_child(b"beta", 3));
            store.write_node(1, &root).unwrap();

            written_page = store.alloc_page().unwrap();
            let mut leaf = TrieNode::new_terminal();
            leaf.flags |= FLAG_HAS_VALUE;
            store.write_node(written_page, &leaf).unwrap();

            store.flush().unwrap();
        }

        // Re-open and verify everything survived.
        {
            let store = PageStore::open(&path).unwrap();

            let root = store.read_node(1).unwrap();
            assert_eq!(root.children.len(), 2);
            assert_eq!(root.children[0].segment(), b"alpha");
            assert_eq!(root.children[0].child_page, 2);
            assert_eq!(root.children[1].segment(), b"beta");
            assert_eq!(root.children[1].child_page, 3);

            let leaf = store.read_node(written_page).unwrap();
            assert!(leaf.is_terminal());
        }
    }

    #[test]
    fn round_trip_all_segment_lengths() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut store = PageStore::open(tmp.path()).unwrap();

        // Write a node whose children have segments of lengths 0, 1, 127, 255, 256.
        let mut node = TrieNode::new();
        for &len in &[0usize, 1, 127, 255, MAX_SEG_LEN] {
            let seg: Vec<u8> = (0..len).map(|i| i as u8).collect();
            node.children.push(make_child(&seg, len as u64 + 10));
        }

        let page_idx = store.alloc_page().unwrap();
        store.write_node(page_idx, &node).unwrap();

        let decoded = store.read_node(page_idx).unwrap();
        assert_eq!(decoded.children.len(), 5);
        for (i, &len) in [0usize, 1, 127, 255, MAX_SEG_LEN].iter().enumerate() {
            let expected: Vec<u8> = (0..len).map(|j| j as u8).collect();
            assert_eq!(decoded.children[i].segment(), expected.as_slice());
            assert_eq!(decoded.children[i].child_page, len as u64 + 10);
        }
    }

    #[test]
    fn max_children_written_and_read_from_store() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut store = PageStore::open(tmp.path()).unwrap();

        let mut node = TrieNode::new();
        for i in 0..MAX_CHILDREN {
            node.children
                .push(make_child(&[i as u8; 64], (i + 2) as u64));
        }

        let idx = store.alloc_page().unwrap();
        store.write_node(idx, &node).unwrap();

        let decoded = store.read_node(idx).unwrap();
        assert_eq!(decoded.children.len(), MAX_CHILDREN);
        for i in 0..MAX_CHILDREN {
            assert_eq!(decoded.children[i].segment(), &[i as u8; 64]);
        }
    }
}
