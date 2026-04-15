//! The `MappedVarTrie` — the public trie with WAL-protected insert and delete.
//!
//! # Thread safety
//!
//! All mutations are serialised by an internal `Mutex<TrieInner>`.  The mutex
//! is intentionally **not** an `RwLock`: reads (`get`) also acquire it because
//! the value-heap read involves a `seek`, and `grow()` inside the page allocator
//! remaps the mmap, invalidating any concurrent reference into it.
//!
//! # File layout
//!
//! Three files are associated with a database path `{db}`:
//!
//! | File          | Contents                                  |
//! |---------------|-------------------------------------------|
//! | `{db}`        | Node pages (page 0 = header, 1 = root, …) |
//! | `{db}.vals`   | Value heap (append-only, offset-addressed) |
//! | `{db}.wal`    | Write-Ahead Log (present only mid-operation) |
//!
//! # Crash safety protocol
//!
//! 1. **Write WAL** (`fsync`) — records the full intent (segments + value).
//! 2. **Mutate** node pages and the value heap.
//! 3. **Flush** — `msync` the mmap, `fsync` the heap.
//! 4. **Delete WAL** — operation is committed.
//!
//! On `open`, any leftover WAL is replayed before the handle is returned.
//! Both `insert` and `delete` are idempotent under replay:
//! - INSERT: existing child slots are followed, not duplicated; the terminal
//!   node's value fields are unconditionally overwritten.
//! - DELETE: clearing an already-cleared flag is a no-op.
//!
//! # Overflow linked list
//!
//! When a node already holds `MAX_CHILDREN` children and a new distinct segment
//! must be added, a fresh overflow page is allocated, linked via
//! `TrieNode::overflow_page`, and `FLAG_OVERFLOW` is set.  Lookup and insertion
//! traverse the chain transparently.
//!
//! # Empty-node GC linked list
//!
//! When `delete` prunes a node that becomes empty (zero children, no value) the
//! node is added to a trie-level **reuse list** stored in
//! `FileHeader::empty_node_head`.  Subsequent inserts that need a fresh page
//! pop from this list before falling back to the page-level allocator, avoiding
//! file growth.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::{Result, TrieError};
use crate::key::TrieKey;
use crate::node::{
    ChildSlot, TrieNode, FLAG_EMPTY_GC, FLAG_HAS_VALUE, FLAG_OVERFLOW,
    MAX_CHILDREN, MAX_SEG_LEN, NULL_PAGE,
};
use crate::store::PageStore;
use crate::valheap::ValHeap;
use crate::wal::{self, WAL_OP_DELETE, WAL_OP_INSERT};

// ---------------------------------------------------------------------------
// TrieInner — the non-thread-safe core
// ---------------------------------------------------------------------------

struct TrieInner {
    store: PageStore,
    heap: ValHeap,
}

impl TrieInner {
    // -----------------------------------------------------------------------
    // Trie-level allocator
    // -----------------------------------------------------------------------

    /// Allocates a fresh trie node page, preferring the empty-node reuse list.
    ///
    /// If `FileHeader::empty_node_head` is non-null, the head is popped and the
    /// page is re-initialised as an empty node.  Otherwise falls through to the
    /// page-level [`PageStore::alloc_page`].
    fn alloc_trie_node(&mut self) -> Result<u64> {
        let mut hdr = self.store.header()?;
        if hdr.empty_node_head != NULL_PAGE {
            let page = hdr.empty_node_head;
            let gc_node = self.store.read_node(page)?;
            // value_offset holds the next pointer in the GC list.
            hdr.empty_node_head = gc_node.value_offset;
            self.store.write_header(&hdr)?;
            // Re-initialise the reused page as a clean empty node.
            self.store.write_node(page, &TrieNode::new())?;
            Ok(page)
        } else {
            self.store.alloc_page()
        }
    }

    /// Adds `page` to the trie-level empty-node reuse list.
    ///
    /// The page is written with `FLAG_EMPTY_GC` set and `value_offset` pointing
    /// to the previous list head.  Its overflow chain (if any) is freed first.
    fn free_trie_node(&mut self, page: u64) -> Result<()> {
        // Free any overflow pages before recycling the primary page.
        self.free_overflow_chain(page)?;

        let mut hdr = self.store.header()?;
        let old_head = hdr.empty_node_head;

        let mut gc_node = TrieNode::new();
        gc_node.flags = FLAG_EMPTY_GC;
        gc_node.value_offset = old_head; // chain pointer
        self.store.write_node(page, &gc_node)?;

        hdr.empty_node_head = page;
        self.store.write_header(&hdr)?;
        Ok(())
    }

    /// Frees all overflow pages reachable from `start_page.overflow_page`.
    ///
    /// The `start_page` itself is **not** freed; only its overflow chain is.
    fn free_overflow_chain(&mut self, start_page: u64) -> Result<()> {
        let node = self.store.read_node(start_page)?;
        let mut overflow = node.overflow_page;
        while overflow != NULL_PAGE {
            let ov_node = self.store.read_node(overflow)?;
            let next = ov_node.overflow_page;
            // Recursively free this overflow page's own overflow chain before
            // adding it to the GC list.
            let mut hdr = self.store.header()?;
            let old_head = hdr.empty_node_head;
            let mut gc = TrieNode::new();
            gc.flags = FLAG_EMPTY_GC;
            gc.value_offset = old_head;
            self.store.write_node(overflow, &gc)?;
            hdr.empty_node_head = overflow;
            self.store.write_header(&hdr)?;

            overflow = next;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Child traversal helpers
    // -----------------------------------------------------------------------

    /// Finds `segment` in the overflow chain starting at `start_page`.
    ///
    /// Returns `Some(child_page)` on success, `None` if not found.
    fn find_child(store: &PageStore, start_page: u64, segment: &[u8]) -> Result<Option<u64>> {
        let mut page = start_page;
        loop {
            let node = store.read_node(page)?;
            if let Some(slot) = node.children.iter().find(|s| s.segment() == segment) {
                return Ok(Some(slot.child_page));
            }
            if node.overflow_page == NULL_PAGE {
                return Ok(None);
            }
            page = node.overflow_page;
        }
    }

    /// Like [`find_child`] but also returns the host page and slot index.
    ///
    /// Returns `Some((host_page, slot_idx, child_page))`.
    fn find_child_detail(
        store: &PageStore,
        start_page: u64,
        segment: &[u8],
    ) -> Result<Option<(u64, usize, u64)>> {
        let mut page = start_page;
        loop {
            let node = store.read_node(page)?;
            if let Some((idx, slot)) = node
                .children
                .iter()
                .enumerate()
                .find(|(_, s)| s.segment() == segment)
            {
                return Ok(Some((page, idx, slot.child_page)));
            }
            if node.overflow_page == NULL_PAGE {
                return Ok(None);
            }
            page = node.overflow_page;
        }
    }

    /// Counts the total number of children across the overflow chain.
    fn count_children_total(store: &PageStore, start_page: u64) -> Result<usize> {
        let mut total = 0usize;
        let mut page = start_page;
        loop {
            let node = store.read_node(page)?;
            total += node.children.len();
            if node.overflow_page == NULL_PAGE {
                break;
            }
            page = node.overflow_page;
        }
        Ok(total)
    }

    // -----------------------------------------------------------------------
    // Insert
    // -----------------------------------------------------------------------

    /// Follows the overflow chain from `start_page` looking for `segment`.
    ///
    /// If found, returns the existing child page.
    /// If not found, adds the child to the first page with a free slot
    /// (allocating a new overflow page if all pages in the chain are full)
    /// and returns the new child page.
    fn follow_or_create_child(&mut self, start_page: u64, segment: &[u8]) -> Result<u64> {
        let mut page = start_page;
        loop {
            let node = self.store.read_node(page)?;

            // Segment already exists — follow it.
            if let Some(slot) = node.children.iter().find(|s| s.segment() == segment) {
                return Ok(slot.child_page);
            }

            if node.children.len() < MAX_CHILDREN {
                // Room in this page — add the child here.
                let child_page = self.alloc_trie_node()?;
                self.store.write_node(child_page, &TrieNode::new())?;

                // Re-read after potential alloc (mmap may have been remapped).
                let mut node = self.store.read_node(page)?;
                let mut seg_bytes = [0u8; MAX_SEG_LEN];
                seg_bytes[..segment.len()].copy_from_slice(segment);
                node.children.push(ChildSlot {
                    seg_len: segment.len() as u16,
                    seg_bytes,
                    child_page,
                });
                self.store.write_node(page, &node)?;
                return Ok(child_page);
            }

            if node.overflow_page != NULL_PAGE {
                // This page is full; follow the existing overflow link.
                page = node.overflow_page;
            } else {
                // This page is full and has no overflow yet — allocate one.
                let overflow_page = self.alloc_trie_node()?;
                self.store.write_node(overflow_page, &TrieNode::new())?;

                // Link current page → overflow page.
                let mut node = self.store.read_node(page)?;
                node.overflow_page = overflow_page;
                node.flags |= FLAG_OVERFLOW;
                self.store.write_node(page, &node)?;

                page = overflow_page;
                // Continue the loop: add the child in the new overflow page.
            }
        }
    }

    /// Inserts or overwrites the value at the trie path defined by `segments`.
    fn insert(&mut self, segments: &[&[u8]], value: &[u8]) -> Result<()> {
        let root_page = self.store.header()?.root_page;
        let mut current_page = root_page;

        for &segment in segments {
            current_page = self.follow_or_create_child(current_page, segment)?;
        }

        // Write value at the terminal node.
        let mut node = self.store.read_node(current_page)?;
        let was_terminal = node.flags & FLAG_HAS_VALUE != 0;
        let value_offset = self.heap.alloc_value(value)?;
        node.flags = (node.flags & !FLAG_EMPTY_GC) | FLAG_HAS_VALUE;
        node.value_len = value.len() as u16;
        node.value_offset = value_offset;
        self.store.write_node(current_page, &node)?;

        // Increment entry count only when a new terminal is created.
        if !was_terminal {
            let mut hdr = self.store.header()?;
            hdr.entry_count = hdr.entry_count.saturating_add(1);
            self.store.write_header(&hdr)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Delete
    // -----------------------------------------------------------------------

    /// Clears the value at the trie path defined by `segments`, then prunes
    /// any nodes that become empty back up the path.
    ///
    /// Pruned nodes are recycled into the trie-level empty-node reuse list.
    fn delete(&mut self, segments: &[&[u8]]) -> Result<()> {
        let root_page = self.store.header()?.root_page;
        let mut current_page = root_page;

        // Track the path so we can remove child slots and compact overflow
        // chains when pruning upward.
        //
        // `depth_primary` is the first page at that depth (the one reached
        // by following a child slot from the parent).  `host_page` is the
        // page within the overflow chain that actually holds the matching
        // child slot — it equals `depth_primary` unless the key lives in an
        // overflow page.  Using `depth_primary` for `maybe_compact_overflow`
        // ensures we compact the chain that *contains* the (now potentially
        // empty) overflow page, not the overflow page itself.
        struct PathStep {
            depth_primary: u64,
            host_page: u64,
            slot_idx: usize,
        }
        let mut path: Vec<PathStep> = Vec::new();

        for &segment in segments {
            let depth_primary = current_page;
            match Self::find_child_detail(&self.store, current_page, segment)? {
                Some((host_page, slot_idx, child_page)) => {
                    path.push(PathStep { depth_primary, host_page, slot_idx });
                    current_page = child_page;
                }
                None => return Ok(()), // key not present
            }
        }

        // Clear value fields on the terminal node.
        let mut node = self.store.read_node(current_page)?;
        if node.flags & FLAG_HAS_VALUE == 0 {
            return Ok(()); // already absent
        }
        node.flags &= !FLAG_HAS_VALUE;
        node.value_len = 0;
        node.value_offset = 0;
        self.store.write_node(current_page, &node)?;

        // Decrement entry count.
        {
            let mut hdr = self.store.header()?;
            hdr.entry_count = hdr.entry_count.saturating_sub(1);
            self.store.write_header(&hdr)?;
        }

        // Prune empty nodes up the path.
        let mut page_to_prune = current_page;
        for step in path.iter().rev() {
            let pruned_node = self.store.read_node(page_to_prune)?;
            let total_children =
                Self::count_children_total(&self.store, page_to_prune)?;
            if pruned_node.flags & FLAG_HAS_VALUE == 0 && total_children == 0 {
                // Remove the child slot from the host page.
                let mut host_node = self.store.read_node(step.host_page)?;
                host_node.children.remove(step.slot_idx);
                self.store.write_node(step.host_page, &host_node)?;
                // Compact the overflow chain rooted at depth_primary.
                // This handles the case where host_page is itself an overflow
                // page: if it's now empty it will be trimmed from the chain.
                self.maybe_compact_overflow(step.depth_primary)?;
                // Free the pruned node (and its own overflow chain, if any).
                self.free_trie_node(page_to_prune)?;
                // Walk up to the depth-primary for the next pruning check.
                page_to_prune = step.depth_primary;
            } else {
                break; // parent still has data; stop pruning
            }
        }

        Ok(())
    }

    /// Removes trailing empty overflow pages from the chain rooted at
    /// `start_page` and clears `FLAG_OVERFLOW` when the chain becomes empty.
    ///
    /// # Double-free safety
    ///
    /// The chain is re-linked (setting the last kept page's `overflow_page`
    /// to `NULL_PAGE`) **before** any page is freed.  Then each page to be
    /// freed has its own `overflow_page` zeroed before `free_trie_node` is
    /// called, so `free_overflow_chain` inside `free_trie_node` sees a null
    /// chain and does not re-add already-freed pages.
    fn maybe_compact_overflow(&mut self, start_page: u64) -> Result<()> {
        let primary = self.store.read_node(start_page)?;
        if primary.overflow_page == NULL_PAGE {
            return Ok(());
        }

        // Collect the full overflow chain.
        let mut chain: Vec<u64> = Vec::new();
        let mut p = primary.overflow_page;
        while p != NULL_PAGE {
            chain.push(p);
            let n = self.store.read_node(p)?;
            p = n.overflow_page;
        }

        // Determine how many pages to keep (trim empty tail pages).
        let mut keep = chain.len();
        for &page in chain.iter().rev() {
            let n = self.store.read_node(page)?;
            if n.children.is_empty() && n.flags & FLAG_HAS_VALUE == 0 {
                keep -= 1;
            } else {
                break;
            }
        }

        if keep == chain.len() {
            return Ok(()); // nothing to trim
        }

        let to_free = chain[keep..].to_vec();

        // Re-link first: sever the tail from the kept chain, then update
        // start_page.  After this the freed pages are unreachable on disk.
        if keep == 0 {
            let mut node = self.store.read_node(start_page)?;
            node.overflow_page = NULL_PAGE;
            node.flags &= !FLAG_OVERFLOW;
            self.store.write_node(start_page, &node)?;
        } else {
            // Update the last kept page to terminate the chain.
            let last_kept = chain[keep - 1];
            let mut lk = self.store.read_node(last_kept)?;
            lk.overflow_page = NULL_PAGE;
            lk.flags &= !FLAG_OVERFLOW;
            self.store.write_node(last_kept, &lk)?;
            // start_page.overflow_page is unchanged (still chain[0]).
        }

        // Free each unlinked overflow page.  Zero its overflow_page first so
        // free_overflow_chain (called by free_trie_node) sees a null chain and
        // does not follow stale pointers into already-freed pages.
        for &page in &to_free {
            let mut n = self.store.read_node(page)?;
            n.overflow_page = NULL_PAGE;
            n.flags &= !FLAG_OVERFLOW;
            self.store.write_node(page, &n)?;
            self.free_trie_node(page)?;
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Get / Contains
    // -----------------------------------------------------------------------

    /// Returns the value at `segments`, or `None` if the key is absent.
    fn get(&mut self, segments: &[&[u8]]) -> Result<Option<Vec<u8>>> {
        let root_page = self.store.header()?.root_page;
        let mut current_page = root_page;

        for &segment in segments {
            match Self::find_child(&self.store, current_page, segment)? {
                Some(p) => current_page = p,
                None => return Ok(None),
            }
        }

        let node = self.store.read_node(current_page)?;
        if node.flags & FLAG_HAS_VALUE == 0 {
            return Ok(None);
        }
        let val = self.heap.read_value(node.value_offset)?;
        Ok(Some(val))
    }

    /// Returns `true` if `segments` names a key with a stored value.
    fn contains(&self, segments: &[&[u8]]) -> Result<bool> {
        let root_page = self.store.header()?.root_page;
        let mut current_page = root_page;

        for &segment in segments {
            match Self::find_child(&self.store, current_page, segment)? {
                Some(p) => current_page = p,
                None => return Ok(false),
            }
        }

        let node = self.store.read_node(current_page)?;
        Ok(node.flags & FLAG_HAS_VALUE != 0)
    }

    // -----------------------------------------------------------------------
    // Prefix scan
    // -----------------------------------------------------------------------

    /// Collects all key-value pairs in the subtree rooted at `page`.
    ///
    /// `current_prefix` is the path from the trie root to `page`.
    /// Results are appended to `out` as `(full_segments, value)` pairs.
    fn collect_subtree(
        &mut self,
        page: u64,
        current_prefix: &mut Vec<Vec<u8>>,
        out: &mut Vec<(Vec<Vec<u8>>, Vec<u8>)>,
    ) -> Result<()> {
        let node = self.store.read_node(page)?;

        if node.flags & FLAG_HAS_VALUE != 0 {
            let val = self.heap.read_value(node.value_offset)?;
            out.push((current_prefix.clone(), val));
        }

        // Collect all children across the overflow chain first (avoids borrow
        // conflicts when we recurse into `self`).
        let mut children: Vec<(Vec<u8>, u64)> = Vec::new();
        let mut chain_page = page;
        loop {
            let chain_node = self.store.read_node(chain_page)?;
            for child in &chain_node.children {
                children.push((child.segment().to_vec(), child.child_page));
            }
            let next = chain_node.overflow_page;
            if next == NULL_PAGE {
                break;
            }
            chain_page = next;
        }

        for (seg, child_page) in children {
            current_prefix.push(seg);
            self.collect_subtree(child_page, current_prefix, out)?;
            current_prefix.pop();
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MappedVarTrie — the public, thread-safe handle
// ---------------------------------------------------------------------------

/// A persistent, memory-mapped trie whose keys are sequences of byte segments.
///
/// Create or reopen with [`MappedVarTrie::open`].  All public methods take
/// `&self` and acquire the internal mutex — the handle can be shared via `Arc`.
///
/// # Generic API
///
/// For types that implement [`TrieKey`], the `_key` suffixed methods
/// (`insert_key`, `get_key`, `delete_key`, `contains_key_t`, `prefix_scan_key`)
/// accept the typed key directly.  The raw `&[&[u8]]`-based methods are always
/// available.
pub struct MappedVarTrie {
    inner: Mutex<TrieInner>,
    db_path: PathBuf,
}

impl MappedVarTrie {
    /// Opens or creates the trie at `path`.
    ///
    /// Associated files (`{path}.vals`, `{path}.wal`) are created or opened
    /// alongside the main node file.  Any leftover WAL from a previous
    /// crash is replayed before this call returns.
    pub fn open(path: &Path) -> Result<Self> {
        let vals_path = vals_path(path);

        let mut store = PageStore::open(path)?;
        let mut heap = ValHeap::open(&vals_path)?;

        // Replay any pending WAL before handing out a handle.
        if let Some(record) = wal::read_existing(path)? {
            let seg_refs: Vec<&[u8]> = record.segments.iter().map(|s| s.as_slice()).collect();
            let mut inner = TrieInner { store, heap };
            match record.op {
                WAL_OP_INSERT => inner.insert(&seg_refs, &record.value)?,
                WAL_OP_DELETE => inner.delete(&seg_refs)?,
                op => {
                    return Err(TrieError::Corruption(format!(
                        "unknown WAL op: {op:#04x}"
                    )))
                }
            }
            inner.store.flush()?;
            inner.heap.flush()?;
            wal::delete(path)?;
            store = inner.store;
            heap = inner.heap;
        }

        Ok(MappedVarTrie {
            inner: Mutex::new(TrieInner { store, heap }),
            db_path: path.to_path_buf(),
        })
    }

    // -----------------------------------------------------------------------
    // Raw segment API
    // -----------------------------------------------------------------------

    /// Inserts or overwrites the value at the path described by `segments`.
    ///
    /// Each segment may be at most [`MAX_SEG_LEN`] (256) bytes.  `value` may be
    /// up to `u16::MAX` bytes.  Nodes with more than [`MAX_CHILDREN`] children
    /// automatically grow an overflow linked list.
    pub fn insert(&self, segments: &[&[u8]], value: &[u8]) -> Result<()> {
        for &seg in segments {
            if seg.len() > MAX_SEG_LEN {
                return Err(TrieError::SegmentTooLong);
            }
        }

        wal::write_insert(&self.db_path, segments, value)?;

        let mut inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;
        inner.insert(segments, value)?;

        inner.store.flush()?;
        inner.heap.flush()?;
        wal::delete(&self.db_path)?;

        Ok(())
    }

    /// Deletes the value at the path described by `segments`.
    ///
    /// Empty nodes left after deletion are recycled into the reuse list.
    /// If the key is not present this is a no-op.
    pub fn delete(&self, segments: &[&[u8]]) -> Result<()> {
        for &seg in segments {
            if seg.len() > MAX_SEG_LEN {
                return Err(TrieError::SegmentTooLong);
            }
        }

        wal::write_delete(&self.db_path, segments)?;

        let mut inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;
        inner.delete(segments)?;

        inner.store.flush()?;
        wal::delete(&self.db_path)?;

        Ok(())
    }

    /// Returns the value at `segments`, or `None` if the key is absent.
    pub fn get(&self, segments: &[&[u8]]) -> Result<Option<Vec<u8>>> {
        let mut inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;
        inner.get(segments)
    }

    /// Returns `true` if a value is stored at `segments`.
    pub fn contains_key(&self, segments: &[&[u8]]) -> Result<bool> {
        let inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;
        inner.contains(segments)
    }

    /// Returns the number of key-value pairs currently stored.
    ///
    /// The count is maintained incrementally and may be slightly off after a
    /// crash (the WAL protects key integrity; the count is best-effort).
    pub fn len(&self) -> Result<u64> {
        let inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;
        let hdr = inner.store.header()?;
        Ok(hdr.entry_count)
    }

    /// Returns `true` if no key-value pairs are stored.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Collects all key-value pairs whose key starts with `prefix`.
    ///
    /// Returns a `Vec` of `(full_segments, value)` pairs.  `full_segments`
    /// includes the prefix segments followed by the suffix path.
    pub fn prefix_scan(
        &self,
        prefix: &[&[u8]],
    ) -> Result<Vec<(Vec<Vec<u8>>, Vec<u8>)>> {
        let mut inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;

        // Navigate to the node at the end of the prefix.
        let root_page = inner.store.header()?.root_page;
        let mut current_page = root_page;
        let mut prefix_segs: Vec<Vec<u8>> =
            prefix.iter().map(|s| s.to_vec()).collect();

        for &segment in prefix {
            match TrieInner::find_child(&inner.store, current_page, segment)? {
                Some(p) => current_page = p,
                None => return Ok(vec![]), // prefix not present
            }
        }

        let mut results = Vec::new();
        inner.collect_subtree(current_page, &mut prefix_segs, &mut results)?;
        Ok(results)
    }

    // -----------------------------------------------------------------------
    // Generic TrieKey API
    // -----------------------------------------------------------------------

    /// Inserts using a [`TrieKey`]-typed key.
    pub fn insert_key<K: TrieKey>(&self, key: &K, value: &[u8]) -> Result<()> {
        let owned: Vec<K::Segment> = key.segments().collect();
        let segs: Vec<&[u8]> = owned.iter().map(|s| s.as_ref()).collect();
        self.insert(&segs, value)
    }

    /// Retrieves using a [`TrieKey`]-typed key.
    pub fn get_key<K: TrieKey>(&self, key: &K) -> Result<Option<Vec<u8>>> {
        let owned: Vec<K::Segment> = key.segments().collect();
        let segs: Vec<&[u8]> = owned.iter().map(|s| s.as_ref()).collect();
        self.get(&segs)
    }

    /// Deletes using a [`TrieKey`]-typed key.
    pub fn delete_key<K: TrieKey>(&self, key: &K) -> Result<()> {
        let owned: Vec<K::Segment> = key.segments().collect();
        let segs: Vec<&[u8]> = owned.iter().map(|s| s.as_ref()).collect();
        self.delete(&segs)
    }

    /// `contains_key` using a [`TrieKey`]-typed key.
    pub fn contains_key_t<K: TrieKey>(&self, key: &K) -> Result<bool> {
        let owned: Vec<K::Segment> = key.segments().collect();
        let segs: Vec<&[u8]> = owned.iter().map(|s| s.as_ref()).collect();
        self.contains_key(&segs)
    }

    /// Prefix scan using a [`TrieKey`]-typed prefix.
    pub fn prefix_scan_key<K: TrieKey>(
        &self,
        prefix: &K,
    ) -> Result<Vec<(Vec<Vec<u8>>, Vec<u8>)>> {
        let owned: Vec<K::Segment> = prefix.segments().collect();
        let segs: Vec<&[u8]> = owned.iter().map(|s| s.as_ref()).collect();
        self.prefix_scan(&segs)
    }
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn vals_path(db_path: &Path) -> PathBuf {
    let mut s = db_path.as_os_str().to_os_string();
    s.push(".vals");
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn open(dir: &tempfile::TempDir) -> MappedVarTrie {
        MappedVarTrie::open(&dir.path().join("trie.db")).unwrap()
    }

    fn reopen(dir: &tempfile::TempDir) -> MappedVarTrie {
        MappedVarTrie::open(&dir.path().join("trie.db")).unwrap()
    }

    // -----------------------------------------------------------------------
    // Basic insert / get / delete
    // -----------------------------------------------------------------------

    #[test]
    fn insert_and_get_single_segment() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"hello"], b"world").unwrap();
        assert_eq!(t.get(&[b"hello"]).unwrap(), Some(b"world".to_vec()));
    }

    #[test]
    fn insert_and_get_multi_segment() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"a", b"b", b"c"], b"leaf_value").unwrap();
        assert_eq!(
            t.get(&[b"a", b"b", b"c"]).unwrap(),
            Some(b"leaf_value".to_vec())
        );
        assert_eq!(t.get(&[b"a", b"b"]).unwrap(), None);
        assert_eq!(t.get(&[b"a"]).unwrap(), None);
    }

    #[test]
    fn get_absent_key_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);
        assert_eq!(t.get(&[b"missing"]).unwrap(), None);
    }

    #[test]
    fn insert_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"k"], b"v1").unwrap();
        t.insert(&[b"k"], b"v2").unwrap();
        assert_eq!(t.get(&[b"k"]).unwrap(), Some(b"v2".to_vec()));
    }

    #[test]
    fn delete_clears_value() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"k"], b"v").unwrap();
        t.delete(&[b"k"]).unwrap();
        assert_eq!(t.get(&[b"k"]).unwrap(), None);
    }

    #[test]
    fn delete_absent_key_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);
        t.delete(&[b"nope"]).unwrap();
    }

    #[test]
    fn delete_prefix_does_not_affect_longer_key() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"a", b"b"], b"ab_val").unwrap();
        t.delete(&[b"a"]).unwrap(); // "a" has no value, no-op

        assert_eq!(
            t.get(&[b"a", b"b"]).unwrap(),
            Some(b"ab_val".to_vec())
        );
    }

    // -----------------------------------------------------------------------
    // Multiple keys, shared prefixes
    // -----------------------------------------------------------------------

    #[test]
    fn shared_prefix_keys() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"com", b"example", b"www"], b"web").unwrap();
        t.insert(&[b"com", b"example", b"mail"], b"email").unwrap();
        t.insert(&[b"com", b"other"], b"other").unwrap();

        assert_eq!(
            t.get(&[b"com", b"example", b"www"]).unwrap(),
            Some(b"web".to_vec())
        );
        assert_eq!(
            t.get(&[b"com", b"example", b"mail"]).unwrap(),
            Some(b"email".to_vec())
        );
        assert_eq!(
            t.get(&[b"com", b"other"]).unwrap(),
            Some(b"other".to_vec())
        );
        assert_eq!(t.get(&[b"com", b"example"]).unwrap(), None);
    }

    // -----------------------------------------------------------------------
    // Persistence
    // -----------------------------------------------------------------------

    #[test]
    fn persist_across_reopen() {
        let dir = tempfile::tempdir().unwrap();

        {
            let t = open(&dir);
            t.insert(&[b"x", b"y"], b"val_xy").unwrap();
            t.insert(&[b"x", b"z"], b"val_xz").unwrap();
        }

        let t = reopen(&dir);
        assert_eq!(t.get(&[b"x", b"y"]).unwrap(), Some(b"val_xy".to_vec()));
        assert_eq!(t.get(&[b"x", b"z"]).unwrap(), Some(b"val_xz".to_vec()));
    }

    #[test]
    fn delete_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();

        {
            let t = open(&dir);
            t.insert(&[b"k"], b"v").unwrap();
            t.delete(&[b"k"]).unwrap();
        }

        let t = reopen(&dir);
        assert_eq!(t.get(&[b"k"]).unwrap(), None);
    }

    // -----------------------------------------------------------------------
    // WAL crash recovery simulation
    // -----------------------------------------------------------------------

    #[test]
    fn recovery_insert_wal_present() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");

        {
            MappedVarTrie::open(&db_path).unwrap();
        }

        wal::write_insert(&db_path, &[b"crash_key"], b"crash_val").unwrap();

        let t = MappedVarTrie::open(&db_path).unwrap();
        assert_eq!(
            t.get(&[b"crash_key"]).unwrap(),
            Some(b"crash_val".to_vec())
        );
        assert!(!wal::wal_path(&db_path).exists());
    }

    #[test]
    fn recovery_delete_wal_present() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");

        {
            let t = MappedVarTrie::open(&db_path).unwrap();
            t.insert(&[b"dk"], b"dv").unwrap();
        }

        wal::write_delete(&db_path, &[b"dk"]).unwrap();

        let t = MappedVarTrie::open(&db_path).unwrap();
        assert_eq!(t.get(&[b"dk"]).unwrap(), None);
        assert!(!wal::wal_path(&db_path).exists());
    }

    #[test]
    fn recovery_insert_idempotent_with_existing_node() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");

        {
            let t = MappedVarTrie::open(&db_path).unwrap();
            t.insert(&[b"a", b"b"], b"v1").unwrap();
        }

        wal::write_insert(&db_path, &[b"a", b"b"], b"v2").unwrap();

        let t = MappedVarTrie::open(&db_path).unwrap();
        assert_eq!(t.get(&[b"a", b"b"]).unwrap(), Some(b"v2".to_vec()));

        {
            let inner = t.inner.lock().unwrap();
            let root = inner.store.read_node(1).unwrap();
            assert_eq!(root.children.len(), 1);
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn empty_key_zero_segments() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[], b"root_val").unwrap();
        assert_eq!(t.get(&[]).unwrap(), Some(b"root_val".to_vec()));

        t.delete(&[]).unwrap();
        assert_eq!(t.get(&[]).unwrap(), None);
    }

    #[test]
    fn large_value() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        let big = vec![0xBEu8; 60_000];
        t.insert(&[b"big"], &big).unwrap();
        assert_eq!(t.get(&[b"big"]).unwrap(), Some(big));
    }

    #[test]
    fn max_segment_length() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        let seg = [0xFFu8; MAX_SEG_LEN];
        t.insert(&[&seg], b"max_seg_val").unwrap();
        assert_eq!(
            t.get(&[&seg]).unwrap(),
            Some(b"max_seg_val".to_vec())
        );
    }

    #[test]
    fn segment_too_long_rejected_before_wal() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");
        let t = MappedVarTrie::open(&db_path).unwrap();

        let too_long = vec![0u8; MAX_SEG_LEN + 1];
        let result = t.insert(&[&too_long], b"v");
        assert!(matches!(result, Err(TrieError::SegmentTooLong)));
        assert!(!wal::wal_path(&db_path).exists());
    }

    #[test]
    fn many_inserts_and_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        for i in 0u8..10 {
            t.insert(&[&[i]], &[i * 10]).unwrap();
        }
        for i in 0u8..10 {
            assert_eq!(t.get(&[&[i]]).unwrap(), Some(vec![i * 10]));
        }

        for i in (0u8..10).step_by(2) {
            t.delete(&[&[i]]).unwrap();
        }
        for i in 0u8..10 {
            let v = t.get(&[&[i]]).unwrap();
            if i % 2 == 0 {
                assert_eq!(v, None, "key {i} should be deleted");
            } else {
                assert_eq!(v, Some(vec![i * 10]), "key {i} should still exist");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Overflow linked list
    // -----------------------------------------------------------------------

    #[test]
    fn overflow_more_than_max_children() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Insert MAX_CHILDREN + 5 keys that all share the root as parent
        // (single-segment keys).  The root node will overflow.
        let n = MAX_CHILDREN + 5;
        for i in 0..n {
            let seg = (i as u8).to_le_bytes();
            t.insert(&[&seg], &[i as u8]).unwrap();
        }

        for i in 0..n {
            let seg = (i as u8).to_le_bytes();
            assert_eq!(
                t.get(&[&seg]).unwrap(),
                Some(vec![i as u8]),
                "missing key {i}"
            );
        }
    }

    #[test]
    fn overflow_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let n = MAX_CHILDREN + 3;

        {
            let t = open(&dir);
            for i in 0..n {
                t.insert(&[&[i as u8]], &[i as u8]).unwrap();
            }
        }

        let t = reopen(&dir);
        for i in 0..n {
            assert_eq!(
                t.get(&[&[i as u8]]).unwrap(),
                Some(vec![i as u8]),
                "key {i} missing after reopen"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Contains, len, is_empty
    // -----------------------------------------------------------------------

    #[test]
    fn contains_key_basic() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        assert!(!t.contains_key(&[b"k"]).unwrap());
        t.insert(&[b"k"], b"v").unwrap();
        assert!(t.contains_key(&[b"k"]).unwrap());
        t.delete(&[b"k"]).unwrap();
        assert!(!t.contains_key(&[b"k"]).unwrap());
    }

    #[test]
    fn len_and_is_empty() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        assert!(t.is_empty().unwrap());
        assert_eq!(t.len().unwrap(), 0);

        t.insert(&[b"a"], b"1").unwrap();
        assert_eq!(t.len().unwrap(), 1);
        assert!(!t.is_empty().unwrap());

        t.insert(&[b"b"], b"2").unwrap();
        assert_eq!(t.len().unwrap(), 2);

        t.delete(&[b"a"]).unwrap();
        assert_eq!(t.len().unwrap(), 1);

        t.delete(&[b"b"]).unwrap();
        assert_eq!(t.len().unwrap(), 0);
        assert!(t.is_empty().unwrap());
    }

    #[test]
    fn len_not_affected_by_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"k"], b"v1").unwrap();
        assert_eq!(t.len().unwrap(), 1);
        t.insert(&[b"k"], b"v2").unwrap(); // overwrite, not a new key
        assert_eq!(t.len().unwrap(), 1);
    }

    // -----------------------------------------------------------------------
    // Prefix scan
    // -----------------------------------------------------------------------

    #[test]
    fn prefix_scan_basic() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"a", b"x"], b"ax").unwrap();
        t.insert(&[b"a", b"y"], b"ay").unwrap();
        t.insert(&[b"b", b"z"], b"bz").unwrap();

        let mut results = t.prefix_scan(&[b"a"]).unwrap();
        results.sort_by(|a, b| a.0.cmp(&b.0));
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, b"ax");
        assert_eq!(results[1].1, b"ay");
    }

    #[test]
    fn prefix_scan_empty_prefix_returns_all() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"a"], b"va").unwrap();
        t.insert(&[b"b"], b"vb").unwrap();
        t.insert(&[b"c"], b"vc").unwrap();

        let results = t.prefix_scan(&[]).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn prefix_scan_absent_prefix_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"a"], b"va").unwrap();
        let results = t.prefix_scan(&[b"z"]).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Empty-node GC reuse
    // -----------------------------------------------------------------------

    #[test]
    fn empty_node_gc_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Insert then delete a key to populate the GC list.
        t.insert(&[b"tmp"], b"v").unwrap();
        let page_count_before = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };
        t.delete(&[b"tmp"]).unwrap();

        // Inserting a fresh key should reuse the GC page, not grow the file.
        t.insert(&[b"new"], b"w").unwrap();
        let page_count_after = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };

        assert_eq!(
            page_count_before, page_count_after,
            "no new pages should have been allocated"
        );
        assert_eq!(t.get(&[b"new"]).unwrap(), Some(b"w".to_vec()));
    }

    // -----------------------------------------------------------------------
    // TrieKey generic API
    // -----------------------------------------------------------------------

    struct StrKey(Vec<String>);

    impl TrieKey for StrKey {
        type Segment = Vec<u8>;
        type SegmentIter<'a> = std::iter::Map<std::slice::Iter<'a, String>, fn(&String) -> Vec<u8>>;

        fn segments(&self) -> Self::SegmentIter<'_> {
            self.0.iter().map(|s| s.as_bytes().to_vec())
        }
    }

    #[test]
    fn trie_key_insert_get_delete() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        let key = StrKey(vec!["hello".into(), "world".into()]);
        t.insert_key(&key, b"greet").unwrap();
        assert_eq!(t.get_key(&key).unwrap(), Some(b"greet".to_vec()));
        assert!(t.contains_key_t(&key).unwrap());
        t.delete_key(&key).unwrap();
        assert!(!t.contains_key_t(&key).unwrap());
    }

    // -----------------------------------------------------------------------
    // Overflow — intermediate node
    // -----------------------------------------------------------------------

    /// Overflow at a non-root intermediate node (shared prefix at depth 1).
    #[test]
    fn overflow_at_intermediate_node() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // All keys share the first segment "p"; the "p" node gets >15 children.
        let n = MAX_CHILDREN + 4;
        for i in 0..n {
            t.insert(&[b"p", &[i as u8]], &[i as u8]).unwrap();
        }

        // Every key is readable.
        for i in 0..n {
            assert_eq!(
                t.get(&[b"p", &[i as u8]]).unwrap(),
                Some(vec![i as u8]),
                "intermediate-overflow key p/{i} missing"
            );
        }

        // The "p" node must have FLAG_OVERFLOW.
        {
            let inner = t.inner.lock().unwrap();
            let root = inner.store.read_node(1).unwrap();
            assert_eq!(root.children.len(), 1, "root should have exactly one child (p)");
            let p_node = inner.store.read_node(root.children[0].child_page).unwrap();
            assert!(p_node.is_overflow(), "intermediate 'p' node should overflow");
        }
    }

    // -----------------------------------------------------------------------
    // Overflow — three pages (>30 children)
    // -----------------------------------------------------------------------

    #[test]
    fn overflow_three_pages() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // 3*MAX_CHILDREN + 2 = 47 keys → 4 pages in the chain (15+15+15+2)
        let n = 3 * MAX_CHILDREN + 2;
        for i in 0..n {
            // Use two-byte segments to avoid collisions when n > 255.
            let seg = (i as u16).to_le_bytes();
            t.insert(&[seg.as_ref()], &[i as u8]).unwrap();
        }

        // All keys readable.
        for i in 0..n {
            let seg = (i as u16).to_le_bytes();
            let val = t.get(&[seg.as_ref()]).unwrap();
            assert_eq!(val, Some(vec![i as u8]), "key {i} missing in 3-page overflow");
        }

        assert_eq!(t.len().unwrap(), n as u64);
    }

    // -----------------------------------------------------------------------
    // Overflow — delete causes chain compaction
    // -----------------------------------------------------------------------

    /// Deleting the single key stored in an overflow page should compact the
    /// chain: the terminal child is pruned, the overflow page becomes empty
    /// and is trimmed, and the primary node's FLAG_OVERFLOW is cleared.
    #[test]
    fn overflow_chain_compacted_on_last_delete() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // MAX_CHILDREN + 1 keys → root overflow_page has exactly one child.
        let n = MAX_CHILDREN + 1;
        for i in 0..n {
            t.insert(&[&[i as u8]], &[i as u8]).unwrap();
        }

        // Delete the key stored in the overflow page.
        let overflow_key = (n - 1) as u8;
        t.delete(&[&[overflow_key]]).unwrap();

        // Overflow page and the terminal node for overflow_key were both
        // pruned; the root should have no overflow chain left.
        {
            let inner = t.inner.lock().unwrap();
            let root = inner.store.read_node(1).unwrap();
            assert!(
                !root.is_overflow(),
                "FLAG_OVERFLOW should be cleared after overflow page is emptied"
            );
            assert_eq!(root.overflow_page, 0, "overflow_page should be NULL after compaction");
            // The root's GC list should now contain the freed pages.
            let hdr = inner.store.header().unwrap();
            assert_ne!(hdr.empty_node_head, 0, "GC list should have freed pages");
        }

        // The remaining keys should all still be accessible.
        for i in 0..(n - 1) {
            assert_eq!(
                t.get(&[&[i as u8]]).unwrap(),
                Some(vec![i as u8]),
                "key {i} should survive after overflow compaction"
            );
        }
        assert_eq!(t.get(&[&[overflow_key]]).unwrap(), None);
    }

    /// Delete all children from TWO consecutive overflow pages; verifies the
    /// double-free-safe compaction path (the original bug).
    #[test]
    fn overflow_compact_two_tail_pages() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Insert 2*MAX_CHILDREN + 2 keys → root has 3 pages in overflow chain.
        let n = 2 * MAX_CHILDREN + 2;
        for i in 0..n {
            let seg = (i as u16).to_le_bytes();
            t.insert(&[seg.as_ref()], &[i as u8]).unwrap();
        }

        // Delete the last MAX_CHILDREN + 2 keys (the last two overflow pages).
        for i in MAX_CHILDREN..n {
            let seg = (i as u16).to_le_bytes();
            t.delete(&[seg.as_ref()]).unwrap();
        }

        // Root chain should now be compacted to a single overflow page (or none).
        // Keys in the primary root page (0..MAX_CHILDREN-1) should still work.
        for i in 0..MAX_CHILDREN {
            let seg = (i as u16).to_le_bytes();
            assert_eq!(
                t.get(&[seg.as_ref()]).unwrap(),
                Some(vec![i as u8]),
                "key {i} should survive double-tail compaction"
            );
        }
        assert_eq!(t.len().unwrap(), MAX_CHILDREN as u64);
    }

    // -----------------------------------------------------------------------
    // Overflow — delete from overflow node in a deeper trie
    // -----------------------------------------------------------------------

    /// The key's path passes through an overflow page at an intermediate level.
    /// After deletion the path is pruned and the overflow chain at that level
    /// is compacted.
    #[test]
    fn overflow_delete_via_overflow_page() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Fill the root with MAX_CHILDREN + 1 children; the overflow page
        // holds child MAX_CHILDREN.  Each child has its own terminal value.
        let n = MAX_CHILDREN + 1;
        for i in 0..n {
            t.insert(&[&[i as u8]], &[i as u8]).unwrap();
        }

        // The overflow-page child also has sub-children to prevent pruning.
        let overflow_child: u8 = (n - 1) as u8;
        t.insert(&[&[overflow_child], b"sub"], b"sub_val").unwrap();

        // Delete the terminal value of the overflow child (not the sub-child).
        t.delete(&[&[overflow_child]]).unwrap();

        // The overflow child should still exist (it has a sub-child).
        assert_eq!(
            t.get(&[&[overflow_child], b"sub"]).unwrap(),
            Some(b"sub_val".to_vec())
        );
        // But its own value is gone.
        assert_eq!(t.get(&[&[overflow_child]]).unwrap(), None);

        // The overflow chain on the root should still exist (overflow_child
        // node is not empty — it still has a sub-child).
        {
            let inner = t.inner.lock().unwrap();
            let root = inner.store.read_node(1).unwrap();
            assert!(root.is_overflow(), "overflow chain should still exist");
        }
    }

    // -----------------------------------------------------------------------
    // Empty-node GC — persistence across reopen
    // -----------------------------------------------------------------------

    #[test]
    fn gc_list_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();

        // Insert then delete so the GC list is non-empty.
        {
            let t = open(&dir);
            t.insert(&[b"x"], b"v").unwrap();
            t.delete(&[b"x"]).unwrap();
        }

        // After reopen the GC list head should still be set.
        {
            let t = reopen(&dir);
            let inner = t.inner.lock().unwrap();
            let hdr = inner.store.header().unwrap();
            assert_ne!(hdr.empty_node_head, 0, "GC list head must persist across reopen");
        }
    }

    #[test]
    fn gc_list_multiple_entries_reused_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Insert 3 single-segment keys (each allocates one terminal page).
        t.insert(&[b"a"], b"1").unwrap();
        t.insert(&[b"b"], b"2").unwrap();
        t.insert(&[b"c"], b"3").unwrap();

        let pages_before = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };

        // Delete all three — 3 terminal nodes + 3 "a/b/c" child nodes enter GC.
        t.delete(&[b"a"]).unwrap();
        t.delete(&[b"b"]).unwrap();
        t.delete(&[b"c"]).unwrap();

        // Each single-segment key allocates exactly one node (the child is also
        // the terminal), so deleting 3 keys leaves 3 nodes in the GC list.
        let gc_count_after_delete = {
            let inner = t.inner.lock().unwrap();
            let mut count = 0u32;
            let hdr = inner.store.header().unwrap();
            let mut p = hdr.empty_node_head;
            while p != 0 {
                count += 1;
                let n = inner.store.read_node(p).unwrap();
                p = n.value_offset; // next pointer
                assert!(count < 100, "GC list cycle detected");
            }
            count
        };
        assert_eq!(gc_count_after_delete, 3, "3 child/terminal nodes should be in GC");

        // Re-inserting 3 new keys should reuse GC pages without growing file.
        t.insert(&[b"d"], b"4").unwrap();
        t.insert(&[b"e"], b"5").unwrap();
        t.insert(&[b"f"], b"6").unwrap();

        let pages_after_reinsert = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };
        assert_eq!(pages_before, pages_after_reinsert, "file should not grow when GC list has entries");
    }

    // -----------------------------------------------------------------------
    // Deep path pruning
    // -----------------------------------------------------------------------

    #[test]
    fn deep_path_pruning_frees_all_nodes() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // 10-segment key → allocates one node per segment level.
        let segs: [&[u8]; 10] = [
            b"s0", b"s1", b"s2", b"s3", b"s4",
            b"s5", b"s6", b"s7", b"s8", b"s9",
        ];
        t.insert(&segs, b"deep").unwrap();

        let pages_after_insert = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };

        t.delete(&segs).unwrap();

        // GC list should now be non-empty.
        {
            let inner = t.inner.lock().unwrap();
            let hdr = inner.store.header().unwrap();
            assert_ne!(hdr.empty_node_head, 0, "GC list must be non-empty after deep pruning");
            // entry_count should be 0.
            assert_eq!(hdr.entry_count, 0);
        }

        // Inserting the same key again must succeed and reuse GC pages.
        t.insert(&segs, b"reborn").unwrap();
        let pages_after_reinsert = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };
        assert_eq!(
            pages_after_insert, pages_after_reinsert,
            "re-inserting a pruned deep key should reuse GC nodes"
        );
        assert_eq!(t.get(&segs).unwrap(), Some(b"reborn".to_vec()));
    }

    // -----------------------------------------------------------------------
    // Root-level value does not affect child pruning
    // -----------------------------------------------------------------------

    #[test]
    fn root_value_prevents_root_pruning() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Insert value at root and at a child.
        t.insert(&[], b"root_v").unwrap();
        t.insert(&[b"child"], b"child_v").unwrap();

        // Delete only the child — root still has its value, not pruned.
        t.delete(&[b"child"]).unwrap();

        assert_eq!(t.get(&[]).unwrap(), Some(b"root_v".to_vec()));
        assert_eq!(t.len().unwrap(), 1);

        // Delete root value too.
        t.delete(&[]).unwrap();
        assert_eq!(t.get(&[]).unwrap(), None);
        assert_eq!(t.len().unwrap(), 0);

        // Root page is always page 1 — must never be in the GC list.
        {
            let inner = t.inner.lock().unwrap();
            let hdr = inner.store.header().unwrap();
            let mut p = hdr.empty_node_head;
            while p != 0 {
                assert_ne!(p, 1, "root page must never be in the GC list");
                let n = inner.store.read_node(p).unwrap();
                p = n.value_offset;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Contains and len with overflow
    // -----------------------------------------------------------------------

    #[test]
    fn contains_key_finds_all_overflow_slots() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        let n = MAX_CHILDREN + 6;
        for i in 0..n {
            t.insert(&[&[i as u8]], &[i as u8]).unwrap();
        }

        for i in 0..n {
            assert!(t.contains_key(&[&[i as u8]]).unwrap(), "key {i} not found");
        }
        assert!(!t.contains_key(&[&[n as u8]]).unwrap());
    }

    #[test]
    fn len_tracks_overflow_inserts_and_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        let n = MAX_CHILDREN + 3;
        for i in 0..n {
            t.insert(&[&[i as u8]], &[i as u8]).unwrap();
        }
        assert_eq!(t.len().unwrap(), n as u64);

        // Delete keys in the overflow range.
        for i in MAX_CHILDREN..n {
            t.delete(&[&[i as u8]]).unwrap();
        }
        assert_eq!(t.len().unwrap(), MAX_CHILDREN as u64);

        // Re-insert them.
        for i in MAX_CHILDREN..n {
            t.insert(&[&[i as u8]], &[i as u8]).unwrap();
        }
        assert_eq!(t.len().unwrap(), n as u64);
    }

    #[test]
    fn len_persists_across_reopen() {
        let dir = tempfile::tempdir().unwrap();

        {
            let t = open(&dir);
            t.insert(&[b"a"], b"1").unwrap();
            t.insert(&[b"b"], b"2").unwrap();
            t.insert(&[b"c"], b"3").unwrap();
            t.delete(&[b"b"]).unwrap();
        }

        let t = reopen(&dir);
        assert_eq!(t.len().unwrap(), 2);
    }

    // -----------------------------------------------------------------------
    // Prefix scan — thorough
    // -----------------------------------------------------------------------

    #[test]
    fn prefix_scan_includes_prefix_endpoint_value() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // The prefix endpoint itself has a value.
        t.insert(&[b"root"], b"root_val").unwrap();
        t.insert(&[b"root", b"child1"], b"c1_val").unwrap();
        t.insert(&[b"root", b"child2"], b"c2_val").unwrap();

        let results = t.prefix_scan(&[b"root"]).unwrap();
        assert_eq!(results.len(), 3, "scan should include the prefix endpoint itself");

        let values: Vec<&[u8]> = results.iter().map(|(_, v)| v.as_slice()).collect();
        assert!(values.contains(&b"root_val".as_ref()));
        assert!(values.contains(&b"c1_val".as_ref()));
        assert!(values.contains(&b"c2_val".as_ref()));
    }

    #[test]
    fn prefix_scan_with_overflow() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // All keys share the prefix [b"pre"] and have distinct second segments,
        // causing the "pre" node to overflow.
        let n = MAX_CHILDREN + 3;
        for i in 0..n {
            t.insert(&[b"pre", &[i as u8]], &[i as u8]).unwrap();
        }

        let results = t.prefix_scan(&[b"pre"]).unwrap();
        assert_eq!(results.len(), n, "prefix scan should find all overflow children");

        // All segments should be present.
        let mut found: Vec<u8> = results.iter().map(|(segs, _)| segs[1][0]).collect();
        found.sort_unstable();
        let expected: Vec<u8> = (0..n as u8).collect();
        assert_eq!(found, expected);
    }

    #[test]
    fn prefix_scan_deeply_nested() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"a", b"b", b"c"], b"abc").unwrap();
        t.insert(&[b"a", b"b", b"d"], b"abd").unwrap();
        t.insert(&[b"a", b"x"], b"ax").unwrap();

        // Scan from "a/b" — should only see abc and abd.
        let results = t.prefix_scan(&[b"a", b"b"]).unwrap();
        assert_eq!(results.len(), 2);
        let mut vals: Vec<Vec<u8>> = results.into_iter().map(|(_, v)| v).collect();
        vals.sort();
        assert_eq!(vals, vec![b"abc".to_vec(), b"abd".to_vec()]);
    }

    // -----------------------------------------------------------------------
    // Overwrite does not double-count in len
    // -----------------------------------------------------------------------

    #[test]
    fn overwrite_in_overflow_does_not_double_count() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        let n = MAX_CHILDREN + 2;
        for i in 0..n {
            t.insert(&[&[i as u8]], &[i as u8]).unwrap();
        }
        assert_eq!(t.len().unwrap(), n as u64);

        // Overwrite a key in the overflow range — len must stay the same.
        let overflow_key = (n - 1) as u8;
        t.insert(&[&[overflow_key]], b"new_value").unwrap();
        assert_eq!(t.len().unwrap(), n as u64);
        assert_eq!(t.get(&[&[overflow_key]]).unwrap(), Some(b"new_value".to_vec()));
    }

    // -----------------------------------------------------------------------
    // TrieKey prefix_scan_key
    // -----------------------------------------------------------------------

    #[test]
    fn prefix_scan_key_works() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        t.insert(&[b"a", b"b"], b"ab").unwrap();
        t.insert(&[b"a", b"c"], b"ac").unwrap();

        let prefix = StrKey(vec!["a".into()]);
        let results = t.prefix_scan_key(&prefix).unwrap();
        assert_eq!(results.len(), 2);
    }

    // -----------------------------------------------------------------------
    // FileHeader version mismatch rejected
    // -----------------------------------------------------------------------

    #[test]
    fn wrong_version_rejected_on_reopen() {
        use crate::node::{PAGE_SIZE, VERSION};

        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        // Create a valid trie.
        MappedVarTrie::open(&path).unwrap();

        // Manually corrupt the version field (at byte offset 4..6 in page 0).
        {
            use std::io::{Seek, SeekFrom, Write};
            let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&path).unwrap();
            // Rebuild header with version = VERSION + 1.
            let mut hdr_page = [0u8; PAGE_SIZE];
            f.seek(SeekFrom::Start(0)).unwrap();
            std::io::Read::read_exact(&mut f, &mut hdr_page).unwrap();

            // Decode, bump version, re-encode.
            // We write raw bytes: version is at [4..6].
            let bad_version = (VERSION + 1).to_le_bytes();
            hdr_page[4..6].copy_from_slice(&bad_version);
            // Recompute CRC (covers [0..48]).
            let new_crc = crc32fast::hash(&hdr_page[..48]);
            hdr_page[48..52].copy_from_slice(&new_crc.to_le_bytes());

            f.seek(SeekFrom::Start(0)).unwrap();
            f.write_all(&hdr_page).unwrap();
        }

        let result = MappedVarTrie::open(&path);
        assert!(result.is_err(), "opening a file with wrong version should fail");
    }

    // -----------------------------------------------------------------------
    // Stress: many inserts, deletes, re-inserts
    // -----------------------------------------------------------------------

    #[test]
    fn stress_insert_delete_cycle() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Round 1: insert 40 keys.
        for i in 0u8..40 {
            t.insert(&[&[i]], &[i]).unwrap();
        }
        assert_eq!(t.len().unwrap(), 40);

        // Round 2: delete all.
        for i in 0u8..40 {
            t.delete(&[&[i]]).unwrap();
        }
        assert_eq!(t.len().unwrap(), 0);

        // Round 3: insert 40 different keys — should reuse GC pages.
        let pages_before = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };
        for i in 0u8..40 {
            t.insert(&[&[100u8.wrapping_add(i)]], &[i]).unwrap();
        }
        let pages_after = {
            let inner = t.inner.lock().unwrap();
            inner.store.header().unwrap().num_pages
        };
        assert_eq!(pages_before, pages_after, "GC reuse should prevent file growth in round 3");
        assert_eq!(t.len().unwrap(), 40);

        // All round-3 keys accessible.
        for i in 0u8..40 {
            assert_eq!(
                t.get(&[&[100u8.wrapping_add(i)]]).unwrap(),
                Some(vec![i]),
                "round-3 key missing"
            );
        }
    }

    // -----------------------------------------------------------------------
    // WAL replay with overflow-causing insert
    // -----------------------------------------------------------------------

    #[test]
    fn wal_replay_overflow_insert() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");

        // Fill root to MAX_CHILDREN so the next insert will trigger overflow.
        {
            let t = MappedVarTrie::open(&db_path).unwrap();
            for i in 0..MAX_CHILDREN as u8 {
                t.insert(&[&[i]], &[i]).unwrap();
            }
        }

        // Leave a WAL for the overflow-causing insert.
        let overflow_seg: u8 = MAX_CHILDREN as u8;
        wal::write_insert(&db_path, &[&[overflow_seg]], &[overflow_seg]).unwrap();

        // Replay should correctly add the overflow page.
        let t = MappedVarTrie::open(&db_path).unwrap();
        assert_eq!(
            t.get(&[&[overflow_seg]]).unwrap(),
            Some(vec![overflow_seg]),
            "WAL replay should create overflow page"
        );
        assert_eq!(t.len().unwrap(), (MAX_CHILDREN + 1) as u64);
        assert!(!wal::wal_path(&db_path).exists());
    }
}
