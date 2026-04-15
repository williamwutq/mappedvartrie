//! On-disk page format and trie node serialization/deserialization.
//!
//! # File layout
//!
//! ```text
//! Page 0:  FileHeader (PAGE_SIZE bytes; first 52 bytes meaningful)
//! Page 1:  root trie node
//! Page 2+: additional trie nodes or freed pages
//! ```
//!
//! Page index `0` doubles as the null/sentinel value ([`NULL_PAGE`]).
//!
//! # Node page layout (4096 bytes)
//!
//! ```text
//! Offset  Size  Field
//! 0       4     magic: u32  (NODE_MAGIC — identifies this as a trie node page)
//! 4       4     crc32: u32  (CRC32 of the whole page with bytes [4..8] zeroed)
//! 8       1     flags: u8   (FLAG_HAS_VALUE, FLAG_OVERFLOW, FLAG_EMPTY_GC, FLAG_FREE)
//! 9       1     _pad
//! 10      2     n_children: u16
//! 12      2     value_len: u16
//! 14      2     _pad
//! 16      8     value_offset: u64  (heap offset; repurposed as next_empty when
//!                                   FLAG_EMPTY_GC, and as next_free when FLAG_FREE)
//! 24      8     overflow_page: u64 (next overflow node page, or NULL_PAGE)
//! 32      266×n child slots (n ≤ MAX_CHILDREN = 15)
//! ```
//!
//! # Child slot layout (266 bytes, tightly packed)
//!
//! ```text
//! Offset  Size  Field
//! 0       2     seg_len: u16
//! 2       256   seg_bytes: [u8; 256]  (only [0..seg_len] are meaningful)
//! 258     8     child_page: u64
//! ```
//!
//! Tight packing (no alignment padding on `child_page`) gives exactly 15 slots
//! in a 4096-byte page: 32 + 15 × 266 = 4022 bytes, leaving 74 bytes unused.
//!
//! # Overflow linked list
//!
//! When all `MAX_CHILDREN` slots in a node are occupied and a new distinct
//! segment arrives, rather than returning an error a fresh **overflow page** is
//! allocated, linked via `overflow_page`, and `FLAG_OVERFLOW` is set on the
//! parent.  The chain is traversed both during lookup and insertion.
//!
//! # Empty-node GC list
//!
//! When a trie node is pruned (zero children, no value), instead of immediately
//! returning the page to the page-level free list it is added to a trie-level
//! **empty-node reuse list** tracked by `FileHeader::empty_node_head`.  The
//! next pointer is stored in the node's `value_offset` field (safe because
//! `FLAG_EMPTY_GC` and `FLAG_HAS_VALUE` are mutually exclusive).
//!
//! # Freed page overlay
//!
//! When a page is freed it keeps `NODE_MAGIC` at [0..4], has `FLAG_FREE` (0xFF)
//! at byte 8, and stores the next-free pointer at [16..24] (the `value_offset`
//! position).

use crc32fast::Hasher;

use crate::error::{Result, TrieError};

// ---------------------------------------------------------------------------
// Layout constants
// ---------------------------------------------------------------------------

/// Page size in bytes.
pub const PAGE_SIZE: usize = 4096;

/// Page index reserved as "no page" / null pointer.
///
/// Page 0 is always the file header, so it is never a valid trie node page.
pub const NULL_PAGE: u64 = 0;

/// Magic bytes for the file header page (little-endian encoding of `b"MVTF"`).
pub const FILE_MAGIC: u32 = u32::from_le_bytes(*b"MVTF");

/// Magic bytes for a trie node page (little-endian encoding of `b"MVTN"`).
pub const NODE_MAGIC: u32 = u32::from_le_bytes(*b"MVTN");

/// On-disk format version stored in the file header.
pub const VERSION: u16 = 2;

/// Maximum segment length in bytes.
pub const MAX_SEG_LEN: usize = 256;

/// Size of one child slot in bytes: `seg_len`(2) + `seg_bytes`(256) + `child_page`(8).
pub const CHILD_SLOT_SIZE: usize = 2 + MAX_SEG_LEN + 8; // = 266

/// Size of the node page header in bytes (includes `overflow_page` field).
pub const NODE_HDR_SIZE: usize = 32;

/// Maximum number of children per node page.
///
/// `floor((PAGE_SIZE - NODE_HDR_SIZE) / CHILD_SLOT_SIZE)` = `floor(4064 / 266)` = 15.
pub const MAX_CHILDREN: usize = (PAGE_SIZE - NODE_HDR_SIZE) / CHILD_SLOT_SIZE;

// Compile-time sanity checks.
const _LAYOUT_FITS: () = assert!(NODE_HDR_SIZE + MAX_CHILDREN * CHILD_SLOT_SIZE <= PAGE_SIZE);
const _SLOT_SIZE: () = assert!(CHILD_SLOT_SIZE == 266);
const _MAX_CHILDREN: () = assert!(MAX_CHILDREN == 15);

/// Grow the file by this many pages per `grow()` call to amortize remap cost.
pub const GROW_BATCH: u64 = 64;

// Node flag bits.

/// This node is a terminal: it holds a value at the end of a key.
pub const FLAG_HAS_VALUE: u8 = 0x01;

/// This node's `overflow_page` field points to a continuation page.
pub const FLAG_OVERFLOW: u8 = 0x02;

/// This node is in the trie-level empty-node reuse list.
///
/// The `value_offset` field holds the next pointer in the list.
/// Mutually exclusive with [`FLAG_HAS_VALUE`].
pub const FLAG_EMPTY_GC: u8 = 0x04;

/// Sentinel flag stored in freed pages (not a valid live-node flag).
pub const FLAG_FREE: u8 = 0xFF;

// Byte offset of the CRC32 field within a node page.
const CRC_OFFSET: usize = 4;

// Byte offset of the `value_offset` / `next_free` / `next_empty` field.
const VALUE_OFFSET_FIELD: usize = 16;

// Byte offset of the `overflow_page` field.
const OVERFLOW_PAGE_OFFSET: usize = 24;

// ---------------------------------------------------------------------------
// File header (page 0)
// ---------------------------------------------------------------------------

/// In-memory representation of the file header (page 0).
///
/// Serialized to/from a full `PAGE_SIZE` byte array. Only the first 52 bytes
/// contain meaningful data; the rest are zeroed padding.
///
/// On-disk layout:
/// ```text
/// [0..4]   magic:            u32  (FILE_MAGIC)
/// [4..6]   version:          u16
/// [6..8]   _pad:             u16
/// [8..16]  root_page:        u64
/// [16..24] free_list_head:   u64
/// [24..32] num_pages:        u64
/// [32..40] empty_node_head:  u64  (head of the trie-level empty-node reuse list)
/// [40..48] entry_count:      u64  (number of key-value pairs)
/// [48..52] header_crc32:     u32  (CRC32 of bytes [0..48])
/// [52..4096] zeroed padding
/// ```
#[derive(Clone, Copy, Debug)]
pub struct FileHeader {
    pub magic: u32,
    pub version: u16,
    /// Page index of the trie root, or [`NULL_PAGE`] for a brand-new file.
    pub root_page: u64,
    /// Head of the free-page singly-linked list, or [`NULL_PAGE`] if empty.
    pub free_list_head: u64,
    /// Total number of pages in the file (including the header page).
    pub num_pages: u64,
    /// Head of the trie-level empty-node reuse list, or [`NULL_PAGE`] if empty.
    pub empty_node_head: u64,
    /// Number of key-value entries currently stored.
    pub entry_count: u64,
}

// Bytes covered by the header CRC (everything before the CRC field itself).
const HDR_CRC_COVER: usize = 48;

impl FileHeader {
    /// Creates an initial header for a brand-new file.
    pub fn new_empty() -> Self {
        FileHeader {
            magic: FILE_MAGIC,
            version: VERSION,
            root_page: NULL_PAGE,
            free_list_head: NULL_PAGE,
            num_pages: 1,
            empty_node_head: NULL_PAGE,
            entry_count: 0,
        }
    }

    /// Serializes to a full [`PAGE_SIZE`]-byte array.
    ///
    /// CRC32 covers bytes `[0..48]` and is written at `[48..52]`.
    pub fn to_page(&self) -> [u8; PAGE_SIZE] {
        let mut buf = [0u8; PAGE_SIZE];
        buf[0..4].copy_from_slice(&self.magic.to_le_bytes());
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        // [6..8]: _pad = 0
        buf[8..16].copy_from_slice(&self.root_page.to_le_bytes());
        buf[16..24].copy_from_slice(&self.free_list_head.to_le_bytes());
        buf[24..32].copy_from_slice(&self.num_pages.to_le_bytes());
        buf[32..40].copy_from_slice(&self.empty_node_head.to_le_bytes());
        buf[40..48].copy_from_slice(&self.entry_count.to_le_bytes());
        let crc = crc32fast::hash(&buf[..HDR_CRC_COVER]);
        buf[48..52].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Deserializes from a [`PAGE_SIZE`]-byte slice, verifying magic and CRC32.
    pub fn from_page(page: &[u8; PAGE_SIZE]) -> Result<Self> {
        let magic = u32::from_le_bytes(page[0..4].try_into().unwrap());
        if magic != FILE_MAGIC {
            return Err(TrieError::Corruption(format!(
                "bad file header magic: {magic:#010x} (expected {FILE_MAGIC:#010x})"
            )));
        }
        // Verify CRC32: covers [0..48], stored at [48..52].
        let stored_crc = u32::from_le_bytes(page[48..52].try_into().unwrap());
        let computed_crc = crc32fast::hash(&page[..HDR_CRC_COVER]);
        if stored_crc != computed_crc {
            return Err(TrieError::Corruption(format!(
                "file header CRC32 mismatch: stored {stored_crc:#010x}, \
                 computed {computed_crc:#010x}"
            )));
        }
        let version = u16::from_le_bytes(page[4..6].try_into().unwrap());
        if version != VERSION {
            return Err(TrieError::Corruption(format!(
                "unsupported version {version} (expected {VERSION})"
            )));
        }
        Ok(FileHeader {
            magic,
            version,
            root_page: u64::from_le_bytes(page[8..16].try_into().unwrap()),
            free_list_head: u64::from_le_bytes(page[16..24].try_into().unwrap()),
            num_pages: u64::from_le_bytes(page[24..32].try_into().unwrap()),
            empty_node_head: u64::from_le_bytes(page[32..40].try_into().unwrap()),
            entry_count: u64::from_le_bytes(page[40..48].try_into().unwrap()),
        })
    }
}

// ---------------------------------------------------------------------------
// ChildSlot
// ---------------------------------------------------------------------------

/// A single child edge in a trie node: a segment label and a page pointer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChildSlot {
    /// Byte length of the segment stored in this slot (0 ≤ `seg_len` ≤ 256).
    pub seg_len: u16,
    /// Segment bytes; only `[0..seg_len]` are meaningful.
    pub seg_bytes: [u8; MAX_SEG_LEN],
    /// Page index of the child trie node.
    pub child_page: u64,
}

impl ChildSlot {
    /// Returns the live segment bytes for this child edge.
    #[inline]
    pub fn segment(&self) -> &[u8] {
        &self.seg_bytes[..self.seg_len as usize]
    }
}

// ---------------------------------------------------------------------------
// TrieNode
// ---------------------------------------------------------------------------

/// In-memory representation of one trie node page.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrieNode {
    /// Node flags.  Use [`FLAG_HAS_VALUE`] for terminal nodes,
    /// [`FLAG_OVERFLOW`] when `overflow_page` is set,
    /// [`FLAG_EMPTY_GC`] for nodes in the reuse list.
    pub flags: u8,
    /// Byte length of the associated value in the external heap (0 when absent).
    pub value_len: u16,
    /// Byte offset in the external value heap; repurposed as:
    /// - next-empty pointer when `FLAG_EMPTY_GC` is set
    /// - next-free pointer when `FLAG_FREE` is set
    pub value_offset: u64,
    /// Page index of the overflow continuation node, or [`NULL_PAGE`].
    pub overflow_page: u64,
    /// Child edges, up to [`MAX_CHILDREN`].
    pub children: Vec<ChildSlot>,
}

impl TrieNode {
    /// Creates a new, empty interior node (no value, no children, no overflow).
    pub fn new() -> Self {
        TrieNode {
            flags: 0,
            value_len: 0,
            value_offset: 0,
            overflow_page: NULL_PAGE,
            children: Vec::new(),
        }
    }

    /// Creates a new terminal node (has a value, no children initially).
    pub fn new_terminal() -> Self {
        TrieNode {
            flags: FLAG_HAS_VALUE,
            value_len: 0,
            value_offset: 0,
            overflow_page: NULL_PAGE,
            children: Vec::new(),
        }
    }

    /// Returns `true` if this node marks the end of a key (has a value).
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.flags & FLAG_HAS_VALUE != 0
    }

    /// Returns `true` if this node has an overflow continuation page.
    #[inline]
    pub fn is_overflow(&self) -> bool {
        self.flags & FLAG_OVERFLOW != 0
    }

    /// Serializes this node to a 4096-byte page.
    ///
    /// The CRC32 is computed over the full page (with bytes [4..8] treated as
    /// zero) and written at offset 4.
    ///
    /// # Errors
    ///
    /// - [`TrieError::TooManyChildren`] if `children.len() > MAX_CHILDREN`.
    /// - [`TrieError::SegmentTooLong`] if any `seg_len > MAX_SEG_LEN`.
    pub fn to_page(&self) -> Result<[u8; PAGE_SIZE]> {
        if self.children.len() > MAX_CHILDREN {
            return Err(TrieError::TooManyChildren);
        }
        for child in &self.children {
            if child.seg_len as usize > MAX_SEG_LEN {
                return Err(TrieError::SegmentTooLong);
            }
        }

        let mut buf = [0u8; PAGE_SIZE];

        // Header fields (magic written below; crc field left 0 for CRC computation).
        buf[8] = self.flags;
        // [9]: _pad = 0
        buf[10..12].copy_from_slice(&(self.children.len() as u16).to_le_bytes());
        buf[12..14].copy_from_slice(&self.value_len.to_le_bytes());
        // [14..16]: _pad = 0
        buf[VALUE_OFFSET_FIELD..VALUE_OFFSET_FIELD + 8]
            .copy_from_slice(&self.value_offset.to_le_bytes());
        buf[OVERFLOW_PAGE_OFFSET..OVERFLOW_PAGE_OFFSET + 8]
            .copy_from_slice(&self.overflow_page.to_le_bytes());

        // Child slots at [NODE_HDR_SIZE ..]
        let mut off = NODE_HDR_SIZE;
        for child in &self.children {
            buf[off..off + 2].copy_from_slice(&child.seg_len.to_le_bytes());
            buf[off + 2..off + 2 + MAX_SEG_LEN].copy_from_slice(&child.seg_bytes);
            buf[off + 2 + MAX_SEG_LEN..off + CHILD_SLOT_SIZE]
                .copy_from_slice(&child.child_page.to_le_bytes());
            off += CHILD_SLOT_SIZE;
        }

        // Write magic then compute CRC (buf[4..8] is already 0 for the hash).
        buf[0..4].copy_from_slice(&NODE_MAGIC.to_le_bytes());
        let crc = node_crc(&buf);
        buf[CRC_OFFSET..CRC_OFFSET + 4].copy_from_slice(&crc.to_le_bytes());

        Ok(buf)
    }

    /// Deserializes a trie node from a 4096-byte page slice.
    ///
    /// Verifies `NODE_MAGIC` and the CRC32 before reading any fields.
    pub fn from_page(page: &[u8; PAGE_SIZE]) -> Result<Self> {
        let magic = u32::from_le_bytes(page[0..4].try_into().unwrap());
        if magic != NODE_MAGIC {
            return Err(TrieError::Corruption(format!(
                "bad node magic: {magic:#010x} (expected {NODE_MAGIC:#010x})"
            )));
        }

        let stored_crc = u32::from_le_bytes(
            page[CRC_OFFSET..CRC_OFFSET + 4].try_into().unwrap(),
        );
        let computed_crc = node_crc(page);
        if stored_crc != computed_crc {
            return Err(TrieError::Corruption(format!(
                "node CRC32 mismatch: stored {stored_crc:#010x}, \
                 computed {computed_crc:#010x}"
            )));
        }

        let flags = page[8];
        if flags == FLAG_FREE {
            return Err(TrieError::Corruption(
                "attempted to read a freed page as a live node".into(),
            ));
        }

        let n_children = u16::from_le_bytes(page[10..12].try_into().unwrap()) as usize;
        if n_children > MAX_CHILDREN {
            return Err(TrieError::Corruption(format!(
                "n_children {n_children} exceeds MAX_CHILDREN {MAX_CHILDREN}"
            )));
        }
        let value_len = u16::from_le_bytes(page[12..14].try_into().unwrap());
        let value_offset = u64::from_le_bytes(
            page[VALUE_OFFSET_FIELD..VALUE_OFFSET_FIELD + 8].try_into().unwrap(),
        );
        let overflow_page = u64::from_le_bytes(
            page[OVERFLOW_PAGE_OFFSET..OVERFLOW_PAGE_OFFSET + 8].try_into().unwrap(),
        );

        let mut children = Vec::with_capacity(n_children);
        let mut off = NODE_HDR_SIZE;
        for _ in 0..n_children {
            let seg_len = u16::from_le_bytes(page[off..off + 2].try_into().unwrap());
            if seg_len as usize > MAX_SEG_LEN {
                return Err(TrieError::Corruption(format!(
                    "seg_len {seg_len} exceeds MAX_SEG_LEN {MAX_SEG_LEN}"
                )));
            }
            let mut seg_bytes = [0u8; MAX_SEG_LEN];
            seg_bytes.copy_from_slice(&page[off + 2..off + 2 + MAX_SEG_LEN]);
            let child_page = u64::from_le_bytes(
                page[off + 2 + MAX_SEG_LEN..off + CHILD_SLOT_SIZE]
                    .try_into()
                    .unwrap(),
            );
            children.push(ChildSlot { seg_len, seg_bytes, child_page });
            off += CHILD_SLOT_SIZE;
        }

        Ok(TrieNode { flags, value_len, value_offset, overflow_page, children })
    }
}

impl Default for TrieNode {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CRC32 helpers
// ---------------------------------------------------------------------------

/// Computes the canonical CRC32 for a node page.
///
/// The checksum field at bytes [`CRC_OFFSET`]..`CRC_OFFSET+4` is treated as
/// zero so the hash is independent of the stored value.  This matches what
/// `to_page` writes and what `from_page` verifies.
#[inline]
pub fn node_crc(page: &[u8; PAGE_SIZE]) -> u32 {
    let mut h = Hasher::new();
    h.update(&page[..CRC_OFFSET]);
    h.update(&[0u8; 4]); // checksum field treated as zero
    h.update(&page[CRC_OFFSET + 4..]);
    h.finalize()
}

/// Writes a freed-page sentinel into `page`.
///
/// - `NODE_MAGIC` at [0..4] so the page is identifiable.
/// - CRC field zeroed (freed pages carry no valid checksum).
/// - `FLAG_FREE` (0xFF) at byte 8.
/// - `next_free` at [16..24] (reusing the `value_offset` position).
pub fn write_free_page(page: &mut [u8], next_free: u64) {
    debug_assert_eq!(page.len(), PAGE_SIZE);
    page.fill(0);
    page[0..4].copy_from_slice(&NODE_MAGIC.to_le_bytes());
    page[8] = FLAG_FREE;
    page[VALUE_OFFSET_FIELD..VALUE_OFFSET_FIELD + 8]
        .copy_from_slice(&next_free.to_le_bytes());
}

/// Reads the `next_free` pointer from a freed page.
pub fn read_next_free(page: &[u8]) -> u64 {
    debug_assert_eq!(page.len(), PAGE_SIZE);
    u64::from_le_bytes(
        page[VALUE_OFFSET_FIELD..VALUE_OFFSET_FIELD + 8]
            .try_into()
            .unwrap(),
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_child(segment: &[u8], child_page: u64) -> ChildSlot {
        assert!(segment.len() <= MAX_SEG_LEN);
        let mut seg_bytes = [0u8; MAX_SEG_LEN];
        seg_bytes[..segment.len()].copy_from_slice(segment);
        ChildSlot {
            seg_len: segment.len() as u16,
            seg_bytes,
            child_page,
        }
    }

    #[test]
    fn round_trip_empty_node() {
        let node = TrieNode::new();
        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert_eq!(node, decoded);
        assert!(!decoded.is_terminal());
        assert_eq!(decoded.children.len(), 0);
        assert_eq!(decoded.overflow_page, NULL_PAGE);
    }

    #[test]
    fn round_trip_terminal_node() {
        let node = TrieNode::new_terminal();
        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert_eq!(node, decoded);
        assert!(decoded.is_terminal());
    }

    #[test]
    fn round_trip_overflow_flag() {
        let mut node = TrieNode::new();
        node.flags |= FLAG_OVERFLOW;
        node.overflow_page = 42;
        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert!(decoded.is_overflow());
        assert_eq!(decoded.overflow_page, 42);
    }

    #[test]
    fn round_trip_empty_gc_flag() {
        let mut node = TrieNode::new();
        node.flags = FLAG_EMPTY_GC;
        node.value_offset = 99; // next_empty pointer
        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert_eq!(decoded.flags, FLAG_EMPTY_GC);
        assert_eq!(decoded.value_offset, 99);
    }

    #[test]
    fn round_trip_node_with_children() {
        let mut node = TrieNode::new();
        node.children.push(make_child(b"hello", 2));
        node.children.push(make_child(b"world", 3));
        node.children.push(make_child(b"foo", 4));

        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert_eq!(decoded.children.len(), 3);
        assert_eq!(decoded.children[0].segment(), b"hello");
        assert_eq!(decoded.children[0].child_page, 2);
        assert_eq!(decoded.children[1].segment(), b"world");
        assert_eq!(decoded.children[1].child_page, 3);
        assert_eq!(decoded.children[2].segment(), b"foo");
        assert_eq!(decoded.children[2].child_page, 4);
    }

    #[test]
    fn round_trip_max_length_segment() {
        let seg = [0xABu8; MAX_SEG_LEN];
        let mut node = TrieNode::new();
        node.children.push(make_child(&seg, 7));

        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert_eq!(decoded.children[0].segment(), &seg[..]);
        assert_eq!(decoded.children[0].child_page, 7);
    }

    #[test]
    fn round_trip_max_children() {
        let mut node = TrieNode::new();
        for i in 0..MAX_CHILDREN {
            node.children.push(make_child(&[i as u8], (i + 2) as u64));
        }
        assert_eq!(node.children.len(), MAX_CHILDREN);

        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert_eq!(decoded.children.len(), MAX_CHILDREN);
        for i in 0..MAX_CHILDREN {
            assert_eq!(decoded.children[i].segment(), &[i as u8]);
            assert_eq!(decoded.children[i].child_page, (i + 2) as u64);
        }
    }

    #[test]
    fn too_many_children_error() {
        let mut node = TrieNode::new();
        for i in 0..=MAX_CHILDREN {
            node.children.push(make_child(&[i as u8], i as u64 + 2));
        }
        assert!(matches!(node.to_page(), Err(TrieError::TooManyChildren)));
    }

    #[test]
    fn crc32_corruption_detected() {
        let node = TrieNode::new_terminal();
        let mut page = node.to_page().unwrap();
        // Corrupt a data byte (not the CRC field itself).
        page[100] ^= 0xFF;
        assert!(matches!(TrieNode::from_page(&page), Err(TrieError::Corruption(_))));
    }

    #[test]
    fn bad_magic_detected() {
        let node = TrieNode::new();
        let mut page = node.to_page().unwrap();
        page[0..4].copy_from_slice(&[0xDE, 0xAD, 0xBE, 0xEF]);
        assert!(matches!(TrieNode::from_page(&page), Err(TrieError::Corruption(_))));
    }

    #[test]
    fn file_header_round_trip() {
        let mut hdr = FileHeader::new_empty();
        hdr.root_page = 1;
        hdr.free_list_head = 2;
        hdr.num_pages = 64;
        hdr.empty_node_head = 5;
        hdr.entry_count = 42;

        let page = hdr.to_page();
        let decoded = FileHeader::from_page(&page).unwrap();
        assert_eq!(decoded.magic, FILE_MAGIC);
        assert_eq!(decoded.version, VERSION);
        assert_eq!(decoded.root_page, 1);
        assert_eq!(decoded.free_list_head, 2);
        assert_eq!(decoded.num_pages, 64);
        assert_eq!(decoded.empty_node_head, 5);
        assert_eq!(decoded.entry_count, 42);
    }

    #[test]
    fn file_header_crc_detected() {
        let hdr = FileHeader::new_empty();
        let mut page = hdr.to_page();
        page[10] ^= 0x01; // corrupt a covered byte
        assert!(matches!(FileHeader::from_page(&page), Err(TrieError::Corruption(_))));
    }

    #[test]
    fn free_page_round_trip() {
        let mut page = [0u8; PAGE_SIZE];
        write_free_page(&mut page, 42);
        assert_eq!(page[8], FLAG_FREE);
        assert_eq!(read_next_free(&page), 42);
    }

    #[test]
    fn free_page_null_chain() {
        let mut page = [0u8; PAGE_SIZE];
        write_free_page(&mut page, NULL_PAGE);
        assert_eq!(read_next_free(&page), NULL_PAGE);
    }

    #[test]
    fn empty_segment_child() {
        let mut node = TrieNode::new();
        node.children.push(make_child(b"", 5));
        let page = node.to_page().unwrap();
        let decoded = TrieNode::from_page(&page).unwrap();
        assert_eq!(decoded.children[0].segment(), b"");
        assert_eq!(decoded.children[0].child_page, 5);
    }
}
