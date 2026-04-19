//! # mappedvartrie
//!
//! A persistent, memory-mapped trie for Rust where keys are sequences of
//! variable-length byte segments.
//!
//! ## Features
//!
//! - One trie node per 4096-byte page, memory-mapped via `memmap2`.
//! - Write-Ahead Log for crash-safe `insert` and `delete` operations.
//! - Append-only value heap for variable-length values (up to `u16::MAX` bytes).
//! - Overflow linked list: nodes with more than [`MAX_CHILDREN`] (15) children
//!   automatically chain overflow pages rather than failing.
//! - Empty-node GC list: nodes emptied by `delete` are recycled before new
//!   pages are allocated, keeping file growth minimal.
//! - Generic [`TrieKey`] trait for typed keys alongside the raw `&[&[u8]]` API.
//!
//! ## On-disk format
//!
//! One node per 4096-byte page.  Each page holds:
//! - a 32-byte header (magic, CRC32, flags, child count, value fields,
//!   overflow-page pointer)
//! - up to 15 child slots of 266 bytes each (2-byte seg_len + 256-byte
//!   seg_bytes + 8-byte child_page pointer)
//!
//! Page 0 is the file header.  Page 1 is the root node.
//!
//! ## Thread safety
//!
//! [`MappedVarTrie`] wraps an internal `Mutex<TrieInner>` — all public methods
//! take `&self` and can be shared across threads via `Arc`.

pub mod error;
pub mod key;
pub mod node;
pub mod store;
pub mod trie;
pub mod valheap;
pub mod wal;

pub use error::{Result, TrieError};
pub use key::TrieKey;
pub use node::{
    CHILD_SLOT_SIZE, ChildSlot, FILE_MAGIC, FLAG_EMPTY_GC, FLAG_FREE, FLAG_HAS_VALUE,
    FLAG_OVERFLOW, FileHeader, GROW_BATCH, MAX_CHILDREN, MAX_SEG_LEN, NODE_HDR_SIZE, NODE_MAGIC,
    NULL_PAGE, PAGE_SIZE, TrieNode, VERSION,
};
pub use store::PageStore;
pub use trie::MappedVarTrie;
pub use valheap::ValHeap;
