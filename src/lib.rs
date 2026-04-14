//! # mappedvartrie
//!
//! A persistent, memory-mapped trie for Rust where keys are sequences of
//! variable-length segments.
//!
//! ## Scope (this version)
//!
//! This version implements only the on-disk node format and page allocator.
//! WAL, value storage, and query logic are not yet implemented.
//!
//! ## On-disk format
//!
//! One node per 4096-byte page.  Each page holds:
//! - a 24-byte header (magic, CRC32, flags, child count, value stub)
//! - up to 15 child slots of 266 bytes each (2-byte seg_len + 256-byte
//!   seg_bytes + 8-byte child_page pointer)
//!
//! Page 0 is the file header.  Page 1 is the root node.
//!
//! ## Thread safety
//!
//! Wrap [`PageStore`] in a `Mutex<PageStore>` at the trie level — see
//! [`store`] module documentation for details.

pub mod error;
pub mod key;
pub mod node;
pub mod store;

pub use error::{Result, TrieError};
pub use key::TrieKey;
pub use node::{
    ChildSlot, FileHeader, TrieNode,
    CHILD_SLOT_SIZE, FLAG_FREE, FLAG_HAS_VALUE, GROW_BATCH,
    MAX_CHILDREN, MAX_SEG_LEN, NODE_HDR_SIZE, NODE_MAGIC, FILE_MAGIC,
    NULL_PAGE, PAGE_SIZE, VERSION,
};
pub use store::PageStore;
