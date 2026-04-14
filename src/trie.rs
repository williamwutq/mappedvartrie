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
//! # Idempotency caveat
//!
//! A replayed INSERT may allocate a fresh value-heap slot even when the node
//! already has a value pointer (the old slot becomes unreferenced garbage).
//! Value compaction is out of scope for this version.

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::{Result, TrieError};
use crate::node::{ChildSlot, TrieNode, FLAG_HAS_VALUE, MAX_CHILDREN, MAX_SEG_LEN};
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
    /// Inserts or overwrites the value at the trie path defined by `segments`.
    ///
    /// At each trie level the algorithm:
    /// 1. Reads the current node.
    /// 2. Searches its children for a slot whose segment matches the next key
    ///    segment.
    /// 3. If found, follows the child pointer.
    /// 4. If not found, allocates a new node, appends a child slot to the
    ///    current node, and writes it back.
    ///
    /// At the terminal node, `FLAG_HAS_VALUE` is set and the value is written
    /// to the heap.
    fn insert(&mut self, segments: &[&[u8]], value: &[u8]) -> Result<()> {
        let root_page = self.store.header()?.root_page;
        let mut current_page = root_page;

        for &segment in segments {
            let mut node = self.store.read_node(current_page)?;

            // Find an existing child whose segment matches.
            if let Some(slot) = node.children.iter().find(|s| s.segment() == segment) {
                current_page = slot.child_page;
            } else {
                // No match: allocate a new child node.
                if node.children.len() >= MAX_CHILDREN {
                    return Err(TrieError::TooManyChildren);
                }

                let new_page = self.store.alloc_page()?;
                self.store.write_node(new_page, &TrieNode::new())?;

                let mut seg_bytes = [0u8; MAX_SEG_LEN];
                seg_bytes[..segment.len()].copy_from_slice(segment);
                node.children.push(ChildSlot {
                    seg_len: segment.len() as u16,
                    seg_bytes,
                    child_page: new_page,
                });
                self.store.write_node(current_page, &node)?;

                current_page = new_page;
            }
        }

        // Write value at the terminal node.
        let mut node = self.store.read_node(current_page)?;
        let value_offset = self.heap.alloc_value(value)?;
        node.flags |= FLAG_HAS_VALUE;
        node.value_len = value.len() as u16;
        node.value_offset = value_offset;
        self.store.write_node(current_page, &node)?;

        Ok(())
    }

    /// Clears the value at the trie path defined by `segments`.
    ///
    /// If the key is absent (any segment is not found, or the terminal node
    /// does not have `FLAG_HAS_VALUE`), this is a no-op.
    ///
    /// Child nodes are never pruned; empty interior nodes persist until a
    /// future compaction pass (out of scope for this version).
    fn delete(&mut self, segments: &[&[u8]]) -> Result<()> {
        let root_page = self.store.header()?.root_page;
        let mut current_page = root_page;

        for &segment in segments {
            let node = self.store.read_node(current_page)?;
            match node.children.iter().find(|s| s.segment() == segment) {
                Some(slot) => current_page = slot.child_page,
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

        Ok(())
    }

    /// Returns the value at `segments`, or `None` if the key is absent.
    fn get(&mut self, segments: &[&[u8]]) -> Result<Option<Vec<u8>>> {
        let root_page = self.store.header()?.root_page;
        let mut current_page = root_page;

        for &segment in segments {
            let node = self.store.read_node(current_page)?;
            match node.children.iter().find(|s| s.segment() == segment) {
                Some(slot) => current_page = slot.child_page,
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
}

// ---------------------------------------------------------------------------
// MappedVarTrie — the public, thread-safe handle
// ---------------------------------------------------------------------------

/// A persistent, memory-mapped trie whose keys are sequences of byte segments.
///
/// Create or reopen with [`MappedVarTrie::open`].  All public methods take
/// `&self` and acquire the internal mutex themselves — the handle can be
/// freely shared via `Arc`.
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

    /// Inserts or overwrites the value at the path described by `segments`.
    ///
    /// `segments` is a slice of byte slices, each at most `MAX_SEG_LEN` (256)
    /// bytes long.  `value` may be up to `u16::MAX` bytes.
    ///
    /// # Errors
    ///
    /// - [`TrieError::SegmentTooLong`] if any segment exceeds 256 bytes.
    /// - [`TrieError::TooManyChildren`] if a trie node already has
    ///   `MAX_CHILDREN` (15) children and the new segment does not match any
    ///   of them.
    pub fn insert(&self, segments: &[&[u8]], value: &[u8]) -> Result<()> {
        // Validate segments before touching the WAL.
        for &seg in segments {
            if seg.len() > MAX_SEG_LEN {
                return Err(TrieError::SegmentTooLong);
            }
        }

        // 1. Write WAL (fsync).
        wal::write_insert(&self.db_path, segments, value)?;

        // 2. Execute.
        let mut inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;
        inner.insert(segments, value)?;

        // 3. Flush.
        inner.store.flush()?;
        inner.heap.flush()?;

        // 4. Commit: delete the WAL.
        wal::delete(&self.db_path)?;

        Ok(())
    }

    /// Deletes the value at the path described by `segments`.
    ///
    /// If the key is not present this is a no-op.  Empty trie nodes are NOT
    /// pruned after deletion (future compaction concern).
    ///
    /// # Errors
    ///
    /// - [`TrieError::SegmentTooLong`] if any segment exceeds 256 bytes.
    pub fn delete(&self, segments: &[&[u8]]) -> Result<()> {
        for &seg in segments {
            if seg.len() > MAX_SEG_LEN {
                return Err(TrieError::SegmentTooLong);
            }
        }

        // 1. Write WAL (fsync).
        wal::write_delete(&self.db_path, segments)?;

        // 2. Execute.
        let mut inner = self.inner.lock().map_err(|_| {
            TrieError::Corruption("mutex poisoned".into())
        })?;
        inner.delete(segments)?;

        // 3. Flush.
        inner.store.flush()?;
        // No heap changes for delete; a heap flush is not required.

        // 4. Commit.
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
        // Prefix should not have a value.
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
        t.delete(&[b"nope"]).unwrap(); // must not error
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

    /// Simulates a crash after the WAL was written but before the operation
    /// completed: leaves the WAL file on disk, then opens the trie.
    /// `open()` must replay the WAL and bring the trie to the post-operation
    /// state.
    #[test]
    fn recovery_insert_wal_present() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");

        // Create an empty trie so the node file and heap file exist.
        {
            MappedVarTrie::open(&db_path).unwrap();
        }

        // Leave a WAL for an INSERT that was never executed.
        wal::write_insert(&db_path, &[b"crash_key"], b"crash_val").unwrap();
        // The WAL file is now present; the trie pages are in their initial state.

        // Opening should replay the WAL automatically.
        let t = MappedVarTrie::open(&db_path).unwrap();
        assert_eq!(
            t.get(&[b"crash_key"]).unwrap(),
            Some(b"crash_val".to_vec())
        );
        // WAL file must be gone after successful replay.
        assert!(!wal::wal_path(&db_path).exists());
    }

    #[test]
    fn recovery_delete_wal_present() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");

        // Insert a key, close cleanly.
        {
            let t = MappedVarTrie::open(&db_path).unwrap();
            t.insert(&[b"dk"], b"dv").unwrap();
        }

        // Simulate leaving a DELETE WAL behind.
        wal::write_delete(&db_path, &[b"dk"]).unwrap();

        // Open should replay the delete.
        let t = MappedVarTrie::open(&db_path).unwrap();
        assert_eq!(t.get(&[b"dk"]).unwrap(), None);
        assert!(!wal::wal_path(&db_path).exists());
    }

    #[test]
    fn recovery_insert_idempotent_with_existing_node() {
        // Simulate: WAL left behind after the node was partially written
        // (the child already exists in the trie).  Recovery must not
        // create a duplicate child slot.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("trie.db");

        // Insert a multi-segment key cleanly.
        {
            let t = MappedVarTrie::open(&db_path).unwrap();
            t.insert(&[b"a", b"b"], b"v1").unwrap();
        }

        // Simulate a WAL for the same key (as if crash before WAL was deleted).
        wal::write_insert(&db_path, &[b"a", b"b"], b"v2").unwrap();

        let t = MappedVarTrie::open(&db_path).unwrap();
        // Value should be the one in the WAL (v2), not the old one (v1).
        assert_eq!(t.get(&[b"a", b"b"]).unwrap(), Some(b"v2".to_vec()));

        // Root must still have exactly one child for "a" (no duplicate).
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
        // A key with zero segments terminates at the root node itself.
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
        // WAL must NOT have been written.
        assert!(!wal::wal_path(&db_path).exists());
    }

    #[test]
    fn many_inserts_and_deletes() {
        let dir = tempfile::tempdir().unwrap();
        let t = open(&dir);

        // Insert 10 distinct single-segment keys.
        for i in 0u8..10 {
            t.insert(&[&[i]], &[i * 10]).unwrap();
        }
        for i in 0u8..10 {
            assert_eq!(t.get(&[&[i]]).unwrap(), Some(vec![i * 10]));
        }

        // Delete even-indexed keys.
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
}
