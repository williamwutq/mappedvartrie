//! Write-Ahead Log (WAL) for crash-safe trie mutations.
//!
//! # Protocol
//!
//! Before every `insert` or `delete`:
//! 1. Write a WAL record to `{db_path}.wal` and call `sync_all` (fsync).
//! 2. Perform the mmap and value-heap mutations.
//! 3. Flush the mmap (`msync`) and the value heap (`sync_all`).
//! 4. Delete the WAL file.
//!
//! On `open`, if `{db_path}.wal` exists, the pending operation is replayed
//! before the caller receives the trie handle.  Both `insert` and `delete`
//! are idempotent, so replay is always safe regardless of how far the first
//! attempt progressed.
//!
//! # Idempotency guarantee
//!
//! - **INSERT replay**: when walking the trie, any child that already exists
//!   from the partial first attempt is followed rather than re-created.  The
//!   terminal node's value fields are unconditionally overwritten.
//! - **DELETE replay**: clearing `FLAG_HAS_VALUE` when it is already clear is
//!   a no-op.  Walking a missing path returns immediately.
//!
//! # File format
//!
//! ```text
//! [0..4]   magic: [u8; 4]  = b"TVWA"
//! [4..8]   crc32: u32 le   CRC32 of bytes [8 .. total_len]
//! [8]      op: u8          WAL_OP_INSERT = 0x01, WAL_OP_DELETE = 0x02
//! [9..11]  n_segments: u16 le
//! [11..13] value_len: u16 le  (0 for DELETE)
//! For each of n_segments:
//!   seg_len: u16 le
//!   seg_bytes: [u8; seg_len]
//! value_bytes: [u8; value_len]
//! ```
//!
//! The CRC covers everything after the 8-byte fixed header (`[8..end]`).
//! If the file is shorter than the minimum expected size, it was truncated by
//! an interrupted write; [`read_existing`] treats it as absent (`Ok(None)`).

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::error::{Result, TrieError};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes at the start of every WAL file.
pub const WAL_MAGIC: [u8; 4] = *b"TVWA";

/// Op tag: insert (or overwrite) a key-value pair.
pub const WAL_OP_INSERT: u8 = 0x01;

/// Op tag: delete a key.
pub const WAL_OP_DELETE: u8 = 0x02;

/// Size of the fixed WAL header: magic(4) + crc32(4) + op(1) + n_segments(2) + value_len(2).
const WAL_HDR_SIZE: usize = 13;

// ---------------------------------------------------------------------------
// WalRecord
// ---------------------------------------------------------------------------

/// A decoded WAL record ready for replay.
pub struct WalRecord {
    /// `WAL_OP_INSERT` or `WAL_OP_DELETE`.
    pub op: u8,
    /// Key segments, each as an owned byte vector.
    pub segments: Vec<Vec<u8>>,
    /// Value bytes (empty for DELETE).
    pub value: Vec<u8>,
}

// ---------------------------------------------------------------------------
// Path helper
// ---------------------------------------------------------------------------

/// Returns `{db_path}.wal`.
pub fn wal_path(db_path: &Path) -> PathBuf {
    let mut s = db_path.as_os_str().to_os_string();
    s.push(".wal");
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Writes and fsyncs an INSERT WAL record.
///
/// Each element of `segments` must be ≤ 256 bytes (the caller is expected to
/// have validated this already, but the WAL itself does not enforce it).
pub fn write_insert(db_path: &Path, segments: &[&[u8]], value: &[u8]) -> Result<()> {
    write_record(db_path, WAL_OP_INSERT, segments, value)
}

/// Writes and fsyncs a DELETE WAL record (value bytes are empty).
pub fn write_delete(db_path: &Path, segments: &[&[u8]]) -> Result<()> {
    write_record(db_path, WAL_OP_DELETE, segments, &[])
}

/// Reads an existing WAL record.
///
/// Returns `Ok(None)` if:
/// - The WAL file does not exist.
/// - The file is too short (truncated write — incomplete fsync).
/// - The magic bytes do not match.
/// - The CRC32 of the payload is wrong.
///
/// Returns `Ok(Some(record))` when a complete, valid record is found.
pub fn read_existing(db_path: &Path) -> Result<Option<WalRecord>> {
    let path = wal_path(db_path);

    let buf = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(TrieError::from(e)),
    };

    // Need at least the fixed header.
    if buf.len() < WAL_HDR_SIZE {
        return Ok(None); // truncated
    }

    // Validate magic.
    if buf[..4] != WAL_MAGIC {
        return Ok(None);
    }

    let stored_crc = u32::from_le_bytes(buf[4..8].try_into().unwrap());
    let op = buf[8];
    let n_segments = u16::from_le_bytes(buf[9..11].try_into().unwrap()) as usize;
    let value_len = u16::from_le_bytes(buf[11..13].try_into().unwrap()) as usize;

    // Parse segments from the variable-length payload.
    let mut pos = WAL_HDR_SIZE;
    let mut segments: Vec<Vec<u8>> = Vec::with_capacity(n_segments);
    for _ in 0..n_segments {
        if pos + 2 > buf.len() {
            return Ok(None); // truncated
        }
        let seg_len = u16::from_le_bytes(buf[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;
        if pos + seg_len > buf.len() {
            return Ok(None); // truncated
        }
        segments.push(buf[pos..pos + seg_len].to_vec());
        pos += seg_len;
    }

    // Read value bytes.
    if pos + value_len > buf.len() {
        return Ok(None); // truncated
    }
    let value = buf[pos..pos + value_len].to_vec();
    let total_len = pos + value_len;

    // Verify CRC32 over everything after the 8-byte magic+crc prefix.
    let computed_crc = crc32fast::hash(&buf[8..total_len]);
    if stored_crc != computed_crc {
        return Ok(None); // corrupted or truncated payload
    }

    Ok(Some(WalRecord {
        op,
        segments,
        value,
    }))
}

/// Deletes the WAL file.  Returns `Ok(())` if the file is already absent.
pub fn delete(db_path: &Path) -> Result<()> {
    let path = wal_path(db_path);
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(TrieError::from(e)),
    }
}

// ---------------------------------------------------------------------------
// Internals
// ---------------------------------------------------------------------------

fn write_record(db_path: &Path, op: u8, segments: &[&[u8]], value: &[u8]) -> Result<()> {
    // Build the payload (bytes [8..]) for CRC computation.
    // Layout: op(1) + n_segments(2) + value_len(2) + [seg_len(2) + seg_bytes]... + value
    let mut payload: Vec<u8> = Vec::new();
    payload.push(op);
    payload.extend_from_slice(&(segments.len() as u16).to_le_bytes());
    payload.extend_from_slice(&(value.len() as u16).to_le_bytes());
    for seg in segments {
        payload.extend_from_slice(&(seg.len() as u16).to_le_bytes());
        payload.extend_from_slice(seg);
    }
    payload.extend_from_slice(value);

    let crc = crc32fast::hash(&payload);

    let path = wal_path(db_path);
    let mut file = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&path)?;

    file.write_all(&WAL_MAGIC)?;
    file.write_all(&crc.to_le_bytes())?;
    file.write_all(&payload)?;

    // fsync — WAL must be durable before any mmap mutation.
    file.sync_all()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn db(dir: &tempfile::TempDir) -> PathBuf {
        dir.path().join("trie.db")
    }

    #[test]
    fn round_trip_insert() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);

        write_insert(&db, &[b"hello", b"world"], b"myvalue").unwrap();

        let rec = read_existing(&db).unwrap().unwrap();
        assert_eq!(rec.op, WAL_OP_INSERT);
        assert_eq!(rec.segments, vec![b"hello".to_vec(), b"world".to_vec()]);
        assert_eq!(rec.value, b"myvalue");
    }

    #[test]
    fn round_trip_delete() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);

        write_delete(&db, &[b"foo", b"bar", b"baz"]).unwrap();

        let rec = read_existing(&db).unwrap().unwrap();
        assert_eq!(rec.op, WAL_OP_DELETE);
        assert_eq!(rec.segments.len(), 3);
        assert_eq!(rec.segments[0], b"foo");
        assert!(rec.value.is_empty());
    }

    #[test]
    fn empty_segments_and_value() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);

        write_insert(&db, &[], b"").unwrap();

        let rec = read_existing(&db).unwrap().unwrap();
        assert_eq!(rec.op, WAL_OP_INSERT);
        assert!(rec.segments.is_empty());
        assert!(rec.value.is_empty());
    }

    #[test]
    fn missing_wal_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        assert!(read_existing(&db).unwrap().is_none());
    }

    #[test]
    fn truncated_wal_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        // Shorter than WAL_HDR_SIZE.
        std::fs::write(wal_path(&db), b"TVW").unwrap();
        assert!(read_existing(&db).unwrap().is_none());
    }

    #[test]
    fn truncated_payload_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        // Write a record then truncate by 1 byte.
        write_insert(&db, &[b"abc"], b"val").unwrap();
        let mut data = std::fs::read(wal_path(&db)).unwrap();
        data.pop();
        std::fs::write(wal_path(&db), &data).unwrap();
        assert!(read_existing(&db).unwrap().is_none());
    }

    #[test]
    fn bad_magic_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        write_insert(&db, &[b"x"], b"y").unwrap();
        let mut data = std::fs::read(wal_path(&db)).unwrap();
        data[0] = 0xFF; // corrupt magic
        std::fs::write(wal_path(&db), &data).unwrap();
        assert!(read_existing(&db).unwrap().is_none());
    }

    #[test]
    fn crc_mismatch_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        write_insert(&db, &[b"x"], b"y").unwrap();
        let mut data = std::fs::read(wal_path(&db)).unwrap();
        // Corrupt a payload byte after the header.
        let last = data.len() - 1;
        data[last] ^= 0xFF;
        std::fs::write(wal_path(&db), &data).unwrap();
        assert!(read_existing(&db).unwrap().is_none());
    }

    #[test]
    fn overwrite_replaces_old_wal() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);

        write_insert(&db, &[b"old"], b"oldval").unwrap();
        write_delete(&db, &[b"new"]).unwrap();

        let rec = read_existing(&db).unwrap().unwrap();
        assert_eq!(rec.op, WAL_OP_DELETE);
        assert_eq!(rec.segments[0], b"new");
    }

    #[test]
    fn delete_wal_removes_file() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        write_insert(&db, &[b"k"], b"v").unwrap();
        assert!(wal_path(&db).exists());
        delete(&db).unwrap();
        assert!(!wal_path(&db).exists());
    }

    #[test]
    fn delete_absent_wal_is_ok() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        delete(&db).unwrap(); // should not error
    }

    #[test]
    fn round_trip_256_byte_segment() {
        let dir = tempfile::tempdir().unwrap();
        let db = db(&dir);
        let seg = [0xABu8; 256];
        write_insert(&db, &[&seg], b"v").unwrap();
        let rec = read_existing(&db).unwrap().unwrap();
        assert_eq!(rec.segments[0], seg.to_vec());
    }
}
