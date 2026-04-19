//! Value heap: an append-only file that stores variable-length byte values.
//!
//! # File layout
//!
//! ```text
//! [0..4]    magic:     u32 le  = HEAP_MAGIC
//! [4..6]    version:   u16 le  = 1
//! [6..8]    _pad:      u16
//! [8..16]   data_size: u64 le  (bytes written to the data section so far)
//! [16..20]  crc32:     u32 le  (CRC32 of bytes [0..16])
//! [20..4096] zeros
//! [4096..]  data section: values packed end-to-end
//! ```
//!
//! Each stored value occupies `2 + val_len` bytes:
//! ```text
//! val_len:   u16 le
//! val_bytes: [u8; val_len]
//! ```
//!
//! The `value_offset` stored in a trie node is the **absolute file offset** of
//! the `val_len` field.  The first value is always at offset 4096 (= `HEAP_DATA_OFFSET`).
//!
//! # Allocation
//!
//! Allocation is a simple bump pointer.  `data_size` tracks how many bytes of
//! the data section have been written.  To allocate:
//! 1. Compute `offset = HEAP_DATA_OFFSET + data_size`.
//! 2. Write `val_len` (2 bytes) + `val_bytes` at that offset.
//! 3. Update `data_size += 2 + val_len` and rewrite the header.
//!
//! Values are never freed or compacted in this version.  Deleted-key values
//! and WAL-replay duplicates become unreferenced but harmless garbage.
//!
//! # Write ordering
//!
//! Within an operation protected by the WAL, value bytes are written to disk
//! **before** `data_size` is updated in the header.  If the process crashes
//! between the two writes:
//! - `data_size` is stale (points before the orphaned bytes).
//! - On WAL replay, `alloc_value` re-uses the same offset, overwriting the
//!   orphaned bytes with identical data.  No corruption occurs.
//!
//! If the crash happens after `data_size` is updated but before the trie node
//! is written, the WAL replay allocates a fresh offset (the previous allocation
//! becomes unreferenced garbage — space leak, not corruption).

use std::fs::{File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::Path;

use crate::error::{Result, TrieError};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Magic bytes identifying a value heap file (little-endian `b"TVHL"`).
pub const HEAP_MAGIC: u32 = u32::from_le_bytes(*b"TVHL");

/// On-disk format version.
pub const HEAP_VERSION: u16 = 1;

/// Byte offset where value data begins (one full page for the header).
pub const HEAP_DATA_OFFSET: u64 = 4096;

/// Maximum value size in bytes (`u16::MAX`).
pub const MAX_VALUE_LEN: usize = u16::MAX as usize;

// Number of bytes covered by the header CRC: bytes [0..16].
const HDR_CRC_COVER: u64 = 16;

// ---------------------------------------------------------------------------
// ValHeap
// ---------------------------------------------------------------------------

/// Append-only value heap backed by a file.
///
/// Opened via [`ValHeap::open`]; use [`ValHeap::alloc_value`] to store a
/// value and receive its file offset, and [`ValHeap::read_value`] to retrieve
/// it by offset.
pub struct ValHeap {
    file: File,
    /// Cached value of `data_size` from the heap header.
    pub data_size: u64,
}

impl ValHeap {
    /// Opens or creates the heap file at `path`.
    ///
    /// - **New file**: writes an initial header (magic, version, `data_size = 0`).
    /// - **Existing file**: validates magic, version, and header CRC32.
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

    fn create_new(mut file: File) -> Result<Self> {
        let hdr = Self::encode_header(0);
        file.write_all(&hdr)?;
        file.sync_all()?;
        Ok(ValHeap { file, data_size: 0 })
    }

    fn open_existing(mut file: File) -> Result<Self> {
        let mut hdr = [0u8; 4096];
        file.seek(SeekFrom::Start(0))?;
        // It's OK if the file is shorter than 4096 bytes — we only need 20.
        let read = std::io::Read::read(&mut file, &mut hdr)?;
        if read < 20 {
            return Err(TrieError::Corruption("heap file too small".into()));
        }

        let magic = u32::from_le_bytes(hdr[0..4].try_into().unwrap());
        if magic != HEAP_MAGIC {
            return Err(TrieError::Corruption(format!(
                "bad heap magic: {magic:#010x}"
            )));
        }

        // Verify header CRC.
        let stored_crc = u32::from_le_bytes(hdr[16..20].try_into().unwrap());
        let computed_crc = crc32fast::hash(&hdr[..HDR_CRC_COVER as usize]);
        if stored_crc != computed_crc {
            return Err(TrieError::Corruption(format!(
                "heap header CRC mismatch: stored {stored_crc:#010x}, \
                 computed {computed_crc:#010x}"
            )));
        }

        let version = u16::from_le_bytes(hdr[4..6].try_into().unwrap());
        if version != HEAP_VERSION {
            return Err(TrieError::Corruption(format!(
                "unsupported heap version {version}"
            )));
        }

        let data_size = u64::from_le_bytes(hdr[8..16].try_into().unwrap());
        Ok(ValHeap { file, data_size })
    }

    // -----------------------------------------------------------------------
    // Allocation
    // -----------------------------------------------------------------------

    /// Appends `data` to the heap and returns its absolute file offset.
    ///
    /// The stored layout at the returned offset is `val_len: u16 le` followed
    /// by `val_bytes`.  Pass the returned offset to [`ValHeap::read_value`] to
    /// retrieve the value.
    ///
    /// # Errors
    ///
    /// Returns [`TrieError::Corruption`] if `data.len() > MAX_VALUE_LEN`.
    pub fn alloc_value(&mut self, data: &[u8]) -> Result<u64> {
        if data.len() > MAX_VALUE_LEN {
            return Err(TrieError::Corruption(format!(
                "value length {} exceeds maximum {}",
                data.len(),
                MAX_VALUE_LEN
            )));
        }

        let offset = HEAP_DATA_OFFSET + self.data_size;

        // Write val_len + val_bytes to the data section first.
        // This must be durable before data_size is updated so that
        // a crash between the two writes leaves the bytes orphaned
        // rather than corrupting the allocation pointer.
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.write_all(&(data.len() as u16).to_le_bytes())?;
        self.file.write_all(data)?;

        // Update the in-memory and on-disk data_size.
        self.data_size += 2 + data.len() as u64;
        self.write_header()?;

        Ok(offset)
    }

    // -----------------------------------------------------------------------
    // Reading
    // -----------------------------------------------------------------------

    /// Reads and returns the value stored at `offset`.
    ///
    /// `offset` must be a value previously returned by [`ValHeap::alloc_value`].
    pub fn read_value(&mut self, offset: u64) -> Result<Vec<u8>> {
        self.file.seek(SeekFrom::Start(offset))?;
        let mut len_buf = [0u8; 2];
        std::io::Read::read_exact(&mut self.file, &mut len_buf)?;
        let val_len = u16::from_le_bytes(len_buf) as usize;
        let mut val_buf = vec![0u8; val_len];
        if val_len > 0 {
            std::io::Read::read_exact(&mut self.file, &mut val_buf)?;
        }
        Ok(val_buf)
    }

    // -----------------------------------------------------------------------
    // Flush
    // -----------------------------------------------------------------------

    /// Flushes the underlying file to disk (`fsync`).
    pub fn flush(&self) -> Result<()> {
        self.file.sync_all()?;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    /// Encodes a 4096-byte header page with the given `data_size`.
    fn encode_header(data_size: u64) -> [u8; 4096] {
        let mut buf = [0u8; 4096];
        buf[0..4].copy_from_slice(&HEAP_MAGIC.to_le_bytes());
        buf[4..6].copy_from_slice(&HEAP_VERSION.to_le_bytes());
        // [6..8]: _pad = 0
        buf[8..16].copy_from_slice(&data_size.to_le_bytes());
        let crc = crc32fast::hash(&buf[..HDR_CRC_COVER as usize]);
        buf[16..20].copy_from_slice(&crc.to_le_bytes());
        buf
    }

    /// Writes the current `data_size` to the on-disk header and syncs.
    fn write_header(&mut self) -> Result<()> {
        let data_size = self.data_size;
        // Re-encode and rewrite only the 20 meaningful header bytes
        // (the rest of the 4096-byte header page is already zero on disk).
        let mut buf = [0u8; 20];
        buf[0..4].copy_from_slice(&HEAP_MAGIC.to_le_bytes());
        buf[4..6].copy_from_slice(&HEAP_VERSION.to_le_bytes());
        buf[8..16].copy_from_slice(&data_size.to_le_bytes());
        let crc = crc32fast::hash(&{
            // CRC covers [0..16], so build those 16 bytes.
            let mut covered = [0u8; 16];
            covered[..].copy_from_slice(&buf[..16]);
            covered
        });
        buf[16..20].copy_from_slice(&crc.to_le_bytes());

        self.file.seek(SeekFrom::Start(0))?;
        self.file.write_all(&buf)?;
        self.file.sync_all()?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_and_alloc_one() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut heap = ValHeap::open(tmp.path()).unwrap();

        let off = heap.alloc_value(b"hello").unwrap();
        assert_eq!(off, HEAP_DATA_OFFSET);

        let val = heap.read_value(off).unwrap();
        assert_eq!(val, b"hello");
    }

    #[test]
    fn alloc_multiple_sequential() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut heap = ValHeap::open(tmp.path()).unwrap();

        let o1 = heap.alloc_value(b"abc").unwrap();
        let o2 = heap.alloc_value(b"de").unwrap();
        let o3 = heap.alloc_value(b"fghij").unwrap();

        // o1 = 4096, len "abc" = 3, slot = 5 → o2 = 4101
        assert_eq!(o1, 4096);
        assert_eq!(o2, 4096 + 2 + 3); // 4101
        assert_eq!(o3, 4096 + 2 + 3 + 2 + 2); // 4105

        assert_eq!(heap.read_value(o1).unwrap(), b"abc");
        assert_eq!(heap.read_value(o2).unwrap(), b"de");
        assert_eq!(heap.read_value(o3).unwrap(), b"fghij");
    }

    #[test]
    fn empty_value() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut heap = ValHeap::open(tmp.path()).unwrap();

        let off = heap.alloc_value(b"").unwrap();
        assert_eq!(off, HEAP_DATA_OFFSET);
        assert_eq!(heap.data_size, 2); // only the 2-byte length field

        let val = heap.read_value(off).unwrap();
        assert!(val.is_empty());
    }

    #[test]
    fn persist_across_reopen() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let (o1, o2);
        {
            let mut heap = ValHeap::open(&path).unwrap();
            o1 = heap.alloc_value(b"first").unwrap();
            o2 = heap.alloc_value(b"second").unwrap();
        }

        let mut heap = ValHeap::open(&path).unwrap();
        assert_eq!(heap.read_value(o1).unwrap(), b"first");
        assert_eq!(heap.read_value(o2).unwrap(), b"second");
        assert_eq!(heap.data_size, 2 + 5 + 2 + 6); // 2+5 + 2+6 = 15
    }

    #[test]
    fn reopen_continues_allocation() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();

        let o1 = {
            let mut heap = ValHeap::open(&path).unwrap();
            heap.alloc_value(b"abc").unwrap()
        };

        let o2 = {
            let mut heap = ValHeap::open(&path).unwrap();
            heap.alloc_value(b"xyz").unwrap()
        };

        // Second allocation must not overlap with the first.
        assert!(o2 > o1);
        assert_eq!(o2, o1 + 2 + 3);

        let mut heap = ValHeap::open(&path).unwrap();
        assert_eq!(heap.read_value(o1).unwrap(), b"abc");
        assert_eq!(heap.read_value(o2).unwrap(), b"xyz");
    }

    #[test]
    fn max_value_length() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        let mut heap = ValHeap::open(tmp.path()).unwrap();

        let data = vec![0xCDu8; MAX_VALUE_LEN];
        let off = heap.alloc_value(&data).unwrap();
        assert_eq!(heap.read_value(off).unwrap(), data);
    }
}
