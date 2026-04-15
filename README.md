# mappedvartrie

A persistent, memory-mapped trie for Rust where keys are sequences of variable-length byte segments.

Built by Claude.

## Features

- File-backed storage via `mmap` — data survives process restarts
- Crash-safe writes with a write-ahead log (WAL) fsynced before every `insert` or `delete`;
  an interrupted write is automatically replayed on next open, leaving the trie consistent
- Corruption detection — every node page carries a CRC32 checksum; a partial write is
  detected immediately and reported as an error
- Thread-safe: all public methods take `&self` and can be shared via `Arc`
- Overflow linked list — nodes that exceed `MAX_CHILDREN` (15) children automatically
  chain overflow pages rather than failing
- Empty-node GC list — nodes emptied by `delete` are recycled before new pages are
  allocated, keeping file growth minimal
- Generic `TrieKey` trait for typed keys alongside the raw `&[&[u8]]` API
- Prefix scan — retrieve all entries whose key begins with a given prefix

## Constraints

Keys are sequences of byte slices (`&[&[u8]]`), where each segment can be up to 256 bytes.
Values are arbitrary byte slices up to 65535 bytes. Both keys and values are heap-allocated
on read — there is no zero-copy borrow from the mmap.

## On-disk format

Three files are associated with a database path `{db}`:

| File        | Contents                                      |
|-------------|-----------------------------------------------|
| `{db}`      | Node pages (page 0 = file header, 1 = root, …) |
| `{db}.vals` | Value heap (append-only, offset-addressed)    |
| `{db}.wal`  | Write-ahead log (present only mid-operation)  |

Each 4096-byte page holds a 32-byte header and up to 15 child slots of 266 bytes each
(2-byte segment length + 256-byte segment bytes + 8-byte child-page pointer).

## Quick start

```toml
[dependencies]
mappedvartrie = "0.1"
```

```rust
use mappedvartrie::MappedVarTrie;

let trie = MappedVarTrie::open("data.db")?;

// Raw &[&[u8]] API — segments are the path through the trie.
trie.insert(&[b"usr", b"local", b"bin"], b"exec")?;
trie.insert(&[b"usr", b"local", b"lib"], b"libs")?;
trie.insert(&[b"usr", b"share"], b"share")?;

assert_eq!(trie.get(&[b"usr", b"local", b"bin"])?, Some(b"exec".to_vec()));
assert!(trie.contains_key(&[b"usr", b"share"])?);

// Prefix scan — returns all (key_segments, value) pairs under a prefix.
let entries = trie.prefix_scan(&[b"usr", b"local"])?;
assert_eq!(entries.len(), 2);

trie.delete(&[b"usr", b"share"])?;
assert_eq!(trie.len()?, 2);
```

## Generic `TrieKey` API

Implement `TrieKey` for any type that can be decomposed into byte segments:

```rust
use mappedvartrie::TrieKey;

struct PathKey(String);

impl TrieKey for PathKey {
    type Segment = Vec<u8>;
    type SegmentIter<'a> = std::iter::Map<std::str::Split<'a, char>, fn(&str) -> Vec<u8>>;

    fn segments(&self) -> Self::SegmentIter<'_> {
        self.0.split('/').map(|s| s.as_bytes().to_vec())
    }
}

let key = PathKey("usr/local/bin".into());
trie.insert_key(&key, b"exec")?;
assert_eq!(trie.get_key(&key)?, Some(b"exec".to_vec()));
trie.delete_key(&key)?;
```

## License

MIT
