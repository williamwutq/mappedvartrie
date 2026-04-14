//! Error types for `mappedvartrie`.

use std::fmt;

/// All errors that can be returned by this crate.
#[derive(Debug)]
pub enum TrieError {
    /// Underlying I/O failure.
    Io(std::io::Error),
    /// On-disk data is inconsistent (bad magic, CRC mismatch, invalid fields).
    Corruption(String),
    /// Attempted to add more children than a page can hold (`MAX_CHILDREN`).
    TooManyChildren,
    /// A segment exceeded `MAX_SEG_LEN` bytes.
    SegmentTooLong,
}

impl fmt::Display for TrieError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrieError::Io(e) => write!(f, "I/O error: {e}"),
            TrieError::Corruption(msg) => write!(f, "corruption: {msg}"),
            TrieError::TooManyChildren => write!(f, "too many children for one page"),
            TrieError::SegmentTooLong => write!(f, "segment too long (max 256 bytes)"),
        }
    }
}

impl std::error::Error for TrieError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            TrieError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for TrieError {
    fn from(e: std::io::Error) -> Self {
        TrieError::Io(e)
    }
}

pub type Result<T> = std::result::Result<T, TrieError>;
