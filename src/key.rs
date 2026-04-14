//! The `TrieKey` trait: keys are sequences of typed segments, not raw bytes.

/// A key type whose identity is a sequence of variable-length segments.
///
/// # Segment size
///
/// Each segment's byte representation (`AsRef<[u8]>`) must be at most
/// `MAX_SEG_LEN` (256) bytes.  The page format allocates 256 bytes per slot;
/// longer segments cannot be stored and will return `TrieError::SegmentTooLong`.
///
/// # Example
///
/// ```ignore
/// struct PathKey(Vec<String>);
///
/// impl TrieKey for PathKey {
///     type Segment = String;
///     type SegmentIter<'a> = std::iter::Cloned<std::slice::Iter<'a, String>>;
///     fn segments(&self) -> Self::SegmentIter<'_> {
///         self.0.iter().cloned()
///     }
/// }
/// ```
pub trait TrieKey {
    /// A single segment — must be comparable and convertible to bytes.
    type Segment: AsRef<[u8]> + Eq;

    /// Iterator over the key's segments.
    type SegmentIter<'a>: Iterator<Item = Self::Segment>
    where
        Self: 'a;

    /// Returns an iterator over this key's segments in order.
    fn segments(&self) -> Self::SegmentIter<'_>;
}
