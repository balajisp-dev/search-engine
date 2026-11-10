//! Delta (gap) encoding of postings: the first half of what postings.bin stores.
//!
//! A postings list [(pos, tf), ...] is sorted by pos. Instead of the positions themselves we store
//! each one's gap from the previous position (small numbers -> fewer varint bytes in Lesson 3).
//! The first gap is measured from `prev_pos` (the previous block's last doc, or 0).
//!
//! ```text
//! postings  [(3, 1), (7, 2), (20, 1)]   prev_pos = 0
//! gaps      [(3, 1), (4, 2), (13, 1)]   (tf is kept as it is)
//! ```

/// [(pos, tf), ...] sorted by pos -> [(gap, tf), ...], gaps counted from `prev_pos`.
pub fn to_gaps(postings: &[(u32, u32)], prev_pos: u32) -> Vec<(u32, u32)> {
    let mut list: Vec<(u32, u32)> = Vec::with_capacity(postings.len());
    let mut prev = prev_pos;
    for &(pos, tf) in postings {
        list.push((pos - prev, tf));
        prev = pos;
    }
    list
}

/// Inverse of to_gaps: [(gap, tf), ...] -> [(pos, tf), ...], gaps counted from `start`.
pub fn from_gaps(gaps: &[(u32, u32)], start: u32) -> Vec<(u32, u32)> {
    let mut pos = start;
    let mut list: Vec<(u32, u32)> = Vec::with_capacity(gaps.len());
    for &(gap, tf) in gaps {
        pos += gap;
        list.push((pos, tf));
    }
    list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_gaps_example() {
        assert_eq!(
            to_gaps(&[(3, 1), (7, 2), (20, 1)], 0),
            vec![(3, 1), (4, 2), (13, 1)]
        );
    }

    #[test]
    fn to_gaps_from_previous_block() {
        // a later block: its first gap is measured from the previous block's last doc (100)
        assert_eq!(to_gaps(&[(103, 1), (110, 4)], 100), vec![(3, 1), (7, 4)]);
    }

    #[test]
    fn from_gaps_example() {
        assert_eq!(
            from_gaps(&[(3, 1), (4, 2), (13, 1)], 0),
            vec![(3, 1), (7, 2), (20, 1)]
        );
        assert_eq!(from_gaps(&[(3, 1), (7, 4)], 100), vec![(103, 1), (110, 4)]);
    }

    #[test]
    fn empty_lists() {
        assert_eq!(to_gaps(&[], 0), vec![]);
        assert_eq!(from_gaps(&[], 5), vec![]);
    }

    #[test]
    fn doc_zero_gives_gap_zero() {
        // doc position 0 is real (the first doc in the corpus), so a gap of 0 must round-trip
        assert_eq!(to_gaps(&[(0, 2), (1, 1)], 0), vec![(0, 2), (1, 1)]);
    }

    #[test]
    fn round_trip() {
        let postings: Vec<(u32, u32)> = (0..500).map(|i| (i * 7 + 3, i % 5 + 1)).collect();
        for start in [0, 2] {
            assert_eq!(from_gaps(&to_gaps(&postings, start), start), postings);
        }
    }
}
