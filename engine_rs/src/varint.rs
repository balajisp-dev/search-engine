//! Variable-length integers (LEB128 style), the Rust twin of engine_py/varint.py, plus whole
//! blocks of postings (delta + varint), the twin of encode_postings / decode_postings.
//!
//! Each byte holds 7 bits of the number, lowest group first. The top bit (0x80) is a flag:
//! 1 = another byte follows, 0 = this is the last byte of the number.
//!
//! ```text
//! 5    -> [0|0000101]                  [0x05]
//! 128  -> [1|0000000] [0|0000001]      [0x80, 0x01]
//! 300  -> [1|0101100] [0|0000010]      [0xAC, 0x02]
//! ```
//!
//! Toolkit: n & 0x7F (lowest 7 bits), n >> 7 (drop them), b | 0x80 (set flag),
//! b & 0x80 (is flag set?), (b & 0x7F) << shift (place a group back).

/// Append the varint bytes of `n` to `buf`.
pub fn encode_varint(n: u32, buf: &mut Vec<u8>) {
    let mut n = n;
    loop {
        let last_7_bits = n & 0x7F;
        n >>= 7;
        if n > 0 {
            buf.push(last_7_bits as u8 | 0x80);
        } else {
            buf.push(last_7_bits as u8);
            break;
        }
    }
}

/// Read one varint starting at `data[i]`. Return (value, index of the byte after it).
pub fn decode_varint(data: &[u8], i: usize) -> (u32, usize) {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    let mut i = i;
    loop {
        result |= ((data[i] & 0x7F) as u32) << shift;
        let flag = data[i] & 0x80;
        if flag != 0 {
            i += 1;
            shift += 7;
        } else {
            break;
        }
    }
    (result, i + 1)
}

/// [(pos, tf), ...] sorted by pos -> block bytes: gap, tf, gap, tf, ... as varints,
/// gaps counted from `prev_pos`. Same bytes as Python's encode_postings.
pub fn encode_block(postings: &[(u32, u32)], prev_pos: u32) -> Vec<u8> {
    let gaps = crate::delta::to_gaps(postings, prev_pos);
    let mut result: Vec<u8> = Vec::with_capacity(postings.len() * 2);
    for &(gap, tf) in &gaps {
        encode_varint(gap, &mut result);
        encode_varint(tf, &mut result);
    }
    result
}

/// Inverse of encode_block: read `count` (gap, tf) pairs from `data`, gaps counted from `start`.
pub fn decode_block(data: &[u8], count: usize, start: u32) -> Vec<(u32, u32)> {
    let mut gaps: Vec<(u32, u32)> = Vec::with_capacity(count);
    let mut i = 0;
    for _ in 0..count {
        let (gap, next) = decode_varint(data, i);
        i = next;
        let (tf, next) = decode_varint(data, i);
        i = next;
        gaps.push((gap, tf));
    }
    crate::delta::from_gaps(&gaps, start)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(n: u32) -> Vec<u8> {
        let mut buf = Vec::new();
        encode_varint(n, &mut buf);
        buf
    }

    // ---- part 1: one number ----

    #[test]
    fn encode_bytes() {
        assert_eq!(encode(0), vec![0x00]); // zero still takes one byte
        assert_eq!(encode(5), vec![0x05]);
        assert_eq!(encode(127), vec![0x7F]); // largest 1-byte value
        assert_eq!(encode(128), vec![0x80, 0x01]); // smallest 2-byte value
        assert_eq!(encode(300), vec![0xAC, 0x02]);
        assert_eq!(encode(16_384), vec![0x80, 0x80, 0x01]);
        assert_eq!(encode(u32::MAX), vec![0xFF, 0xFF, 0xFF, 0xFF, 0x0F]); // 5 bytes at most
    }

    #[test]
    fn encode_appends_to_existing_buffer() {
        let mut buf = vec![0x01];
        encode_varint(300, &mut buf);
        assert_eq!(buf, vec![0x01, 0xAC, 0x02]);
    }

    #[test]
    fn decode_returns_next_index() {
        assert_eq!(decode_varint(&[0xAC, 0x02], 0), (300, 2));
    }

    #[test]
    fn decode_mid_stream() {
        // [5][300][7]: start reading at index 1, the 300
        let data = [0x05, 0xAC, 0x02, 0x07];
        assert_eq!(decode_varint(&data, 1), (300, 3));
        assert_eq!(decode_varint(&data, 3), (7, 4));
    }

    #[test]
    fn decode_largest() {
        assert_eq!(
            decode_varint(&[0xFF, 0xFF, 0xFF, 0xFF, 0x0F], 0),
            (u32::MAX, 5)
        );
    }

    #[test]
    fn round_trip() {
        for n in [
            0,
            1,
            127,
            128,
            255,
            300,
            16_383,
            16_384,
            1 << 21,
            123_456_789,
            u32::MAX,
        ] {
            let data = encode(n);
            assert_eq!(decode_varint(&data, 0), (n, data.len()), "n = {n}");
        }
    }

    // ---- part 2: whole blocks, checked against bytes Python wrote ----

    // "aspirin" in the SciFact segment: 18 postings, one block, bytes copied from postings.bin
    const ASPIRIN_BYTES: [u8; 49] = [
        76, 1, 29, 1, 205, 4, 3, 232, 3, 4, 196, 1, 12, 165, 2, 1, 213, 2, 1, 133, 4, 15, 249, 1,
        1, 228, 3, 1, 128, 3, 9, 226, 3, 1, 23, 1, 172, 1, 1, 254, 2, 9, 4, 1, 176, 1, 10, 31, 1,
    ];
    const ASPIRIN_POSTINGS: [(u32, u32); 18] = [
        (76, 1),
        (105, 1),
        (694, 3),
        (1182, 4),
        (1378, 12),
        (1671, 1),
        (2012, 1),
        (2529, 15),
        (2778, 1),
        (3262, 1),
        (3646, 9),
        (4128, 1),
        (4151, 1),
        (4323, 1),
        (4705, 9),
        (4709, 1),
        (4885, 10),
        (4916, 1),
    ];

    #[test]
    fn decode_block_matches_python() {
        assert_eq!(
            decode_block(&ASPIRIN_BYTES, 18, 0),
            ASPIRIN_POSTINGS.to_vec()
        );
    }

    #[test]
    fn encode_block_matches_python() {
        assert_eq!(encode_block(&ASPIRIN_POSTINGS, 0), ASPIRIN_BYTES.to_vec());
    }

    #[test]
    fn block_from_previous_block() {
        // a later block: first gap from the previous block's last doc
        let postings = [(1003, 2), (1010, 1)];
        let bytes = encode_block(&postings, 1000);
        assert_eq!(bytes, vec![3, 2, 7, 1]);
        assert_eq!(decode_block(&bytes, 2, 1000), postings.to_vec());
    }
}
