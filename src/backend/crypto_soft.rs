//! Software fallback implementations for AES round operations and carry-less multiply.
//!
//! Used when hardware AES-NI / VAES / PCLMULQDQ / VPCLMULQDQ is not available.
//! All operations work on 128-bit (16-byte) blocks, matching the AES block size.

// AES forward S-box.
#[rustfmt::skip]
const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
    0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4,
    0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71,
    0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6,
    0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
    0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
    0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44,
    0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a,
    0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49,
    0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
    0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
    0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1,
    0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
    0x16,
];

// AES inverse S-box.

#[rustfmt::skip]
const INV_SBOX: [u8; 256] = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7,
    0xfb, 0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde,
    0xe9, 0xcb, 0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42,
    0xfa, 0xc3, 0x4e, 0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49,
    0x6d, 0x8b, 0xd1, 0x25, 0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c,
    0xcc, 0x5d, 0x65, 0xb6, 0x92, 0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15,
    0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84, 0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7,
    0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06, 0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02,
    0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b, 0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc,
    0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73, 0x96, 0xac, 0x74, 0x22, 0xe7, 0xad,
    0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e, 0x47, 0xf1, 0x1a, 0x71, 0x1d,
    0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b, 0xfc, 0x56, 0x3e, 0x4b,
    0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4, 0x1f, 0xdd, 0xa8,
    0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f, 0x60, 0x51,
    0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef, 0xa0,
    0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c,
    0x7d,
];

/// Multiply by 2 in GF(2^8) with polynomial x^8 + x^4 + x^3 + x + 1.
#[inline(always)]
fn xtime(x: u8) -> u8 {
    let shifted = (x as u16) << 1;
    (shifted ^ (if x & 0x80 != 0 { 0x1B } else { 0 })) as u8
}

/// SubBytes: apply S-box to each byte.
#[inline(always)]
fn sub_bytes(block: &mut [u8; 16]) {
    for b in block.iter_mut() {
        *b = SBOX[*b as usize];
    }
}

/// InvSubBytes: apply inverse S-box to each byte.
#[inline(always)]
fn inv_sub_bytes(block: &mut [u8; 16]) {
    for b in block.iter_mut() {
        *b = INV_SBOX[*b as usize];
    }
}

/// ShiftRows: cyclically shift rows of the 4x4 state (column-major layout).
///
/// Permutation: [0, 5, 10, 15, 4, 9, 14, 3, 8, 13, 2, 7, 12, 1, 6, 11]
#[inline(always)]
fn shift_rows(block: &mut [u8; 16]) {
    let t = *block;
    block[0] = t[0];
    block[1] = t[5];
    block[2] = t[10];
    block[3] = t[15];
    block[4] = t[4];
    block[5] = t[9];
    block[6] = t[14];
    block[7] = t[3];
    block[8] = t[8];
    block[9] = t[13];
    block[10] = t[2];
    block[11] = t[7];
    block[12] = t[12];
    block[13] = t[1];
    block[14] = t[6];
    block[15] = t[11];
}

/// InvShiftRows: inverse of ShiftRows.
///
/// Permutation: [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]
#[inline(always)]
fn inv_shift_rows(block: &mut [u8; 16]) {
    let t = *block;
    block[0] = t[0];
    block[1] = t[13];
    block[2] = t[10];
    block[3] = t[7];
    block[4] = t[4];
    block[5] = t[1];
    block[6] = t[14];
    block[7] = t[11];
    block[8] = t[8];
    block[9] = t[5];
    block[10] = t[2];
    block[11] = t[15];
    block[12] = t[12];
    block[13] = t[9];
    block[14] = t[6];
    block[15] = t[3];
}

/// MixColumns: multiply each column by the MDS matrix in GF(2^8).
///
/// Matrix: [[2,3,1,1],[1,2,3,1],[1,1,2,3],[3,1,1,2]]
#[inline(always)]
fn mix_columns(block: &mut [u8; 16]) {
    for col in 0..4 {
        let i = col * 4;
        let b0 = block[i];
        let b1 = block[i + 1];
        let b2 = block[i + 2];
        let b3 = block[i + 3];
        let d0 = xtime(b0);
        let d1 = xtime(b1);
        let d2 = xtime(b2);
        let d3 = xtime(b3);
        block[i] = d0 ^ d1 ^ b1 ^ b2 ^ b3; // 2*b0 + 3*b1 + b2 + b3
        block[i + 1] = b0 ^ d1 ^ d2 ^ b2 ^ b3; // b0 + 2*b1 + 3*b2 + b3
        block[i + 2] = b0 ^ b1 ^ d2 ^ d3 ^ b3; // b0 + b1 + 2*b2 + 3*b3
        block[i + 3] = d0 ^ b0 ^ b1 ^ b2 ^ d3; // 3*b0 + b1 + b2 + 2*b3
    }
}

/// InvMixColumns: multiply each column by the inverse MDS matrix in GF(2^8).
///
/// Matrix: [[14,11,13,9],[9,14,11,13],[13,9,14,11],[11,13,9,14]]
#[inline(always)]
fn inv_mix_columns(block: &mut [u8; 16]) {
    for col in 0..4 {
        let i = col * 4;
        let b0 = block[i];
        let b1 = block[i + 1];
        let b2 = block[i + 2];
        let b3 = block[i + 3];
        // Compute multiples in GF(2^8)
        let x2_0 = xtime(b0);
        let x4_0 = xtime(x2_0);
        let x8_0 = xtime(x4_0);
        let x2_1 = xtime(b1);
        let x4_1 = xtime(x2_1);
        let x8_1 = xtime(x4_1);
        let x2_2 = xtime(b2);
        let x4_2 = xtime(x2_2);
        let x8_2 = xtime(x4_2);
        let x2_3 = xtime(b3);
        let x4_3 = xtime(x2_3);
        let x8_3 = xtime(x4_3);
        // 9*x  = 8*x ^ x
        // 11*x = 8*x ^ 2*x ^ x
        // 13*x = 8*x ^ 4*x ^ x
        // 14*x = 8*x ^ 4*x ^ 2*x
        block[i] = (x8_0 ^ x4_0 ^ x2_0)
            ^ (x8_1 ^ x2_1 ^ b1)
            ^ (x8_2 ^ x4_2 ^ b2)
            ^ (x8_3 ^ b3);
        block[i + 1] = (x8_0 ^ b0)
            ^ (x8_1 ^ x4_1 ^ x2_1)
            ^ (x8_2 ^ x2_2 ^ b2)
            ^ (x8_3 ^ x4_3 ^ b3);
        block[i + 2] = (x8_0 ^ x4_0 ^ b0)
            ^ (x8_1 ^ b1)
            ^ (x8_2 ^ x4_2 ^ x2_2)
            ^ (x8_3 ^ x2_3 ^ b3);
        block[i + 3] = (x8_0 ^ x2_0 ^ b0)
            ^ (x8_1 ^ x4_1 ^ b1)
            ^ (x8_2 ^ b2)
            ^ (x8_3 ^ x4_3 ^ x2_3);
    }
}

/// AddRoundKey: XOR state with round key.
#[inline(always)]
fn add_round_key(block: &mut [u8; 16], key: &[u8; 16]) {
    for i in 0..16 {
        block[i] ^= key[i];
    }
}

/// One AES encryption round: SubBytes + ShiftRows + MixColumns + AddRoundKey.
pub(crate) fn aes_round(state: &mut [u8; 16], round_key: &[u8; 16]) {
    sub_bytes(state);
    shift_rows(state);
    mix_columns(state);
    add_round_key(state, round_key);
}

/// Last AES encryption round: SubBytes + ShiftRows + AddRoundKey (no MixColumns).
pub(crate) fn aes_last_round(state: &mut [u8; 16], round_key: &[u8; 16]) {
    sub_bytes(state);
    shift_rows(state);
    add_round_key(state, round_key);
}

/// One AES decryption round: InvSubBytes + InvShiftRows + InvMixColumns + AddRoundKey.
pub(crate) fn aes_round_inv(state: &mut [u8; 16], round_key: &[u8; 16]) {
    inv_sub_bytes(state);
    inv_shift_rows(state);
    inv_mix_columns(state);
    add_round_key(state, round_key);
}

/// Last AES decryption round: InvSubBytes + InvShiftRows + AddRoundKey (no InvMixColumns).
pub(crate) fn aes_last_round_inv(state: &mut [u8; 16], round_key: &[u8; 16]) {
    inv_sub_bytes(state);
    inv_shift_rows(state);
    add_round_key(state, round_key);
}

/// AES KeyGenAssist: SubWord + RotWord + Rcon XOR on 128-bit block.
///
/// Output:
///   - bytes [0..3] = SubWord(input[4..7])
///   - bytes [4..7] = RotWord(SubWord(input[4..7])) XOR Rcon
///   - bytes [8..11] = SubWord(input[12..15])
///   - bytes [12..15] = RotWord(SubWord(input[12..15])) XOR Rcon
pub(crate) fn aes_key_gen_assist(block: &[u8; 16], rcon: u8) -> [u8; 16] {
    let mut result = [0u8; 16];
    // SubWord on bytes [4..7]
    let x0 = SBOX[block[4] as usize];
    let x1 = SBOX[block[5] as usize];
    let x2 = SBOX[block[6] as usize];
    let x3 = SBOX[block[7] as usize];
    result[0] = x0;
    result[1] = x1;
    result[2] = x2;
    result[3] = x3;
    // RotWord(SubWord) XOR Rcon: RotWord([x0,x1,x2,x3]) = [x1,x2,x3,x0]
    result[4] = x1 ^ rcon;
    result[5] = x2;
    result[6] = x3;
    result[7] = x0;
    // SubWord on bytes [12..15]
    let y0 = SBOX[block[12] as usize];
    let y1 = SBOX[block[13] as usize];
    let y2 = SBOX[block[14] as usize];
    let y3 = SBOX[block[15] as usize];
    result[8] = y0;
    result[9] = y1;
    result[10] = y2;
    result[11] = y3;
    // RotWord(SubWord) XOR Rcon
    result[12] = y1 ^ rcon;
    result[13] = y2;
    result[14] = y3;
    result[15] = y0;
    result
}

/// AES InvMixColumns: apply inverse MixColumns to a 128-bit block.
/// Used for decryption key expansion.
pub(crate) fn aes_inv_mix_columns_block(block: &mut [u8; 16]) {
    inv_mix_columns(block);
}

/// 64-bit carry-less multiply, producing a 128-bit result as (lo, hi).
pub(crate) fn clmul_64(a: u64, b: u64) -> (u64, u64) {
    let mut result = 0u128;
    let a_wide = a as u128;
    for i in 0..64 {
        if (b >> i) & 1 != 0 {
            result ^= a_wide << i;
        }
    }
    (result as u64, (result >> 64) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sub_bytes_roundtrip() {
        let original: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa,
            0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        let mut state = original;
        sub_bytes(&mut state);
        assert_ne!(state, original);
        inv_sub_bytes(&mut state);
        assert_eq!(state, original);
    }

    #[test]
    fn test_shift_rows_roundtrip() {
        let original: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa,
            0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        let mut state = original;
        shift_rows(&mut state);
        assert_ne!(state, original);
        inv_shift_rows(&mut state);
        assert_eq!(state, original);
    }

    #[test]
    fn test_mix_columns_roundtrip() {
        let original: [u8; 16] = [
            0xdb, 0x13, 0x53, 0x45, 0xf2, 0x0a, 0x22, 0x5c, 0x01, 0x01, 0x01,
            0x01, 0xc6, 0xc6, 0xc6, 0xc6,
        ];
        let mut state = original;
        mix_columns(&mut state);
        assert_ne!(state, original);
        inv_mix_columns(&mut state);
        assert_eq!(state, original);
    }

    #[test]
    fn test_mix_columns_known_vector() {
        // FIPS-197 Section 5.1.3 example:
        // Input column: [db, 13, 53, 45] -> Output column: [8e, 4d, a1, bc]
        let mut state = [0u8; 16];
        state[0] = 0xdb;
        state[1] = 0x13;
        state[2] = 0x53;
        state[3] = 0x45;
        mix_columns(&mut state);
        assert_eq!(state[0], 0x8e);
        assert_eq!(state[1], 0x4d);
        assert_eq!(state[2], 0xa1);
        assert_eq!(state[3], 0xbc);
    }

    #[test]
    fn test_aes_round_with_zero_key() {
        // AESRound with zero key = SubBytes + ShiftRows + MixColumns (no key effect)
        let input: [u8; 16] = [
            0x19, 0x3d, 0xe3, 0xbe, 0xa0, 0xf4, 0xe2, 0x2b, 0x9a, 0xc6, 0x8d,
            0x2a, 0xe9, 0xf8, 0x48, 0x08,
        ];
        let zero_key = [0u8; 16];
        let mut state = input;
        aes_round(&mut state, &zero_key);
        // After SubBytes
        let mut expected = input;
        sub_bytes(&mut expected);
        shift_rows(&mut expected);
        mix_columns(&mut expected);
        assert_eq!(state, expected);
    }

    #[test]
    fn test_aes_round_inv_with_adjusted_key() {
        // AESRoundInv with InvMixColumns(key) should invert AESRound
        let original: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa,
            0xbb, 0xcc, 0xdd, 0xee, 0xff,
        ];
        let key: [u8; 16] = [
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a,
            0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
        ];
        let mut state = original;
        aes_round(&mut state, &key);
        assert_ne!(state, original);
        // To invert: XOR key, InvMixColumns, InvShiftRows, InvSubBytes
        add_round_key(&mut state, &key);
        inv_mix_columns(&mut state);
        inv_shift_rows(&mut state);
        inv_sub_bytes(&mut state);
        assert_eq!(state, original);
    }

    #[test]
    fn test_aes_last_round_inv_inverts() {
        // AESLastRound: SubBytes + ShiftRows + XOR key
        // Inverse: XOR key, InvShiftRows, InvSubBytes
        let original: [u8; 16] = [
            0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x01, 0x23, 0x45,
            0x67, 0x89, 0xab, 0xcd, 0xef,
        ];
        let key: [u8; 16] = [0x42; 16];
        let mut state = original;
        aes_last_round(&mut state, &key);
        assert_ne!(state, original);
        // Manual inverse
        add_round_key(&mut state, &key);
        inv_shift_rows(&mut state);
        inv_sub_bytes(&mut state);
        assert_eq!(state, original);
    }

    #[test]
    fn test_clmul_known_values() {
        // clmul(1, x) = x for any x
        assert_eq!(clmul_64(1, 0x1234567890abcdef), (0x1234567890abcdef, 0));
        // clmul(x, 1) = x for any x
        assert_eq!(clmul_64(0xdeadbeef, 1), (0xdeadbeef, 0));
        // clmul(2, 3) = 2*3 in GF(2)[x]: (x) * (x+1) = x^2 + x = 0b110 = 6
        assert_eq!(clmul_64(2, 3), (6, 0));
        // clmul(0x8000000000000000, 2) = x^64 -> hi=1, lo=0
        assert_eq!(clmul_64(0x8000000000000000, 2), (0, 1));
    }
}
