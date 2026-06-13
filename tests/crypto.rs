#![allow(clippy::undocumented_unsafe_blocks, clippy::needless_range_loop)]
//! Tests for SimdCrypto operations (AES round, carry-less multiply).
//!
//! Since SimdCrypto is not part of SimdOps (scalar can't implement it),
//! these tests use backend types directly rather than the dispatch mechanism.

#[cfg(target_arch = "x86_64")]
mod x86_tests {
    use highway::backend::avx2::Avx2;
    use highway::backend::avx512::Avx512;
    use highway::backend::sse2::Sse2;
    use highway::ops::{SimdCore, SimdCrypto, SimdMemory};

    fn has_sse2() -> bool {
        is_x86_feature_detected!("sse2")
    }
    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
    }
    fn has_avx512() -> bool {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512cd")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("avx512vl")
    }

    // -----------------------------------------------------------------------
    // Helper: extract bytes/u64s from vectors using SimdCore::extract_lane
    // -----------------------------------------------------------------------

    fn extract_bytes<S: SimdCore + SimdCrypto>(s: S, v: S::Vec<u8>) -> Vec<u8> {
        let n = S::VECTOR_BYTES;
        let mut out = vec![0u8; n];
        for i in 0..n {
            out[i] = unsafe { s.extract_lane(v, i) };
        }
        out
    }

    fn extract_u64s<S: SimdCore + SimdCrypto>(s: S, v: S::Vec<u64>) -> Vec<u64> {
        let n = S::VECTOR_BYTES / 8;
        let mut out = vec![0u64; n];
        for i in 0..n {
            out[i] = unsafe { s.extract_lane(v, i) };
        }
        out
    }

    // -----------------------------------------------------------------------
    // AES round: basic sanity (output != input, output != all zeros)
    // -----------------------------------------------------------------------

    #[test]
    fn test_aes_round_sse2_changes_state() {
        if !has_sse2() {
            return;
        }
        let s = Sse2;
        unsafe {
            let state = s.splat::<u8>(0x42);
            let key = s.splat::<u8>(0x00);
            let result = s.aes_round(state, key);
            let state_bytes = extract_bytes(s, state);
            let result_bytes = extract_bytes(s, result);
            assert_ne!(
                state_bytes, result_bytes,
                "AES round should change the state"
            );
            assert_ne!(
                result_bytes,
                vec![0u8; 16],
                "AES round should not produce all zeros"
            );
        }
    }

    #[test]
    fn test_aes_round_avx2_changes_state() {
        if !has_avx2() {
            return;
        }
        let s = Avx2;
        unsafe {
            let state = s.splat::<u8>(0x42);
            let key = s.splat::<u8>(0x00);
            let result = s.aes_round(state, key);
            let state_bytes = extract_bytes(s, state);
            let result_bytes = extract_bytes(s, result);
            assert_ne!(state_bytes, result_bytes);
        }
    }

    #[test]
    fn test_aes_round_avx512_changes_state() {
        if !has_avx512() {
            return;
        }
        let s = Avx512;
        unsafe {
            let state = s.splat::<u8>(0x42);
            let key = s.splat::<u8>(0x00);
            let result = s.aes_round(state, key);
            let state_bytes = extract_bytes(s, state);
            let result_bytes = extract_bytes(s, result);
            assert_ne!(state_bytes, result_bytes);
        }
    }

    // -----------------------------------------------------------------------
    // AES: all backends produce the same result per 128-bit block
    // -----------------------------------------------------------------------

    #[test]
    fn test_aes_round_backends_match() {
        if !has_sse2() {
            return;
        }

        let input: [u8; 16] = [
            0x19, 0x3d, 0xe3, 0xbe, 0xa0, 0xf4, 0xe2, 0x2b, 0x9a, 0xc6, 0x8d, 0x2a, 0xe9, 0xf8,
            0x48, 0x08,
        ];
        let key: [u8; 16] = [
            0xa0, 0xfa, 0xfe, 0x17, 0x88, 0x54, 0x2c, 0xb1, 0x23, 0xa3, 0x39, 0x39, 0x2a, 0x6c,
            0x76, 0x05,
        ];

        // SSE2 reference
        let sse2_result = unsafe {
            let s = Sse2;
            let state = s.load_u::<u8>(input.as_ptr());
            let rk = s.load_u::<u8>(key.as_ptr());
            extract_bytes(s, s.aes_round(state, rk))
        };

        // AVX2: first 128-bit block should match SSE2
        if has_avx2() {
            let avx2_result = unsafe {
                let s = Avx2;
                let mut buf = [0u8; 32];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 32];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_round(state, rk))
            };
            assert_eq!(
                &sse2_result[..],
                &avx2_result[..16],
                "AVX2 lower 128 bits should match SSE2"
            );
        }

        // AVX-512: first 128-bit block should match SSE2
        if has_avx512() {
            let avx512_result = unsafe {
                let s = Avx512;
                let mut buf = [0u8; 64];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 64];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_round(state, rk))
            };
            assert_eq!(
                &sse2_result[..],
                &avx512_result[..16],
                "AVX-512 lower 128 bits should match SSE2"
            );
        }
    }

    #[test]
    fn test_aes_last_round_backends_match() {
        if !has_sse2() {
            return;
        }

        let input: [u8; 16] = [
            0xeb, 0x40, 0xf2, 0x1e, 0x59, 0x2e, 0x38, 0x84, 0x8b, 0xa1, 0x13, 0xe7, 0x1b, 0xc3,
            0x42, 0xd2,
        ];
        let key: [u8; 16] = [
            0xd0, 0x14, 0xf9, 0xa8, 0xc9, 0xee, 0x25, 0x89, 0xe1, 0x3f, 0x0c, 0xc8, 0xb6, 0x63,
            0x0c, 0xa6,
        ];

        let sse2_result = unsafe {
            let s = Sse2;
            let state = s.load_u::<u8>(input.as_ptr());
            let rk = s.load_u::<u8>(key.as_ptr());
            extract_bytes(s, s.aes_last_round(state, rk))
        };

        if has_avx2() {
            let avx2_result = unsafe {
                let s = Avx2;
                let mut buf = [0u8; 32];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 32];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_last_round(state, rk))
            };
            assert_eq!(&sse2_result[..], &avx2_result[..16]);
        }

        if has_avx512() {
            let avx512_result = unsafe {
                let s = Avx512;
                let mut buf = [0u8; 64];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 64];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_last_round(state, rk))
            };
            assert_eq!(&sse2_result[..], &avx512_result[..16]);
        }
    }

    #[test]
    fn test_aes_round_inv_backends_match() {
        if !has_sse2() {
            return;
        }

        let input: [u8; 16] = [
            0x7a, 0x9f, 0x10, 0x27, 0x89, 0xd5, 0xf5, 0x0b, 0x2b, 0xef, 0xfd, 0x9f, 0x3d, 0xca,
            0x4e, 0xa7,
        ];
        let key: [u8; 16] = [0x13; 16];

        let sse2_result = unsafe {
            let s = Sse2;
            let state = s.load_u::<u8>(input.as_ptr());
            let rk = s.load_u::<u8>(key.as_ptr());
            extract_bytes(s, s.aes_round_inv(state, rk))
        };

        if has_avx2() {
            let avx2_result = unsafe {
                let s = Avx2;
                let mut buf = [0u8; 32];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 32];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_round_inv(state, rk))
            };
            assert_eq!(&sse2_result[..], &avx2_result[..16]);
        }

        if has_avx512() {
            let avx512_result = unsafe {
                let s = Avx512;
                let mut buf = [0u8; 64];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 64];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_round_inv(state, rk))
            };
            assert_eq!(&sse2_result[..], &avx512_result[..16]);
        }
    }

    #[test]
    fn test_aes_last_round_inv_backends_match() {
        if !has_sse2() {
            return;
        }

        let input: [u8; 16] = [
            0x63, 0x53, 0xe0, 0x8c, 0x09, 0x60, 0xe1, 0x04, 0xcd, 0x70, 0xb7, 0x51, 0xba, 0xca,
            0xd0, 0xe7,
        ];
        let key: [u8; 16] = [0xab; 16];

        let sse2_result = unsafe {
            let s = Sse2;
            let state = s.load_u::<u8>(input.as_ptr());
            let rk = s.load_u::<u8>(key.as_ptr());
            extract_bytes(s, s.aes_last_round_inv(state, rk))
        };

        if has_avx2() {
            let avx2_result = unsafe {
                let s = Avx2;
                let mut buf = [0u8; 32];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 32];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_last_round_inv(state, rk))
            };
            assert_eq!(&sse2_result[..], &avx2_result[..16]);
        }

        if has_avx512() {
            let avx512_result = unsafe {
                let s = Avx512;
                let mut buf = [0u8; 64];
                buf[..16].copy_from_slice(&input);
                let state = s.load_u::<u8>(buf.as_ptr());
                let mut kbuf = [0u8; 64];
                kbuf[..16].copy_from_slice(&key);
                let rk = s.load_u::<u8>(kbuf.as_ptr());
                extract_bytes(s, s.aes_last_round_inv(state, rk))
            };
            assert_eq!(&sse2_result[..], &avx512_result[..16]);
        }
    }

    // -----------------------------------------------------------------------
    // AES: multi-block consistency (AVX2 = 2 blocks, AVX-512 = 4 blocks)
    // -----------------------------------------------------------------------

    #[test]
    fn test_aes_round_multiblock_avx2() {
        if !has_avx2() || !has_sse2() {
            return;
        }

        let block0: [u8; 16] = [
            0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd,
            0xee, 0xff,
        ];
        let block1: [u8; 16] = [
            0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22,
            0x11, 0x00,
        ];
        let key: [u8; 16] = [
            0xde, 0xad, 0xbe, 0xef, 0xca, 0xfe, 0xba, 0xbe, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab,
            0xcd, 0xef,
        ];

        unsafe {
            // SSE2: process each block separately
            let s1 = Sse2;
            let r0 = extract_bytes(
                s1,
                s1.aes_round(
                    s1.load_u::<u8>(block0.as_ptr()),
                    s1.load_u::<u8>(key.as_ptr()),
                ),
            );
            let r1 = extract_bytes(
                s1,
                s1.aes_round(
                    s1.load_u::<u8>(block1.as_ptr()),
                    s1.load_u::<u8>(key.as_ptr()),
                ),
            );

            // AVX2: process both blocks at once
            let s2 = Avx2;
            let mut combined_state = [0u8; 32];
            combined_state[..16].copy_from_slice(&block0);
            combined_state[16..].copy_from_slice(&block1);
            let mut combined_key = [0u8; 32];
            combined_key[..16].copy_from_slice(&key);
            combined_key[16..].copy_from_slice(&key);

            let avx2_result = extract_bytes(
                s2,
                s2.aes_round(
                    s2.load_u::<u8>(combined_state.as_ptr()),
                    s2.load_u::<u8>(combined_key.as_ptr()),
                ),
            );

            assert_eq!(&r0[..], &avx2_result[..16], "AVX2 block 0 mismatch");
            assert_eq!(&r1[..], &avx2_result[16..], "AVX2 block 1 mismatch");
        }
    }

    #[test]
    fn test_aes_round_multiblock_avx512() {
        if !has_avx512() || !has_sse2() {
            return;
        }

        let blocks: [[u8; 16]; 4] = [
            [0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff],
            [0xff, 0xee, 0xdd, 0xcc, 0xbb, 0xaa, 0x99, 0x88, 0x77, 0x66, 0x55, 0x44, 0x33, 0x22, 0x11, 0x00],
            [0xde, 0xad, 0xbe, 0xef, 0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef, 0xfe, 0xdc, 0xba, 0x98],
            [0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42, 0x42],
        ];
        let key: [u8; 16] = [
            0xca, 0xfe, 0xba, 0xbe, 0xde, 0xad, 0xf0, 0x0d,
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
        ];

        unsafe {
            // SSE2: process each block separately
            let s1 = Sse2;
            let rk = s1.load_u::<u8>(key.as_ptr());
            let mut sse2_results = Vec::new();
            for block in &blocks {
                let state = s1.load_u::<u8>(block.as_ptr());
                sse2_results.push(extract_bytes(s1, s1.aes_round(state, rk)));
            }

            // AVX-512: process all 4 blocks at once
            let s5 = Avx512;
            let mut combined_state = [0u8; 64];
            for (i, block) in blocks.iter().enumerate() {
                combined_state[i * 16..(i + 1) * 16].copy_from_slice(block);
            }
            let mut combined_key = [0u8; 64];
            for i in 0..4 {
                combined_key[i * 16..(i + 1) * 16].copy_from_slice(&key);
            }

            let avx512_result = extract_bytes(
                s5,
                s5.aes_round(
                    s5.load_u::<u8>(combined_state.as_ptr()),
                    s5.load_u::<u8>(combined_key.as_ptr()),
                ),
            );

            for (i, sse2_r) in sse2_results.iter().enumerate() {
                assert_eq!(
                    &sse2_r[..],
                    &avx512_result[i * 16..(i + 1) * 16],
                    "AVX-512 block {} mismatch",
                    i
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // CLMul: known values
    // -----------------------------------------------------------------------

    #[test]
    fn test_clmul_lower_sse2() {
        if !has_sse2() {
            return;
        }
        let s = Sse2;
        unsafe {
            // clmul(1, x) = x
            let a = s.load_u::<u64>([1u64, 0xdead].as_ptr());
            let b = s.load_u::<u64>([0x1234567890abcdefu64, 0xffff].as_ptr());
            let r = extract_u64s(s, s.cl_mul_lower(a, b));
            assert_eq!(r[0], 0x1234567890abcdef);
            assert_eq!(r[1], 0);

            // clmul(2, 3) = 6 in GF(2)[x]
            let a = s.load_u::<u64>([2u64, 0].as_ptr());
            let b = s.load_u::<u64>([3u64, 0].as_ptr());
            let r = extract_u64s(s, s.cl_mul_lower(a, b));
            assert_eq!(r[0], 6);
            assert_eq!(r[1], 0);
        }
    }

    #[test]
    fn test_clmul_upper_sse2() {
        if !has_sse2() {
            return;
        }
        let s = Sse2;
        unsafe {
            // clmul_upper uses the upper u64 of each 128-bit block
            let a = s.load_u::<u64>([0u64, 1].as_ptr()); // upper = 1
            let b = s.load_u::<u64>([0u64, 0xdeadbeef].as_ptr()); // upper = 0xdeadbeef
            let r = extract_u64s(s, s.cl_mul_upper(a, b));
            assert_eq!(r[0], 0xdeadbeef);
            assert_eq!(r[1], 0);
        }
    }

    #[test]
    fn test_clmul_overflow() {
        if !has_sse2() {
            return;
        }
        let s = Sse2;
        unsafe {
            // clmul(0x8000000000000000, 2) should overflow into hi
            let a = s.load_u::<u64>([0x8000000000000000u64, 0].as_ptr());
            let b = s.load_u::<u64>([2u64, 0].as_ptr());
            let r = extract_u64s(s, s.cl_mul_lower(a, b));
            assert_eq!(r[0], 0, "lo should be 0");
            assert_eq!(r[1], 1, "hi should be 1 (x^64)");
        }
    }

    #[test]
    fn test_clmul_backends_match() {
        if !has_sse2() {
            return;
        }

        let a_lo = 0x123456789abcdef0u64;
        let a_hi = 0xfedcba9876543210u64;
        let b_lo = 0x0f0f0f0f0f0f0f0fu64;
        let b_hi = 0xf0f0f0f0f0f0f0f0u64;

        // SSE2 reference
        let (sse2_lower, sse2_upper) = unsafe {
            let s = Sse2;
            let a = s.load_u::<u64>([a_lo, a_hi].as_ptr());
            let b = s.load_u::<u64>([b_lo, b_hi].as_ptr());
            (
                extract_u64s(s, s.cl_mul_lower(a, b)),
                extract_u64s(s, s.cl_mul_upper(a, b)),
            )
        };

        // AVX2: each 128-bit block should independently match SSE2
        if has_avx2() {
            let s2 = Avx2;
            let (avx2_lower, avx2_upper) = unsafe {
                let a = s2.load_u::<u64>([a_lo, a_hi, a_lo, a_hi].as_ptr());
                let b = s2.load_u::<u64>([b_lo, b_hi, b_lo, b_hi].as_ptr());
                (
                    extract_u64s(s2, s2.cl_mul_lower(a, b)),
                    extract_u64s(s2, s2.cl_mul_upper(a, b)),
                )
            };
            // Block 0
            assert_eq!(sse2_lower[0], avx2_lower[0], "CLMul lower block0 lo");
            assert_eq!(sse2_lower[1], avx2_lower[1], "CLMul lower block0 hi");
            assert_eq!(sse2_upper[0], avx2_upper[0], "CLMul upper block0 lo");
            assert_eq!(sse2_upper[1], avx2_upper[1], "CLMul upper block0 hi");
            // Block 1 (same data)
            assert_eq!(sse2_lower[0], avx2_lower[2]);
            assert_eq!(sse2_lower[1], avx2_lower[3]);
        }

        // AVX-512
        if has_avx512() {
            let s5 = Avx512;
            let (avx512_lower, avx512_upper) = unsafe {
                let a = s5.load_u::<u64>(
                    [a_lo, a_hi, a_lo, a_hi, a_lo, a_hi, a_lo, a_hi].as_ptr(),
                );
                let b = s5.load_u::<u64>(
                    [b_lo, b_hi, b_lo, b_hi, b_lo, b_hi, b_lo, b_hi].as_ptr(),
                );
                (
                    extract_u64s(s5, s5.cl_mul_lower(a, b)),
                    extract_u64s(s5, s5.cl_mul_upper(a, b)),
                )
            };
            assert_eq!(sse2_lower[0], avx512_lower[0]);
            assert_eq!(sse2_lower[1], avx512_lower[1]);
            assert_eq!(sse2_upper[0], avx512_upper[0]);
            assert_eq!(sse2_upper[1], avx512_upper[1]);
        }
    }
}
