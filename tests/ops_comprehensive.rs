#![allow(
    unused_unsafe,
    clippy::undocumented_unsafe_blocks,
    clippy::needless_range_loop,
    clippy::manual_div_ceil,
    clippy::too_many_lines
)]
//! Comprehensive tests for ALL SimdOps operations across all available targets.
//!
//! Each test exercises a specific SIMD operation, computes expected results
//! using plain Rust, and compares against the SIMD output on every available target.

use highway::ops::{A128, Aligned};
use highway::{SimdCrypto, SimdOps, TargetId, WithSimd, dispatch_to};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn available_targets() -> Vec<TargetId> {
    let mut targets = vec![TargetId::Scalar];
    let best = highway::dispatch::detect_best_target();
    match best {
        TargetId::Avx512 => {
            targets.push(TargetId::Sse2);
            targets.push(TargetId::Avx2);
            targets.push(TargetId::Avx512);
        }
        TargetId::Avx2 => {
            targets.push(TargetId::Sse2);
            targets.push(TargetId::Avx2);
        }
        TargetId::Sse2 => {
            targets.push(TargetId::Sse2);
        }
        _ => {}
    }
    targets
}

// Helper to get lane count for a given type on a target.
struct LaneCount<T: highway::Lane>(core::marker::PhantomData<T>);
impl<T: highway::Lane> WithSimd for LaneCount<T> {
    type Output = usize;
    fn with_simd<S: SimdOps>(self, s: S) -> usize {
        s.lanes::<T>()
    }
}

fn lanes_for<T: highway::Lane>(target: TargetId) -> usize {
    dispatch_to(LaneCount::<T>(core::marker::PhantomData), target)
}

// =========================================================================
// SimdCore tests
// =========================================================================

// ---------------------------------------------------------------------------
// iota: check lane[i] == base + i
// ---------------------------------------------------------------------------

struct IotaKernel<'a> {
    base: u32,
    out: &'a mut [u32],
}

impl WithSimd for IotaKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.iota(self.base);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(v, i);
            }
        }
    }
}

#[test]
fn test_iota() {
    let base = 10u32;
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let mut out = vec![0u32; lanes];
        dispatch_to(
            IotaKernel {
                base,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i],
                base + i as u32,
                "iota: lane {i} wrong for {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// get_lane: check returns lane 0
// ---------------------------------------------------------------------------

struct GetLaneKernel {
    value: i32,
}

impl WithSimd for GetLaneKernel {
    type Output = i32;
    fn with_simd<S: SimdOps>(self, s: S) -> i32 {
        unsafe {
            let v = s.splat(self.value);
            s.get_lane(v)
        }
    }
}

#[test]
fn test_get_lane() {
    let value = 42i32;
    for target in available_targets() {
        let result = dispatch_to(GetLaneKernel { value }, target);
        assert_eq!(result, value, "get_lane failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// bitcast: u32 -> f32 -> u32 roundtrip
// ---------------------------------------------------------------------------

struct BitcastRoundtripKernel {
    value: u32,
}

impl WithSimd for BitcastRoundtripKernel {
    type Output = u32;
    fn with_simd<S: SimdOps>(self, s: S) -> u32 {
        unsafe {
            let v_u32 = s.splat(self.value);
            let v_f32: S::Vec<f32> = s.bitcast(v_u32);
            let v_u32_back: S::Vec<u32> = s.bitcast(v_f32);
            s.extract_lane(v_u32_back, 0)
        }
    }
}

#[test]
fn test_bitcast_roundtrip() {
    let value = 0x40490FDBu32; // f32 PI bit pattern
    for target in available_targets() {
        let result = dispatch_to(BitcastRoundtripKernel { value }, target);
        assert_eq!(result, value, "bitcast roundtrip failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// insert_lane + extract_lane roundtrip
// ---------------------------------------------------------------------------

struct InsertExtractKernel {
    value: u32,
}

impl WithSimd for InsertExtractKernel {
    type Output = u32;
    fn with_simd<S: SimdOps>(self, s: S) -> u32 {
        unsafe {
            let v = s.zero::<u32>();
            let v = s.insert_lane(v, 0, self.value);
            s.extract_lane(v, 0)
        }
    }
}

#[test]
fn test_insert_extract_lane() {
    let value = 12345u32;
    for target in available_targets() {
        let result = dispatch_to(InsertExtractKernel { value }, target);
        assert_eq!(
            result, value,
            "insert_lane + extract_lane failed for {target:?}"
        );
    }
}

// =========================================================================
// SimdMemory tests
// =========================================================================

// ---------------------------------------------------------------------------
// stream: non-temporal store, verify data matches store_u
// ---------------------------------------------------------------------------

struct StreamKernel<'a> {
    data: &'a [u32],
    stream_out: &'a mut [u32],
    store_out: &'a mut [u32],
}

impl WithSimd for StreamKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.data.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.data.as_ptr());
            s.stream(v, self.stream_out.as_mut_ptr());
            s.store_u(v, self.store_out.as_mut_ptr());
        }
    }
}

#[test]
#[cfg_attr(miri, ignore)]
fn test_stream() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let data: Vec<u32> = (1..=64).collect();
        // stream requires an aligned pointer — use a stack-aligned array
        let mut stream_buf = Aligned::<A128, [u32; 16]>::new([0u32; 16]);
        let mut store_out = vec![0u32; lanes];
        dispatch_to(
            StreamKernel {
                data: &data,
                stream_out: &mut stream_buf.value[..lanes],
                store_out: &mut store_out,
            },
            target,
        );
        assert_eq!(
            &stream_buf.value[..lanes],
            &store_out[..lanes],
            "stream vs store_u mismatch for {target:?}"
        );
    }
}

// =========================================================================
// SimdArith tests
// =========================================================================

// ---------------------------------------------------------------------------
// neg: negate f32 values
// ---------------------------------------------------------------------------

struct NegF32Kernel<'a> {
    input: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for NegF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.neg(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = -self.input[i];
            i += 1;
        }
    }
}

#[test]
fn test_neg_f32() {
    let n = 32;
    let input: Vec<f32> = (0..n).map(|i| i as f32 * 1.5 - 10.0).collect();
    let expected: Vec<f32> = input.iter().map(|x| -x).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            NegF32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-6,
                "neg f32 mismatch at {j} for {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// sub: f32 subtraction
// ---------------------------------------------------------------------------

struct SubF32Kernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for SubF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let r = s.sub(va, vb);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i] - self.b[i];
            i += 1;
        }
    }
}

#[test]
fn test_sub_f32() {
    let n = 32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 2.5).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 1.1 + 3.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x - y).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            SubF32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-4,
                "sub f32 mismatch at {j} for {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// div: f32 division
// ---------------------------------------------------------------------------

struct DivF32Kernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for DivF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let r = s.div(va, vb);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i] / self.b[i];
            i += 1;
        }
    }
}

#[test]
fn test_div_f32() {
    let n = 32;
    let a: Vec<f32> = (1..=n as i32).map(|i| i as f32 * 10.0).collect();
    let b: Vec<f32> = (1..=n as i32).map(|i| i as f32 * 0.5 + 1.0).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x / y).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            DivF32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-4,
                "div f32 mismatch at {j} for {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// mul_high: i16 high multiplication
// ---------------------------------------------------------------------------

struct MulHighI16Kernel<'a> {
    a: &'a [i16],
    b: &'a [i16],
    out: &'a mut [i16],
}

impl WithSimd for MulHighI16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i16>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let r = s.mul_high(va, vb);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = ((self.a[i] as i32 * self.b[i] as i32) >> 16) as i16;
            i += 1;
        }
    }
}

#[test]
fn test_mul_high_i16() {
    let n = 32;
    let a: Vec<i16> = (0..n as i16).map(|i| i * 100 - 500).collect();
    let b: Vec<i16> = (0..n as i16).map(|i| i * 50 + 100).collect();
    let expected: Vec<i16> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| ((*x as i32 * *y as i32) >> 16) as i16)
        .collect();

    for target in available_targets() {
        let mut out = vec![0i16; n];
        dispatch_to(
            MulHighI16Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "mul_high i16 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// average_round: u8 averaging
// ---------------------------------------------------------------------------

struct AverageRoundU8Kernel<'a> {
    a: &'a [u8],
    b: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for AverageRoundU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let r = s.average_round(va, vb);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] =
                (self.a[i] as u16 + self.b[i] as u16).div_ceil(2) as u8;
            i += 1;
        }
    }
}

#[test]
fn test_average_round_u8() {
    let n = 64;
    let a: Vec<u8> = (0..n as u8).map(|i| i.wrapping_mul(3)).collect();
    let b: Vec<u8> = (0..n as u8).map(|i| 200u8.wrapping_sub(i)).collect();
    let expected: Vec<u8> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (*x as u16 + *y as u16).div_ceil(2) as u8)
        .collect();

    for target in available_targets() {
        let mut out = vec![0u8; n];
        dispatch_to(
            AverageRoundU8Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "average_round u8 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// mul_even: u32 -> u64 even multiplication
// ---------------------------------------------------------------------------

struct MulEvenU32Kernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u64],
}

impl WithSimd for MulEvenU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_u32 = s.lanes::<u32>();
        let lanes_u64 = s.lanes::<u64>();
        if lanes_u32 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_even(va, vb);
            for i in 0..lanes_u64.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_even_u32() {
    for target in available_targets() {
        let lanes_u32 = lanes_for::<u32>(target);
        let lanes_u64 = lanes_for::<u64>(target);
        let a: Vec<u32> = (0..lanes_u32 as u32).map(|i| i * 100 + 50).collect();
        let b: Vec<u32> =
            (0..lanes_u32 as u32).map(|i| i * 200 + 100).collect();
        let mut out = vec![0u64; lanes_u64];

        dispatch_to(
            MulEvenU32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        // mul_even multiplies lanes at even indices (0, 2, 4, ...) to produce wide results
        for i in 0..lanes_u64 {
            let src_idx = i * 2;
            if src_idx < lanes_u32 {
                let expected = a[src_idx] as u64 * b[src_idx] as u64;
                assert_eq!(
                    out[i], expected,
                    "mul_even u32 lane {i} (src {src_idx}) wrong for {target:?}: got {}, expected {}",
                    out[i], expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// saturated_sub: u8 saturating subtraction
// ---------------------------------------------------------------------------

struct SaturatedSubU8Kernel<'a> {
    a: &'a [u8],
    b: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for SaturatedSubU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let r = s.saturated_sub(va, vb);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i].saturating_sub(self.b[i]);
            i += 1;
        }
    }
}

#[test]
fn test_saturated_sub_u8() {
    let n = 64;
    let a: Vec<u8> = (0..n as u8).map(|i| i.wrapping_mul(5)).collect();
    let b: Vec<u8> = (0..n as u8).map(|i| 100u8.wrapping_add(i)).collect();
    let expected: Vec<u8> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x.saturating_sub(*y))
        .collect();

    for target in available_targets() {
        let mut out = vec![0u8; n];
        dispatch_to(
            SaturatedSubU8Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "saturated_sub u8 failed for {target:?}");
    }
}

// =========================================================================
// SimdBitwise tests
// =========================================================================

// ---------------------------------------------------------------------------
// and_not: verify ~a & b
// ---------------------------------------------------------------------------

struct AndNotKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for AndNotKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let r = s.and_not(va, vb);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = !self.a[i] & self.b[i];
            i += 1;
        }
    }
}

#[test]
fn test_and_not() {
    let n = 32;
    let a: Vec<u32> = (0..n as u32)
        .map(|i| 0xFF00_FF00 ^ i.wrapping_mul(0x0101_0101))
        .collect();
    let b: Vec<u32> = (0..n as u32)
        .map(|i| 0x00FF_00FF | i.wrapping_mul(0x1010_1010))
        .collect();
    let expected: Vec<u32> = a.iter().zip(&b).map(|(x, y)| !x & y).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            AndNotKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "and_not failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// rotate_right: u32 rotation
// ---------------------------------------------------------------------------

struct RotateRightKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for RotateRightKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r: S::Vec<u32> = s.rotate_right::<u32, 7>(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].rotate_right(7);
            i += 1;
        }
    }
}

#[test]
fn test_rotate_right() {
    let n = 32;
    let input: Vec<u32> = (0..n as u32)
        .map(|i| 0xDEAD_BEEF ^ i.wrapping_mul(0x1234_5678))
        .collect();
    let expected: Vec<u32> = input.iter().map(|x| x.rotate_right(7)).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            RotateRightKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "rotate_right failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// shift_left_bytes / shift_right_bytes: byte-level shift within 128-bit blocks
// ---------------------------------------------------------------------------

struct ShiftLeftBytesKernel<'a> {
    input: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for ShiftLeftBytesKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r: S::Vec<u8> = s.shift_left_bytes::<u8, 2>(v);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

struct ShiftRightBytesKernel<'a> {
    input: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for ShiftRightBytesKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r: S::Vec<u8> = s.shift_right_bytes::<u8, 2>(v);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_shift_left_bytes() {
    for target in available_targets() {
        let lanes = lanes_for::<u8>(target);
        let input: Vec<u8> = (1..=64).collect();
        let mut out = vec![0u8; lanes];
        dispatch_to(
            ShiftLeftBytesKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // Within each 128-bit block (16 bytes), shifting left by 2 bytes:
        // bytes shift toward higher indices, low 2 bytes become 0
        for block_start in (0..lanes).step_by(16) {
            let block_end = (block_start + 16).min(lanes);
            for i in block_start..block_end {
                let offset_in_block = i - block_start;
                let expected = if offset_in_block < 2 {
                    0u8
                } else {
                    input[i - 2]
                };
                assert_eq!(
                    out[i], expected,
                    "shift_left_bytes: byte {i} wrong for {target:?}: got {}, expected {}",
                    out[i], expected
                );
            }
        }
    }
}

#[test]
fn test_shift_right_bytes() {
    for target in available_targets() {
        let lanes = lanes_for::<u8>(target);
        let input: Vec<u8> = (1..=64).collect();
        let mut out = vec![0u8; lanes];
        dispatch_to(
            ShiftRightBytesKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // Within each 128-bit block (16 bytes), shifting right by 2 bytes:
        // bytes shift toward lower indices, top 2 bytes become 0
        for block_start in (0..lanes).step_by(16) {
            let block_end = (block_start + 16).min(lanes);
            let block_size = block_end - block_start;
            for i in block_start..block_end {
                let offset_in_block = i - block_start;
                let expected = if offset_in_block + 2 >= block_size {
                    0u8
                } else {
                    input[i + 2]
                };
                assert_eq!(
                    out[i], expected,
                    "shift_right_bytes: byte {i} wrong for {target:?}: got {}, expected {}",
                    out[i], expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// trailing_zero_count: u32
// ---------------------------------------------------------------------------

struct TrailingZeroCountKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for TrailingZeroCountKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.trailing_zero_count(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].trailing_zeros();
            i += 1;
        }
    }
}

#[test]
fn test_trailing_zero_count() {
    let n = 32;
    let input: Vec<u32> = (0..n as u32)
        .map(|i| {
            if i == 0 {
                0x80000000 // trailing zeros = 31
            } else {
                1u32 << (i % 32)
            }
        })
        .collect();
    let expected: Vec<u32> = input.iter().map(|x| x.trailing_zeros()).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            TrailingZeroCountKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "trailing_zero_count failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// reverse_bits: u8
// ---------------------------------------------------------------------------

struct ReverseBitsU8Kernel<'a> {
    input: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for ReverseBitsU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.reverse_bits(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].reverse_bits();
            i += 1;
        }
    }
}

#[test]
fn test_reverse_bits_u8() {
    let n = 64;
    let input: Vec<u8> = (0..n as u8).collect();
    let expected: Vec<u8> = input.iter().map(|x| x.reverse_bits()).collect();

    for target in available_targets() {
        let mut out = vec![0u8; n];
        dispatch_to(
            ReverseBitsU8Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "reverse_bits u8 failed for {target:?}");
    }
}

// =========================================================================
// SimdCompare tests
// =========================================================================

// ---------------------------------------------------------------------------
// ne, le, ge comparisons
// ---------------------------------------------------------------------------

struct CompareKernel<'a> {
    a: &'a [i32],
    b: &'a [i32],
}

impl WithSimd for CompareKernel<'_> {
    type Output = (Vec<bool>, Vec<bool>, Vec<bool>); // (ne, le, ge)
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        let n = lanes.min(self.a.len());
        let mut ne_results = Vec::new();
        let mut le_results = Vec::new();
        let mut ge_results = Vec::new();
        if n == 0 {
            return (ne_results, le_results, ge_results);
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());

            let ne_mask = s.ne(va, vb);
            let le_mask = s.le(va, vb);
            let ge_mask = s.ge(va, vb);

            let ne_vec = s.vec_from_mask(ne_mask);
            let le_vec = s.vec_from_mask(le_mask);
            let ge_vec = s.vec_from_mask(ge_mask);

            for i in 0..n {
                ne_results.push(s.extract_lane(ne_vec, i) != 0);
                le_results.push(s.extract_lane(le_vec, i) != 0);
                ge_results.push(s.extract_lane(ge_vec, i) != 0);
            }
        }
        (ne_results, le_results, ge_results)
    }
}

#[test]
fn test_ne_le_ge() {
    let a = vec![1i32, 5, 3, 7, 10, -1, 0, 4, 2, 6, 8, 9, -5, 3, 7, 11];
    let b = vec![1i32, 3, 5, 7, 2, -1, 1, 4, 9, 6, 0, 9, 5, 3, -7, 11];

    for target in available_targets() {
        let (ne_r, le_r, ge_r) =
            dispatch_to(CompareKernel { a: &a, b: &b }, target);
        let lanes = ne_r.len();
        for i in 0..lanes {
            assert_eq!(ne_r[i], a[i] != b[i], "ne wrong at {i} for {target:?}");
            assert_eq!(le_r[i], a[i] <= b[i], "le wrong at {i} for {target:?}");
            assert_eq!(ge_r[i], a[i] >= b[i], "ge wrong at {i} for {target:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// test_bit: check specific bit positions
// ---------------------------------------------------------------------------

struct TestBitKernel<'a> {
    values: &'a [u32],
    bits: &'a [u32],
}

impl WithSimd for TestBitKernel<'_> {
    type Output = Vec<bool>;
    fn with_simd<S: SimdOps>(self, s: S) -> Vec<bool> {
        let lanes = s.lanes::<u32>();
        let n = lanes.min(self.values.len());
        let mut results = Vec::new();
        if n == 0 {
            return results;
        }
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            let b = s.load_u(self.bits.as_ptr());
            let mask = s.test_bit(v, b);
            let vec = s.vec_from_mask(mask);
            for i in 0..n {
                results.push(s.extract_lane(vec, i) != 0);
            }
        }
        results
    }
}

#[test]
fn test_test_bit() {
    // test_bit checks if (v & bit) != 0 — bit is a bitmask, not a position
    let values = vec![
        0b1010u32, 0b0101, 0b1111, 0b0000, 0xFF, 0x80, 0x01, 0xAB, 0b1010,
        0b0101, 0b1111, 0b0000, 0xFF, 0x80, 0x01, 0xAB,
    ];
    let bits = vec![
        0b0010u32, 0b0001, 0b1000, 0b0001, 0x80, 0x80, 0x01, 0x20, 0b0010,
        0b0001, 0b1000, 0b0001, 0x80, 0x80, 0x01, 0x20,
    ];

    for target in available_targets() {
        let results = dispatch_to(
            TestBitKernel {
                values: &values,
                bits: &bits,
            },
            target,
        );
        let lanes = results.len();
        for i in 0..lanes {
            let expected = (values[i] & bits[i]) != 0;
            assert_eq!(
                results[i], expected,
                "test_bit wrong at {i} for {target:?}: value={:#b}, bit={:#b}",
                values[i], bits[i]
            );
        }
    }
}

// =========================================================================
// SimdMask tests
// =========================================================================

// ---------------------------------------------------------------------------
// mask_from_vec + vec_from_mask roundtrip
// ---------------------------------------------------------------------------

struct MaskRoundtripKernel<'a> {
    input: &'a [i32],
}

impl WithSimd for MaskRoundtripKernel<'_> {
    type Output = Vec<i32>;
    fn with_simd<S: SimdOps>(self, s: S) -> Vec<i32> {
        let lanes = s.lanes::<i32>();
        let n = lanes.min(self.input.len());
        let mut results = Vec::new();
        if n == 0 {
            return results;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let mask = s.mask_from_vec(v);
            let v_back = s.vec_from_mask(mask);
            for i in 0..n {
                results.push(s.extract_lane(v_back, i));
            }
        }
        results
    }
}

#[test]
fn test_mask_from_vec_roundtrip() {
    // mask_from_vec: lane != 0 -> true; vec_from_mask: true -> all-ones, false -> zero
    let input =
        vec![0i32, 1, -1, 0, 42, 0, -100, 7, 0, 1, -1, 0, 42, 0, -100, 7];

    for target in available_targets() {
        let results =
            dispatch_to(MaskRoundtripKernel { input: &input }, target);
        let lanes = results.len();
        for i in 0..lanes {
            if input[i] != 0 {
                // true -> all-ones (as i32, that is -1)
                assert_eq!(
                    results[i], -1,
                    "mask_from_vec+vec_from_mask: non-zero should give all-ones at {i} for {target:?}"
                );
            } else {
                assert_eq!(
                    results[i], 0,
                    "mask_from_vec+vec_from_mask: zero should give zero at {i} for {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// first_n: check first N lanes set
// ---------------------------------------------------------------------------

struct FirstNKernel {
    n: usize,
}

impl WithSimd for FirstNKernel {
    type Output = Vec<bool>;
    fn with_simd<S: SimdOps>(self, s: S) -> Vec<bool> {
        let lanes = s.lanes::<u32>();
        let mut results = Vec::new();
        unsafe {
            let mask: S::Mask<u32> = s.first_n(self.n);
            let vec = s.vec_from_mask(mask);
            for i in 0..lanes {
                results.push(s.extract_lane(vec, i) != 0);
            }
        }
        results
    }
}

#[test]
fn test_first_n() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        // Test with n = half of lanes
        let n_set = (lanes + 1) / 2;
        let results = dispatch_to(FirstNKernel { n: n_set }, target);
        assert_eq!(results.len(), lanes);
        for i in 0..lanes {
            if i < n_set {
                assert!(
                    results[i],
                    "first_n: lane {i} should be set for n={n_set}, {target:?}"
                );
            } else {
                assert!(
                    !results[i],
                    "first_n: lane {i} should NOT be set for n={n_set}, {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// if_then_else_zero, if_then_zero_else
// ---------------------------------------------------------------------------

struct IfThenElseZeroKernel<'a> {
    mask_input: &'a [i32],
    values: &'a [i32],
}

impl WithSimd for IfThenElseZeroKernel<'_> {
    type Output = (Vec<i32>, Vec<i32>); // (if_then_else_zero, if_then_zero_else)
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        let n = lanes.min(self.mask_input.len());
        let mut iftez = Vec::new();
        let mut iftze = Vec::new();
        if n == 0 {
            return (iftez, iftze);
        }
        unsafe {
            let mask_vec = s.load_u(self.mask_input.as_ptr());
            let mask = s.mask_from_vec(mask_vec);
            let vals = s.load_u(self.values.as_ptr());

            let r1 = s.if_then_else_zero(mask, vals);
            let r2 = s.if_then_zero_else(mask, vals);

            for i in 0..n {
                iftez.push(s.extract_lane(r1, i));
                iftze.push(s.extract_lane(r2, i));
            }
        }
        (iftez, iftze)
    }
}

#[test]
fn test_if_then_else_zero_and_if_then_zero_else() {
    let mask_input =
        vec![0i32, -1, 0, -1, 1, 0, -1, 1, 0, -1, 0, -1, 1, 0, -1, 1];
    let values = vec![
        10i32, 20, 30, 40, 50, 60, 70, 80, 10, 20, 30, 40, 50, 60, 70, 80,
    ];

    for target in available_targets() {
        let (iftez, iftze) = dispatch_to(
            IfThenElseZeroKernel {
                mask_input: &mask_input,
                values: &values,
            },
            target,
        );
        let lanes = iftez.len();
        for i in 0..lanes {
            let mask_true = mask_input[i] != 0;
            if mask_true {
                assert_eq!(
                    iftez[i], values[i],
                    "if_then_else_zero: true lane {i} should have value, {target:?}"
                );
                assert_eq!(
                    iftze[i], 0,
                    "if_then_zero_else: true lane {i} should be zero, {target:?}"
                );
            } else {
                assert_eq!(
                    iftez[i], 0,
                    "if_then_else_zero: false lane {i} should be zero, {target:?}"
                );
                assert_eq!(
                    iftze[i], values[i],
                    "if_then_zero_else: false lane {i} should have value, {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// all_true, all_false
// ---------------------------------------------------------------------------

struct AllTrueFalseKernel<'a> {
    input: &'a [i32],
}

impl WithSimd for AllTrueFalseKernel<'_> {
    type Output = (bool, bool); // (all_true, all_false)
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        if lanes > self.input.len() {
            return (false, true);
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let mask = s.mask_from_vec(v);
            (s.all_true(mask), s.all_false(mask))
        }
    }
}

#[test]
fn test_all_true_all_false() {
    // All-ones
    let all_nonzero = vec![-1i32; 64];
    // All-zeros
    let all_zero = vec![0i32; 64];
    // Mixed
    let mixed = vec![1i32, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0];

    for target in available_targets() {
        let (at, af) = dispatch_to(
            AllTrueFalseKernel {
                input: &all_nonzero,
            },
            target,
        );
        assert!(at, "all_true should be true for all-nonzero, {target:?}");
        assert!(!af, "all_false should be false for all-nonzero, {target:?}");

        let (at, af) =
            dispatch_to(AllTrueFalseKernel { input: &all_zero }, target);
        assert!(!at, "all_true should be false for all-zero, {target:?}");
        assert!(af, "all_false should be true for all-zero, {target:?}");

        let lanes = lanes_for::<i32>(target);
        if lanes > 1 {
            let (at, af) =
                dispatch_to(AllTrueFalseKernel { input: &mixed }, target);
            assert!(!at, "all_true should be false for mixed, {target:?}");
            assert!(!af, "all_false should be false for mixed, {target:?}");
        }
    }
}

// =========================================================================
// SimdConvert tests
// =========================================================================

// ---------------------------------------------------------------------------
// promote_to: u16 -> u32
// ---------------------------------------------------------------------------

struct PromoteU16Kernel<'a> {
    input: &'a [u16],
    out: &'a mut [u32],
}

impl WithSimd for PromoteU16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_narrow = s.lanes::<u16>();
        let lanes_wide = s.lanes::<u32>();
        if lanes_narrow > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let promoted: S::Vec<u32> = s.promote_to(v);
            for i in 0..lanes_wide.min(self.out.len()) {
                self.out[i] = s.extract_lane(promoted, i);
            }
        }
    }
}

#[test]
fn test_promote_u16_to_u32() {
    for target in available_targets() {
        let lanes_u16 = lanes_for::<u16>(target);
        let lanes_u32 = lanes_for::<u32>(target);
        let input: Vec<u16> =
            (0..lanes_u16 as u16).map(|i| i * 1000 + 500).collect();
        let mut out = vec![0u32; lanes_u32];

        dispatch_to(
            PromoteU16Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        // promote takes the lower half of the narrow vector
        for i in 0..lanes_u32 {
            if i < lanes_u16 {
                assert_eq!(
                    out[i], input[i] as u32,
                    "promote_to u16->u32: lane {i} wrong for {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// demote_to: i32 -> i16 (with saturation)
// ---------------------------------------------------------------------------

struct DemoteI32Kernel<'a> {
    input: &'a [i32],
    out: &'a mut [i16],
}

impl WithSimd for DemoteI32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_wide = s.lanes::<i32>();
        let lanes_narrow = s.lanes::<i16>();
        if lanes_wide > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let demoted: S::Vec<i16> = s.demote_to(v);
            for i in 0..lanes_narrow.min(self.out.len()) {
                self.out[i] = s.extract_lane(demoted, i);
            }
        }
    }
}

#[test]
fn test_demote_i32_to_i16() {
    for target in available_targets() {
        let lanes_i32 = lanes_for::<i32>(target);
        let lanes_i16 = lanes_for::<i16>(target);
        // Use values that both fit and overflow i16 range
        let input: Vec<i32> = (0..lanes_i32 as i32)
            .map(|i| {
                if i % 3 == 0 {
                    i * 100 // fits in i16
                } else if i % 3 == 1 {
                    40000 // overflows -> saturates to 32767
                } else {
                    -40000 // underflows -> saturates to -32768
                }
            })
            .collect();
        let mut out = vec![0i16; lanes_i16];

        dispatch_to(
            DemoteI32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        // demote fills the lower half of the narrow vector from the wide vector
        for i in 0..lanes_i32.min(lanes_i16) {
            let expected =
                input[i].clamp(i16::MIN as i32, i16::MAX as i32) as i16;
            assert_eq!(
                out[i], expected,
                "demote_to i32->i16: lane {i} wrong for {target:?}, input={}, got={}, expected={}",
                input[i], out[i], expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// convert_to_int + convert_to_float roundtrip
// ---------------------------------------------------------------------------

struct ConvertRoundtripKernel<'a> {
    input: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for ConvertRoundtripKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v_f32 = s.load_u(self.input.as_ptr().add(i));
                let v_i32: S::Vec<i32> = s.convert_to_int(v_f32);
                let v_f32_back: S::Vec<f32> = s.convert_to_float(v_i32);
                s.store_u(v_f32_back, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = (self.input[i] as i32) as f32;
            i += 1;
        }
    }
}

#[test]
fn test_convert_roundtrip() {
    let n = 32;
    // Use integer-valued floats so the roundtrip is exact
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 10.0 - 100.0).collect();
    let expected: Vec<f32> = input.iter().map(|x| (*x as i32) as f32).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            ConvertRoundtripKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "convert roundtrip failed for {target:?}");
    }
}

// =========================================================================
// SimdShuffle tests
// =========================================================================

// ---------------------------------------------------------------------------
// reverse: full vector reverse
// ---------------------------------------------------------------------------

struct ReverseKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ReverseKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.reverse(v);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_reverse() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (0..lanes as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            ReverseKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let expected: Vec<u32> = input.iter().copied().rev().collect();
        assert_eq!(out, expected, "reverse failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// broadcast_lane: broadcast lane 0
// ---------------------------------------------------------------------------

struct BroadcastLane0Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for BroadcastLane0Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r: S::Vec<u32> = s.broadcast_lane::<u32, 0>(v);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_broadcast_lane() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (100..100 + lanes as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            BroadcastLane0Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i], input[0],
                "broadcast_lane::<0>: lane {i} should be {}, got {} for {target:?}",
                input[0], out[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// interleave_lower, interleave_upper
// ---------------------------------------------------------------------------

struct InterleaveLowerKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for InterleaveLowerKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.interleave_lower(va, vb);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

struct InterleaveUpperKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for InterleaveUpperKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.interleave_upper(va, vb);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_interleave_lower() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let a: Vec<u32> = (0..lanes as u32).map(|i| 100 + i).collect();
        let b: Vec<u32> = (0..lanes as u32).map(|i| 200 + i).collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            InterleaveLowerKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        // interleave_lower (x86 unpacklo) operates within each 128-bit lane.
        // For 4 u32 lanes (SSE2): [a0,b0,a1,b1]
        // For 8 u32 lanes (AVX2): [a0,b0,a1,b1, a4,b4,a5,b5]
        //   (lower half of each 128-bit lane interleaved)
        if lanes > 1 {
            let lanes_per_128 = 128 / (core::mem::size_of::<u32>() * 8); // 4 for u32
            let _half_128 = lanes_per_128 / 2;
            for block in 0..(lanes / lanes_per_128) {
                let block_start = block * lanes_per_128;
                let src_block_start = block * lanes_per_128;
                for j in 0..lanes_per_128 {
                    let expected = if j % 2 == 0 {
                        a[src_block_start + j / 2]
                    } else {
                        b[src_block_start + j / 2]
                    };
                    assert_eq!(
                        out[block_start + j],
                        expected,
                        "interleave_lower: lane {} wrong for {target:?}",
                        block_start + j,
                    );
                }
            }
        }
    }
}

#[test]
fn test_interleave_upper() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes <= 1 {
            continue; // No meaningful upper half for scalar
        }
        let a: Vec<u32> = (0..lanes as u32).map(|i| 100 + i).collect();
        let b: Vec<u32> = (0..lanes as u32).map(|i| 200 + i).collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            InterleaveUpperKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        // interleave_upper (x86 unpackhi) operates within each 128-bit lane.
        // For 4 u32 lanes (SSE2): [a2,b2,a3,b3]
        // For 8 u32 lanes (AVX2): [a2,b2,a3,b3, a6,b6,a7,b7]
        let lanes_per_128 = 128 / (core::mem::size_of::<u32>() * 8); // 4 for u32
        let half_128 = lanes_per_128 / 2;
        for block in 0..(lanes / lanes_per_128) {
            let block_start = block * lanes_per_128;
            let src_block_start = block * lanes_per_128;
            for j in 0..lanes_per_128 {
                let expected = if j % 2 == 0 {
                    a[src_block_start + half_128 + j / 2]
                } else {
                    b[src_block_start + half_128 + j / 2]
                };
                assert_eq!(
                    out[block_start + j],
                    expected,
                    "interleave_upper: lane {} wrong for {target:?}",
                    block_start + j,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// zip_lower, zip_upper
// ---------------------------------------------------------------------------

struct ZipLowerKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ZipLowerKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.zip_lower(va, vb);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

struct ZipUpperKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ZipUpperKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.zip_upper(va, vb);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_zip_lower() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let a: Vec<u32> = (0..lanes as u32).map(|i| 100 + i).collect();
        let b: Vec<u32> = (0..lanes as u32).map(|i| 200 + i).collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            ZipLowerKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        // zip_lower delegates to interleave_lower — operates within 128-bit lanes
        if lanes > 1 {
            let lanes_per_128 = 128 / (core::mem::size_of::<u32>() * 8);
            for block in 0..(lanes / lanes_per_128) {
                let block_start = block * lanes_per_128;
                let src_block_start = block * lanes_per_128;
                for j in 0..lanes_per_128 {
                    let expected = if j % 2 == 0 {
                        a[src_block_start + j / 2]
                    } else {
                        b[src_block_start + j / 2]
                    };
                    assert_eq!(
                        out[block_start + j],
                        expected,
                        "zip_lower: lane {} wrong for {target:?}",
                        block_start + j,
                    );
                }
            }
        }
    }
}

#[test]
fn test_zip_upper() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes <= 1 {
            continue;
        }
        let a: Vec<u32> = (0..lanes as u32).map(|i| 100 + i).collect();
        let b: Vec<u32> = (0..lanes as u32).map(|i| 200 + i).collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            ZipUpperKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        // zip_upper delegates to interleave_upper — operates within 128-bit lanes
        let lanes_per_128 = 128 / (core::mem::size_of::<u32>() * 8);
        let half_128 = lanes_per_128 / 2;
        for block in 0..(lanes / lanes_per_128) {
            let block_start = block * lanes_per_128;
            let src_block_start = block * lanes_per_128;
            for j in 0..lanes_per_128 {
                let expected = if j % 2 == 0 {
                    a[src_block_start + half_128 + j / 2]
                } else {
                    b[src_block_start + half_128 + j / 2]
                };
                assert_eq!(
                    out[block_start + j],
                    expected,
                    "zip_upper: lane {} wrong for {target:?}",
                    block_start + j,
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// concat_even, concat_odd
// ---------------------------------------------------------------------------

struct ConcatEvenKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ConcatEvenKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.concat_even(va, vb);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

struct ConcatOddKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ConcatOddKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.concat_odd(va, vb);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_concat_even() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let a: Vec<u32> = (0..lanes as u32).map(|i| 100 + i).collect();
        let b: Vec<u32> = (0..lanes as u32).map(|i| 200 + i).collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            ConcatEvenKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        // concat_even: even-indexed lanes from a (lower half), even-indexed from b (upper half)
        if lanes > 1 {
            let half = lanes / 2;
            for i in 0..lanes {
                let expected = if i < half {
                    a[i * 2] // even lanes from a
                } else {
                    b[(i - half) * 2] // even lanes from b
                };
                assert_eq!(
                    out[i], expected,
                    "concat_even: lane {i} wrong for {target:?}"
                );
            }
        }
    }
}

#[test]
fn test_concat_odd() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes <= 1 {
            continue;
        }
        let a: Vec<u32> = (0..lanes as u32).map(|i| 100 + i).collect();
        let b: Vec<u32> = (0..lanes as u32).map(|i| 200 + i).collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            ConcatOddKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        // concat_odd: odd-indexed lanes from a (lower half), odd-indexed from b (upper half)
        let half = lanes / 2;
        for i in 0..lanes {
            let expected = if i < half {
                a[i * 2 + 1]
            } else {
                b[(i - half) * 2 + 1]
            };
            assert_eq!(
                out[i], expected,
                "concat_odd: lane {i} wrong for {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// slide_up_lanes
// ---------------------------------------------------------------------------

struct SlideUpKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
    n_slide: usize,
}

impl WithSimd for SlideUpKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.slide_up_lanes(v, self.n_slide);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_slide_up_lanes() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (1..=lanes as u32).collect();
        let slide_n = 1;
        let mut out = vec![0u32; lanes];

        dispatch_to(
            SlideUpKernel {
                input: &input,
                out: &mut out,
                n_slide: slide_n,
            },
            target,
        );

        // After slide_up by 1: low lanes become 0, rest shift up
        for i in 0..lanes {
            let expected = if i < slide_n { 0 } else { input[i - slide_n] };
            assert_eq!(
                out[i], expected,
                "slide_up_lanes: lane {i} wrong for {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// table_lookup_bytes: simple identity permutation
// ---------------------------------------------------------------------------

struct TableLookupBytesKernel<'a> {
    table: &'a [u8],
    idx: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for TableLookupBytesKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        if lanes > self.table.len() {
            return;
        }
        unsafe {
            let table_v = s.load_u(self.table.as_ptr());
            let idx_v = s.load_u(self.idx.as_ptr());
            let r = s.table_lookup_bytes(table_v, idx_v);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_table_lookup_bytes_identity() {
    for target in available_targets() {
        let lanes = lanes_for::<u8>(target);
        // Identity permutation within each 128-bit block
        let table: Vec<u8> = (0..lanes as u8)
            .map(|i| i.wrapping_mul(3).wrapping_add(10))
            .collect();
        // Indices: identity within each 16-byte block
        let idx: Vec<u8> = (0..lanes).map(|i| (i % 16) as u8).collect();
        let mut out = vec![0u8; lanes];

        dispatch_to(
            TableLookupBytesKernel {
                table: &table,
                idx: &idx,
                out: &mut out,
            },
            target,
        );

        // For identity permutation within each 128-bit block, output should match table
        for block_start in (0..lanes).step_by(16) {
            let block_end = (block_start + 16).min(lanes);
            for i in block_start..block_end {
                let offset = i - block_start;
                assert_eq!(
                    out[i],
                    table[block_start + offset],
                    "table_lookup_bytes identity: byte {i} wrong for {target:?}"
                );
            }
        }
    }
}

// =========================================================================
// SimdReduce tests
// =========================================================================

// ---------------------------------------------------------------------------
// min_of_lanes, max_of_lanes
// ---------------------------------------------------------------------------

struct MinMaxOfLanesKernel<'a> {
    input: &'a [f32],
}

impl WithSimd for MinMaxOfLanesKernel<'_> {
    type Output = (f32, f32);
    fn with_simd<S: SimdOps>(self, s: S) -> (f32, f32) {
        let lanes = s.lanes::<f32>();
        if lanes > self.input.len() {
            return (f32::MAX, f32::MIN);
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            (s.min_of_lanes(v), s.max_of_lanes(v))
        }
    }
}

#[test]
fn test_min_max_of_lanes() {
    let input = vec![
        5.0f32, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0, 10.0, 0.5, 11.0, -1.0,
        15.0, 12.0, 0.1,
    ];

    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let (min_v, max_v) =
            dispatch_to(MinMaxOfLanesKernel { input: &input }, target);

        let expected_min =
            input[..lanes].iter().copied().fold(f32::MAX, f32::min);
        let expected_max =
            input[..lanes].iter().copied().fold(f32::MIN, f32::max);

        assert!(
            (min_v - expected_min).abs() < 1e-6,
            "min_of_lanes: got {min_v}, expected {expected_min} for {target:?}"
        );
        assert!(
            (max_v - expected_max).abs() < 1e-6,
            "max_of_lanes: got {max_v}, expected {expected_max} for {target:?}"
        );
    }
}

// =========================================================================
// SimdFloat tests
// =========================================================================

// ---------------------------------------------------------------------------
// round, trunc, ceil, floor
// ---------------------------------------------------------------------------

struct RoundingKernel<'a> {
    input: &'a [f32],
    round_out: &'a mut [f32],
    trunc_out: &'a mut [f32],
    ceil_out: &'a mut [f32],
    floor_out: &'a mut [f32],
}

impl WithSimd for RoundingKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                s.store_u(s.round(v), self.round_out.as_mut_ptr().add(i));
                s.store_u(s.trunc(v), self.trunc_out.as_mut_ptr().add(i));
                s.store_u(s.ceil(v), self.ceil_out.as_mut_ptr().add(i));
                s.store_u(s.floor(v), self.floor_out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.round_out[i] = self.input[i].round();
            self.trunc_out[i] = self.input[i].trunc();
            self.ceil_out[i] = self.input[i].ceil();
            self.floor_out[i] = self.input[i].floor();
            i += 1;
        }
    }
}

#[test]
fn test_rounding() {
    let n = 16;
    let input: Vec<f32> = vec![
        1.3, 1.5, 1.7, -1.3, -1.5, -1.7, 2.0, 0.0, 2.5, -2.5, 3.5, -3.5, 0.1,
        -0.1, 99.9, -99.9,
    ];
    let exp_round: Vec<f32> = input.iter().map(|x| x.round()).collect();
    let exp_trunc: Vec<f32> = input.iter().map(|x| x.trunc()).collect();
    let exp_ceil: Vec<f32> = input.iter().map(|x| x.ceil()).collect();
    let exp_floor: Vec<f32> = input.iter().map(|x| x.floor()).collect();

    for target in available_targets() {
        let mut round_out = vec![0f32; n];
        let mut trunc_out = vec![0f32; n];
        let mut ceil_out = vec![0f32; n];
        let mut floor_out = vec![0f32; n];
        dispatch_to(
            RoundingKernel {
                input: &input,
                round_out: &mut round_out,
                trunc_out: &mut trunc_out,
                ceil_out: &mut ceil_out,
                floor_out: &mut floor_out,
            },
            target,
        );
        let lanes = lanes_for::<f32>(target);
        for j in 0..n.min(lanes * (n / lanes)) {
            // round: SIMD uses round-to-nearest-even (banker's rounding),
            // while Rust's f32::round uses round-half-away-from-zero.
            // So for half-integer values, we allow the difference.
            let fractional = (input[j] - input[j].trunc()).abs();
            let is_half = (fractional - 0.5).abs() < 1e-6;
            if !is_half {
                assert!(
                    (round_out[j] - exp_round[j]).abs() < 1e-4,
                    "round mismatch at {j} for {target:?}: got {}, expected {}",
                    round_out[j],
                    exp_round[j]
                );
            }
            assert!(
                (trunc_out[j] - exp_trunc[j]).abs() < 1e-4,
                "trunc mismatch at {j} for {target:?}: got {}, expected {}",
                trunc_out[j],
                exp_trunc[j]
            );
            assert!(
                (ceil_out[j] - exp_ceil[j]).abs() < 1e-4,
                "ceil mismatch at {j} for {target:?}: got {}, expected {}",
                ceil_out[j],
                exp_ceil[j]
            );
            assert!(
                (floor_out[j] - exp_floor[j]).abs() < 1e-4,
                "floor mismatch at {j} for {target:?}: got {}, expected {}",
                floor_out[j],
                exp_floor[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// mul_add: FMA accuracy
// ---------------------------------------------------------------------------

struct MulAddKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for MulAddKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let vc = s.load_u(self.c.as_ptr().add(i));
                let r = s.mul_add(va, vb, vc);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i] * self.b[i] + self.c[i];
            i += 1;
        }
    }
}

#[test]
fn test_mul_add() {
    let n = 32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.3 + 1.0).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.7 - 2.0).collect();
    let c: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let expected: Vec<f32> = a
        .iter()
        .zip(&b)
        .zip(&c)
        .map(|((x, y), z)| x * y + z)
        .collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            MulAddKernel {
                a: &a,
                b: &b,
                c: &c,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-3,
                "mul_add mismatch at {j} for {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// copy_sign
// ---------------------------------------------------------------------------

struct CopySignKernel<'a> {
    mag: &'a [f32],
    sign: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for CopySignKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.mag.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let vm = s.load_u(self.mag.as_ptr().add(i));
                let vs = s.load_u(self.sign.as_ptr().add(i));
                let r = s.copy_sign(vm, vs);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.mag[i].copysign(self.sign[i]);
            i += 1;
        }
    }
}

#[test]
fn test_copy_sign() {
    let n = 16;
    let mag: Vec<f32> = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
        7.0, 8.0,
    ];
    let sign: Vec<f32> = vec![
        1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0,
        1.0, -1.0, 1.0,
    ];
    let expected: Vec<f32> =
        mag.iter().zip(&sign).map(|(m, s)| m.copysign(*s)).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            CopySignKernel {
                mag: &mag,
                sign: &sign,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-6,
                "copy_sign mismatch at {j} for {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// is_nan, is_inf
// ---------------------------------------------------------------------------

struct IsNanInfKernel<'a> {
    input: &'a [f32],
}

impl WithSimd for IsNanInfKernel<'_> {
    type Output = (Vec<bool>, Vec<bool>); // (is_nan, is_inf)
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = lanes.min(self.input.len());
        let mut nan_results = Vec::new();
        let mut inf_results = Vec::new();
        if n == 0 {
            return (nan_results, inf_results);
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let nan_mask = s.is_nan(v);
            let inf_mask = s.is_inf(v);
            let nan_vec: S::Vec<f32> = s.vec_from_mask(nan_mask);
            let inf_vec: S::Vec<f32> = s.vec_from_mask(inf_mask);
            for i in 0..n {
                let nan_bits: u32 = s.extract_lane(s.bitcast(nan_vec), i);
                let inf_bits: u32 = s.extract_lane(s.bitcast(inf_vec), i);
                nan_results.push(nan_bits != 0);
                inf_results.push(inf_bits != 0);
            }
        }
        (nan_results, inf_results)
    }
}

#[test]
fn test_is_nan_is_inf() {
    let input = vec![
        1.0f32,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::NAN,
        42.0,
        1.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::NAN,
        42.0,
    ];

    for target in available_targets() {
        let (nan_r, inf_r) =
            dispatch_to(IsNanInfKernel { input: &input }, target);
        let lanes = nan_r.len();
        for i in 0..lanes {
            assert_eq!(
                nan_r[i],
                input[i].is_nan(),
                "is_nan wrong at {i} for {target:?}: input={}",
                input[i]
            );
            assert_eq!(
                inf_r[i],
                input[i].is_infinite(),
                "is_inf wrong at {i} for {target:?}: input={}",
                input[i]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// approx_reciprocal: check within 1% tolerance
// ---------------------------------------------------------------------------

struct ApproxReciprocalKernel<'a> {
    input: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for ApproxReciprocalKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.approx_reciprocal(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = 1.0 / self.input[i];
            i += 1;
        }
    }
}

#[test]
fn test_approx_reciprocal() {
    let n = 16;
    let input: Vec<f32> = (1..=n as i32).map(|i| i as f32 * 2.0).collect();
    let expected: Vec<f32> = input.iter().map(|x| 1.0 / x).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            ApproxReciprocalKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            let rel_error = ((out[j] - expected[j]) / expected[j]).abs();
            assert!(
                rel_error < 0.01,
                "approx_reciprocal: relative error {rel_error:.4} > 1% at {j} for {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// =========================================================================
// Additional tests for previously untested methods
// =========================================================================

// ---------------------------------------------------------------------------
// undefined: returns a valid vector (doesn't crash, can be operated on)
// ---------------------------------------------------------------------------

struct UndefinedKernel<'a> {
    out: &'a mut [u32],
}

impl WithSimd for UndefinedKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        unsafe {
            let u: S::Vec<u32> = s.undefined();
            let zero = s.zero::<u32>();
            let _result = s.add(u, zero);
            if !self.out.is_empty() {
                self.out[0] = lanes as u32;
            }
        }
    }
}

#[test]
fn test_undefined() {
    for target in available_targets() {
        let mut out = vec![0u32; 1];
        dispatch_to(UndefinedKernel { out: &mut out }, target);
        assert!(out[0] > 0, "undefined test didn't execute for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// mul_odd: u32 -> u64 odd-lane multiplication
// ---------------------------------------------------------------------------

struct MulOddU32Kernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u64],
}

impl WithSimd for MulOddU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_u32 = s.lanes::<u32>();
        let lanes_u64 = s.lanes::<u64>();
        if lanes_u32 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_odd(va, vb);
            for i in 0..lanes_u64.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_odd_u32() {
    for target in available_targets() {
        let lanes_u32 = lanes_for::<u32>(target);
        let lanes_u64 = lanes_for::<u64>(target);
        let a: Vec<u32> = (0..lanes_u32 as u32).map(|i| i * 100 + 50).collect();
        let b: Vec<u32> =
            (0..lanes_u32 as u32).map(|i| i * 200 + 100).collect();
        let mut out = vec![0u64; lanes_u64];

        dispatch_to(
            MulOddU32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_u64 {
            let src_idx = i * 2 + 1;
            if src_idx < lanes_u32 {
                let expected = a[src_idx] as u64 * b[src_idx] as u64;
                assert_eq!(
                    out[i], expected,
                    "mul_odd u32 lane {i} (src {src_idx}) wrong for {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// shift_left_same / shift_right_same: runtime variable shift
// ---------------------------------------------------------------------------

struct ShiftSameKernel<'a> {
    input: &'a [u32],
    out_left: &'a mut [u32],
    out_right: &'a mut [u32],
    bits: u32,
}

impl WithSimd for ShiftSameKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let l = s.shift_left_same(v, self.bits);
            let r = s.shift_right_same(v, self.bits);
            s.store_u(l, self.out_left.as_mut_ptr());
            s.store_u(r, self.out_right.as_mut_ptr());
        }
    }
}

#[test]
fn test_shift_left_right_same() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> =
            (0..lanes as u32).map(|i| 0x0000_FF00 | (i + 1)).collect();
        let mut out_left = vec![0u32; lanes];
        let mut out_right = vec![0u32; lanes];
        let bits = 4u32;

        dispatch_to(
            ShiftSameKernel {
                input: &input,
                out_left: &mut out_left,
                out_right: &mut out_right,
                bits,
            },
            target,
        );

        for i in 0..lanes {
            let expected_left = input[i] << bits;
            let expected_right = input[i] >> bits;
            assert_eq!(
                out_left[i], expected_left,
                "shift_left_same lane {i} wrong for {target:?}"
            );
            assert_eq!(
                out_right[i], expected_right,
                "shift_right_same lane {i} wrong for {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// and_mask / or_mask / not_mask
// ---------------------------------------------------------------------------

struct MaskLogicKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out_and: &'a mut [bool],
    out_or: &'a mut [bool],
    out_not_a: &'a mut [bool],
}

impl WithSimd for MaskLogicKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let threshold = s.splat(5u32);
            let mask_a = s.gt(va, threshold);
            let mask_b = s.gt(vb, threshold);

            let m_and = s.and_mask(mask_a, mask_b);
            let m_or = s.or_mask(mask_a, mask_b);
            let m_not_a = s.not_mask(mask_a);

            for i in 0..lanes.min(self.out_and.len()) {
                let v_and: u32 =
                    s.extract_lane(s.vec_from_mask::<u32>(m_and), i);
                self.out_and[i] = v_and != 0;
                let v_or: u32 = s.extract_lane(s.vec_from_mask::<u32>(m_or), i);
                self.out_or[i] = v_or != 0;
                let v_not: u32 =
                    s.extract_lane(s.vec_from_mask::<u32>(m_not_a), i);
                self.out_not_a[i] = v_not != 0;
            }
        }
    }
}

#[test]
fn test_mask_logic() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let a: Vec<u32> = (0..lanes as u32).map(|i| i * 3).collect();
        let b: Vec<u32> =
            (0..lanes as u32).map(|i| 10u32.wrapping_sub(i)).collect();
        let mut out_and = vec![false; lanes];
        let mut out_or = vec![false; lanes];
        let mut out_not_a = vec![false; lanes];

        dispatch_to(
            MaskLogicKernel {
                a: &a,
                b: &b,
                out_and: &mut out_and,
                out_or: &mut out_or,
                out_not_a: &mut out_not_a,
            },
            target,
        );

        for i in 0..lanes {
            let a_gt5 = a[i] > 5;
            let b_gt5 = b[i] > 5;
            assert_eq!(
                out_and[i],
                a_gt5 && b_gt5,
                "and_mask lane {i} for {target:?}"
            );
            assert_eq!(
                out_or[i],
                a_gt5 || b_gt5,
                "or_mask lane {i} for {target:?}"
            );
            assert_eq!(
                out_not_a[i], !a_gt5,
                "not_mask lane {i} for {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// reverse4: reverse groups of 4 lanes
// ---------------------------------------------------------------------------

struct Reverse4Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for Reverse4Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.reverse4(v);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_reverse4() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 4 {
            continue;
        }
        let input: Vec<u32> = (0..lanes as u32).collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            Reverse4Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        for g in 0..(lanes / 4) {
            let b = g * 4;
            assert_eq!(out[b], input[b + 3], "reverse4 g{g} p0 {target:?}");
            assert_eq!(out[b + 1], input[b + 2], "reverse4 g{g} p1 {target:?}");
            assert_eq!(out[b + 2], input[b + 1], "reverse4 g{g} p2 {target:?}");
            assert_eq!(out[b + 3], input[b], "reverse4 g{g} p3 {target:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// reverse8: reverse groups of 8 lanes
// ---------------------------------------------------------------------------

struct Reverse8Kernel<'a> {
    input: &'a [u16],
    out: &'a mut [u16],
}

impl WithSimd for Reverse8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u16>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.reverse8(v);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_reverse8() {
    for target in available_targets() {
        let lanes = lanes_for::<u16>(target);
        if lanes < 8 {
            continue;
        }
        let input: Vec<u16> = (0..lanes as u16).collect();
        let mut out = vec![0u16; lanes];

        dispatch_to(
            Reverse8Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        for g in 0..(lanes / 8) {
            let b = g * 8;
            for j in 0..8 {
                assert_eq!(
                    out[b + j],
                    input[b + 7 - j],
                    "reverse8 g{g} p{j} {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// concat_upper_lower / concat_lower_upper
// ---------------------------------------------------------------------------

struct ConcatHalfKernel<'a> {
    hi: &'a [u32],
    lo: &'a [u32],
    out_ul: &'a mut [u32],
    out_lu: &'a mut [u32],
}

impl WithSimd for ConcatHalfKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.hi.len() {
            return;
        }
        unsafe {
            let vhi = s.load_u(self.hi.as_ptr());
            let vlo = s.load_u(self.lo.as_ptr());
            let r_ul = s.concat_upper_lower(vhi, vlo);
            let r_lu = s.concat_lower_upper(vhi, vlo);
            s.store_u(r_ul, self.out_ul.as_mut_ptr());
            s.store_u(r_lu, self.out_lu.as_mut_ptr());
        }
    }
}

#[test]
fn test_concat_upper_lower() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 2 {
            continue; // No upper/lower distinction for 1-lane vectors
        }
        let half = lanes / 2;
        let hi: Vec<u32> = (0..lanes as u32).map(|i| 100 + i).collect();
        let lo: Vec<u32> = (0..lanes as u32).map(|i| 200 + i).collect();
        let mut out_ul = vec![0u32; lanes];
        let mut out_lu = vec![0u32; lanes];

        dispatch_to(
            ConcatHalfKernel {
                hi: &hi,
                lo: &lo,
                out_ul: &mut out_ul,
                out_lu: &mut out_lu,
            },
            target,
        );

        // concat_upper_lower: upper half from hi, lower half from lo
        for i in 0..half {
            assert_eq!(
                out_ul[i], lo[i],
                "concat_upper_lower lo lane {i} {target:?}"
            );
        }
        for i in half..lanes {
            assert_eq!(
                out_ul[i], hi[i],
                "concat_upper_lower hi lane {i} {target:?}"
            );
        }
        // concat_lower_upper: lower half from hi, upper half from lo
        for i in 0..half {
            assert_eq!(
                out_lu[i], hi[i],
                "concat_lower_upper lo lane {i} {target:?}"
            );
        }
        for i in half..lanes {
            assert_eq!(
                out_lu[i], lo[i],
                "concat_lower_upper hi lane {i} {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// table_lookup_lanes: lane-level table lookup by index
// ---------------------------------------------------------------------------

struct TableLookupLanesKernel<'a> {
    table: &'a [u32],
    indices: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for TableLookupLanesKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.table.len() {
            return;
        }
        unsafe {
            let vtable = s.load_u(self.table.as_ptr());
            let vidx = s.load_u(self.indices.as_ptr());
            let r = s.table_lookup_lanes::<u32, u32>(vtable, vidx);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_table_lookup_lanes() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let table: Vec<u32> = (0..lanes as u32).map(|i| (i + 1) * 10).collect();
        let indices: Vec<u32> = (0..lanes as u32).rev().collect();
        let mut out = vec![0u32; lanes];

        dispatch_to(
            TableLookupLanesKernel {
                table: &table,
                indices: &indices,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes {
            let expected = table[indices[i] as usize];
            assert_eq!(
                out[i], expected,
                "table_lookup_lanes lane {i} {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// mul_sub: a * b - c
// ---------------------------------------------------------------------------

struct MulSubKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for MulSubKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let vc = s.load_u(self.c.as_ptr());
            let r = s.mul_sub(va, vb, vc);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_mul_sub() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let a: Vec<f32> = (0..lanes).map(|i| (i as f32 + 1.0) * 2.0).collect();
        let b: Vec<f32> = (0..lanes).map(|i| i as f32 + 0.5).collect();
        let c: Vec<f32> = (0..lanes).map(|i| i as f32 * 3.0).collect();
        let mut out = vec![0f32; lanes];

        dispatch_to(
            MulSubKernel {
                a: &a,
                b: &b,
                c: &c,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes {
            let expected = a[i] * b[i] - c[i];
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "mul_sub lane {i} {target:?}: got {}, expected {}",
                out[i],
                expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// neg_mul_sub: -(a * b) - c
// ---------------------------------------------------------------------------

struct NegMulSubKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for NegMulSubKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let vc = s.load_u(self.c.as_ptr());
            let r = s.neg_mul_sub(va, vb, vc);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_neg_mul_sub() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let a: Vec<f32> = (0..lanes).map(|i| (i as f32 + 1.0) * 2.0).collect();
        let b: Vec<f32> = (0..lanes).map(|i| i as f32 + 0.5).collect();
        let c: Vec<f32> = (0..lanes).map(|i| i as f32 * 3.0).collect();
        let mut out = vec![0f32; lanes];

        dispatch_to(
            NegMulSubKernel {
                a: &a,
                b: &b,
                c: &c,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes {
            let expected = -(a[i] * b[i]) - c[i];
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "neg_mul_sub lane {i} {target:?}: got {}, expected {}",
                out[i],
                expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// approx_reciprocal_sqrt: 1/sqrt(x) within tolerance
// ---------------------------------------------------------------------------

struct ApproxRsqrtKernel<'a> {
    input: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for ApproxRsqrtKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.approx_reciprocal_sqrt(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = 1.0 / self.input[i].sqrt();
            i += 1;
        }
    }
}

#[test]
fn test_approx_reciprocal_sqrt() {
    let n = 16;
    let input: Vec<f32> = (1..=n as i32).map(|i| (i as f32) * 4.0).collect();
    let expected: Vec<f32> = input.iter().map(|x| 1.0 / x.sqrt()).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            ApproxRsqrtKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            let rel_error = ((out[j] - expected[j]) / expected[j]).abs();
            assert!(
                rel_error < 0.02,
                "approx_reciprocal_sqrt: error {rel_error:.4} > 2% at {j} {target:?}"
            );
        }
    }
}

// ===========================================================================
// Additional correctness tests for previously untested methods
// ===========================================================================

// ---------------------------------------------------------------------------
// not: bitwise NOT
// ---------------------------------------------------------------------------

struct NotKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for NotKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.not(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = !self.input[i];
            i += 1;
        }
    }
}

#[test]
fn test_not() {
    let n = 32;
    let input: Vec<u32> =
        (0..n as u32).map(|i| i.wrapping_mul(0x1234_5678)).collect();
    let expected: Vec<u32> = input.iter().map(|x| !x).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            NotKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "not failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// lt: less-than comparison
// ---------------------------------------------------------------------------

struct LtKernel<'a> {
    a: &'a [i32],
    b: &'a [i32],
    out: &'a mut [i32],
}

impl WithSimd for LtKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let mask = s.lt(va, vb);
                let result = s.vec_from_mask::<i32>(mask);
                s.store_u(result, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = if self.a[i] < self.b[i] { -1 } else { 0 };
            i += 1;
        }
    }
}

#[test]
fn test_lt() {
    let n = 32;
    let a: Vec<i32> = (0..n as i32).map(|i| i * 3 - 20).collect();
    let b: Vec<i32> = (0..n as i32).map(|i| 30 - i * 2).collect();
    // expected: -1 (all bits set) where a < b, 0 otherwise
    let expected: Vec<i32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| if x < y { -1i32 } else { 0 })
        .collect();

    for target in available_targets() {
        let mut out = vec![0i32; n];
        dispatch_to(
            LtKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "lt failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// if_then_else: ternary select by mask
// ---------------------------------------------------------------------------

struct IfThenElseKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    threshold: u32,
    out: &'a mut [u32],
}

impl WithSimd for IfThenElseKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let thresh = s.splat(self.threshold);
                let mask = s.gt(va, thresh); // mask = a > threshold
                let r = s.if_then_else(mask, va, vb); // true->a, false->b
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = if self.a[i] > self.threshold {
                self.a[i]
            } else {
                self.b[i]
            };
            i += 1;
        }
    }
}

#[test]
fn test_if_then_else() {
    let n = 32;
    let a: Vec<u32> = (0..n as u32).collect();
    let b: Vec<u32> = (0..n as u32).map(|i| 100 + i).collect();
    let threshold = 15u32;
    let expected: Vec<u32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| if *x > threshold { *x } else { *y })
        .collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            IfThenElseKernel {
                a: &a,
                b: &b,
                threshold,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "if_then_else failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// compress: pack lanes where mask is true (returns vector)
// ---------------------------------------------------------------------------

struct CompressKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
    count_out: &'a mut usize,
}

impl WithSimd for CompressKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            // Keep only values > 10
            let thresh = s.splat(10u32);
            let mask = s.gt(v, thresh);
            let compressed = s.compress(v, mask);
            s.store_u(compressed, self.out.as_mut_ptr());
            *self.count_out = s.count_true(mask);
        }
    }
}

#[test]
fn test_compress() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        // Values: 5, 15, 3, 20, 8, 25, 1, 30, ...
        let input: Vec<u32> = (0..lanes as u32)
            .map(|i| if i % 2 == 0 { i + 1 } else { (i + 1) * 5 })
            .collect();
        let mut out = vec![0u32; lanes];
        let mut count = 0usize;

        dispatch_to(
            CompressKernel {
                input: &input,
                out: &mut out,
                count_out: &mut count,
            },
            target,
        );

        let expected: Vec<u32> =
            input.iter().copied().filter(|x| *x > 10).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress count wrong for {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress values wrong for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// neg_mul_add: -(a * b) + c
// ---------------------------------------------------------------------------

struct NegMulAddKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for NegMulAddKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        if lanes > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let vc = s.load_u(self.c.as_ptr());
            let r = s.neg_mul_add(va, vb, vc);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_neg_mul_add() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let a: Vec<f32> = (0..lanes).map(|i| (i as f32 + 1.0) * 2.0).collect();
        let b: Vec<f32> = (0..lanes).map(|i| i as f32 + 0.5).collect();
        let c: Vec<f32> = (0..lanes).map(|i| i as f32 * 3.0 + 10.0).collect();
        let mut out = vec![0f32; lanes];

        dispatch_to(
            NegMulAddKernel {
                a: &a,
                b: &b,
                c: &c,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes {
            let expected = -(a[i] * b[i]) + c[i];
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "neg_mul_add lane {i} {target:?}: got {}, expected {}",
                out[i],
                expected
            );
        }
    }
}

// =========================================================================
// Additional coverage tests for all type variants
// =========================================================================

// ---------------------------------------------------------------------------
// round/trunc/ceil/floor for f64
// ---------------------------------------------------------------------------

struct RoundingF64Kernel<'a> {
    input: &'a [f64],
    round_out: &'a mut [f64],
    trunc_out: &'a mut [f64],
    ceil_out: &'a mut [f64],
    floor_out: &'a mut [f64],
}

impl WithSimd for RoundingF64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f64>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                s.store_u(s.round(v), self.round_out.as_mut_ptr().add(i));
                s.store_u(s.trunc(v), self.trunc_out.as_mut_ptr().add(i));
                s.store_u(s.ceil(v), self.ceil_out.as_mut_ptr().add(i));
                s.store_u(s.floor(v), self.floor_out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.round_out[i] = self.input[i].round();
            self.trunc_out[i] = self.input[i].trunc();
            self.ceil_out[i] = self.input[i].ceil();
            self.floor_out[i] = self.input[i].floor();
            i += 1;
        }
    }
}

#[test]
fn test_rounding_f64() {
    let input: Vec<f64> = vec![
        1.3,
        1.5,
        1.7,
        -1.3,
        -1.5,
        -1.7,
        2.0,
        0.0,
        2.5,
        -2.5,
        3.5,
        -3.5,
        0.1,
        -0.1,
        99.9,
        -99.9,
        // Large values near 2^52 boundary
        4503599627370496.0,
        4503599627370495.5,
        -4503599627370496.0,
        1e15,
        -1e15,
        1e18,
        -1e18,
        0.0,
    ];
    let n = input.len();
    let exp_trunc: Vec<f64> = input.iter().map(|x| x.trunc()).collect();
    let exp_ceil: Vec<f64> = input.iter().map(|x| x.ceil()).collect();
    let exp_floor: Vec<f64> = input.iter().map(|x| x.floor()).collect();

    for target in available_targets() {
        let mut round_out = vec![0f64; n];
        let mut trunc_out = vec![0f64; n];
        let mut ceil_out = vec![0f64; n];
        let mut floor_out = vec![0f64; n];
        dispatch_to(
            RoundingF64Kernel {
                input: &input,
                round_out: &mut round_out,
                trunc_out: &mut trunc_out,
                ceil_out: &mut ceil_out,
                floor_out: &mut floor_out,
            },
            target,
        );
        let lanes = lanes_for::<f64>(target);
        for j in 0..n.min(lanes * (n / lanes)) {
            // round: SIMD uses round-to-nearest-even (banker's rounding)
            let fractional = (input[j] - input[j].trunc()).abs();
            let is_half = (fractional - 0.5).abs() < 1e-10;
            if !is_half {
                let exp_round = input[j].round();
                assert!(
                    (round_out[j] - exp_round).abs() < 1e-6,
                    "round f64 mismatch at {j} for {target:?}: got {}, expected {}, input={}",
                    round_out[j],
                    exp_round,
                    input[j]
                );
            }
            assert!(
                (trunc_out[j] - exp_trunc[j]).abs() < 1e-6,
                "trunc f64 mismatch at {j} for {target:?}: got {}, expected {}, input={}",
                trunc_out[j],
                exp_trunc[j],
                input[j]
            );
            assert!(
                (ceil_out[j] - exp_ceil[j]).abs() < 1e-6,
                "ceil f64 mismatch at {j} for {target:?}: got {}, expected {}, input={}",
                ceil_out[j],
                exp_ceil[j],
                input[j]
            );
            assert!(
                (floor_out[j] - exp_floor[j]).abs() < 1e-6,
                "floor f64 mismatch at {j} for {target:?}: got {}, expected {}, input={}",
                floor_out[j],
                exp_floor[j],
                input[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// demote_to: u64 -> u32 (unsigned saturating)
// ---------------------------------------------------------------------------

struct DemoteU64ToU32Kernel<'a> {
    input: &'a [u64],
    out: &'a mut [u32],
}

impl WithSimd for DemoteU64ToU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_wide = s.lanes::<u64>();
        let lanes_narrow = s.lanes::<u32>();
        if lanes_wide > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let demoted: S::Vec<u32> = s.demote_to(v);
            for i in 0..lanes_narrow.min(self.out.len()) {
                self.out[i] = s.extract_lane(demoted, i);
            }
        }
    }
}

#[test]
fn test_demote_u64_to_u32() {
    for target in available_targets() {
        let lanes_wide = lanes_for::<u64>(target);
        let lanes_narrow = lanes_for::<u32>(target);
        // Test values: within range, at boundary, and overflow
        let mut input = vec![0u64; lanes_wide];
        for i in 0..lanes_wide {
            input[i] = match i % 4 {
                0 => 42,                     // fits
                1 => u32::MAX as u64,        // exactly at max
                2 => u32::MAX as u64 + 1000, // overflows -> saturates to u32::MAX
                3 => 0xFFFF_FFFF_FFFF,       // way over -> saturates
                _ => unreachable!(),
            };
        }
        let mut out = vec![0u32; lanes_narrow];

        dispatch_to(
            DemoteU64ToU32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_wide.min(lanes_narrow) {
            let expected = input[i].min(u32::MAX as u64) as u32;
            assert_eq!(
                out[i], expected,
                "demote u64->u32 lane {i} wrong for {target:?}: input={}, got={}, expected={}",
                input[i], out[i], expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// demote_to: i64 -> i32 (signed saturating)
// ---------------------------------------------------------------------------

struct DemoteI64ToI32Kernel<'a> {
    input: &'a [i64],
    out: &'a mut [i32],
}

impl WithSimd for DemoteI64ToI32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_wide = s.lanes::<i64>();
        let lanes_narrow = s.lanes::<i32>();
        if lanes_wide > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let demoted: S::Vec<i32> = s.demote_to(v);
            for i in 0..lanes_narrow.min(self.out.len()) {
                self.out[i] = s.extract_lane(demoted, i);
            }
        }
    }
}

#[test]
fn test_demote_i64_to_i32() {
    for target in available_targets() {
        let lanes_wide = lanes_for::<i64>(target);
        let lanes_narrow = lanes_for::<i32>(target);
        let mut input = vec![0i64; lanes_wide];
        for i in 0..lanes_wide {
            input[i] = match i % 6 {
                0 => 42,                     // fits
                1 => i32::MAX as i64,        // exactly at max
                2 => i32::MIN as i64,        // exactly at min
                3 => i32::MAX as i64 + 5000, // overflows -> saturates to i32::MAX
                4 => i32::MIN as i64 - 5000, // underflows -> saturates to i32::MIN
                5 => -100,                   // negative that fits
                _ => unreachable!(),
            };
        }
        let mut out = vec![0i32; lanes_narrow];

        dispatch_to(
            DemoteI64ToI32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_wide.min(lanes_narrow) {
            let expected =
                input[i].clamp(i32::MIN as i64, i32::MAX as i64) as i32;
            assert_eq!(
                out[i], expected,
                "demote i64->i32 lane {i} wrong for {target:?}: input={}, got={}, expected={}",
                input[i], out[i], expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// demote_to: u32 -> u16 (unsigned saturating)
// ---------------------------------------------------------------------------

struct DemoteU32ToU16Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u16],
}

impl WithSimd for DemoteU32ToU16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_wide = s.lanes::<u32>();
        let lanes_narrow = s.lanes::<u16>();
        if lanes_wide > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let demoted: S::Vec<u16> = s.demote_to(v);
            for i in 0..lanes_narrow.min(self.out.len()) {
                self.out[i] = s.extract_lane(demoted, i);
            }
        }
    }
}

#[test]
fn test_demote_u32_to_u16() {
    for target in available_targets() {
        let lanes_wide = lanes_for::<u32>(target);
        let lanes_narrow = lanes_for::<u16>(target);
        let mut input = vec![0u32; lanes_wide];
        for i in 0..lanes_wide {
            input[i] = match i % 4 {
                0 => 100,                   // fits
                1 => u16::MAX as u32,       // exactly at max
                2 => u16::MAX as u32 + 500, // overflows -> saturates
                3 => 0,                     // zero
                _ => unreachable!(),
            };
        }
        let mut out = vec![0u16; lanes_narrow];

        dispatch_to(
            DemoteU32ToU16Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_wide.min(lanes_narrow) {
            let expected = input[i].min(u16::MAX as u32) as u16;
            assert_eq!(
                out[i], expected,
                "demote u32->u16 lane {i} wrong for {target:?}: input={}, got={}, expected={}",
                input[i], out[i], expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// demote_to: f64 -> f32
// ---------------------------------------------------------------------------

struct DemoteF64ToF32Kernel<'a> {
    input: &'a [f64],
    out: &'a mut [f32],
}

impl WithSimd for DemoteF64ToF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_wide = s.lanes::<f64>();
        let lanes_narrow = s.lanes::<f32>();
        if lanes_wide > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let demoted: S::Vec<f32> = s.demote_to(v);
            for i in 0..lanes_narrow.min(self.out.len()) {
                self.out[i] = s.extract_lane(demoted, i);
            }
        }
    }
}

#[test]
fn test_demote_f64_to_f32() {
    for target in available_targets() {
        let lanes_wide = lanes_for::<f64>(target);
        let lanes_narrow = lanes_for::<f32>(target);
        let mut input = vec![0.0f64; lanes_wide];
        for i in 0..lanes_wide {
            input[i] = match i % 4 {
                0 => 3.125,
                1 => -2.75,
                2 => 0.0,
                3 => 1e10,
                _ => unreachable!(),
            };
        }
        let mut out = vec![0f32; lanes_narrow];

        dispatch_to(
            DemoteF64ToF32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_wide.min(lanes_narrow) {
            let expected = input[i] as f32;
            assert!(
                (out[i] - expected).abs() < 1e-4,
                "demote f64->f32 lane {i} wrong for {target:?}: input={}, got={}, expected={}",
                input[i],
                out[i],
                expected
            );
        }
    }
}

// ---------------------------------------------------------------------------
// compress: i32 (tests the new SSE2 shuffle path)
// ---------------------------------------------------------------------------

struct CompressI32Kernel<'a> {
    input: &'a [i32],
    out: &'a mut [i32],
    count_out: &'a mut usize,
}

impl WithSimd for CompressI32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(0i32);
            let mask = s.gt(v, thresh); // keep positive values
            let compressed = s.compress(v, mask);
            s.store_u(compressed, self.out.as_mut_ptr());
            *self.count_out = s.count_true(mask);
        }
    }
}

#[test]
fn test_compress_i32() {
    for target in available_targets() {
        let lanes = lanes_for::<i32>(target);
        // Alternate positive and negative
        let input: Vec<i32> = (0..lanes as i32)
            .map(|i| if i % 2 == 0 { -(i + 1) } else { (i + 1) * 10 })
            .collect();
        let mut out = vec![0i32; lanes];
        let mut count = 0usize;

        dispatch_to(
            CompressI32Kernel {
                input: &input,
                out: &mut out,
                count_out: &mut count,
            },
            target,
        );

        let expected: Vec<i32> =
            input.iter().copied().filter(|x| *x > 0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress i32 count wrong for {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress i32 values wrong for {target:?}: input={input:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress: all 16 mask patterns for u32 (exhaustive test for SSE2 shuffle table)
// ---------------------------------------------------------------------------

struct CompressU32AllMasksKernel<'a> {
    input: &'a [u32],
    mask_bits: u8,
    out: &'a mut [u32],
    count_out: &'a mut usize,
}

impl WithSimd for CompressU32AllMasksKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes < 4 || self.input.len() < lanes {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            // Build mask from mask_bits: set lanes where bit is 1
            let mut mask_arr = vec![0u32; lanes];
            for i in 0..4.min(lanes) {
                if self.mask_bits & (1 << i) != 0 {
                    mask_arr[i] = u32::MAX;
                }
            }
            let mask_vec = s.load_u(mask_arr.as_ptr());
            let mask = s.mask_from_vec(mask_vec);
            let compressed = s.compress(v, mask);
            s.store_u(compressed, self.out.as_mut_ptr());
            *self.count_out = s.count_true(mask);
        }
    }
}

#[test]
fn test_compress_u32_all_masks() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 4 {
            continue;
        }
        let input: Vec<u32> =
            (0..lanes as u32).map(|i| (i + 1) * 100).collect();

        // Test all 16 mask patterns for the first 4 lanes
        for mask_bits in 0u8..16 {
            let mut out = vec![0u32; lanes];
            let mut count = 0usize;

            dispatch_to(
                CompressU32AllMasksKernel {
                    input: &input,
                    mask_bits,
                    out: &mut out,
                    count_out: &mut count,
                },
                target,
            );

            let expected: Vec<u32> = (0..4)
                .filter(|i| mask_bits & (1 << i) != 0)
                .map(|i| input[i as usize])
                .collect();
            assert_eq!(
                count,
                expected.len(),
                "compress u32 mask=0b{mask_bits:04b} count wrong for {target:?}"
            );
            assert_eq!(
                &out[..count],
                &expected[..],
                "compress u32 mask=0b{mask_bits:04b} values wrong for {target:?}: input={:?}",
                &input[..4]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// compress: u16 (scalar fallback path)
// ---------------------------------------------------------------------------

struct CompressU16Kernel<'a> {
    input: &'a [u16],
    out: &'a mut [u16],
    count_out: &'a mut usize,
}

impl WithSimd for CompressU16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u16>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(100u16);
            let mask = s.gt(v, thresh);
            let compressed = s.compress(v, mask);
            s.store_u(compressed, self.out.as_mut_ptr());
            *self.count_out = s.count_true(mask);
        }
    }
}

#[test]
fn test_compress_u16() {
    for target in available_targets() {
        let lanes = lanes_for::<u16>(target);
        let input: Vec<u16> = (0..lanes as u16)
            .map(|i| if i % 2 == 0 { 50 + i } else { 150 + i })
            .collect();
        let mut out = vec![0u16; lanes];
        let mut count = 0usize;

        dispatch_to(
            CompressU16Kernel {
                input: &input,
                out: &mut out,
                count_out: &mut count,
            },
            target,
        );

        let expected: Vec<u16> =
            input.iter().copied().filter(|x| *x > 100).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress u16 count wrong for {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress u16 values wrong for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress: u8
// ---------------------------------------------------------------------------

struct CompressU8Kernel<'a> {
    input: &'a [u8],
    out: &'a mut [u8],
    count_out: &'a mut usize,
}

impl WithSimd for CompressU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(100u8);
            let mask = s.gt(v, thresh);
            let compressed = s.compress(v, mask);
            s.store_u(compressed, self.out.as_mut_ptr());
            *self.count_out = s.count_true(mask);
        }
    }
}

#[test]
fn test_compress_u8() {
    for target in available_targets() {
        let lanes = lanes_for::<u8>(target);
        let input: Vec<u8> = (0..lanes as u8)
            .map(|i| {
                if i % 2 == 0 {
                    50_u8.wrapping_add(i)
                } else {
                    150_u8.wrapping_add(i)
                }
            })
            .collect();
        let mut out = vec![0u8; lanes];
        let mut count = 0usize;

        dispatch_to(
            CompressU8Kernel {
                input: &input,
                out: &mut out,
                count_out: &mut count,
            },
            target,
        );

        let expected: Vec<u8> =
            input.iter().copied().filter(|x| *x > 100).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress u8 count wrong for {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress u8 values wrong for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress: i8
// ---------------------------------------------------------------------------

struct CompressI8Kernel<'a> {
    input: &'a [i8],
    out: &'a mut [i8],
    count_out: &'a mut usize,
}

impl WithSimd for CompressI8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i8>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(0i8);
            let mask = s.gt(v, thresh);
            let compressed = s.compress(v, mask);
            s.store_u(compressed, self.out.as_mut_ptr());
            *self.count_out = s.count_true(mask);
        }
    }
}

#[test]
fn test_compress_i8() {
    for target in available_targets() {
        let lanes = lanes_for::<i8>(target);
        let input: Vec<i8> = (0..lanes)
            .map(|i| {
                if i % 2 == 0 {
                    -50 + (i as i8)
                } else {
                    50 + (i as i8 % 70)
                }
            })
            .collect();
        let mut out = vec![0i8; lanes];
        let mut count = 0usize;

        dispatch_to(
            CompressI8Kernel {
                input: &input,
                out: &mut out,
                count_out: &mut count,
            },
            target,
        );

        let expected: Vec<i8> =
            input.iter().copied().filter(|x| *x > 0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress i8 count wrong for {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress i8 values wrong for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress: i16
// ---------------------------------------------------------------------------

struct CompressI16Kernel<'a> {
    input: &'a [i16],
    out: &'a mut [i16],
    count_out: &'a mut usize,
}

impl WithSimd for CompressI16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i16>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(0i16);
            let mask = s.gt(v, thresh);
            let compressed = s.compress(v, mask);
            s.store_u(compressed, self.out.as_mut_ptr());
            *self.count_out = s.count_true(mask);
        }
    }
}

#[test]
fn test_compress_i16() {
    for target in available_targets() {
        let lanes = lanes_for::<i16>(target);
        let input: Vec<i16> = (0..lanes as i16)
            .map(|i| if i % 2 == 0 { -100 + i } else { 100 + i })
            .collect();
        let mut out = vec![0i16; lanes];
        let mut count = 0usize;

        dispatch_to(
            CompressI16Kernel {
                input: &input,
                out: &mut out,
                count_out: &mut count,
            },
            target,
        );

        let expected: Vec<i16> =
            input.iter().copied().filter(|x| *x > 0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress i16 count wrong for {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress i16 values wrong for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// mul_even / mul_odd: u8 -> u16
// ---------------------------------------------------------------------------

struct MulEvenU8Kernel<'a> {
    a: &'a [u8],
    b: &'a [u8],
    out: &'a mut [u16],
}

impl WithSimd for MulEvenU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_u8 = s.lanes::<u8>();
        let lanes_u16 = s.lanes::<u16>();
        if lanes_u8 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_even(va, vb);
            for i in 0..lanes_u16.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

struct MulOddU8Kernel<'a> {
    a: &'a [u8],
    b: &'a [u8],
    out: &'a mut [u16],
}

impl WithSimd for MulOddU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_u8 = s.lanes::<u8>();
        let lanes_u16 = s.lanes::<u16>();
        if lanes_u8 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_odd(va, vb);
            for i in 0..lanes_u16.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_even_u8() {
    for target in available_targets() {
        let lanes_u8 = lanes_for::<u8>(target);
        let lanes_u16 = lanes_for::<u16>(target);
        let a: Vec<u8> = (0..lanes_u8)
            .map(|i| (i as u8).wrapping_mul(3).wrapping_add(10))
            .collect();
        let b: Vec<u8> = (0..lanes_u8)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(5))
            .collect();
        let mut out = vec![0u16; lanes_u16];

        dispatch_to(
            MulEvenU8Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_u16 {
            let src = i * 2;
            if src < lanes_u8 {
                let expected = a[src] as u16 * b[src] as u16;
                assert_eq!(
                    out[i], expected,
                    "mul_even u8 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

#[test]
fn test_mul_odd_u8() {
    for target in available_targets() {
        let lanes_u8 = lanes_for::<u8>(target);
        let lanes_u16 = lanes_for::<u16>(target);
        let a: Vec<u8> = (0..lanes_u8)
            .map(|i| (i as u8).wrapping_mul(3).wrapping_add(10))
            .collect();
        let b: Vec<u8> = (0..lanes_u8)
            .map(|i| (i as u8).wrapping_mul(7).wrapping_add(5))
            .collect();
        let mut out = vec![0u16; lanes_u16];

        dispatch_to(
            MulOddU8Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_u16 {
            let src = i * 2 + 1;
            if src < lanes_u8 {
                let expected = a[src] as u16 * b[src] as u16;
                assert_eq!(
                    out[i], expected,
                    "mul_odd u8 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_even / mul_odd: i8 -> i16
// ---------------------------------------------------------------------------

struct MulEvenI8Kernel<'a> {
    a: &'a [i8],
    b: &'a [i8],
    out: &'a mut [i16],
}

impl WithSimd for MulEvenI8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_i8 = s.lanes::<i8>();
        let lanes_i16 = s.lanes::<i16>();
        if lanes_i8 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_even(va, vb);
            for i in 0..lanes_i16.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

struct MulOddI8Kernel<'a> {
    a: &'a [i8],
    b: &'a [i8],
    out: &'a mut [i16],
}

impl WithSimd for MulOddI8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_i8 = s.lanes::<i8>();
        let lanes_i16 = s.lanes::<i16>();
        if lanes_i8 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_odd(va, vb);
            for i in 0..lanes_i16.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_even_i8() {
    for target in available_targets() {
        let lanes_i8 = lanes_for::<i8>(target);
        let lanes_i16 = lanes_for::<i16>(target);
        let a: Vec<i8> = (0..lanes_i8)
            .map(|i| (i as i8).wrapping_mul(3).wrapping_sub(50))
            .collect();
        let b: Vec<i8> = (0..lanes_i8)
            .map(|i| (i as i8).wrapping_mul(5).wrapping_add(20))
            .collect();
        let mut out = vec![0i16; lanes_i16];

        dispatch_to(
            MulEvenI8Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_i16 {
            let src = i * 2;
            if src < lanes_i8 {
                let expected = a[src] as i16 * b[src] as i16;
                assert_eq!(
                    out[i], expected,
                    "mul_even i8 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

#[test]
fn test_mul_odd_i8() {
    for target in available_targets() {
        let lanes_i8 = lanes_for::<i8>(target);
        let lanes_i16 = lanes_for::<i16>(target);
        let a: Vec<i8> = (0..lanes_i8)
            .map(|i| (i as i8).wrapping_mul(3).wrapping_sub(50))
            .collect();
        let b: Vec<i8> = (0..lanes_i8)
            .map(|i| (i as i8).wrapping_mul(5).wrapping_add(20))
            .collect();
        let mut out = vec![0i16; lanes_i16];

        dispatch_to(
            MulOddI8Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_i16 {
            let src = i * 2 + 1;
            if src < lanes_i8 {
                let expected = a[src] as i16 * b[src] as i16;
                assert_eq!(
                    out[i], expected,
                    "mul_odd i8 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_even / mul_odd: u16 -> u32
// ---------------------------------------------------------------------------

struct MulEvenU16Kernel<'a> {
    a: &'a [u16],
    b: &'a [u16],
    out: &'a mut [u32],
}

impl WithSimd for MulEvenU16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_u16 = s.lanes::<u16>();
        let lanes_u32 = s.lanes::<u32>();
        if lanes_u16 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_even(va, vb);
            for i in 0..lanes_u32.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

struct MulOddU16Kernel<'a> {
    a: &'a [u16],
    b: &'a [u16],
    out: &'a mut [u32],
}

impl WithSimd for MulOddU16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_u16 = s.lanes::<u16>();
        let lanes_u32 = s.lanes::<u32>();
        if lanes_u16 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_odd(va, vb);
            for i in 0..lanes_u32.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_even_u16() {
    for target in available_targets() {
        let lanes_u16 = lanes_for::<u16>(target);
        let lanes_u32 = lanes_for::<u32>(target);
        let a: Vec<u16> =
            (0..lanes_u16).map(|i| (i as u16) * 200 + 100).collect();
        let b: Vec<u16> =
            (0..lanes_u16).map(|i| (i as u16) * 300 + 50).collect();
        let mut out = vec![0u32; lanes_u32];

        dispatch_to(
            MulEvenU16Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_u32 {
            let src = i * 2;
            if src < lanes_u16 {
                let expected = a[src] as u32 * b[src] as u32;
                assert_eq!(
                    out[i], expected,
                    "mul_even u16 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

#[test]
fn test_mul_odd_u16() {
    for target in available_targets() {
        let lanes_u16 = lanes_for::<u16>(target);
        let lanes_u32 = lanes_for::<u32>(target);
        let a: Vec<u16> =
            (0..lanes_u16).map(|i| (i as u16) * 200 + 100).collect();
        let b: Vec<u16> =
            (0..lanes_u16).map(|i| (i as u16) * 300 + 50).collect();
        let mut out = vec![0u32; lanes_u32];

        dispatch_to(
            MulOddU16Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_u32 {
            let src = i * 2 + 1;
            if src < lanes_u16 {
                let expected = a[src] as u32 * b[src] as u32;
                assert_eq!(
                    out[i], expected,
                    "mul_odd u16 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_even / mul_odd: i16 -> i32
// ---------------------------------------------------------------------------

struct MulEvenI16Kernel<'a> {
    a: &'a [i16],
    b: &'a [i16],
    out: &'a mut [i32],
}

impl WithSimd for MulEvenI16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_i16 = s.lanes::<i16>();
        let lanes_i32 = s.lanes::<i32>();
        if lanes_i16 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_even(va, vb);
            for i in 0..lanes_i32.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

struct MulOddI16Kernel<'a> {
    a: &'a [i16],
    b: &'a [i16],
    out: &'a mut [i32],
}

impl WithSimd for MulOddI16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_i16 = s.lanes::<i16>();
        let lanes_i32 = s.lanes::<i32>();
        if lanes_i16 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_odd(va, vb);
            for i in 0..lanes_i32.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_even_i16() {
    for target in available_targets() {
        let lanes_i16 = lanes_for::<i16>(target);
        let lanes_i32 = lanes_for::<i32>(target);
        let a: Vec<i16> =
            (0..lanes_i16).map(|i| (i as i16) * 100 - 500).collect();
        let b: Vec<i16> =
            (0..lanes_i16).map(|i| (i as i16) * 50 + 200).collect();
        let mut out = vec![0i32; lanes_i32];

        dispatch_to(
            MulEvenI16Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_i32 {
            let src = i * 2;
            if src < lanes_i16 {
                let expected = a[src] as i32 * b[src] as i32;
                assert_eq!(
                    out[i], expected,
                    "mul_even i16 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

#[test]
fn test_mul_odd_i16() {
    for target in available_targets() {
        let lanes_i16 = lanes_for::<i16>(target);
        let lanes_i32 = lanes_for::<i32>(target);
        let a: Vec<i16> =
            (0..lanes_i16).map(|i| (i as i16) * 100 - 500).collect();
        let b: Vec<i16> =
            (0..lanes_i16).map(|i| (i as i16) * 50 + 200).collect();
        let mut out = vec![0i32; lanes_i32];

        dispatch_to(
            MulOddI16Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_i32 {
            let src = i * 2 + 1;
            if src < lanes_i16 {
                let expected = a[src] as i32 * b[src] as i32;
                assert_eq!(
                    out[i], expected,
                    "mul_odd i16 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src], b[src], out[i], expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_even / mul_odd: i32 -> i64
// ---------------------------------------------------------------------------

struct MulEvenI32Kernel<'a> {
    a: &'a [i32],
    b: &'a [i32],
    out: &'a mut [i64],
}

impl WithSimd for MulEvenI32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_i32 = s.lanes::<i32>();
        let lanes_i64 = s.lanes::<i64>();
        if lanes_i32 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_even(va, vb);
            for i in 0..lanes_i64.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

struct MulOddI32Kernel<'a> {
    a: &'a [i32],
    b: &'a [i32],
    out: &'a mut [i64],
}

impl WithSimd for MulOddI32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_i32 = s.lanes::<i32>();
        let lanes_i64 = s.lanes::<i64>();
        if lanes_i32 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_odd(va, vb);
            for i in 0..lanes_i64.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_even_i32() {
    for target in available_targets() {
        let lanes_i32 = lanes_for::<i32>(target);
        let lanes_i64 = lanes_for::<i64>(target);
        let a: Vec<i32> =
            (0..lanes_i32 as i32).map(|i| i * 1000 - 5000).collect();
        let b: Vec<i32> =
            (0..lanes_i32 as i32).map(|i| i * 2000 + 100).collect();
        let mut out = vec![0i64; lanes_i64];

        dispatch_to(
            MulEvenI32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_i64 {
            let src = i * 2;
            if src < lanes_i32 {
                let expected = a[src] as i64 * b[src] as i64;
                assert_eq!(
                    out[i], expected,
                    "mul_even i32 lane {i} (src {src}) wrong for {target:?}"
                );
            }
        }
    }
}

#[test]
fn test_mul_odd_i32() {
    for target in available_targets() {
        let lanes_i32 = lanes_for::<i32>(target);
        let lanes_i64 = lanes_for::<i64>(target);
        let a: Vec<i32> =
            (0..lanes_i32 as i32).map(|i| i * 1000 - 5000).collect();
        let b: Vec<i32> =
            (0..lanes_i32 as i32).map(|i| i * 2000 + 100).collect();
        let mut out = vec![0i64; lanes_i64];

        dispatch_to(
            MulOddI32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_i64 {
            let src = i * 2 + 1;
            if src < lanes_i32 {
                let expected = a[src] as i64 * b[src] as i64;
                assert_eq!(
                    out[i], expected,
                    "mul_odd i32 lane {i} (src {src}) wrong for {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// mul_even / mul_odd: f32 -> f64
// ---------------------------------------------------------------------------

struct MulEvenF32Kernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f64],
}

impl WithSimd for MulEvenF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_f32 = s.lanes::<f32>();
        let lanes_f64 = s.lanes::<f64>();
        if lanes_f32 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_even(va, vb);
            for i in 0..lanes_f64.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

struct MulOddF32Kernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f64],
}

impl WithSimd for MulOddF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes_f32 = s.lanes::<f32>();
        let lanes_f64 = s.lanes::<f64>();
        if lanes_f32 > self.a.len() {
            return;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.mul_odd(va, vb);
            for i in 0..lanes_f64.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_mul_even_f32() {
    for target in available_targets() {
        let lanes_f32 = lanes_for::<f32>(target);
        let lanes_f64 = lanes_for::<f64>(target);
        let a: Vec<f32> =
            (0..lanes_f32).map(|i| (i as f32) * 1.5 + 0.5).collect();
        let b: Vec<f32> =
            (0..lanes_f32).map(|i| (i as f32) * 2.5 + 1.0).collect();
        let mut out = vec![0.0f64; lanes_f64];

        dispatch_to(
            MulEvenF32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_f64 {
            let src = i * 2;
            if src < lanes_f32 {
                let expected = a[src] as f64 * b[src] as f64;
                assert!(
                    (out[i] - expected).abs() < 1e-6,
                    "mul_even f32 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src],
                    b[src],
                    out[i],
                    expected
                );
            }
        }
    }
}

#[test]
fn test_mul_odd_f32() {
    for target in available_targets() {
        let lanes_f32 = lanes_for::<f32>(target);
        let lanes_f64 = lanes_for::<f64>(target);
        let a: Vec<f32> =
            (0..lanes_f32).map(|i| (i as f32) * 1.5 + 0.5).collect();
        let b: Vec<f32> =
            (0..lanes_f32).map(|i| (i as f32) * 2.5 + 1.0).collect();
        let mut out = vec![0.0f64; lanes_f64];

        dispatch_to(
            MulOddF32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );

        for i in 0..lanes_f64 {
            let src = i * 2 + 1;
            if src < lanes_f32 {
                let expected = a[src] as f64 * b[src] as f64;
                assert!(
                    (out[i] - expected).abs() < 1e-6,
                    "mul_odd f32 lane {i} (src {src}) wrong for {target:?}: a={}, b={}, got={}, expected={}",
                    a[src],
                    b[src],
                    out[i],
                    expected
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// convert_to_int: f64 -> i64
// ---------------------------------------------------------------------------

struct ConvertF64ToI64Kernel<'a> {
    input: &'a [f64],
    out: &'a mut [i64],
}

impl WithSimd for ConvertF64ToI64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f64>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let converted: S::Vec<i64> = s.convert_to_int(v);
                s.store_u(converted, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i] as i64;
            i += 1;
        }
    }
}

#[test]
fn test_convert_f64_to_i64() {
    let input: Vec<f64> = vec![
        0.0,
        1.0,
        -1.0,
        42.7,
        -42.7,
        1e10,
        -1e10,
        123456789.0,
        -123456789.0,
        0.5,
        -0.5,
        1e15,
        -1e15,
        1e18,
        -1e18,
        0.0,
    ];
    let n = input.len();
    let expected: Vec<i64> = input.iter().map(|x| *x as i64).collect();

    for target in available_targets() {
        let mut out = vec![0i64; n];
        dispatch_to(
            ConvertF64ToI64Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let lanes = lanes_for::<f64>(target);
        for j in 0..n.min(lanes * (n / lanes)) {
            assert_eq!(
                out[j], expected[j],
                "convert_to_int f64->i64 lane {j} for {target:?}: input={}, got={}, expected={}",
                input[j], out[j], expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// bits_from_mask: all lane sizes
// ---------------------------------------------------------------------------

struct BitsFromMaskU8Kernel<'a> {
    input: &'a [u8],
    threshold: u8,
    out: &'a mut u64,
}

impl WithSimd for BitsFromMaskU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(self.threshold);
            let mask = s.gt(v, thresh);
            *self.out = s.bits_from_mask(mask);
        }
    }
}

#[test]
fn test_bits_from_mask_u8() {
    for target in available_targets() {
        let lanes = lanes_for::<u8>(target);
        let input: Vec<u8> = (0..lanes).map(|i| (i * 7 + 3) as u8).collect();
        let threshold = 100u8;
        let mut result = 0u64;

        dispatch_to(
            BitsFromMaskU8Kernel {
                input: &input,
                threshold,
                out: &mut result,
            },
            target,
        );

        let mut expected = 0u64;
        for i in 0..lanes {
            if input[i] > threshold {
                expected |= 1 << i;
            }
        }
        assert_eq!(
            result, expected,
            "bits_from_mask u8 wrong for {target:?}: got 0b{result:b}, expected 0b{expected:b}"
        );
    }
}

struct BitsFromMaskU16Kernel<'a> {
    input: &'a [u16],
    threshold: u16,
    out: &'a mut u64,
}

impl WithSimd for BitsFromMaskU16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u16>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(self.threshold);
            let mask = s.gt(v, thresh);
            *self.out = s.bits_from_mask(mask);
        }
    }
}

#[test]
fn test_bits_from_mask_u16() {
    for target in available_targets() {
        let lanes = lanes_for::<u16>(target);
        let input: Vec<u16> =
            (0..lanes).map(|i| (i * 1000 + 500) as u16).collect();
        let threshold = 5000u16;
        let mut result = 0u64;

        dispatch_to(
            BitsFromMaskU16Kernel {
                input: &input,
                threshold,
                out: &mut result,
            },
            target,
        );

        let mut expected = 0u64;
        for i in 0..lanes {
            if input[i] > threshold {
                expected |= 1 << i;
            }
        }
        assert_eq!(
            result, expected,
            "bits_from_mask u16 wrong for {target:?}: got 0b{result:b}, expected 0b{expected:b}"
        );
    }
}

struct BitsFromMaskU32Kernel<'a> {
    input: &'a [u32],
    threshold: u32,
    out: &'a mut u64,
}

impl WithSimd for BitsFromMaskU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(self.threshold);
            let mask = s.gt(v, thresh);
            *self.out = s.bits_from_mask(mask);
        }
    }
}

#[test]
fn test_bits_from_mask_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (0..lanes as u32).map(|i| i * 10).collect();
        let threshold = 25u32;
        let mut result = 0u64;

        dispatch_to(
            BitsFromMaskU32Kernel {
                input: &input,
                threshold,
                out: &mut result,
            },
            target,
        );

        let mut expected = 0u64;
        for i in 0..lanes {
            if input[i] > threshold {
                expected |= 1 << i;
            }
        }
        assert_eq!(
            result, expected,
            "bits_from_mask u32 wrong for {target:?}: got 0b{result:b}, expected 0b{expected:b}"
        );
    }
}

struct BitsFromMaskU64Kernel<'a> {
    input: &'a [u64],
    threshold: u64,
    out: &'a mut u64,
}

impl WithSimd for BitsFromMaskU64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u64>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let thresh = s.splat(self.threshold);
            let mask = s.gt(v, thresh);
            *self.out = s.bits_from_mask(mask);
        }
    }
}

#[test]
fn test_bits_from_mask_u64() {
    for target in available_targets() {
        let lanes = lanes_for::<u64>(target);
        let input: Vec<u64> = (0..lanes as u64).map(|i| i * 100).collect();
        let threshold = 150u64;
        let mut result = 0u64;

        dispatch_to(
            BitsFromMaskU64Kernel {
                input: &input,
                threshold,
                out: &mut result,
            },
            target,
        );

        let mut expected = 0u64;
        for i in 0..lanes {
            if input[i] > threshold {
                expected |= 1 << i;
            }
        }
        assert_eq!(
            result, expected,
            "bits_from_mask u64 wrong for {target:?}: got 0b{result:b}, expected 0b{expected:b}"
        );
    }
}

// ---------------------------------------------------------------------------
// bits_from_mask: u16 exhaustive (all mask patterns for SSE2 8-lane)
// ---------------------------------------------------------------------------

struct BitsFromMaskU16ExhaustiveKernel<'a> {
    mask_arr: &'a [u16],
    out: &'a mut u64,
}

impl WithSimd for BitsFromMaskU16ExhaustiveKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u16>();
        if lanes > self.mask_arr.len() {
            return;
        }
        unsafe {
            let mask_vec = s.load_u(self.mask_arr.as_ptr());
            let mask = s.mask_from_vec(mask_vec);
            *self.out = s.bits_from_mask(mask);
        }
    }
}

#[test]
fn test_bits_from_mask_u16_exhaustive() {
    for target in available_targets() {
        let lanes = lanes_for::<u16>(target);
        // Test all 2^min(lanes,8) patterns for the first min(lanes,8) lanes
        let test_bits = lanes.min(8);
        let num_patterns = 1u32 << test_bits;

        for pattern in 0..num_patterns {
            let mut mask_arr = vec![0u16; lanes];
            for i in 0..test_bits {
                if pattern & (1 << i) != 0 {
                    mask_arr[i] = u16::MAX;
                }
            }
            let mut result = 0u64;

            dispatch_to(
                BitsFromMaskU16ExhaustiveKernel {
                    mask_arr: &mask_arr,
                    out: &mut result,
                },
                target,
            );

            // Expected: same pattern bits
            let expected = pattern as u64;
            let mask = (1u64 << test_bits) - 1;
            assert_eq!(
                result & mask,
                expected & mask,
                "bits_from_mask u16 pattern 0b{pattern:0width$b} wrong for {target:?}: got 0b{result:0width$b}, expected 0b{expected:0width$b}",
                width = test_bits
            );
        }
    }
}

// =========================================================================
// Exhaustive type-variant coverage tests
// =========================================================================

// ---------------------------------------------------------------------------
// Helper macros for multi-type testing
// ---------------------------------------------------------------------------

/// Generate a saturated_add test for an unsigned integer type.
macro_rules! test_saturated_add_unsigned {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.saturated_add(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> =
                (0..n).map(|i| <$T>::MAX - (i as $T) % 10).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) % 20).collect();
            let expected: Vec<$T> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.saturating_add(y))
                .collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "saturated_add {} failed for {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

/// Generate a saturated_add test for a signed integer type.
macro_rules! test_saturated_add_signed {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.saturated_add(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> =
                (0..n).map(|i| <$T>::MAX - (i as $T) % 10).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) % 20 - 5).collect();
            let expected: Vec<$T> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.saturating_add(y))
                .collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "saturated_add {} failed for {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

/// Generate a saturated_sub test for an unsigned integer type.
macro_rules! test_saturated_sub_unsigned {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.saturated_sub(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> = (0..n).map(|i| (i as $T) % 10).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) % 20).collect();
            let expected: Vec<$T> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.saturating_sub(y))
                .collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "saturated_sub {} failed for {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

/// Generate a saturated_sub test for a signed integer type.
macro_rules! test_saturated_sub_signed {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.saturated_sub(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> =
                (0..n).map(|i| <$T>::MIN + (i as $T) % 10).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) % 20 - 5).collect();
            let expected: Vec<$T> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| x.saturating_sub(y))
                .collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "saturated_sub {} failed for {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_saturated_add_unsigned!(test_saturated_add_u8, u8);
test_saturated_add_unsigned!(test_saturated_add_u16, u16);
test_saturated_add_unsigned!(test_saturated_add_u32, u32);
test_saturated_add_unsigned!(test_saturated_add_u64, u64);
test_saturated_add_signed!(test_saturated_add_i8, i8);
test_saturated_add_signed!(test_saturated_add_i16, i16);
test_saturated_add_signed!(test_saturated_add_i32, i32);
test_saturated_add_signed!(test_saturated_add_i64, i64);
test_saturated_sub_unsigned!(test_saturated_sub_u16, u16);
test_saturated_sub_unsigned!(test_saturated_sub_u32, u32);
test_saturated_sub_unsigned!(test_saturated_sub_u64, u64);
test_saturated_sub_signed!(test_saturated_sub_i8, i8);
test_saturated_sub_signed!(test_saturated_sub_i16, i16);
test_saturated_sub_signed!(test_saturated_sub_i32, i32);
test_saturated_sub_signed!(test_saturated_sub_i64, i64);

// ---------------------------------------------------------------------------
// mul_high: all integer types that have type-specific paths
// ---------------------------------------------------------------------------

macro_rules! test_mul_high {
    ($name:ident, $T:ty, $wide:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.mul_high(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let bits = core::mem::size_of::<$T>() * 8;
            let a: Vec<$T> =
                (0..n).map(|i| ((i as $wide) * 13 + 100) as $T).collect();
            let b: Vec<$T> =
                (0..n).map(|i| ((i as $wide) * 7 + 50) as $T).collect();
            let expected: Vec<$T> = a
                .iter()
                .zip(&b)
                .map(|(&x, &y)| {
                    ((x as $wide).wrapping_mul(y as $wide) >> bits) as $T
                })
                .collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "mul_high {} failed for {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_mul_high!(test_mul_high_u8, u8, u16);
test_mul_high!(test_mul_high_i8, i8, i16);
test_mul_high!(test_mul_high_u16, u16, u32);
test_mul_high!(test_mul_high_u32, u32, u64);
test_mul_high!(test_mul_high_i32, i32, i64);

// ---------------------------------------------------------------------------
// average_round: u16 (u8 already tested)
// ---------------------------------------------------------------------------

#[test]
fn test_average_round_u16() {
    struct K<'a> {
        a: &'a [u16],
        b: &'a [u16],
        out: &'a mut [u16],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u16>();
            let n = self.a.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let va = s.load_u(self.a.as_ptr().add(i));
                    let vb = s.load_u(self.b.as_ptr().add(i));
                    s.store_u(
                        s.average_round(va, vb),
                        self.out.as_mut_ptr().add(i),
                    );
                }
                i += lanes;
            }
        }
    }
    let n = 32;
    let a: Vec<u16> = (0..n).map(|i| (i * 1000 + 100) as u16).collect();
    let b: Vec<u16> = (0..n).map(|i| (i * 500 + 200) as u16).collect();
    let expected: Vec<u16> = a
        .iter()
        .zip(&b)
        .map(|(&x, &y)| ((x as u32 + y as u32 + 1) / 2) as u16)
        .collect();
    for target in available_targets() {
        let mut out = vec![0u16; n];
        let lanes = lanes_for::<u16>(target);
        dispatch_to(
            K {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        assert_eq!(
            &out[..check],
            &expected[..check],
            "average_round u16 failed for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// promote_to: all missing pairs
// ---------------------------------------------------------------------------

macro_rules! test_promote {
    ($name:ident, $N:ty, $W:ty, $gen:expr) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$N],
                out: &'a mut [$W],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes_n = s.lanes::<$N>();
                    let lanes_w = s.lanes::<$W>();
                    if lanes_n > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        let promoted: S::Vec<$W> = s.promote_to(v);
                        for i in 0..lanes_w.min(self.out.len()) {
                            self.out[i] = s.extract_lane(promoted, i);
                        }
                    }
                }
            }
            for target in available_targets() {
                let lanes_n = lanes_for::<$N>(target);
                let lanes_w = lanes_for::<$W>(target);
                let gen_fn: fn(usize) -> $N = $gen;
                let input: Vec<$N> = (0..lanes_n).map(gen_fn).collect();
                let mut out = vec![<$W>::default(); lanes_w];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                for i in 0..lanes_w.min(lanes_n) {
                    let expected = input[i] as $W;
                    assert_eq!(
                        out[i],
                        expected,
                        "promote {} -> {} lane {i} wrong for {target:?}: got {:?}, expected {:?}",
                        stringify!($N),
                        stringify!($W),
                        out[i],
                        expected
                    );
                }
            }
        }
    };
}

test_promote!(test_promote_u8_to_u16, u8, u16, |i| (i as u8)
    .wrapping_mul(17));
test_promote!(test_promote_i8_to_i16, i8, i16, |i| (i as i8)
    .wrapping_mul(13)
    .wrapping_sub(60));
test_promote!(test_promote_i16_to_i32, i16, i32, |i| (i as i16) * 100
    - 500);
test_promote!(test_promote_u32_to_u64, u32, u64, |i| i as u32 * 100000
    + 42);
test_promote!(test_promote_i32_to_i64, i32, i64, |i| i as i32 * 100000
    - 500000);
test_promote!(test_promote_f32_to_f64, f32, f64, |i| i as f32 * 1.5 - 10.0);

// ---------------------------------------------------------------------------
// demote_to: missing pairs (i16->i8, u16->u8)
// ---------------------------------------------------------------------------

macro_rules! test_demote {
    ($name:ident, $W:ty, $N:ty, $gen:expr, $clamp:expr) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$W],
                out: &'a mut [$N],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes_w = s.lanes::<$W>();
                    let lanes_n = s.lanes::<$N>();
                    if lanes_w > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        let demoted: S::Vec<$N> = s.demote_to(v);
                        for i in 0..lanes_n.min(self.out.len()) {
                            self.out[i] = s.extract_lane(demoted, i);
                        }
                    }
                }
            }
            for target in available_targets() {
                let lanes_w = lanes_for::<$W>(target);
                let lanes_n = lanes_for::<$N>(target);
                let gen_fn: fn(usize) -> $W = $gen;
                let input: Vec<$W> = (0..lanes_w).map(gen_fn).collect();
                let mut out = vec![<$N>::default(); lanes_n];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                let clamp_fn: fn($W) -> $N = $clamp;
                for i in 0..lanes_w.min(lanes_n) {
                    let expected = clamp_fn(input[i]);
                    assert_eq!(
                        out[i],
                        expected,
                        "demote {} -> {} lane {i} wrong for {target:?}: input={:?}, got={:?}, expected={:?}",
                        stringify!($W),
                        stringify!($N),
                        input[i],
                        out[i],
                        expected
                    );
                }
            }
        }
    };
}

test_demote!(
    test_demote_i16_to_i8,
    i16,
    i8,
    |i| match i % 4 {
        0 => 42,
        1 => 200,
        2 => -200,
        _ => -1,
    },
    |v: i16| v.clamp(i8::MIN as i16, i8::MAX as i16) as i8
);

test_demote!(
    test_demote_u16_to_u8,
    u16,
    u8,
    |i| match i % 4 {
        0 => 42,
        1 => 255,
        2 => 300,
        _ => 0,
    },
    |v: u16| v.min(u8::MAX as u16) as u8
);

// ---------------------------------------------------------------------------
// neg: f64 and integer types
// ---------------------------------------------------------------------------

#[test]
fn test_neg_f64() {
    struct K<'a> {
        input: &'a [f64],
        out: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.input.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let v = s.load_u(self.input.as_ptr().add(i));
                    s.store_u(s.neg(v), self.out.as_mut_ptr().add(i));
                }
                i += lanes;
            }
        }
    }
    let input: Vec<f64> = vec![1.0, -2.5, 0.0, 1e15, -1e15, 3.125, -0.0, 42.0];
    let n = input.len();
    let expected: Vec<f64> = input.iter().map(|x| -x).collect();
    for target in available_targets() {
        let mut out = vec![0.0f64; n];
        let lanes = lanes_for::<f64>(target);
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        for j in 0..check {
            assert!(
                out[j].to_bits() == expected[j].to_bits(),
                "neg f64 lane {j} {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

macro_rules! test_neg_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.input.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let v = s.load_u(self.input.as_ptr().add(i));
                            s.store_u(s.neg(v), self.out.as_mut_ptr().add(i));
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let input: Vec<$T> = (0..n)
                .map(|i| (i as $T).wrapping_mul(7).wrapping_sub(30))
                .collect();
            let expected: Vec<$T> =
                input.iter().map(|x| x.wrapping_neg()).collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "neg {} failed for {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_neg_int!(test_neg_i8, i8);
test_neg_int!(test_neg_i16, i16);
test_neg_int!(test_neg_i32, i32);
test_neg_int!(test_neg_i64, i64);

// ---------------------------------------------------------------------------
// div: f64
// ---------------------------------------------------------------------------

#[test]
fn test_div_f64() {
    struct K<'a> {
        a: &'a [f64],
        b: &'a [f64],
        out: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.a.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let va = s.load_u(self.a.as_ptr().add(i));
                    let vb = s.load_u(self.b.as_ptr().add(i));
                    s.store_u(s.div(va, vb), self.out.as_mut_ptr().add(i));
                }
                i += lanes;
            }
        }
    }
    let a: Vec<f64> = vec![10.0, -20.0, 100.0, 1e15, 0.0, 3.125, -1.0, 42.0];
    let b: Vec<f64> = vec![2.0, 4.0, -5.0, 1e10, 1.0, 2.0, -0.5, 7.0];
    let n = a.len();
    let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x / y).collect();
    for target in available_targets() {
        let mut out = vec![0.0f64; n];
        let lanes = lanes_for::<f64>(target);
        dispatch_to(
            K {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        for j in 0..check {
            assert!(
                (out[j] - expected[j]).abs() < 1e-6,
                "div f64 lane {j} {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// sum_of_lanes: all types with distinct paths
// ---------------------------------------------------------------------------

macro_rules! test_sum_of_lanes_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut $T,
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    if lanes > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        *self.out = s.sum_of_lanes(v);
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> = (0..lanes)
                    .map(|i| (i as $T).wrapping_mul(3).wrapping_add(1))
                    .collect();
                let expected: $T = input.iter().copied().fold(0 as $T, |a, b| a.wrapping_add(b));
                let mut result: $T = 0;
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut result,
                    },
                    target,
                );
                assert_eq!(
                    result,
                    expected,
                    "sum_of_lanes {} failed for {target:?}: got {:?}, expected {:?}",
                    stringify!($T),
                    result,
                    expected
                );
            }
        }
    };
}

macro_rules! test_sum_of_lanes_float {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut $T,
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    if lanes > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        *self.out = s.sum_of_lanes(v);
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> = (0..lanes).map(|i| i as $T * 1.5 + 1.0).collect();
                let expected: $T = input.iter().copied().sum();
                let mut result: $T = 0.0;
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut result,
                    },
                    target,
                );
                assert!(
                    (result - expected).abs() < 0.01,
                    "sum_of_lanes {} failed for {target:?}: got {result}, expected {expected}",
                    stringify!($T)
                );
            }
        }
    };
}

test_sum_of_lanes_int!(test_sum_of_lanes_u8, u8);
test_sum_of_lanes_int!(test_sum_of_lanes_u16, u16);
test_sum_of_lanes_int!(test_sum_of_lanes_u32, u32);
test_sum_of_lanes_int!(test_sum_of_lanes_u64, u64);
test_sum_of_lanes_int!(test_sum_of_lanes_i8, i8);
test_sum_of_lanes_int!(test_sum_of_lanes_i16, i16);
test_sum_of_lanes_int!(test_sum_of_lanes_i64, i64);
test_sum_of_lanes_float!(test_sum_of_lanes_f32, f32);
test_sum_of_lanes_float!(test_sum_of_lanes_f64, f64);

// ---------------------------------------------------------------------------
// min_of_lanes / max_of_lanes: all types
// ---------------------------------------------------------------------------

macro_rules! test_min_max_of_lanes {
    ($name:ident, $T:ty, $gen:expr) => {
        #[test]
        fn $name() {
            struct KMin<'a> {
                input: &'a [$T],
                out: &'a mut $T,
            }
            struct KMax<'a> {
                input: &'a [$T],
                out: &'a mut $T,
            }
            impl WithSimd for KMin<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    if lanes > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        *self.out = s.min_of_lanes(v);
                    }
                }
            }
            impl WithSimd for KMax<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    if lanes > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        *self.out = s.max_of_lanes(v);
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let gen_fn: fn(usize, usize) -> $T = $gen;
                let input: Vec<$T> =
                    (0..lanes).map(|i| gen_fn(i, lanes)).collect();
                let exp_min = *input
                    .iter()
                    .min_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let exp_max = *input
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap())
                    .unwrap();
                let mut got_min: $T = input[0];
                let mut got_max: $T = input[0];
                dispatch_to(
                    KMin {
                        input: &input,
                        out: &mut got_min,
                    },
                    target,
                );
                dispatch_to(
                    KMax {
                        input: &input,
                        out: &mut got_max,
                    },
                    target,
                );
                assert_eq!(
                    got_min,
                    exp_min,
                    "min_of_lanes {} failed for {target:?}: input={input:?}",
                    stringify!($T)
                );
                assert_eq!(
                    got_max,
                    exp_max,
                    "max_of_lanes {} failed for {target:?}: input={input:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_min_max_of_lanes!(
    test_min_max_of_lanes_u8,
    u8,
    |i, l| ((i * 37 + 13) % l) as u8
);
test_min_max_of_lanes!(test_min_max_of_lanes_u16, u16, |i, l| ((i * 37 + 13)
    % l) as u16
    * 100);
test_min_max_of_lanes!(test_min_max_of_lanes_u32, u32, |i, l| ((i * 37 + 13)
    % l) as u32
    * 10000);
test_min_max_of_lanes!(test_min_max_of_lanes_u64, u64, |i, l| ((i * 37 + 13)
    % l) as u64
    * 100000);
test_min_max_of_lanes!(test_min_max_of_lanes_i8, i8, |i, l| {
    (((i * 37 + 13) % l) as i8).wrapping_sub(60)
});
test_min_max_of_lanes!(test_min_max_of_lanes_i16, i16, |i, l| ((i * 37 + 13)
    % l) as i16
    * 100
    - 5000);
test_min_max_of_lanes!(test_min_max_of_lanes_i64, i64, |i, l| ((i * 37 + 13)
    % l) as i64
    * 100000
    - 500000);
test_min_max_of_lanes!(test_min_max_of_lanes_f32, f32, |i, l| ((i * 37 + 13)
    % l) as f32
    * 1.5
    - 10.0);
test_min_max_of_lanes!(test_min_max_of_lanes_f64, f64, |i, l| ((i * 37 + 13)
    % l) as f64
    * 2.5
    - 20.0);

// ---------------------------------------------------------------------------
// population_count / leading_zero_count / trailing_zero_count: all integer types
// ---------------------------------------------------------------------------

macro_rules! test_bitop {
    ($name:ident, $T:ty, $op:ident, $ref_fn:expr) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.input.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let v = s.load_u(self.input.as_ptr().add(i));
                            s.store_u(s.$op(v), self.out.as_mut_ptr().add(i));
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let input: Vec<$T> = (0..n)
                .map(|i| (i as $T).wrapping_mul(0x37u8 as $T) | (1 as $T))
                .collect();
            let ref_fn: fn($T) -> $T = $ref_fn;
            let expected: Vec<$T> = input.iter().map(|&x| ref_fn(x)).collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "{} {} failed for {target:?}",
                    stringify!($op),
                    stringify!($T)
                );
            }
        }
    };
}

test_bitop!(
    test_popcount_u8,
    u8,
    population_count,
    |x: u8| x.count_ones() as u8
);
test_bitop!(
    test_popcount_u16,
    u16,
    population_count,
    |x: u16| x.count_ones() as u16
);
test_bitop!(test_popcount_u32, u32, population_count, |x: u32| {
    x.count_ones()
});
test_bitop!(
    test_popcount_u64,
    u64,
    population_count,
    |x: u64| x.count_ones() as u64
);

test_bitop!(
    test_lzcnt_u8,
    u8,
    leading_zero_count,
    |x: u8| x.leading_zeros() as u8
);
test_bitop!(
    test_lzcnt_u16,
    u16,
    leading_zero_count,
    |x: u16| x.leading_zeros() as u16
);
test_bitop!(test_lzcnt_u32, u32, leading_zero_count, |x: u32| {
    x.leading_zeros()
});
test_bitop!(
    test_lzcnt_u64,
    u64,
    leading_zero_count,
    |x: u64| x.leading_zeros() as u64
);

test_bitop!(
    test_tzcnt_u8,
    u8,
    trailing_zero_count,
    |x: u8| x.trailing_zeros() as u8
);
test_bitop!(test_tzcnt_u16, u16, trailing_zero_count, |x: u16| x
    .trailing_zeros()
    as u16);
test_bitop!(test_tzcnt_u32, u32, trailing_zero_count, |x: u32| {
    x.trailing_zeros()
});
test_bitop!(test_tzcnt_u64, u64, trailing_zero_count, |x: u64| x
    .trailing_zeros()
    as u64);

test_bitop!(test_reverse_bits_u16, u16, reverse_bits, |x: u16| x
    .reverse_bits());
test_bitop!(test_reverse_bits_u32, u32, reverse_bits, |x: u32| x
    .reverse_bits());
test_bitop!(test_reverse_bits_u64, u64, reverse_bits, |x: u64| x
    .reverse_bits());

// ---------------------------------------------------------------------------
// convert_to_float: f64 path (i64 -> f64, the Wim trick)
// ---------------------------------------------------------------------------

#[test]
fn test_convert_to_float_f64() {
    struct K<'a> {
        input: &'a [i64],
        out: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.input.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let vi = s.load_u(self.input.as_ptr().add(i));
                    let vf: S::Vec<f64> = s.convert_to_float(vi);
                    s.store_u(vf, self.out.as_mut_ptr().add(i));
                }
                i += lanes;
            }
        }
    }
    let input: Vec<i64> = vec![
        0,
        1,
        -1,
        42,
        -42,
        1000000,
        -1000000,
        123456789,
        -123456789,
        i32::MAX as i64,
        i32::MIN as i64,
        0,
        1,
        -1,
        100,
        -100,
    ];
    let n = input.len();
    let expected: Vec<f64> = input.iter().map(|&x| x as f64).collect();
    for target in available_targets() {
        let mut out = vec![0.0f64; n];
        let lanes = lanes_for::<f64>(target);
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        for j in 0..check {
            assert!(
                (out[j] - expected[j]).abs() < 1.0,
                "convert_to_float f64 lane {j} {target:?}: got {}, expected {}, input={}",
                out[j],
                expected[j],
                input[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// compress: f32, u64, i64
// ---------------------------------------------------------------------------

#[test]
fn test_compress_f32() {
    struct K<'a> {
        input: &'a [f32],
        out: &'a mut [f32],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f32>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0.0f32);
                let mask = s.gt(v, thresh);
                let compressed = s.compress(v, mask);
                s.store_u(compressed, self.out.as_mut_ptr());
                *self.count = s.count_true(mask);
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let input: Vec<f32> = (0..lanes)
            .map(|i| {
                if i % 2 == 0 {
                    -(i as f32) - 1.0
                } else {
                    (i as f32) * 10.0
                }
            })
            .collect();
        let mut out = vec![0.0f32; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<f32> =
            input.iter().copied().filter(|x| *x > 0.0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress f32 count wrong for {target:?}"
        );
        for j in 0..count {
            assert!(
                (out[j] - expected[j]).abs() < 1e-4,
                "compress f32 values wrong at {j} for {target:?}"
            );
        }
    }
}

#[test]
fn test_compress_u64() {
    struct K<'a> {
        input: &'a [u64],
        out: &'a mut [u64],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u64>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(100u64);
                let mask = s.gt(v, thresh);
                let compressed = s.compress(v, mask);
                s.store_u(compressed, self.out.as_mut_ptr());
                *self.count = s.count_true(mask);
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u64>(target);
        let input: Vec<u64> = (0..lanes as u64).map(|i| i * 80).collect();
        let mut out = vec![0u64; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<u64> =
            input.iter().copied().filter(|x| *x > 100).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress u64 count wrong for {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress u64 values wrong for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// shift_left / shift_right: types with distinct code paths
// ---------------------------------------------------------------------------

macro_rules! test_shift {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct KL<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            struct KR<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for KL<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    if lanes > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        let r = s.shift_left::<$T, 2>(v);
                        s.store_u(r, self.out.as_mut_ptr());
                    }
                }
            }
            impl WithSimd for KR<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    if lanes > self.input.len() {
                        return;
                    }
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        let r = s.shift_right::<$T, 2>(v);
                        s.store_u(r, self.out.as_mut_ptr());
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> = (0..lanes)
                    .map(|i| ((i as $T).wrapping_mul(17)).wrapping_add(5))
                    .collect();
                let mut out_l = vec![0 as $T; lanes];
                let mut out_r = vec![0 as $T; lanes];
                dispatch_to(
                    KL {
                        input: &input,
                        out: &mut out_l,
                    },
                    target,
                );
                dispatch_to(
                    KR {
                        input: &input,
                        out: &mut out_r,
                    },
                    target,
                );
                for j in 0..lanes {
                    let exp_l = input[j].wrapping_shl(2);
                    let exp_r = input[j] >> 2; // logical for unsigned, arithmetic for signed
                    assert_eq!(
                        out_l[j],
                        exp_l,
                        "shift_left {} lane {j} {target:?}: got {:?}, expected {:?}",
                        stringify!($T),
                        out_l[j],
                        exp_l
                    );
                    assert_eq!(
                        out_r[j],
                        exp_r,
                        "shift_right {} lane {j} {target:?}: got {:?}, expected {:?}",
                        stringify!($T),
                        out_r[j],
                        exp_r
                    );
                }
            }
        }
    };
}

test_shift!(test_shift_u8, u8);
test_shift!(test_shift_u16, u16);
test_shift!(test_shift_u32, u32);
test_shift!(test_shift_u64, u64);
test_shift!(test_shift_i8, i8);
test_shift!(test_shift_i16, i16);
test_shift!(test_shift_i32, i32);
test_shift!(test_shift_i64, i64);

// =========================================================================
// Batch 1: add / sub / mul / abs for all types
// =========================================================================

macro_rules! test_binop_int {
    ($name:ident, $T:ty, $op:ident, $ref_fn:expr) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.$op(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> = (0..n)
                .map(|i| (i as $T).wrapping_mul(7).wrapping_add(3))
                .collect();
            let b: Vec<$T> = (0..n)
                .map(|i| (i as $T).wrapping_mul(3).wrapping_add(1))
                .collect();
            let ref_fn: fn($T, $T) -> $T = $ref_fn;
            let expected: Vec<$T> =
                a.iter().zip(&b).map(|(&x, &y)| ref_fn(x, y)).collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "{} {} failed for {target:?}",
                    stringify!($op),
                    stringify!($T)
                );
            }
        }
    };
}

macro_rules! test_binop_float {
    ($name:ident, $T:ty, $op:ident, $ref_fn:expr) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.$op(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> = (0..n).map(|i| (i as $T) * 1.5 + 0.25).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) * 0.75 + 1.0).collect();
            let ref_fn: fn($T, $T) -> $T = $ref_fn;
            let expected: Vec<$T> =
                a.iter().zip(&b).map(|(&x, &y)| ref_fn(x, y)).collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                for j in 0..check {
                    assert!(
                        (out[j] - expected[j]).abs() < 1e-4 as $T,
                        "{} {} lane {j} {target:?}: got {}, expected {}",
                        stringify!($op),
                        stringify!($T),
                        out[j],
                        expected[j]
                    );
                }
            }
        }
    };
}

// add
test_binop_int!(test_add_u8, u8, add, |a: u8, b: u8| a.wrapping_add(b));
test_binop_int!(test_add_u16, u16, add, |a: u16, b: u16| a.wrapping_add(b));
test_binop_int!(test_add_u32, u32, add, |a: u32, b: u32| a.wrapping_add(b));
test_binop_int!(test_add_u64, u64, add, |a: u64, b: u64| a.wrapping_add(b));
test_binop_int!(test_add_i8, i8, add, |a: i8, b: i8| a.wrapping_add(b));
test_binop_int!(test_add_i16, i16, add, |a: i16, b: i16| a.wrapping_add(b));
test_binop_int!(test_add_i32, i32, add, |a: i32, b: i32| a.wrapping_add(b));
test_binop_int!(test_add_i64, i64, add, |a: i64, b: i64| a.wrapping_add(b));
test_binop_float!(test_add_f32, f32, add, |a: f32, b: f32| a + b);
test_binop_float!(test_add_f64, f64, add, |a: f64, b: f64| a + b);

// sub
test_binop_int!(test_sub_u8, u8, sub, |a: u8, b: u8| a.wrapping_sub(b));
test_binop_int!(test_sub_u16, u16, sub, |a: u16, b: u16| a.wrapping_sub(b));
test_binop_int!(test_sub_u32, u32, sub, |a: u32, b: u32| a.wrapping_sub(b));
test_binop_int!(test_sub_u64, u64, sub, |a: u64, b: u64| a.wrapping_sub(b));
test_binop_int!(test_sub_i8, i8, sub, |a: i8, b: i8| a.wrapping_sub(b));
test_binop_int!(test_sub_i16, i16, sub, |a: i16, b: i16| a.wrapping_sub(b));
test_binop_int!(test_sub_i32, i32, sub, |a: i32, b: i32| a.wrapping_sub(b));
test_binop_int!(test_sub_i64, i64, sub, |a: i64, b: i64| a.wrapping_sub(b));
test_binop_float!(test_sub_f64, f64, sub, |a: f64, b: f64| a - b);

// mul
test_binop_int!(test_mul_u8, u8, mul, |a: u8, b: u8| a.wrapping_mul(b));
test_binop_int!(test_mul_u16, u16, mul, |a: u16, b: u16| a.wrapping_mul(b));
test_binop_int!(test_mul_u32, u32, mul, |a: u32, b: u32| a.wrapping_mul(b));
test_binop_int!(test_mul_i8, i8, mul, |a: i8, b: i8| a.wrapping_mul(b));
test_binop_int!(test_mul_i16, i16, mul, |a: i16, b: i16| a.wrapping_mul(b));
test_binop_int!(test_mul_i32, i32, mul, |a: i32, b: i32| a.wrapping_mul(b));
test_binop_float!(test_mul_f32, f32, mul, |a: f32, b: f32| a * b);
test_binop_float!(test_mul_f64, f64, mul, |a: f64, b: f64| a * b);

// abs
macro_rules! test_abs_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.input.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let v = s.load_u(self.input.as_ptr().add(i));
                            s.store_u(s.abs(v), self.out.as_mut_ptr().add(i));
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            // For signed: test negative, positive, zero, MIN, MAX
            // For unsigned: abs is identity
            let input: Vec<$T> = (0..n)
                .map(|i| {
                    let v = (i as $T).wrapping_mul(11).wrapping_sub(30 as $T);
                    v
                })
                .collect();
            #[allow(unused_comparisons)]
            let expected: Vec<$T> = input
                .iter()
                .map(|&x| {
                    if <$T>::MIN == 0 {
                        x
                    } else if x < 0 {
                        (0 as $T).wrapping_sub(x)
                    } else {
                        x
                    }
                })
                .collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "abs {} failed for {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

macro_rules! test_abs_float {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.input.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let v = s.load_u(self.input.as_ptr().add(i));
                            s.store_u(s.abs(v), self.out.as_mut_ptr().add(i));
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let input: Vec<$T> =
                (0..n).map(|i| (i as $T) * 2.5 - 80.0).collect();
            let expected: Vec<$T> = input.iter().map(|x| x.abs()).collect();
            for target in available_targets() {
                let mut out = vec![0 as $T; n];
                let lanes = lanes_for::<$T>(target);
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                let check = lanes * (n / lanes);
                for j in 0..check {
                    assert!(
                        (out[j] - expected[j]).abs() < 1e-6 as $T,
                        "abs {} lane {j} {target:?}: got {}, expected {}",
                        stringify!($T),
                        out[j],
                        expected[j]
                    );
                }
            }
        }
    };
}

test_abs_int!(test_abs_u8, u8);
test_abs_int!(test_abs_u16, u16);
test_abs_int!(test_abs_u32, u32);
test_abs_int!(test_abs_u64, u64);
test_abs_int!(test_abs_i8, i8);
test_abs_int!(test_abs_i16, i16);
test_abs_int!(test_abs_i32, i32);
test_abs_int!(test_abs_i64, i64);
test_abs_float!(test_abs_f32, f32);
test_abs_float!(test_abs_f64, f64);

// =========================================================================
// Batch 2: bitwise ops, comparisons, min/max, abs_diff, clamp
// =========================================================================

// ---------------------------------------------------------------------------
// and / or / xor / not / and_not for all integer types
// ---------------------------------------------------------------------------

macro_rules! test_bitwise_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out_and: &'a mut [$T],
                out_or: &'a mut [$T],
                out_xor: &'a mut [$T],
                out_not: &'a mut [$T],
                out_andn: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.and(va, vb),
                                self.out_and.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.or(va, vb),
                                self.out_or.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.xor(va, vb),
                                self.out_xor.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.not(va),
                                self.out_not.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.and_not(va, vb),
                                self.out_andn.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> =
                (0..n).map(|i| (i as $T).wrapping_mul(0x5B)).collect();
            let b: Vec<$T> = (0..n)
                .map(|i| (i as $T).wrapping_mul(0x3D).wrapping_add(0x11))
                .collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut o_and = vec![0 as $T; n];
                let mut o_or = vec![0 as $T; n];
                let mut o_xor = vec![0 as $T; n];
                let mut o_not = vec![0 as $T; n];
                let mut o_andn = vec![0 as $T; n];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out_and: &mut o_and,
                        out_or: &mut o_or,
                        out_xor: &mut o_xor,
                        out_not: &mut o_not,
                        out_andn: &mut o_andn,
                    },
                    target,
                );
                for j in 0..check {
                    let ea = a[j];
                    let eb = b[j];
                    assert_eq!(
                        o_and[j],
                        ea & eb,
                        "and {} lane {j} {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_or[j],
                        ea | eb,
                        "or {} lane {j} {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_xor[j],
                        ea ^ eb,
                        "xor {} lane {j} {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_not[j],
                        !ea,
                        "not {} lane {j} {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_andn[j],
                        !ea & eb,
                        "and_not {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_bitwise_type!(test_bitwise_u8, u8);
test_bitwise_type!(test_bitwise_u16, u16);
test_bitwise_type!(test_bitwise_u32, u32);
test_bitwise_type!(test_bitwise_u64, u64);
test_bitwise_type!(test_bitwise_i8, i8);
test_bitwise_type!(test_bitwise_i16, i16);
test_bitwise_type!(test_bitwise_i32, i32);
test_bitwise_type!(test_bitwise_i64, i64);

// ---------------------------------------------------------------------------
// eq / ne / lt / le / gt / ge for all types
// ---------------------------------------------------------------------------

macro_rules! test_compare_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                eq: &'a mut [$T],
                ne: &'a mut [$T],
                lt: &'a mut [$T],
                le: &'a mut [$T],
                gt: &'a mut [$T],
                ge: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.vec_from_mask(s.eq(va, vb)),
                                self.eq.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.vec_from_mask(s.ne(va, vb)),
                                self.ne.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.vec_from_mask(s.lt(va, vb)),
                                self.lt.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.vec_from_mask(s.le(va, vb)),
                                self.le.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.vec_from_mask(s.gt(va, vb)),
                                self.gt.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.vec_from_mask(s.ge(va, vb)),
                                self.ge.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> =
                (0..n).map(|i| (i as $T).wrapping_mul(3)).collect();
            let b: Vec<$T> = (0..n)
                .map(|i| {
                    (i as $T).wrapping_mul(3).wrapping_add(
                        if i % 3 == 0 {
                            0
                        } else if i % 3 == 1 {
                            1 as $T
                        } else {
                            <$T>::MAX
                        }, // 0, +1, -1(wrapping)
                    )
                })
                .collect();
            let all_set: $T = !0;
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut o_eq = vec![0 as $T; n];
                let mut o_ne = vec![0 as $T; n];
                let mut o_lt = vec![0 as $T; n];
                let mut o_le = vec![0 as $T; n];
                let mut o_gt = vec![0 as $T; n];
                let mut o_ge = vec![0 as $T; n];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        eq: &mut o_eq,
                        ne: &mut o_ne,
                        lt: &mut o_lt,
                        le: &mut o_le,
                        gt: &mut o_gt,
                        ge: &mut o_ge,
                    },
                    target,
                );
                for j in 0..check {
                    let ea = a[j];
                    let eb = b[j];
                    let expect =
                        |c: bool| -> $T { if c { all_set } else { 0 } };
                    assert_eq!(
                        o_eq[j],
                        expect(ea == eb),
                        "eq {} lane {j} {target:?}: a={ea}, b={eb}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_ne[j],
                        expect(ea != eb),
                        "ne {} lane {j} {target:?}: a={ea}, b={eb}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_lt[j],
                        expect(ea < eb),
                        "lt {} lane {j} {target:?}: a={ea}, b={eb}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_le[j],
                        expect(ea <= eb),
                        "le {} lane {j} {target:?}: a={ea}, b={eb}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_gt[j],
                        expect(ea > eb),
                        "gt {} lane {j} {target:?}: a={ea}, b={eb}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_ge[j],
                        expect(ea >= eb),
                        "ge {} lane {j} {target:?}: a={ea}, b={eb}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

macro_rules! test_compare_float {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                eq: &'a mut [u8],
                ne: &'a mut [u8],
                lt: &'a mut [u8],
                le: &'a mut [u8],
                gt: &'a mut [u8],
                ge: &'a mut [u8],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            self.eq[i] =
                                if s.all_true(s.eq(va, vb)) { 1 } else { 0 };
                            self.ne[i] =
                                if s.all_true(s.ne(va, vb)) { 1 } else { 0 };
                            self.lt[i] =
                                if s.all_true(s.lt(va, vb)) { 1 } else { 0 };
                            self.le[i] =
                                if s.all_true(s.le(va, vb)) { 1 } else { 0 };
                            self.gt[i] =
                                if s.all_true(s.gt(va, vb)) { 1 } else { 0 };
                            self.ge[i] =
                                if s.all_true(s.ge(va, vb)) { 1 } else { 0 };
                        }
                        i += lanes;
                    }
                }
            }
            let test_cases: Vec<($T, $T)> = vec![
                (1.0, 1.0),
                (1.0, 2.0),
                (2.0, 1.0),
                (-1.0, 1.0),
                (0.0, -0.0),
            ];
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                for &(av, bv) in &test_cases {
                    let a = vec![av; lanes];
                    let b = vec![bv; lanes];
                    let mut o_eq = vec![0u8; lanes];
                    let mut o_ne = vec![0u8; lanes];
                    let mut o_lt = vec![0u8; lanes];
                    let mut o_le = vec![0u8; lanes];
                    let mut o_gt = vec![0u8; lanes];
                    let mut o_ge = vec![0u8; lanes];
                    dispatch_to(
                        K {
                            a: &a,
                            b: &b,
                            eq: &mut o_eq,
                            ne: &mut o_ne,
                            lt: &mut o_lt,
                            le: &mut o_le,
                            gt: &mut o_gt,
                            ge: &mut o_ge,
                        },
                        target,
                    );
                    let chk = |name: &str, got: u8, exp: bool| {
                        assert_eq!(
                            got != 0,
                            exp,
                            "{name} {} a={av} b={bv} {target:?}",
                            stringify!($T)
                        );
                    };
                    chk("eq", o_eq[0], av == bv);
                    chk("ne", o_ne[0], av != bv);
                    chk("lt", o_lt[0], av < bv);
                    chk("le", o_le[0], av <= bv);
                    chk("gt", o_gt[0], av > bv);
                    chk("ge", o_ge[0], av >= bv);
                }
            }
        }
    };
}

test_compare_int!(test_cmp_u8, u8);
test_compare_int!(test_cmp_u16, u16);
test_compare_int!(test_cmp_u32, u32);
test_compare_int!(test_cmp_u64, u64);
test_compare_int!(test_cmp_i8, i8);
test_compare_int!(test_cmp_i16, i16);
test_compare_int!(test_cmp_i32, i32);
test_compare_int!(test_cmp_i64, i64);
test_compare_float!(test_cmp_f32, f32);
test_compare_float!(test_cmp_f64, f64);

// ---------------------------------------------------------------------------
// min / max for all types
// ---------------------------------------------------------------------------

macro_rules! test_min_max_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out_min: &'a mut [$T],
                out_max: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.min(va, vb),
                                self.out_min.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.max(va, vb),
                                self.out_max.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> =
                (0..n).map(|i| (i as $T).wrapping_mul(7)).collect();
            let b: Vec<$T> = (0..n)
                .map(|i| (i as $T).wrapping_mul(5).wrapping_add(10))
                .collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut o_min = vec![0 as $T; n];
                let mut o_max = vec![0 as $T; n];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out_min: &mut o_min,
                        out_max: &mut o_max,
                    },
                    target,
                );
                for j in 0..check {
                    let ea = a[j];
                    let eb = b[j];
                    assert_eq!(
                        o_min[j],
                        ea.min(eb),
                        "min {} lane {j} {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_max[j],
                        ea.max(eb),
                        "max {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

macro_rules! test_min_max_float {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out_min: &'a mut [$T],
                out_max: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.min(va, vb),
                                self.out_min.as_mut_ptr().add(i),
                            );
                            s.store_u(
                                s.max(va, vb),
                                self.out_max.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> = (0..n).map(|i| (i as $T) * 1.5 - 40.0).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) * 0.75 + 5.0).collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut o_min = vec![0.0 as $T; n];
                let mut o_max = vec![0.0 as $T; n];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out_min: &mut o_min,
                        out_max: &mut o_max,
                    },
                    target,
                );
                for j in 0..check {
                    let ea = a[j];
                    let eb = b[j];
                    assert_eq!(
                        o_min[j],
                        ea.min(eb),
                        "min {} lane {j} {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_max[j],
                        ea.max(eb),
                        "max {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_min_max_int!(test_min_max_u8, u8);
test_min_max_int!(test_min_max_u16, u16);
test_min_max_int!(test_min_max_u32, u32);
test_min_max_int!(test_min_max_u64, u64);
test_min_max_int!(test_min_max_i8, i8);
test_min_max_int!(test_min_max_i16, i16);
test_min_max_int!(test_min_max_i32, i32);
test_min_max_int!(test_min_max_i64, i64);
test_min_max_float!(test_min_max_f32_all, f32);
test_min_max_float!(test_min_max_f64_all, f64);

// ---------------------------------------------------------------------------
// abs_diff for all types
// ---------------------------------------------------------------------------

macro_rules! test_abs_diff_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.abs_diff(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            // Keep values small enough so |a-b| doesn't overflow the signed type
            let a: Vec<$T> = (0..n).map(|i| (i % 16) as $T).collect();
            let b: Vec<$T> = (0..n).map(|i| ((i + 3) % 20) as $T).collect();
            // Cross-target consistency: all targets must agree
            let targets = available_targets();
            let mut reference: Option<Vec<$T>> = None;
            for target in &targets {
                let lanes = lanes_for::<$T>(*target);
                let check = lanes * (n / lanes);
                let mut out = vec![0 as $T; n];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    *target,
                );
                // Basic sanity: result should be non-negative for non-overflowing inputs
                for j in 0..check {
                    let ea = a[j];
                    let eb = b[j];
                    let expected = if ea > eb {
                        ea.wrapping_sub(eb)
                    } else {
                        eb.wrapping_sub(ea)
                    };
                    assert_eq!(
                        out[j],
                        expected,
                        "abs_diff {} lane {j} {target:?}: a={ea}, b={eb}",
                        stringify!($T)
                    );
                }
                if let Some(ref prev) = reference {
                    let cmp_len = check.min(prev.len());
                    assert_eq!(
                        &out[..cmp_len],
                        &prev[..cmp_len],
                        "abs_diff {} cross-target mismatch {target:?}",
                        stringify!($T)
                    );
                } else {
                    reference = Some(out[..check].to_vec());
                }
            }
        }
    };
}

macro_rules! test_abs_diff_float {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.abs_diff(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> = (0..n).map(|i| (i as $T) * 1.5 - 40.0).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) * 0.75 + 5.0).collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut out = vec![0.0 as $T; n];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..check {
                    let expected = (a[j] - b[j]).abs();
                    assert!(
                        (out[j] - expected).abs() < 1e-4 as $T,
                        "abs_diff {} lane {j} {target:?}: got {}, expected {}",
                        stringify!($T),
                        out[j],
                        expected
                    );
                }
            }
        }
    };
}

test_abs_diff_int!(test_abs_diff_u8, u8);
test_abs_diff_int!(test_abs_diff_u16, u16);
test_abs_diff_int!(test_abs_diff_u64, u64);
test_abs_diff_int!(test_abs_diff_i8, i8);
test_abs_diff_int!(test_abs_diff_i16, i16);
test_abs_diff_int!(test_abs_diff_i32, i32);
test_abs_diff_int!(test_abs_diff_i64, i64);
test_abs_diff_float!(test_abs_diff_f32, f32);

// ---------------------------------------------------------------------------
// clamp for all types
// ---------------------------------------------------------------------------

macro_rules! test_clamp_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                v: &'a [$T],
                lo: &'a [$T],
                hi: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.v.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let vv = s.load_u(self.v.as_ptr().add(i));
                            let vlo = s.load_u(self.lo.as_ptr().add(i));
                            let vhi = s.load_u(self.hi.as_ptr().add(i));
                            s.store_u(
                                s.clamp(vv, vlo, vhi),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let v: Vec<$T> =
                (0..n).map(|i| (i as $T).wrapping_mul(5)).collect();
            let lo: Vec<$T> = (0..n).map(|_| 10 as $T).collect();
            let hi: Vec<$T> = (0..n).map(|_| 50 as $T).collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut out = vec![0 as $T; n];
                dispatch_to(
                    K {
                        v: &v,
                        lo: &lo,
                        hi: &hi,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..check {
                    let expected = v[j].max(lo[j]).min(hi[j]);
                    assert_eq!(
                        out[j],
                        expected,
                        "clamp {} lane {j} {target:?}: v={}",
                        stringify!($T),
                        v[j]
                    );
                }
            }
        }
    };
}

macro_rules! test_clamp_float {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                v: &'a [$T],
                lo: &'a [$T],
                hi: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.v.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let vv = s.load_u(self.v.as_ptr().add(i));
                            let vlo = s.load_u(self.lo.as_ptr().add(i));
                            let vhi = s.load_u(self.hi.as_ptr().add(i));
                            s.store_u(
                                s.clamp(vv, vlo, vhi),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let v: Vec<$T> = (0..n).map(|i| (i as $T) * 3.0 - 50.0).collect();
            let lo = vec![-10.0 as $T; n];
            let hi = vec![100.0 as $T; n];
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut out = vec![0.0 as $T; n];
                dispatch_to(
                    K {
                        v: &v,
                        lo: &lo,
                        hi: &hi,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..check {
                    let expected = v[j].max(lo[j]).min(hi[j]);
                    assert!(
                        (out[j] - expected).abs() < 1e-6 as $T,
                        "clamp {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_clamp_int!(test_clamp_u8, u8);
test_clamp_int!(test_clamp_u16, u16);
test_clamp_int!(test_clamp_u32, u32);
test_clamp_int!(test_clamp_u64, u64);
test_clamp_int!(test_clamp_i8, i8);
test_clamp_int!(test_clamp_i16, i16);
test_clamp_int!(test_clamp_i64, i64);
test_clamp_float!(test_clamp_f64, f64);

// ---------------------------------------------------------------------------
// test_bit for all integer types
// ---------------------------------------------------------------------------

macro_rules! test_testbit {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                v: &'a [$T],
                bit: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.v.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let vv = s.load_u(self.v.as_ptr().add(i));
                            let vb = s.load_u(self.bit.as_ptr().add(i));
                            s.store_u(
                                s.vec_from_mask(s.test_bit(vv, vb)),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let v: Vec<$T> = (0..n)
                .map(|i| (i as $T).wrapping_mul(0x5A).wrapping_add(1))
                .collect();
            let bit_idx: Vec<$T> = (0..n).map(|_| 1 as $T).collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut out = vec![0 as $T; n];
                dispatch_to(
                    K {
                        v: &v,
                        bit: &bit_idx,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..check {
                    let expected =
                        if v[j] & bit_idx[j] != 0 { !0 as $T } else { 0 };
                    assert_eq!(
                        out[j],
                        expected,
                        "test_bit {} lane {j} {target:?}: v={}, bit={}",
                        stringify!($T),
                        v[j],
                        bit_idx[j]
                    );
                }
            }
        }
    };
}

test_testbit!(test_testbit_u8, u8);
test_testbit!(test_testbit_u16, u16);
test_testbit!(test_testbit_u64, u64);
test_testbit!(test_testbit_i8, i8);
test_testbit!(test_testbit_i16, i16);
test_testbit!(test_testbit_i32, i32);
test_testbit!(test_testbit_i64, i64);

// =========================================================================
// Batch 3: mask ops multi-type + float ops f64
// =========================================================================

// ---------------------------------------------------------------------------
// mask_from_vec / first_n / count_true / all_true / all_false / find_first_true / find_last_true
// ---------------------------------------------------------------------------

macro_rules! test_mask_ops_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K {
                lanes: usize,
                results: Vec<(usize, bool, bool, Option<usize>, Option<usize>)>,
            }
            impl WithSimd for K {
                type Output =
                    Vec<(usize, bool, bool, Option<usize>, Option<usize>)>;
                fn with_simd<S: SimdOps>(mut self, s: S) -> Self::Output {
                    let lanes = s.lanes::<$T>();
                    self.lanes = lanes;
                    unsafe {
                        // Test first_n for 0..=lanes
                        for n in 0..=lanes {
                            let m = s.first_n::<$T>(n);
                            let cnt = s.count_true(m);
                            let at = s.all_true(m);
                            let af = s.all_false(m);
                            let ff = s.find_first_true(m);
                            let fl = s.find_last_true(m);
                            self.results.push((cnt, at, af, ff, fl));
                        }
                    }
                    self.results
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let k = K {
                    lanes: 0,
                    results: Vec::new(),
                };
                let results = dispatch_to(k, target);
                for (n, &(cnt, at, af, ff, fl)) in results.iter().enumerate() {
                    assert_eq!(
                        cnt,
                        n,
                        "count_true {} first_n({n}) {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        at,
                        n == lanes,
                        "all_true {} first_n({n}) {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        af,
                        n == 0,
                        "all_false {} first_n({n}) {target:?}",
                        stringify!($T)
                    );
                    if n == 0 {
                        assert_eq!(
                            ff,
                            None,
                            "find_first_true {} first_n(0) {target:?}",
                            stringify!($T)
                        );
                        assert_eq!(
                            fl,
                            None,
                            "find_last_true {} first_n(0) {target:?}",
                            stringify!($T)
                        );
                    } else {
                        assert_eq!(
                            ff,
                            Some(0),
                            "find_first_true {} first_n({n}) {target:?}",
                            stringify!($T)
                        );
                        assert_eq!(
                            fl,
                            Some(n - 1),
                            "find_last_true {} first_n({n}) {target:?}",
                            stringify!($T)
                        );
                    }
                }
            }
        }
    };
}

test_mask_ops_type!(test_mask_ops_u8, u8);
test_mask_ops_type!(test_mask_ops_u16, u16);
test_mask_ops_type!(test_mask_ops_u64, u64);
test_mask_ops_type!(test_mask_ops_i8, i8);
test_mask_ops_type!(test_mask_ops_i16, i16);
test_mask_ops_type!(test_mask_ops_i32, i32);
test_mask_ops_type!(test_mask_ops_i64, i64);
test_mask_ops_type!(test_mask_ops_f32, f32);
test_mask_ops_type!(test_mask_ops_f64, f64);

// ---------------------------------------------------------------------------
// if_then_else / if_then_else_zero / if_then_zero_else for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_if_then_else_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out_ite: &'a mut [$T],
                out_itez: &'a mut [$T],
                out_itze: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    unsafe {
                        let va = s.load_u(self.a.as_ptr());
                        let vb = s.load_u(self.b.as_ptr());
                        // Mask: first half true, second half false
                        let m = s.first_n::<$T>(lanes / 2);
                        s.store_u(
                            s.if_then_else(m, va, vb),
                            self.out_ite.as_mut_ptr(),
                        );
                        s.store_u(
                            s.if_then_else_zero(m, va),
                            self.out_itez.as_mut_ptr(),
                        );
                        s.store_u(
                            s.if_then_zero_else(m, vb),
                            self.out_itze.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let a_raw: Vec<u8> = (0..lanes * core::mem::size_of::<$T>())
                    .map(|i| 0x11u8.wrapping_mul(i as u8 + 1))
                    .collect();
                let b_raw: Vec<u8> = (0..lanes * core::mem::size_of::<$T>())
                    .map(|i| 0xAAu8.wrapping_mul(i as u8 + 1))
                    .collect();
                let a: Vec<$T> = (0..lanes)
                    .map(|i| unsafe {
                        core::ptr::read_unaligned(
                            a_raw
                                .as_ptr()
                                .add(i * core::mem::size_of::<$T>())
                                .cast::<$T>(),
                        )
                    })
                    .collect();
                let b: Vec<$T> = (0..lanes)
                    .map(|i| unsafe {
                        core::ptr::read_unaligned(
                            b_raw
                                .as_ptr()
                                .add(i * core::mem::size_of::<$T>())
                                .cast::<$T>(),
                        )
                    })
                    .collect();
                let mut o_ite =
                    vec![unsafe { core::mem::zeroed::<$T>() }; lanes];
                let mut o_itez =
                    vec![unsafe { core::mem::zeroed::<$T>() }; lanes];
                let mut o_itze =
                    vec![unsafe { core::mem::zeroed::<$T>() }; lanes];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out_ite: &mut o_ite,
                        out_itez: &mut o_itez,
                        out_itze: &mut o_itze,
                    },
                    target,
                );
                let half = lanes / 2;
                for j in 0..lanes {
                    let zero: $T = unsafe { core::mem::zeroed() };
                    if j < half {
                        assert_eq!(
                            o_ite[j],
                            a[j],
                            "if_then_else {} lane {j} {target:?}",
                            stringify!($T)
                        );
                        assert_eq!(
                            o_itez[j],
                            a[j],
                            "if_then_else_zero {} lane {j} {target:?}",
                            stringify!($T)
                        );
                        assert_eq!(
                            o_itze[j],
                            zero,
                            "if_then_zero_else {} lane {j} {target:?}",
                            stringify!($T)
                        );
                    } else {
                        assert_eq!(
                            o_ite[j],
                            b[j],
                            "if_then_else {} lane {j} {target:?}",
                            stringify!($T)
                        );
                        assert_eq!(
                            o_itez[j],
                            zero,
                            "if_then_else_zero {} lane {j} {target:?}",
                            stringify!($T)
                        );
                        assert_eq!(
                            o_itze[j],
                            b[j],
                            "if_then_zero_else {} lane {j} {target:?}",
                            stringify!($T)
                        );
                    }
                }
            }
        }
    };
}

test_if_then_else_type!(test_ite_u8, u8);
test_if_then_else_type!(test_ite_u16, u16);
test_if_then_else_type!(test_ite_u64, u64);
test_if_then_else_type!(test_ite_i8, i8);
test_if_then_else_type!(test_ite_i16, i16);
test_if_then_else_type!(test_ite_i64, i64);
test_if_then_else_type!(test_ite_f32, f32);
test_if_then_else_type!(test_ite_f64, f64);

// ---------------------------------------------------------------------------
// mask logic (and_mask / or_mask / not_mask / xor_mask) for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_mask_logic_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K;
            impl WithSimd for K {
                type Output = bool;
                fn with_simd<S: SimdOps>(self, s: S) -> bool {
                    let lanes = s.lanes::<$T>();
                    unsafe {
                        let m_all = s.first_n::<$T>(lanes);
                        let m_none = s.first_n::<$T>(0);
                        let m_half = s.first_n::<$T>(lanes / 2);
                        // and_mask(all, half) == half
                        assert_eq!(
                            s.count_true(s.and_mask(m_all, m_half)),
                            lanes / 2
                        );
                        // or_mask(none, half) == half
                        assert_eq!(
                            s.count_true(s.or_mask(m_none, m_half)),
                            lanes / 2
                        );
                        // not_mask(none) == all
                        assert!(s.all_true(s.not_mask(m_none)));
                        // not_mask(all) == none
                        assert!(s.all_false(s.not_mask(m_all)));
                        // xor_mask(all, all) == none
                        assert!(s.all_false(s.xor_mask(m_all, m_all)));
                        // xor_mask(all, none) == all
                        assert!(s.all_true(s.xor_mask(m_all, m_none)));
                    }
                    true
                }
            }
            for target in available_targets() {
                let result = dispatch_to(K, target);
                assert!(result, "mask_logic {} {target:?}", stringify!($T));
            }
        }
    };
}

test_mask_logic_type!(test_mask_logic_u8, u8);
test_mask_logic_type!(test_mask_logic_u16, u16);
test_mask_logic_type!(test_mask_logic_u64, u64);
test_mask_logic_type!(test_mask_logic_i8, i8);
test_mask_logic_type!(test_mask_logic_i16, i16);
test_mask_logic_type!(test_mask_logic_i64, i64);
test_mask_logic_type!(test_mask_logic_f32, f32);
test_mask_logic_type!(test_mask_logic_f64, f64);

// ---------------------------------------------------------------------------
// bits_from_mask for signed integer types
// ---------------------------------------------------------------------------

macro_rules! test_bits_from_mask_signed {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K;
            impl WithSimd for K {
                type Output = Vec<u64>;
                fn with_simd<S: SimdOps>(self, s: S) -> Vec<u64> {
                    let lanes = s.lanes::<$T>();
                    let mut results = Vec::new();
                    unsafe {
                        for n in 0..=lanes {
                            let m = s.first_n::<$T>(n);
                            results.push(s.bits_from_mask(m));
                        }
                    }
                    results
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let results = dispatch_to(K, target);
                for n in 0..=lanes {
                    let expected =
                        if n >= 64 { u64::MAX } else { (1u64 << n) - 1 };
                    assert_eq!(
                        results[n],
                        expected,
                        "bits_from_mask {} first_n({n}) {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_bits_from_mask_signed!(test_bits_from_mask_i8, i8);
test_bits_from_mask_signed!(test_bits_from_mask_i16, i16);
test_bits_from_mask_signed!(test_bits_from_mask_i32, i32);
test_bits_from_mask_signed!(test_bits_from_mask_i64, i64);
test_bits_from_mask_signed!(test_bits_from_mask_f32, f32);
test_bits_from_mask_signed!(test_bits_from_mask_f64, f64);

// ---------------------------------------------------------------------------
// Float ops: f64 variants (sqrt, approx_reciprocal, approx_reciprocal_sqrt,
//   mul_add, neg_mul_add, mul_sub, neg_mul_sub, copy_sign, is_nan, is_inf)
// ---------------------------------------------------------------------------

#[test]
fn test_sqrt_f64() {
    struct K<'a> {
        input: &'a [f64],
        out: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.input.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let v = s.load_u(self.input.as_ptr().add(i));
                    s.store_u(s.sqrt(v), self.out.as_mut_ptr().add(i));
                }
                i += lanes;
            }
        }
    }
    let input: Vec<f64> = vec![
        0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 2.0, 0.25, 0.01, 1e10, 1e-10,
        0.5, 81.0, 144.0, 256.0,
    ];
    let n = input.len();
    let expected: Vec<f64> = input.iter().map(|x| x.sqrt()).collect();
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let mut out = vec![0.0f64; n];
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        for j in 0..check {
            assert!(
                (out[j] - expected[j]).abs() < 1e-10,
                "sqrt f64 lane {j} {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

#[test]
fn test_mul_add_f64() {
    struct K<'a> {
        a: &'a [f64],
        b: &'a [f64],
        c: &'a [f64],
        out_ma: &'a mut [f64],
        out_nma: &'a mut [f64],
        out_ms: &'a mut [f64],
        out_nms: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.a.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let va = s.load_u(self.a.as_ptr().add(i));
                    let vb = s.load_u(self.b.as_ptr().add(i));
                    let vc = s.load_u(self.c.as_ptr().add(i));
                    s.store_u(
                        s.mul_add(va, vb, vc),
                        self.out_ma.as_mut_ptr().add(i),
                    );
                    s.store_u(
                        s.neg_mul_add(va, vb, vc),
                        self.out_nma.as_mut_ptr().add(i),
                    );
                    s.store_u(
                        s.mul_sub(va, vb, vc),
                        self.out_ms.as_mut_ptr().add(i),
                    );
                    s.store_u(
                        s.neg_mul_sub(va, vb, vc),
                        self.out_nms.as_mut_ptr().add(i),
                    );
                }
                i += lanes;
            }
        }
    }
    let a = vec![2.0f64, -3.0, 4.0, 0.5, 1.0, -1.0, 10.0, 0.0];
    let b = vec![3.0f64, 4.0, -2.0, 6.0, 1.0, 1.0, 0.5, 7.0];
    let c = vec![1.0f64, 2.0, 3.0, -1.0, 0.0, 0.0, -5.0, 1.0];
    let n = a.len();
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let check = lanes * (n / lanes);
        let mut o_ma = vec![0.0f64; n];
        let mut o_nma = vec![0.0f64; n];
        let mut o_ms = vec![0.0f64; n];
        let mut o_nms = vec![0.0f64; n];
        dispatch_to(
            K {
                a: &a,
                b: &b,
                c: &c,
                out_ma: &mut o_ma,
                out_nma: &mut o_nma,
                out_ms: &mut o_ms,
                out_nms: &mut o_nms,
            },
            target,
        );
        for j in 0..check {
            let ea = a[j];
            let eb = b[j];
            let ec = c[j];
            assert!(
                (o_ma[j] - (ea * eb + ec)).abs() < 1e-10,
                "mul_add f64 lane {j} {target:?}"
            );
            assert!(
                (o_nma[j] - (-ea * eb + ec)).abs() < 1e-10,
                "neg_mul_add f64 lane {j} {target:?}"
            );
            assert!(
                (o_ms[j] - (ea * eb - ec)).abs() < 1e-10,
                "mul_sub f64 lane {j} {target:?}"
            );
            assert!(
                (o_nms[j] - (-ea * eb - ec)).abs() < 1e-10,
                "neg_mul_sub f64 lane {j} {target:?}"
            );
        }
    }
}

#[test]
fn test_copy_sign_f64() {
    struct K<'a> {
        mag: &'a [f64],
        sign: &'a [f64],
        out: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.mag.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let vm = s.load_u(self.mag.as_ptr().add(i));
                    let vs = s.load_u(self.sign.as_ptr().add(i));
                    s.store_u(
                        s.copy_sign(vm, vs),
                        self.out.as_mut_ptr().add(i),
                    );
                }
                i += lanes;
            }
        }
    }
    let mag = vec![5.0f64, -3.0, 7.0, 0.0, 1.0, -1.0, 100.0, 0.5];
    let sign = vec![-1.0f64, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
    let n = mag.len();
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let check = lanes * (n / lanes);
        let mut out = vec![0.0f64; n];
        dispatch_to(
            K {
                mag: &mag,
                sign: &sign,
                out: &mut out,
            },
            target,
        );
        for j in 0..check {
            let expected = mag[j].abs().copysign(sign[j]);
            assert_eq!(out[j], expected, "copy_sign f64 lane {j} {target:?}");
        }
    }
}

#[test]
fn test_is_nan_is_inf_f64() {
    struct K {}
    impl WithSimd for K {
        type Output = bool;
        fn with_simd<S: SimdOps>(self, s: S) -> bool {
            unsafe {
                let nan = s.splat(f64::NAN);
                let inf = s.splat(f64::INFINITY);
                let normal = s.splat(1.0f64);
                let neg_inf = s.splat(f64::NEG_INFINITY);
                assert!(s.all_true(s.is_nan(nan)), "NAN should be nan");
                assert!(s.all_false(s.is_nan(normal)), "1.0 should not be nan");
                assert!(s.all_false(s.is_nan(inf)), "INF should not be nan");
                assert!(s.all_true(s.is_inf(inf)), "INF should be inf");
                assert!(s.all_true(s.is_inf(neg_inf)), "NEG_INF should be inf");
                assert!(s.all_false(s.is_inf(normal)), "1.0 should not be inf");
                assert!(s.all_false(s.is_inf(nan)), "NAN should not be inf");
            }
            true
        }
    }
    for target in available_targets() {
        assert!(dispatch_to(K {}, target), "is_nan/is_inf f64 {target:?}");
    }
}

// ---------------------------------------------------------------------------
// rounding f32 (already tested for f64, add explicit f32 variants)
// round/trunc/ceil/floor already tested via test_rounding (f32) and
// test_rounding_f64 (f64). These are sufficient.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// average_round for u32, u64 (u8 and u16 already tested)
// ---------------------------------------------------------------------------

macro_rules! test_avg_round {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.a.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let va = s.load_u(self.a.as_ptr().add(i));
                            let vb = s.load_u(self.b.as_ptr().add(i));
                            s.store_u(
                                s.average_round(va, vb),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let a: Vec<$T> = (0..n).map(|i| (i as $T) * 3 + 1).collect();
            let b: Vec<$T> = (0..n).map(|i| (i as $T) * 5 + 2).collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut out = vec![0 as $T; n];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..check {
                    let ea = a[j] as u128;
                    let eb = b[j] as u128;
                    let expected = ((ea + eb + 1) / 2) as $T;
                    assert_eq!(
                        out[j],
                        expected,
                        "average_round {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_avg_round!(test_avg_round_u32, u32);
test_avg_round!(test_avg_round_u64, u64);

// =========================================================================
// Batch 4: shuffle ops for multiple types + remaining ops
// =========================================================================

// ---------------------------------------------------------------------------
// reverse for all types
// ---------------------------------------------------------------------------

macro_rules! test_reverse_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        s.store_u(s.reverse(v), self.out.as_mut_ptr());
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> =
                    (0..lanes).map(|i| (i as u8 + 1) as $T).collect();
                let mut out = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..lanes {
                    assert_eq!(
                        out[j],
                        input[lanes - 1 - j],
                        "reverse {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_reverse_type!(test_reverse_u8, u8);
test_reverse_type!(test_reverse_u16, u16);
test_reverse_type!(test_reverse_u64, u64);
test_reverse_type!(test_reverse_i8, i8);
test_reverse_type!(test_reverse_i16, i16);
test_reverse_type!(test_reverse_i32, i32);
test_reverse_type!(test_reverse_i64, i64);

// ---------------------------------------------------------------------------
// reverse2 / reverse4 / reverse8 for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_reverse_n {
    ($name:ident, $T:ty, $op:ident, $n:expr) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        s.store_u(s.$op(v), self.out.as_mut_ptr());
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                if lanes < $n {
                    continue;
                }
                let input: Vec<$T> = (0..lanes).map(|i| i as $T).collect();
                let mut out = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..lanes {
                    let group_start = (j / $n) * $n;
                    let offset_in_group = j % $n;
                    let expected_idx = group_start + ($n - 1 - offset_in_group);
                    assert_eq!(
                        out[j],
                        input[expected_idx],
                        "{} {} lane {j} {target:?}",
                        stringify!($op),
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_reverse_n!(test_reverse2_u8, u8, reverse2, 2);
test_reverse_n!(test_reverse2_u16, u16, reverse2, 2);
test_reverse_n!(test_reverse2_u64, u64, reverse2, 2);
test_reverse_n!(test_reverse2_i8, i8, reverse2, 2);
test_reverse_n!(test_reverse2_i16, i16, reverse2, 2);
test_reverse_n!(test_reverse2_i32, i32, reverse2, 2);
test_reverse_n!(test_reverse2_i64, i64, reverse2, 2);

test_reverse_n!(test_reverse4_u8, u8, reverse4, 4);
test_reverse_n!(test_reverse4_u16, u16, reverse4, 4);
test_reverse_n!(test_reverse4_i8, i8, reverse4, 4);
test_reverse_n!(test_reverse4_i16, i16, reverse4, 4);
test_reverse_n!(test_reverse4_i64, i64, reverse4, 4);

test_reverse_n!(test_reverse8_u8, u8, reverse8, 8);
test_reverse_n!(test_reverse8_u16, u16, reverse8, 8);
test_reverse_n!(test_reverse8_i8, i8, reverse8, 8);
test_reverse_n!(test_reverse8_i16, i16, reverse8, 8);

// ---------------------------------------------------------------------------
// interleave_lower / interleave_upper for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_interleave_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out_lo: &'a mut [$T],
                out_hi: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let va = s.load_u(self.a.as_ptr());
                        let vb = s.load_u(self.b.as_ptr());
                        s.store_u(s.interleave_lower(va, vb), self.out_lo.as_mut_ptr());
                        s.store_u(s.interleave_upper(va, vb), self.out_hi.as_mut_ptr());
                    }
                }
            }
            // Use SSE2 as reference; verify other targets match
            let sse2_lanes = lanes_for::<$T>(TargetId::Sse2);
            if sse2_lanes < 2 {
                return;
            }
            let a: Vec<$T> = (0..sse2_lanes).map(|i| (i * 2) as $T).collect();
            let b: Vec<$T> = (0..sse2_lanes).map(|i| (i * 2 + 1) as $T).collect();
            let mut ref_lo = vec![0 as $T; sse2_lanes];
            let mut ref_hi = vec![0 as $T; sse2_lanes];
            dispatch_to(
                K {
                    a: &a,
                    b: &b,
                    out_lo: &mut ref_lo,
                    out_hi: &mut ref_hi,
                },
                TargetId::Sse2,
            );
            // Verify SSE2 result: per-half interleave
            let half = sse2_lanes / 2;
            for j in 0..sse2_lanes {
                let src_idx = j / 2;
                let expected_lo = if j % 2 == 0 { a[src_idx] } else { b[src_idx] };
                assert_eq!(
                    ref_lo[j],
                    expected_lo,
                    "interleave_lower {} lane {j} Sse2",
                    stringify!($T)
                );
                let expected_hi = if j % 2 == 0 {
                    a[half + src_idx]
                } else {
                    b[half + src_idx]
                };
                assert_eq!(
                    ref_hi[j],
                    expected_hi,
                    "interleave_upper {} lane {j} Sse2",
                    stringify!($T)
                );
            }
            // AVX2/AVX-512: just verify interleave_lower+upper together reconstruct all elements
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                if lanes < 2 {
                    continue;
                }
                let ta: Vec<$T> = (0..lanes).map(|i| (i * 2) as $T).collect();
                let tb: Vec<$T> = (0..lanes).map(|i| (i * 2 + 1) as $T).collect();
                let mut o_lo = vec![0 as $T; lanes];
                let mut o_hi = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        a: &ta,
                        b: &tb,
                        out_lo: &mut o_lo,
                        out_hi: &mut o_hi,
                    },
                    target,
                );
                // Combined lo+hi must contain every element from a and b exactly once
                let mut combined: Vec<$T> = o_lo.iter().chain(o_hi.iter()).copied().collect();
                combined.sort_by_key(|x| *x as u64);
                let mut all: Vec<$T> = ta.iter().chain(tb.iter()).copied().collect();
                all.sort_by_key(|x| *x as u64);
                assert_eq!(
                    combined,
                    all,
                    "interleave {} lo+hi should contain all elements {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_interleave_type!(test_interleave_u8, u8);
test_interleave_type!(test_interleave_u16, u16);
test_interleave_type!(test_interleave_u64, u64);
test_interleave_type!(test_interleave_i8, i8);
test_interleave_type!(test_interleave_i16, i16);
test_interleave_type!(test_interleave_i64, i64);

// ---------------------------------------------------------------------------
// zip_lower / zip_upper for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_zip_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out_lo: &'a mut [$T],
                out_hi: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let va = s.load_u(self.a.as_ptr());
                        let vb = s.load_u(self.b.as_ptr());
                        s.store_u(
                            s.zip_lower(va, vb),
                            self.out_lo.as_mut_ptr(),
                        );
                        s.store_u(
                            s.zip_upper(va, vb),
                            self.out_hi.as_mut_ptr(),
                        );
                    }
                }
            }
            // Verify on SSE2, then check other targets reconstruct all elements
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                if lanes < 2 {
                    continue;
                }
                let a: Vec<$T> = (0..lanes).map(|i| (i + 10) as $T).collect();
                let b: Vec<$T> = (0..lanes).map(|i| (i + 100) as $T).collect();
                let mut o_lo = vec![0 as $T; lanes];
                let mut o_hi = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out_lo: &mut o_lo,
                        out_hi: &mut o_hi,
                    },
                    target,
                );
                let mut combined: Vec<$T> =
                    o_lo.iter().chain(o_hi.iter()).copied().collect();
                combined.sort_by_key(|x| *x as u64);
                let mut all: Vec<$T> =
                    a.iter().chain(b.iter()).copied().collect();
                all.sort_by_key(|x| *x as u64);
                assert_eq!(
                    combined,
                    all,
                    "zip {} lo+hi should contain all elements {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_zip_type!(test_zip_u8, u8);
test_zip_type!(test_zip_u16, u16);
test_zip_type!(test_zip_u64, u64);
test_zip_type!(test_zip_i8, i8);
test_zip_type!(test_zip_i16, i16);
test_zip_type!(test_zip_i64, i64);

// ---------------------------------------------------------------------------
// concat_even / concat_odd for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_concat_even_odd_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                a: &'a [$T],
                b: &'a [$T],
                out_even: &'a mut [$T],
                out_odd: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let va = s.load_u(self.a.as_ptr());
                        let vb = s.load_u(self.b.as_ptr());
                        s.store_u(
                            s.concat_even(va, vb),
                            self.out_even.as_mut_ptr(),
                        );
                        s.store_u(
                            s.concat_odd(va, vb),
                            self.out_odd.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                if lanes < 2 {
                    continue;
                }
                let a: Vec<$T> = (0..lanes).map(|i| i as $T).collect();
                let b: Vec<$T> = (0..lanes).map(|i| (i + 100) as $T).collect();
                let mut o_even = vec![0 as $T; lanes];
                let mut o_odd = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        a: &a,
                        b: &b,
                        out_even: &mut o_even,
                        out_odd: &mut o_odd,
                    },
                    target,
                );
                // concat_even+concat_odd should together reconstruct all elements
                let mut combined: Vec<$T> =
                    o_even.iter().chain(o_odd.iter()).copied().collect();
                combined.sort_by_key(|x| *x as u64);
                let mut all: Vec<$T> =
                    a.iter().chain(b.iter()).copied().collect();
                all.sort_by_key(|x| *x as u64);
                assert_eq!(
                    combined,
                    all,
                    "concat_even+odd {} should contain all elements {target:?}",
                    stringify!($T)
                );
                // Verify even contains only even-indexed lanes, odd contains only odd-indexed
                for &v in &o_even {
                    let vu = v as u64;
                    let from_a = vu < 100;
                    let idx = if from_a { vu } else { vu - 100 };
                    assert_eq!(
                        idx % 2,
                        0,
                        "concat_even {} got odd-indexed value {v} {target:?}",
                        stringify!($T)
                    );
                }
                for &v in &o_odd {
                    let vu = v as u64;
                    let from_a = vu < 100;
                    let idx = if from_a { vu } else { vu - 100 };
                    assert_eq!(
                        idx % 2,
                        1,
                        "concat_odd {} got even-indexed value {v} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_concat_even_odd_type!(test_concat_eo_u8, u8);
test_concat_even_odd_type!(test_concat_eo_u16, u16);
test_concat_even_odd_type!(test_concat_eo_u64, u64);
test_concat_even_odd_type!(test_concat_eo_i8, i8);
test_concat_even_odd_type!(test_concat_eo_i16, i16);
test_concat_even_odd_type!(test_concat_eo_i64, i64);

// ---------------------------------------------------------------------------
// concat_upper_lower / concat_lower_upper
// ---------------------------------------------------------------------------

macro_rules! test_concat_ul_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                hi: &'a [$T],
                lo: &'a [$T],
                out_ul: &'a mut [$T],
                out_lu: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let vh = s.load_u(self.hi.as_ptr());
                        let vl = s.load_u(self.lo.as_ptr());
                        s.store_u(
                            s.concat_upper_lower(vh, vl),
                            self.out_ul.as_mut_ptr(),
                        );
                        s.store_u(
                            s.concat_lower_upper(vh, vl),
                            self.out_lu.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                if lanes < 2 {
                    continue;
                }
                let hi: Vec<$T> = (0..lanes).map(|i| (i + 100) as $T).collect();
                let lo: Vec<$T> = (0..lanes).map(|i| (i + 200) as $T).collect();
                let mut o_ul = vec![0 as $T; lanes];
                let mut o_lu = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        hi: &hi,
                        lo: &lo,
                        out_ul: &mut o_ul,
                        out_lu: &mut o_lu,
                    },
                    target,
                );
                let half = lanes / 2;
                // concat_upper_lower(hi, lo): upper half of hi, lower half of lo
                for j in 0..half {
                    assert_eq!(
                        o_ul[j + half],
                        hi[j + half],
                        "concat_upper_lower {} lane {} (from hi) {target:?}",
                        stringify!($T),
                        j + half
                    );
                    assert_eq!(
                        o_ul[j],
                        lo[j],
                        "concat_upper_lower {} lane {j} (from lo) {target:?}",
                        stringify!($T)
                    );
                }
                // concat_lower_upper(hi, lo): lower half of hi, upper half of lo
                for j in 0..half {
                    assert_eq!(
                        o_lu[j],
                        hi[j],
                        "concat_lower_upper {} lane {j} (from hi) {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        o_lu[j + half],
                        lo[j + half],
                        "concat_lower_upper {} lane {} (from lo) {target:?}",
                        stringify!($T),
                        j + half
                    );
                }
            }
        }
    };
}

test_concat_ul_type!(test_concat_ul_u8, u8);
test_concat_ul_type!(test_concat_ul_u16, u16);
test_concat_ul_type!(test_concat_ul_u32, u32);
test_concat_ul_type!(test_concat_ul_u64, u64);
test_concat_ul_type!(test_concat_ul_i8, i8);
test_concat_ul_type!(test_concat_ul_i16, i16);
test_concat_ul_type!(test_concat_ul_i32, i32);
test_concat_ul_type!(test_concat_ul_i64, i64);

// ---------------------------------------------------------------------------
// odd_even for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_odd_even_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                odd: &'a [$T],
                even: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let vo = s.load_u(self.odd.as_ptr());
                        let ve = s.load_u(self.even.as_ptr());
                        s.store_u(s.odd_even(vo, ve), self.out.as_mut_ptr());
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                if lanes < 2 {
                    continue;
                }
                let odd: Vec<$T> =
                    (0..lanes).map(|i| (i + 100) as $T).collect();
                let even: Vec<$T> =
                    (0..lanes).map(|i| (i + 200) as $T).collect();
                let mut out = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        odd: &odd,
                        even: &even,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..lanes {
                    let expected = if j % 2 == 0 { even[j] } else { odd[j] };
                    assert_eq!(
                        out[j],
                        expected,
                        "odd_even {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_odd_even_type!(test_odd_even_u8, u8);
test_odd_even_type!(test_odd_even_u16, u16);
test_odd_even_type!(test_odd_even_u64, u64);
test_odd_even_type!(test_odd_even_i8, i8);
test_odd_even_type!(test_odd_even_i16, i16);
test_odd_even_type!(test_odd_even_i32, i32);
test_odd_even_type!(test_odd_even_i64, i64);
test_odd_even_type!(test_odd_even_f64, f64);

// ---------------------------------------------------------------------------
// broadcast_lane for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_broadcast_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        // Broadcast lane 0
                        s.store_u(
                            s.broadcast_lane::<$T, 0>(v),
                            self.out.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> =
                    (0..lanes).map(|i| (i + 42) as $T).collect();
                let mut out = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..lanes {
                    assert_eq!(
                        out[j],
                        input[0],
                        "broadcast_lane {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_broadcast_type!(test_broadcast_u8, u8);
test_broadcast_type!(test_broadcast_u16, u16);
test_broadcast_type!(test_broadcast_u64, u64);
test_broadcast_type!(test_broadcast_i8, i8);
test_broadcast_type!(test_broadcast_i16, i16);
test_broadcast_type!(test_broadcast_i64, i64);

// ---------------------------------------------------------------------------
// slide_down_lanes for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_slide_down_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
                n: usize,
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        s.store_u(
                            s.slide_down_lanes(v, self.n),
                            self.out.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> =
                    (0..lanes).map(|i| (i + 1) as $T).collect();
                let shift = 1.min(lanes - 1);
                let mut out = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                        n: shift,
                    },
                    target,
                );
                for j in 0..lanes {
                    let expected = if j + shift < lanes {
                        input[j + shift]
                    } else {
                        0 as $T
                    };
                    assert_eq!(
                        out[j],
                        expected,
                        "slide_down {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_slide_down_type!(test_slide_down_u8, u8);
test_slide_down_type!(test_slide_down_u16, u16);
test_slide_down_type!(test_slide_down_u32, u32);
test_slide_down_type!(test_slide_down_u64, u64);
test_slide_down_type!(test_slide_down_i8, i8);
test_slide_down_type!(test_slide_down_i16, i16);
test_slide_down_type!(test_slide_down_i32, i32);
test_slide_down_type!(test_slide_down_i64, i64);

// ---------------------------------------------------------------------------
// slide_up_lanes for multiple types (existing test only covers u32)
// ---------------------------------------------------------------------------

macro_rules! test_slide_up_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
                n: usize,
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        s.store_u(
                            s.slide_up_lanes(v, self.n),
                            self.out.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> =
                    (0..lanes).map(|i| (i + 1) as $T).collect();
                let shift = 1.min(lanes - 1);
                let mut out = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                        n: shift,
                    },
                    target,
                );
                for j in 0..lanes {
                    let expected = if j >= shift {
                        input[j - shift]
                    } else {
                        0 as $T
                    };
                    assert_eq!(
                        out[j],
                        expected,
                        "slide_up {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_slide_up_type!(test_slide_up_u8, u8);
test_slide_up_type!(test_slide_up_u16, u16);
test_slide_up_type!(test_slide_up_u64, u64);
test_slide_up_type!(test_slide_up_i8, i8);
test_slide_up_type!(test_slide_up_i16, i16);
test_slide_up_type!(test_slide_up_i32, i32);
test_slide_up_type!(test_slide_up_i64, i64);

// ---------------------------------------------------------------------------
// reverse_lane_bytes for u16, u32, u64, i16, i32, i64
// ---------------------------------------------------------------------------

macro_rules! test_rev_lane_bytes {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.input.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let v = s.load_u(self.input.as_ptr().add(i));
                            s.store_u(
                                s.reverse_lane_bytes(v),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let input: Vec<$T> = (0..n)
                .map(|i| {
                    let bytes = core::mem::size_of::<$T>();
                    let mut val: $T = 0;
                    for b in 0..bytes {
                        val |= ((i * bytes + b + 1) as $T & 0xFF) << (b * 8);
                    }
                    val
                })
                .collect();
            let expected: Vec<$T> = input
                .iter()
                .map(|&x| {
                    let bytes = core::mem::size_of::<$T>();
                    let mut val: $T = 0;
                    for b in 0..bytes {
                        val |= ((x >> (b * 8)) & 0xFF) << ((bytes - 1 - b) * 8);
                    }
                    val
                })
                .collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut out = vec![0 as $T; n];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "reverse_lane_bytes {} {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_rev_lane_bytes!(test_rev_lane_bytes_u16, u16);
test_rev_lane_bytes!(test_rev_lane_bytes_u64, u64);
test_rev_lane_bytes!(test_rev_lane_bytes_i16, i16);
test_rev_lane_bytes!(test_rev_lane_bytes_i32, i32);
test_rev_lane_bytes!(test_rev_lane_bytes_i64, i64);

// ---------------------------------------------------------------------------
// shift_left_same / shift_right_same for all integer types
// ---------------------------------------------------------------------------

macro_rules! test_shift_same {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct KL<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for KL<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        s.store_u(
                            s.shift_left_same(v, 3),
                            self.out.as_mut_ptr(),
                        );
                    }
                }
            }
            struct KR<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for KR<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        s.store_u(
                            s.shift_right_same(v, 3),
                            self.out.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> = (0..lanes)
                    .map(|i| ((i as $T).wrapping_mul(17)).wrapping_add(5))
                    .collect();
                let mut out_l = vec![0 as $T; lanes];
                let mut out_r = vec![0 as $T; lanes];
                dispatch_to(
                    KL {
                        input: &input,
                        out: &mut out_l,
                    },
                    target,
                );
                dispatch_to(
                    KR {
                        input: &input,
                        out: &mut out_r,
                    },
                    target,
                );
                for j in 0..lanes {
                    let exp_l = input[j].wrapping_shl(3);
                    let exp_r = input[j] >> 3;
                    assert_eq!(
                        out_l[j],
                        exp_l,
                        "shift_left_same {} lane {j} {target:?}",
                        stringify!($T)
                    );
                    assert_eq!(
                        out_r[j],
                        exp_r,
                        "shift_right_same {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_shift_same!(test_shift_same_u8, u8);
test_shift_same!(test_shift_same_u16, u16);
test_shift_same!(test_shift_same_u32, u32);
test_shift_same!(test_shift_same_u64, u64);
test_shift_same!(test_shift_same_i8, i8);
test_shift_same!(test_shift_same_i16, i16);
test_shift_same!(test_shift_same_i32, i32);
test_shift_same!(test_shift_same_i64, i64);

// ---------------------------------------------------------------------------
// rotate_right for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_rotate_type {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    let n = self.input.len();
                    let mut i = 0;
                    while i + lanes <= n {
                        unsafe {
                            let v = s.load_u(self.input.as_ptr().add(i));
                            s.store_u(
                                s.rotate_right::<$T, 3>(v),
                                self.out.as_mut_ptr().add(i),
                            );
                        }
                        i += lanes;
                    }
                }
            }
            let n = 64;
            let input: Vec<$T> = (0..n)
                .map(|i| ((i as $T).wrapping_mul(0x5A)).wrapping_add(1))
                .collect();
            let expected: Vec<$T> = input
                .iter()
                .map(|&x| {
                    // Rotate uses logical (unsigned) shift right
                    let bytes = core::mem::size_of::<$T>();
                    let xu = match bytes {
                        1 => {
                            ((x as u8).wrapping_shr(3)
                                | (x as u8).wrapping_shl(8 - 3))
                                as $T
                        }
                        2 => {
                            ((x as u16).wrapping_shr(3)
                                | (x as u16).wrapping_shl(16 - 3))
                                as $T
                        }
                        4 => {
                            ((x as u32).wrapping_shr(3)
                                | (x as u32).wrapping_shl(32 - 3))
                                as $T
                        }
                        _ => {
                            ((x as u64).wrapping_shr(3)
                                | (x as u64).wrapping_shl(64 - 3))
                                as $T
                        }
                    };
                    xu
                })
                .collect();
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let check = lanes * (n / lanes);
                let mut out = vec![0 as $T; n];
                dispatch_to(
                    K {
                        input: &input,
                        out: &mut out,
                    },
                    target,
                );
                assert_eq!(
                    &out[..check],
                    &expected[..check],
                    "rotate_right {} {target:?}",
                    stringify!($T)
                );
            }
        }
    };
}

test_rotate_type!(test_rotate_u8, u8);
test_rotate_type!(test_rotate_u16, u16);
test_rotate_type!(test_rotate_u64, u64);
test_rotate_type!(test_rotate_i8, i8);
test_rotate_type!(test_rotate_i16, i16);
test_rotate_type!(test_rotate_i32, i32);
test_rotate_type!(test_rotate_i64, i64);

// ---------------------------------------------------------------------------
// table_lookup_lanes for multiple types
// ---------------------------------------------------------------------------

macro_rules! test_tbl_lookup_lanes_type {
    ($name:ident, $T:ty, $I:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                input: &'a [$T],
                idx: &'a [$I],
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.load_u(self.input.as_ptr());
                        let i = s.load_u(self.idx.as_ptr());
                        s.store_u(
                            s.table_lookup_lanes(v, i),
                            self.out.as_mut_ptr(),
                        );
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let input: Vec<$T> =
                    (0..lanes).map(|i| (i * 10 + 1) as $T).collect();
                // Reverse order index
                let idx: Vec<$I> =
                    (0..lanes).map(|i| (lanes - 1 - i) as $I).collect();
                let mut out = vec![0 as $T; lanes];
                dispatch_to(
                    K {
                        input: &input,
                        idx: &idx,
                        out: &mut out,
                    },
                    target,
                );
                for j in 0..lanes {
                    assert_eq!(
                        out[j],
                        input[lanes - 1 - j],
                        "table_lookup_lanes {} lane {j} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

test_tbl_lookup_lanes_type!(test_tbl_lookup_u8, u8, u8);
test_tbl_lookup_lanes_type!(test_tbl_lookup_u16, u16, u16);
test_tbl_lookup_lanes_type!(test_tbl_lookup_u64, u64, u64);
test_tbl_lookup_lanes_type!(test_tbl_lookup_i8, i8, i8);
test_tbl_lookup_lanes_type!(test_tbl_lookup_i32, i32, i32);
test_tbl_lookup_lanes_type!(test_tbl_lookup_i64, i64, i64);

// ---------------------------------------------------------------------------
// compress_store for u32
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_u32() {
    struct K<'a> {
        input: &'a [u32],
        out: &'a mut [u32],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u32>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(5u32);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (0..lanes).map(|i| (i as u32) * 3).collect();
        let mut out = vec![0u32; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<u32> =
            input.iter().copied().filter(|&x| x > 5).collect();
        assert_eq!(count, expected.len(), "compress_store count {target:?}");
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store values {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress_store for u8
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_u8() {
    struct K<'a> {
        input: &'a [u8],
        out: &'a mut [u8],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u8>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(100u8);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u8>(target);
        let input: Vec<u8> = (0..lanes)
            .map(|i| {
                if i % 2 == 0 {
                    50u8.wrapping_add(i as u8)
                } else {
                    150u8.wrapping_add(i as u8)
                }
            })
            .collect();
        let mut out = vec![0u8; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<u8> =
            input.iter().copied().filter(|&x| x > 100).collect();
        assert_eq!(count, expected.len(), "compress_store u8 count {target:?}");
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store u8 values {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress_store for i8
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_i8() {
    struct K<'a> {
        input: &'a [i8],
        out: &'a mut [i8],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<i8>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0i8);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<i8>(target);
        let input: Vec<i8> = (0..lanes)
            .map(|i| {
                if i % 2 == 0 {
                    -50 + (i as i8)
                } else {
                    50 + (i as i8 % 70)
                }
            })
            .collect();
        let mut out = vec![0i8; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<i8> =
            input.iter().copied().filter(|&x| x > 0).collect();
        assert_eq!(count, expected.len(), "compress_store i8 count {target:?}");
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store i8 values {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress_store for u16
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_u16() {
    struct K<'a> {
        input: &'a [u16],
        out: &'a mut [u16],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u16>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(100u16);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u16>(target);
        let input: Vec<u16> = (0..lanes as u16)
            .map(|i| if i % 2 == 0 { 50 + i } else { 150 + i })
            .collect();
        let mut out = vec![0u16; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<u16> =
            input.iter().copied().filter(|&x| x > 100).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress_store u16 count {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store u16 values {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress_store for i16
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_i16() {
    struct K<'a> {
        input: &'a [i16],
        out: &'a mut [i16],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<i16>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0i16);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<i16>(target);
        let input: Vec<i16> = (0..lanes as i16)
            .map(|i| if i % 2 == 0 { -100 + i } else { 100 + i })
            .collect();
        let mut out = vec![0i16; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<i16> =
            input.iter().copied().filter(|&x| x > 0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress_store i16 count {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store i16 values {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress_store for u64
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_u64() {
    struct K<'a> {
        input: &'a [u64],
        out: &'a mut [u64],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u64>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(5u64);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u64>(target);
        let input: Vec<u64> = (0..lanes).map(|i| (i as u64) * 3).collect();
        let mut out = vec![0u64; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<u64> =
            input.iter().copied().filter(|&x| x > 5).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress_store u64 count {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store u64 values {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// compress_store for i64
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_i64() {
    struct K<'a> {
        input: &'a [i64],
        out: &'a mut [i64],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<i64>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0i64);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<i64>(target);
        let input: Vec<i64> = (0..lanes).map(|i| (i as i64) * 5 - 10).collect();
        let mut out = vec![0i64; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<i64> =
            input.iter().copied().filter(|&x| x > 0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress_store i64 count {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store i64 values {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// iota: all lane types
// ---------------------------------------------------------------------------

macro_rules! test_iota_int {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    let lanes = s.lanes::<$T>();
                    unsafe {
                        let v = s.iota(5 as $T);
                        s.store_u(v, self.out.as_mut_ptr());
                        let _ = lanes;
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let mut out = vec![0 as $T; lanes];
                dispatch_to(K { out: &mut out }, target);
                for i in 0..lanes {
                    assert_eq!(
                        out[i],
                        5 as $T + i as $T,
                        "iota {} lane {i} {target:?}",
                        stringify!($T)
                    );
                }
            }
        }
    };
}

macro_rules! test_iota_float {
    ($name:ident, $T:ty) => {
        #[test]
        fn $name() {
            struct K<'a> {
                out: &'a mut [$T],
            }
            impl WithSimd for K<'_> {
                type Output = ();
                fn with_simd<S: SimdOps>(self, s: S) {
                    unsafe {
                        let v = s.iota(2.5 as $T);
                        s.store_u(v, self.out.as_mut_ptr());
                    }
                }
            }
            for target in available_targets() {
                let lanes = lanes_for::<$T>(target);
                let mut out = vec![0.0 as $T; lanes];
                dispatch_to(K { out: &mut out }, target);
                for i in 0..lanes {
                    let expected = 2.5 as $T + i as $T;
                    assert!(
                        (out[i] - expected).abs() < 1e-6 as $T,
                        "iota {} lane {i} {target:?}: got {}, expected {}",
                        stringify!($T),
                        out[i],
                        expected
                    );
                }
            }
        }
    };
}

test_iota_int!(test_iota_u8, u8);
test_iota_int!(test_iota_u16, u16);
test_iota_int!(test_iota_u64, u64);
test_iota_int!(test_iota_i8, i8);
test_iota_int!(test_iota_i16, i16);
test_iota_int!(test_iota_i32, i32);
test_iota_int!(test_iota_i64, i64);
test_iota_float!(test_iota_f32, f32);
test_iota_float!(test_iota_f64, f64);

// =========================================================================
// Gap coverage: missing type variants and operations
// =========================================================================

// ---------------------------------------------------------------------------
// compress: i64, f64
// ---------------------------------------------------------------------------

#[test]
fn test_compress_i64() {
    struct K<'a> {
        input: &'a [i64],
        out: &'a mut [i64],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<i64>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0i64);
                let mask = s.gt(v, thresh);
                let compressed = s.compress(v, mask);
                s.store_u(compressed, self.out.as_mut_ptr());
                *self.count = s.count_true(mask);
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<i64>(target);
        let input: Vec<i64> = (0..lanes).map(|i| (i as i64) * 5 - 10).collect();
        let mut out = vec![0i64; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<i64> =
            input.iter().copied().filter(|&x| x > 0).collect();
        assert_eq!(count, expected.len(), "compress i64 count {target:?}");
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress i64 values {target:?}"
        );
    }
}

#[test]
fn test_compress_f64() {
    struct K<'a> {
        input: &'a [f64],
        out: &'a mut [f64],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0.0f64);
                let mask = s.gt(v, thresh);
                let compressed = s.compress(v, mask);
                s.store_u(compressed, self.out.as_mut_ptr());
                *self.count = s.count_true(mask);
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let input: Vec<f64> = (0..lanes)
            .map(|i| {
                if i % 2 == 0 {
                    -(i as f64) - 1.0
                } else {
                    (i as f64) * 10.0
                }
            })
            .collect();
        let mut out = vec![0.0f64; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<f64> =
            input.iter().copied().filter(|&x| x > 0.0).collect();
        assert_eq!(count, expected.len(), "compress f64 count {target:?}");
        for j in 0..count {
            assert!(
                (out[j] - expected[j]).abs() < 1e-10,
                "compress f64 value {j} {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// compress_store: i32, f32, f64
// ---------------------------------------------------------------------------

#[test]
fn test_compress_store_i32() {
    struct K<'a> {
        input: &'a [i32],
        out: &'a mut [i32],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<i32>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0i32);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<i32>(target);
        let input: Vec<i32> = (0..lanes as i32).map(|i| i * 3 - 5).collect();
        let mut out = vec![0i32; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<i32> =
            input.iter().copied().filter(|&x| x > 0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress_store i32 count {target:?}"
        );
        assert_eq!(
            &out[..count],
            &expected[..],
            "compress_store i32 values {target:?}"
        );
    }
}

#[test]
fn test_compress_store_f32() {
    struct K<'a> {
        input: &'a [f32],
        out: &'a mut [f32],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f32>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0.0f32);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let input: Vec<f32> = (0..lanes)
            .map(|i| {
                if i % 2 == 0 {
                    -(i as f32) - 1.0
                } else {
                    (i as f32) * 10.0
                }
            })
            .collect();
        let mut out = vec![0.0f32; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<f32> =
            input.iter().copied().filter(|&x| x > 0.0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress_store f32 count {target:?}"
        );
        for j in 0..count {
            assert!(
                (out[j] - expected[j]).abs() < 1e-4,
                "compress_store f32 value {j} {target:?}"
            );
        }
    }
}

#[test]
fn test_compress_store_f64() {
    struct K<'a> {
        input: &'a [f64],
        out: &'a mut [f64],
        count: &'a mut usize,
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            if lanes > self.input.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.input.as_ptr());
                let thresh = s.splat(0.0f64);
                let mask = s.gt(v, thresh);
                *self.count = s.compress_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let input: Vec<f64> = (0..lanes)
            .map(|i| {
                if i % 2 == 0 {
                    -(i as f64) - 1.0
                } else {
                    (i as f64) * 10.0
                }
            })
            .collect();
        let mut out = vec![0.0f64; lanes];
        let mut count = 0usize;
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
                count: &mut count,
            },
            target,
        );
        let expected: Vec<f64> =
            input.iter().copied().filter(|&x| x > 0.0).collect();
        assert_eq!(
            count,
            expected.len(),
            "compress_store f64 count {target:?}"
        );
        for j in 0..count {
            assert!(
                (out[j] - expected[j]).abs() < 1e-10,
                "compress_store f64 value {j} {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// blended_store, masked_load, load_dup128
// ---------------------------------------------------------------------------

#[test]
fn test_blended_store() {
    struct K<'a> {
        data: &'a [u32],
        out: &'a mut [u32],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u32>();
            if lanes > self.data.len() {
                return;
            }
            unsafe {
                let v = s.load_u(self.data.as_ptr());
                let thresh = s.splat(5u32);
                let mask = s.gt(v, thresh);
                s.blended_store(v, mask, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let data: Vec<u32> = (0..lanes).map(|i| (i as u32) * 3).collect();
        let mut out = vec![999u32; lanes];
        dispatch_to(
            K {
                data: &data,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            if data[i] > 5 {
                assert_eq!(
                    out[i], data[i],
                    "blended_store lane {i} should be written for {target:?}"
                );
            } else {
                assert_eq!(
                    out[i], 999,
                    "blended_store lane {i} should be untouched for {target:?}"
                );
            }
        }
    }
}

#[test]
fn test_masked_load() {
    struct K<'a> {
        data: &'a [u32],
        out: &'a mut [u32],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<u32>();
            if lanes > self.data.len() {
                return;
            }
            unsafe {
                let all = s.load_u(self.data.as_ptr());
                let thresh = s.splat(5u32);
                let mask = s.gt(all, thresh);
                let loaded = s.masked_load(mask, self.data.as_ptr());
                s.store_u(loaded, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let data: Vec<u32> = (0..lanes).map(|i| (i as u32) * 3).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            K {
                data: &data,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            if data[i] > 5 {
                assert_eq!(
                    out[i], data[i],
                    "masked_load lane {i} should be loaded for {target:?}"
                );
            } else {
                assert_eq!(
                    out[i], 0,
                    "masked_load lane {i} should be zero for {target:?}"
                );
            }
        }
    }
}

#[test]
fn test_load_dup128() {
    struct K<'a> {
        data: &'a [u32],
        out: &'a mut [u32],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            unsafe {
                let v = s.load_dup128(self.data.as_ptr());
                s.store_u(v, self.out.as_mut_ptr());
            }
        }
    }
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let data: Vec<u32> = vec![10, 20, 30, 40];
        let mut out = vec![0u32; lanes];
        dispatch_to(
            K {
                data: &data,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(out[i], data[i % 4], "load_dup128 lane {i} {target:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// sqrt f32
// ---------------------------------------------------------------------------

#[test]
fn test_sqrt_f32() {
    struct K<'a> {
        input: &'a [f32],
        out: &'a mut [f32],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f32>();
            let n = self.input.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let v = s.load_u(self.input.as_ptr().add(i));
                    s.store_u(s.sqrt(v), self.out.as_mut_ptr().add(i));
                }
                i += lanes;
            }
        }
    }
    let input: Vec<f32> = vec![
        0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 100.0, 2.0, 0.25, 0.01, 1e6, 1e-6, 0.5,
        81.0, 144.0, 256.0,
    ];
    let n = input.len();
    let expected: Vec<f32> = input.iter().map(|x| x.sqrt()).collect();
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let mut out = vec![0.0f32; n];
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        for j in 0..check {
            assert!(
                (out[j] - expected[j]).abs() < 1e-4,
                "sqrt f32 lane {j} {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// approx_reciprocal f64
// ---------------------------------------------------------------------------

#[test]
fn test_approx_reciprocal_f64() {
    struct K<'a> {
        input: &'a [f64],
        out: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.input.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let v = s.load_u(self.input.as_ptr().add(i));
                    s.store_u(
                        s.approx_reciprocal(v),
                        self.out.as_mut_ptr().add(i),
                    );
                }
                i += lanes;
            }
        }
    }
    let input: Vec<f64> = (1..=16).map(|i| i as f64 * 2.0).collect();
    let expected: Vec<f64> = input.iter().map(|x| 1.0 / x).collect();
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let n = input.len();
        let mut out = vec![0.0f64; n];
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        for j in 0..check {
            let rel_error = ((out[j] - expected[j]) / expected[j]).abs();
            assert!(
                rel_error < 0.01,
                "approx_reciprocal f64 error {rel_error:.4} at {j} {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// approx_reciprocal_sqrt f64
// ---------------------------------------------------------------------------

#[test]
fn test_approx_reciprocal_sqrt_f64() {
    struct K<'a> {
        input: &'a [f64],
        out: &'a mut [f64],
    }
    impl WithSimd for K<'_> {
        type Output = ();
        fn with_simd<S: SimdOps>(self, s: S) {
            let lanes = s.lanes::<f64>();
            let n = self.input.len();
            let mut i = 0;
            while i + lanes <= n {
                unsafe {
                    let v = s.load_u(self.input.as_ptr().add(i));
                    s.store_u(
                        s.approx_reciprocal_sqrt(v),
                        self.out.as_mut_ptr().add(i),
                    );
                }
                i += lanes;
            }
        }
    }
    let input: Vec<f64> = (1..=16).map(|i| (i as f64) * 4.0).collect();
    let expected: Vec<f64> = input.iter().map(|x| 1.0 / x.sqrt()).collect();
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let n = input.len();
        let mut out = vec![0.0f64; n];
        dispatch_to(
            K {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let check = lanes * (n / lanes);
        for j in 0..check {
            let rel_error = ((out[j] - expected[j]) / expected[j]).abs();
            assert!(
                rel_error < 0.02,
                "approx_reciprocal_sqrt f64 error {rel_error:.4} at {j} {target:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// mul u64, i64
// ---------------------------------------------------------------------------

test_binop_int!(test_mul_u64, u64, mul, |a: u64, b: u64| a.wrapping_mul(b));
test_binop_int!(test_mul_i64, i64, mul, |a: i64, b: i64| a.wrapping_mul(b));

// ---------------------------------------------------------------------------
// abs_diff: u32, f64
// ---------------------------------------------------------------------------

test_abs_diff_int!(test_abs_diff_u32, u32);
test_abs_diff_float!(test_abs_diff_f64, f64);

// ---------------------------------------------------------------------------
// clamp: i32, f32
// ---------------------------------------------------------------------------

test_clamp_int!(test_clamp_i32, i32);
test_clamp_float!(test_clamp_f32, f32);

// ---------------------------------------------------------------------------
// sum/min/max_of_lanes: i32
// ---------------------------------------------------------------------------

test_sum_of_lanes_int!(test_sum_of_lanes_i32, i32);
test_min_max_of_lanes!(test_min_max_of_lanes_i32, i32, |i, l| ((i * 37 + 13)
    % l) as i32
    * 1000
    - 50000);

// ---------------------------------------------------------------------------
// mask_ops, if_then_else, mask_logic: u32, i32
// ---------------------------------------------------------------------------

test_mask_ops_type!(test_mask_ops_u32, u32);
test_if_then_else_type!(test_ite_u32, u32);
test_if_then_else_type!(test_ite_i32, i32);
test_mask_logic_type!(test_mask_logic_u32, u32);
test_mask_logic_type!(test_mask_logic_i32, i32);

// ---------------------------------------------------------------------------
// popcount/lzcnt/tzcnt/reverse_bits: signed variants
// ---------------------------------------------------------------------------

test_bitop!(
    test_popcount_i8,
    i8,
    population_count,
    |x: i8| x.count_ones() as i8
);
test_bitop!(
    test_popcount_i16,
    i16,
    population_count,
    |x: i16| x.count_ones() as i16
);
test_bitop!(test_popcount_i32, i32, population_count, |x: i32| {
    x.count_ones() as i32
});
test_bitop!(
    test_popcount_i64,
    i64,
    population_count,
    |x: i64| x.count_ones() as i64
);

test_bitop!(
    test_lzcnt_i8,
    i8,
    leading_zero_count,
    |x: i8| x.leading_zeros() as i8
);
test_bitop!(
    test_lzcnt_i16,
    i16,
    leading_zero_count,
    |x: i16| x.leading_zeros() as i16
);
test_bitop!(test_lzcnt_i32, i32, leading_zero_count, |x: i32| {
    x.leading_zeros() as i32
});
test_bitop!(
    test_lzcnt_i64,
    i64,
    leading_zero_count,
    |x: i64| x.leading_zeros() as i64
);

test_bitop!(
    test_tzcnt_i8,
    i8,
    trailing_zero_count,
    |x: i8| x.trailing_zeros() as i8
);
test_bitop!(test_tzcnt_i16, i16, trailing_zero_count, |x: i16| x
    .trailing_zeros()
    as i16);
test_bitop!(test_tzcnt_i32, i32, trailing_zero_count, |x: i32| {
    x.trailing_zeros() as i32
});
test_bitop!(test_tzcnt_i64, i64, trailing_zero_count, |x: i64| x
    .trailing_zeros()
    as i64);

test_bitop!(test_reverse_bits_i8, i8, reverse_bits, |x: i8| x
    .reverse_bits());
test_bitop!(test_reverse_bits_i16, i16, reverse_bits, |x: i16| x
    .reverse_bits());
test_bitop!(test_reverse_bits_i32, i32, reverse_bits, |x: i32| x
    .reverse_bits());
test_bitop!(test_reverse_bits_i64, i64, reverse_bits, |x: i64| x
    .reverse_bits());

// ---------------------------------------------------------------------------
// reverse_lane_bytes: u32
// ---------------------------------------------------------------------------

test_rev_lane_bytes!(test_rev_lane_bytes_u32, u32);

// ---------------------------------------------------------------------------
// shuffle: missing f32/f64/i32 type variants
// ---------------------------------------------------------------------------

test_reverse_type!(test_reverse_f32, f32);
test_reverse_type!(test_reverse_f64, f64);

test_reverse_n!(test_reverse2_f32, f32, reverse2, 2);
test_reverse_n!(test_reverse2_f64, f64, reverse2, 2);

test_reverse_n!(test_reverse4_f32, f32, reverse4, 4);

test_interleave_type!(test_interleave_i32, i32);
test_interleave_type!(test_interleave_f32, f32);
test_interleave_type!(test_interleave_f64, f64);

test_zip_type!(test_zip_i32, i32);
test_zip_type!(test_zip_f32, f32);
test_zip_type!(test_zip_f64, f64);

test_concat_even_odd_type!(test_concat_eo_i32, i32);
test_concat_even_odd_type!(test_concat_eo_f32, f32);
test_concat_even_odd_type!(test_concat_eo_f64, f64);

test_concat_ul_type!(test_concat_ul_f32, f32);
test_concat_ul_type!(test_concat_ul_f64, f64);

test_odd_even_type!(test_odd_even_u32, u32);
test_odd_even_type!(test_odd_even_f32, f32);

test_broadcast_type!(test_broadcast_i32, i32);
test_broadcast_type!(test_broadcast_f32, f32);
test_broadcast_type!(test_broadcast_f64, f64);

test_slide_down_type!(test_slide_down_f32, f32);
test_slide_down_type!(test_slide_down_f64, f64);

test_slide_up_type!(test_slide_up_u32, u32);
test_slide_up_type!(test_slide_up_f32, f32);
test_slide_up_type!(test_slide_up_f64, f64);

test_tbl_lookup_lanes_type!(test_tbl_lookup_i16, i16, i16);
test_tbl_lookup_lanes_type!(test_tbl_lookup_u32, u32, u32);
test_tbl_lookup_lanes_type!(test_tbl_lookup_f32, f32, u32);
test_tbl_lookup_lanes_type!(test_tbl_lookup_f64, f64, u64);
// =========================================================================
// SimdMemory: gather_index
// =========================================================================

struct GatherIndexKernel<'a> {
    table: &'a [u32],
    indices: &'a [i32],
    out: &'a mut [u32],
}

impl WithSimd for GatherIndexKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            // Build index vector from the first `lanes` indices
            let mut idx_buf = vec![0i32; lanes];
            for i in 0..lanes.min(self.indices.len()) {
                idx_buf[i] = self.indices[i];
            }
            let idx = s.load_u(idx_buf.as_ptr());
            let result = s.gather_index(self.table.as_ptr(), idx);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_gather_index_u32() {
    let table: Vec<u32> = (0..64).map(|i| i * 10).collect();
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let indices: Vec<i32> = (0..lanes).map(|i| (i * 3) as i32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            GatherIndexKernel {
                table: &table,
                indices: &indices,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = table[indices[i] as usize];
            assert_eq!(
                out[i], expected,
                "gather_index lane {i} wrong for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdMemory: scatter_index
// =========================================================================

struct ScatterIndexKernel<'a> {
    values: &'a [u32],
    indices: &'a [i32],
    out: &'a mut [u32],
}

impl WithSimd for ScatterIndexKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            let mut idx_buf = vec![0i32; lanes];
            for i in 0..lanes.min(self.indices.len()) {
                idx_buf[i] = self.indices[i];
            }
            let idx = s.load_u(idx_buf.as_ptr());
            s.scatter_index(v, self.out.as_mut_ptr(), idx);
        }
    }
}

#[test]
fn test_scatter_index_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let values: Vec<u32> =
            (0..lanes).map(|i| (i + 1) as u32 * 100).collect();
        let indices: Vec<i32> = (0..lanes).map(|i| (i * 2) as i32).collect();
        let mut out = vec![0u32; 64];
        dispatch_to(
            ScatterIndexKernel {
                values: &values,
                indices: &indices,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let idx = indices[i] as usize;
            assert_eq!(
                out[idx], values[i],
                "scatter_index: out[{idx}] wrong for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdMemory: load/store interleaved 2
// =========================================================================

struct Interleave2Kernel<'a> {
    input: &'a [u32],
    out0: &'a mut [u32],
    out1: &'a mut [u32],
}

impl WithSimd for Interleave2Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let (v0, v1) = s.load_interleaved_2(self.input.as_ptr());
            for i in 0..lanes {
                self.out0[i] = s.extract_lane(v0, i);
                self.out1[i] = s.extract_lane(v1, i);
            }
        }
    }
}

#[test]
fn test_load_interleaved_2_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        // Interleaved data: [a0, b0, a1, b1, ...]
        let input: Vec<u32> = (0..2 * lanes).map(|i| i as u32).collect();
        let mut out0 = vec![0u32; lanes];
        let mut out1 = vec![0u32; lanes];
        dispatch_to(
            Interleave2Kernel {
                input: &input,
                out0: &mut out0,
                out1: &mut out1,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out0[i],
                input[2 * i],
                "interleave2 ch0 lane {i} for {target:?}"
            );
            assert_eq!(
                out1[i],
                input[2 * i + 1],
                "interleave2 ch1 lane {i} for {target:?}"
            );
        }
    }
}

struct StoreInterleave2Kernel<'a> {
    ch0: &'a [u32],
    ch1: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for StoreInterleave2Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        unsafe {
            let v0 = s.load_u(self.ch0.as_ptr());
            let v1 = s.load_u(self.ch1.as_ptr());
            s.store_interleaved_2(v0, v1, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_store_interleaved_2_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let ch0: Vec<u32> = (0..lanes).map(|i| i as u32 * 10).collect();
        let ch1: Vec<u32> = (0..lanes).map(|i| i as u32 * 10 + 1).collect();
        let mut out = vec![0u32; 2 * lanes];
        dispatch_to(
            StoreInterleave2Kernel {
                ch0: &ch0,
                ch1: &ch1,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[2 * i],
                ch0[i],
                "store_interleave2 ch0 lane {i} for {target:?}"
            );
            assert_eq!(
                out[2 * i + 1],
                ch1[i],
                "store_interleave2 ch1 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdMemory: load/store interleaved 3
// =========================================================================

struct Interleave3Kernel<'a> {
    input: &'a [u32],
    out0: &'a mut [u32],
    out1: &'a mut [u32],
    out2: &'a mut [u32],
}

impl WithSimd for Interleave3Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let (v0, v1, v2) = s.load_interleaved_3(self.input.as_ptr());
            for i in 0..lanes {
                self.out0[i] = s.extract_lane(v0, i);
                self.out1[i] = s.extract_lane(v1, i);
                self.out2[i] = s.extract_lane(v2, i);
            }
        }
    }
}

#[test]
fn test_load_interleaved_3_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (0..3 * lanes).map(|i| i as u32).collect();
        let mut out0 = vec![0u32; lanes];
        let mut out1 = vec![0u32; lanes];
        let mut out2 = vec![0u32; lanes];
        dispatch_to(
            Interleave3Kernel {
                input: &input,
                out0: &mut out0,
                out1: &mut out1,
                out2: &mut out2,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out0[i],
                input[3 * i],
                "interleave3 ch0 lane {i} for {target:?}"
            );
            assert_eq!(
                out1[i],
                input[3 * i + 1],
                "interleave3 ch1 lane {i} for {target:?}"
            );
            assert_eq!(
                out2[i],
                input[3 * i + 2],
                "interleave3 ch2 lane {i} for {target:?}"
            );
        }
    }
}

struct StoreInterleave3Kernel<'a> {
    ch0: &'a [u32],
    ch1: &'a [u32],
    ch2: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for StoreInterleave3Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        unsafe {
            let v0 = s.load_u(self.ch0.as_ptr());
            let v1 = s.load_u(self.ch1.as_ptr());
            let v2 = s.load_u(self.ch2.as_ptr());
            s.store_interleaved_3(v0, v1, v2, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_store_interleaved_3_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let ch0: Vec<u32> = (0..lanes).map(|i| i as u32 * 100).collect();
        let ch1: Vec<u32> = (0..lanes).map(|i| i as u32 * 100 + 1).collect();
        let ch2: Vec<u32> = (0..lanes).map(|i| i as u32 * 100 + 2).collect();
        let mut out = vec![0u32; 3 * lanes];
        dispatch_to(
            StoreInterleave3Kernel {
                ch0: &ch0,
                ch1: &ch1,
                ch2: &ch2,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[3 * i],
                ch0[i],
                "store_interleave3 ch0 lane {i} for {target:?}"
            );
            assert_eq!(
                out[3 * i + 1],
                ch1[i],
                "store_interleave3 ch1 lane {i} for {target:?}"
            );
            assert_eq!(
                out[3 * i + 2],
                ch2[i],
                "store_interleave3 ch2 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdMemory: load/store interleaved 4
// =========================================================================

struct Interleave4Kernel<'a> {
    input: &'a [u32],
    out0: &'a mut [u32],
    out1: &'a mut [u32],
    out2: &'a mut [u32],
    out3: &'a mut [u32],
}

impl WithSimd for Interleave4Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let (v0, v1, v2, v3) = s.load_interleaved_4(self.input.as_ptr());
            for i in 0..lanes {
                self.out0[i] = s.extract_lane(v0, i);
                self.out1[i] = s.extract_lane(v1, i);
                self.out2[i] = s.extract_lane(v2, i);
                self.out3[i] = s.extract_lane(v3, i);
            }
        }
    }
}

#[test]
fn test_load_interleaved_4_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (0..4 * lanes).map(|i| i as u32).collect();
        let mut out0 = vec![0u32; lanes];
        let mut out1 = vec![0u32; lanes];
        let mut out2 = vec![0u32; lanes];
        let mut out3 = vec![0u32; lanes];
        dispatch_to(
            Interleave4Kernel {
                input: &input,
                out0: &mut out0,
                out1: &mut out1,
                out2: &mut out2,
                out3: &mut out3,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out0[i],
                input[4 * i],
                "interleave4 ch0 lane {i} for {target:?}"
            );
            assert_eq!(
                out1[i],
                input[4 * i + 1],
                "interleave4 ch1 lane {i} for {target:?}"
            );
            assert_eq!(
                out2[i],
                input[4 * i + 2],
                "interleave4 ch2 lane {i} for {target:?}"
            );
            assert_eq!(
                out3[i],
                input[4 * i + 3],
                "interleave4 ch3 lane {i} for {target:?}"
            );
        }
    }
}

struct StoreInterleave4Kernel<'a> {
    ch0: &'a [u32],
    ch1: &'a [u32],
    ch2: &'a [u32],
    ch3: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for StoreInterleave4Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        unsafe {
            let v0 = s.load_u(self.ch0.as_ptr());
            let v1 = s.load_u(self.ch1.as_ptr());
            let v2 = s.load_u(self.ch2.as_ptr());
            let v3 = s.load_u(self.ch3.as_ptr());
            s.store_interleaved_4(v0, v1, v2, v3, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_store_interleaved_4_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let ch0: Vec<u32> = (0..lanes).map(|i| i as u32 * 1000).collect();
        let ch1: Vec<u32> = (0..lanes).map(|i| i as u32 * 1000 + 1).collect();
        let ch2: Vec<u32> = (0..lanes).map(|i| i as u32 * 1000 + 2).collect();
        let ch3: Vec<u32> = (0..lanes).map(|i| i as u32 * 1000 + 3).collect();
        let mut out = vec![0u32; 4 * lanes];
        dispatch_to(
            StoreInterleave4Kernel {
                ch0: &ch0,
                ch1: &ch1,
                ch2: &ch2,
                ch3: &ch3,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[4 * i],
                ch0[i],
                "store_interleave4 ch0 lane {i} for {target:?}"
            );
            assert_eq!(
                out[4 * i + 1],
                ch1[i],
                "store_interleave4 ch1 lane {i} for {target:?}"
            );
            assert_eq!(
                out[4 * i + 2],
                ch2[i],
                "store_interleave4 ch2 lane {i} for {target:?}"
            );
            assert_eq!(
                out[4 * i + 3],
                ch3[i],
                "store_interleave4 ch3 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// load/store interleaved roundtrip: store then load should give back original
// =========================================================================

struct InterleaveRoundtripKernel<'a> {
    ch0: &'a [u32],
    ch1: &'a [u32],
    out0: &'a mut [u32],
    out1: &'a mut [u32],
}

impl WithSimd for InterleaveRoundtripKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v0 = s.load_u(self.ch0.as_ptr());
            let v1 = s.load_u(self.ch1.as_ptr());
            let mut buf = vec![0u32; 2 * lanes];
            s.store_interleaved_2(v0, v1, buf.as_mut_ptr());
            let (r0, r1) = s.load_interleaved_2(buf.as_ptr());
            for i in 0..lanes {
                self.out0[i] = s.extract_lane(r0, i);
                self.out1[i] = s.extract_lane(r1, i);
            }
        }
    }
}

#[test]
fn test_interleave_2_roundtrip() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let ch0: Vec<u32> = (0..lanes).map(|i| (i + 10) as u32).collect();
        let ch1: Vec<u32> = (0..lanes).map(|i| (i + 100) as u32).collect();
        let mut out0 = vec![0u32; lanes];
        let mut out1 = vec![0u32; lanes];
        dispatch_to(
            InterleaveRoundtripKernel {
                ch0: &ch0,
                ch1: &ch1,
                out0: &mut out0,
                out1: &mut out1,
            },
            target,
        );
        assert_eq!(out0, ch0, "interleave2 roundtrip ch0 for {target:?}");
        assert_eq!(out1, ch1, "interleave2 roundtrip ch1 for {target:?}");
    }
}

// =========================================================================
// SimdArith: widen_mul_pairwise_add_i16
// =========================================================================

struct WidenMulPairwiseAddKernel<'a> {
    a: &'a [i16],
    b: &'a [i16],
    out: &'a mut [i32],
}

impl WithSimd for WidenMulPairwiseAddKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_i32 = s.lanes::<i32>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let result = s.widen_mul_pairwise_add_i16(va, vb);
            for i in 0..lanes_i32.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_widen_mul_pairwise_add_i16() {
    for target in available_targets() {
        let lanes_i16 = lanes_for::<i16>(target);
        let lanes_i32 = lanes_for::<i32>(target);
        if lanes_i16 < 2 {
            continue; // pairwise needs at least 2 narrow lanes
        }
        let a: Vec<i16> = (0..lanes_i16).map(|i| (i + 1) as i16).collect();
        let b: Vec<i16> = (0..lanes_i16).map(|i| (i + 2) as i16).collect();
        let mut out = vec![0i32; lanes_i32];
        dispatch_to(
            WidenMulPairwiseAddKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes_i32 {
            let expected = (a[2 * i] as i32) * (b[2 * i] as i32)
                + (a[2 * i + 1] as i32) * (b[2 * i + 1] as i32);
            assert_eq!(
                out[i], expected,
                "widen_mul_pairwise_add lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdArith: sat_widen_mul_pairwise_add (u8 * i8 -> i16)
// =========================================================================

struct SatWidenMulPairwiseAddKernel<'a> {
    a: &'a [u8],
    b: &'a [i8],
    out: &'a mut [i16],
}

impl WithSimd for SatWidenMulPairwiseAddKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_i16 = s.lanes::<i16>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let result = s.sat_widen_mul_pairwise_add(va, vb);
            for i in 0..lanes_i16.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_sat_widen_mul_pairwise_add() {
    for target in available_targets() {
        let lanes_u8 = lanes_for::<u8>(target);
        let lanes_i16 = lanes_for::<i16>(target);
        if lanes_u8 < 2 {
            continue; // pairwise needs at least 2 narrow lanes
        }
        // Use small values to avoid saturation in expected computation
        let a: Vec<u8> = (0..lanes_u8).map(|i| (i % 10 + 1) as u8).collect();
        let b: Vec<i8> = (0..lanes_u8).map(|i| (i % 10) as i8 - 5).collect();
        let mut out = vec![0i16; lanes_i16];
        dispatch_to(
            SatWidenMulPairwiseAddKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes_i16 {
            let sum = (a[2 * i] as i32) * (b[2 * i] as i32)
                + (a[2 * i + 1] as i32) * (b[2 * i + 1] as i32);
            let expected = sum.clamp(-32768, 32767) as i16;
            assert_eq!(
                out[i], expected,
                "sat_widen_mul_pairwise_add lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdArith: mul_fixed_point_15
// =========================================================================

struct MulFixedPoint15Kernel<'a> {
    a: &'a [i16],
    b: &'a [i16],
    out: &'a mut [i16],
}

impl WithSimd for MulFixedPoint15Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<i16>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let result = s.mul_fixed_point_15(va, vb);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_mul_fixed_point_15() {
    for target in available_targets() {
        let lanes = lanes_for::<i16>(target);
        let a: Vec<i16> = (0..lanes).map(|i| (i as i16 + 1) * 100).collect();
        let b: Vec<i16> = (0..lanes).map(|i| (i as i16 + 1) * 50).collect();
        let mut out = vec![0i16; lanes];
        dispatch_to(
            MulFixedPoint15Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected =
                (((a[i] as i32) * (b[i] as i32) + (1 << 14)) >> 15) as i16;
            assert_eq!(
                out[i], expected,
                "mul_fixed_point_15 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdArith: reorder_widen_mul_accumulate
// =========================================================================

struct ReorderWidenMulAccKernel<'a> {
    a: &'a [i16],
    b: &'a [i16],
    sum_init: &'a [i32],
    out: &'a mut [i32],
}

impl WithSimd for ReorderWidenMulAccKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_i32 = s.lanes::<i32>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let vsum = s.load_u(self.sum_init.as_ptr());
            let result = s.reorder_widen_mul_accumulate(va, vb, vsum);
            for i in 0..lanes_i32.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_reorder_widen_mul_accumulate() {
    for target in available_targets() {
        let lanes_i16 = lanes_for::<i16>(target);
        let lanes_i32 = lanes_for::<i32>(target);
        if lanes_i16 < 2 {
            continue;
        }
        let a: Vec<i16> = (0..lanes_i16).map(|i| (i + 1) as i16).collect();
        let b: Vec<i16> = (0..lanes_i16).map(|i| (i + 2) as i16).collect();
        let sum_init: Vec<i32> =
            (0..lanes_i32).map(|i| (i * 1000) as i32).collect();
        let mut out = vec![0i32; lanes_i32];
        dispatch_to(
            ReorderWidenMulAccKernel {
                a: &a,
                b: &b,
                sum_init: &sum_init,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes_i32 {
            let madd = (a[2 * i] as i32) * (b[2 * i] as i32)
                + (a[2 * i + 1] as i32) * (b[2 * i + 1] as i32);
            let expected = sum_init[i] + madd;
            assert_eq!(
                out[i], expected,
                "reorder_widen_mul_accumulate lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdBitwise: shl (per-lane variable left shift)
// =========================================================================

struct ShlKernel<'a> {
    values: &'a [u32],
    shifts: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ShlKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            let bits = s.load_u(self.shifts.as_ptr());
            let result = s.shl(v, bits);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_shl_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let values: Vec<u32> = (0..lanes).map(|i| 1u32 << (i % 8)).collect();
        let shifts: Vec<u32> = (0..lanes).map(|i| (i % 16) as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            ShlKernel {
                values: &values,
                shifts: &shifts,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = values[i].wrapping_shl(shifts[i]);
            assert_eq!(out[i], expected, "shl lane {i} for {target:?}");
        }
    }
}

// =========================================================================
// SimdBitwise: shr (per-lane variable right shift)
// =========================================================================

struct ShrKernel<'a> {
    values: &'a [u32],
    shifts: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ShrKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            let bits = s.load_u(self.shifts.as_ptr());
            let result = s.shr(v, bits);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_shr_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let values: Vec<u32> =
            (0..lanes).map(|i| 0x8000_0000u32 >> (i % 8)).collect();
        let shifts: Vec<u32> = (0..lanes).map(|i| (i % 16) as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            ShrKernel {
                values: &values,
                shifts: &shifts,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = values[i].wrapping_shr(shifts[i]);
            assert_eq!(out[i], expected, "shr lane {i} for {target:?}");
        }
    }
}

// Also test signed shr (arithmetic shift)
struct ShrSignedKernel<'a> {
    values: &'a [i32],
    shifts: &'a [i32],
    out: &'a mut [i32],
}

impl WithSimd for ShrSignedKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<i32>();
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            let bits = s.load_u(self.shifts.as_ptr());
            let result = s.shr(v, bits);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_shr_i32_arithmetic() {
    for target in available_targets() {
        let lanes = lanes_for::<i32>(target);
        let values: Vec<i32> =
            (0..lanes).map(|i| -1000i32 * (i as i32 + 1)).collect();
        let shifts: Vec<i32> = (0..lanes).map(|i| (i % 16) as i32).collect();
        let mut out = vec![0i32; lanes];
        dispatch_to(
            ShrSignedKernel {
                values: &values,
                shifts: &shifts,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = values[i].wrapping_shr(shifts[i] as u32);
            assert_eq!(out[i], expected, "shr_signed lane {i} for {target:?}");
        }
    }
}

// =========================================================================
// SimdConvert: truncate_to (u32 -> u16)
// =========================================================================

struct TruncateU32ToU16Kernel<'a> {
    values: &'a [u32],
    out: &'a mut [u16],
}

impl WithSimd for TruncateU32ToU16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            let result: S::Vec<u16> = s.truncate_to(v);
            // The result has lanes of u16 — extract the lower half (lanes_u32 worth)
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_truncate_u32_to_u16() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let values: Vec<u32> = (0..lanes)
            .map(|i| 0x0001_0000u32 + i as u32 * 0x1234)
            .collect();
        let mut out = vec![0u16; lanes];
        dispatch_to(
            TruncateU32ToU16Kernel {
                values: &values,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = values[i] as u16; // truncation: keep low 16 bits
            assert_eq!(
                out[i], expected,
                "truncate u32->u16 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdConvert: truncate_to (u16 -> u8)
// =========================================================================

struct TruncateU16ToU8Kernel<'a> {
    values: &'a [u16],
    out: &'a mut [u8],
}

impl WithSimd for TruncateU16ToU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u16>();
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            let result: S::Vec<u8> = s.truncate_to(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_truncate_u16_to_u8() {
    for target in available_targets() {
        let lanes = lanes_for::<u16>(target);
        let values: Vec<u16> =
            (0..lanes).map(|i| 0x0100u16 + i as u16 * 17).collect();
        let mut out = vec![0u8; lanes];
        dispatch_to(
            TruncateU16ToU8Kernel {
                values: &values,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = values[i] as u8;
            assert_eq!(
                out[i], expected,
                "truncate u16->u8 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdConvert: ordered_demote_2_to (i32 -> i16)
// =========================================================================

struct OrderedDemote2Kernel<'a> {
    lo: &'a [i32],
    hi: &'a [i32],
    out: &'a mut [i16],
}

impl WithSimd for OrderedDemote2Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_i32 = s.lanes::<i32>();
        unsafe {
            let vlo = s.load_u(self.lo.as_ptr());
            let vhi = s.load_u(self.hi.as_ptr());
            let result: S::Vec<i16> = s.ordered_demote_2_to(vlo, vhi);
            for i in 0..(2 * lanes_i32).min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_ordered_demote_2_i32_to_i16() {
    for target in available_targets() {
        let lanes_i32 = lanes_for::<i32>(target);
        let lanes_i16 = lanes_for::<i16>(target);
        if lanes_i32 < 2 {
            // Scalar: ordered_demote_2 with 1-lane vectors produces 1 i16 lane (just lo)
            let lo = vec![100i32];
            let hi = vec![-100i32];
            let mut out = vec![0i16; 1];
            dispatch_to(
                OrderedDemote2Kernel {
                    lo: &lo,
                    hi: &hi,
                    out: &mut out,
                },
                target,
            );
            assert_eq!(
                out[0], 100i16,
                "ordered_demote_2 scalar for {target:?}"
            );
            continue;
        }
        // Use values that fit in i16
        let lo: Vec<i32> =
            (0..lanes_i32).map(|i| (i as i32 + 1) * 100).collect();
        let hi: Vec<i32> =
            (0..lanes_i32).map(|i| -((i as i32 + 1) * 100)).collect();
        let mut out = vec![0i16; lanes_i16];
        dispatch_to(
            OrderedDemote2Kernel {
                lo: &lo,
                hi: &hi,
                out: &mut out,
            },
            target,
        );
        // lo should be in the lower half, hi in the upper half
        for i in 0..lanes_i32 {
            assert_eq!(
                out[i], lo[i] as i16,
                "ordered_demote_2 lo lane {i} for {target:?}"
            );
        }
        for i in 0..lanes_i32 {
            assert_eq!(
                out[lanes_i32 + i],
                hi[i] as i16,
                "ordered_demote_2 hi lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdShuffle: dup_even
// =========================================================================

struct DupEvenKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for DupEvenKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let result = s.dup_even(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_dup_even_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let input: Vec<u32> = (0..lanes).map(|i| (i * 10 + 1) as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            DupEvenKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // dup_even: [a0, a0, a2, a2, ...] -- each even lane duplicated
        for i in 0..lanes {
            let expected = input[i & !1]; // even index: i & !1 clears bit 0
            assert_eq!(out[i], expected, "dup_even lane {i} for {target:?}");
        }
    }
}

// =========================================================================
// SimdShuffle: dup_odd
// =========================================================================

struct DupOddKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for DupOddKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let result = s.dup_odd(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_dup_odd_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 2 {
            continue; // scalar has only 1 lane, dup_odd is not meaningful
        }
        let input: Vec<u32> = (0..lanes).map(|i| (i * 10 + 1) as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            DupOddKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // dup_odd: [a1, a1, a3, a3, ...]
        for i in 0..lanes {
            let expected = input[i | 1]; // odd index: i | 1 sets bit 0
            assert_eq!(out[i], expected, "dup_odd lane {i} for {target:?}");
        }
    }
}

// =========================================================================
// SimdShuffle: concat_lower_lower
// =========================================================================

struct ConcatLowerLowerKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ConcatLowerLowerKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let result = s.concat_lower_lower(va, vb);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_concat_lower_lower_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let half = lanes / 2;
        if half == 0 {
            continue; // scalar
        }
        let a: Vec<u32> = (0..lanes).map(|i| (i + 100) as u32).collect();
        let b: Vec<u32> = (0..lanes).map(|i| (i + 200) as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            ConcatLowerLowerKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        // Lower half of b, then lower half of a
        for i in 0..half {
            assert_eq!(
                out[i], b[i],
                "concat_lower_lower lo lane {i} for {target:?}"
            );
        }
        for i in 0..half {
            assert_eq!(
                out[half + i],
                a[i],
                "concat_lower_lower hi lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdShuffle: concat_upper_upper
// =========================================================================

struct ConcatUpperUpperKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ConcatUpperUpperKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let result = s.concat_upper_upper(va, vb);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_concat_upper_upper_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let half = lanes / 2;
        if half == 0 {
            continue;
        }
        let a: Vec<u32> = (0..lanes).map(|i| (i + 100) as u32).collect();
        let b: Vec<u32> = (0..lanes).map(|i| (i + 200) as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            ConcatUpperUpperKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        // Upper half of b, then upper half of a
        for i in 0..half {
            assert_eq!(
                out[i],
                b[half + i],
                "concat_upper_upper lo lane {i} for {target:?}"
            );
        }
        for i in 0..half {
            assert_eq!(
                out[half + i],
                a[half + i],
                "concat_upper_upper hi lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdShuffle: slide_1_up
// =========================================================================

struct Slide1UpKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for Slide1UpKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let result = s.slide_1_up(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_slide_1_up_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 2 {
            continue;
        }
        let input: Vec<u32> = (0..lanes).map(|i| (i + 1) as u32 * 10).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            Slide1UpKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // slide_1_up: lane 0 = 0, lane i = input[i-1]
        assert_eq!(out[0], 0, "slide_1_up lane 0 for {target:?}");
        for i in 1..lanes {
            assert_eq!(
                out[i],
                input[i - 1],
                "slide_1_up lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdShuffle: slide_1_down
// =========================================================================

struct Slide1DownKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for Slide1DownKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let result = s.slide_1_down(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_slide_1_down_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 2 {
            continue;
        }
        let input: Vec<u32> = (0..lanes).map(|i| (i + 1) as u32 * 10).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            Slide1DownKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // slide_1_down: lane i = input[i+1], last lane = 0
        for i in 0..lanes - 1 {
            assert_eq!(
                out[i],
                input[i + 1],
                "slide_1_down lane {i} for {target:?}"
            );
        }
        assert_eq!(out[lanes - 1], 0, "slide_1_down last lane for {target:?}");
    }
}

// =========================================================================
// SimdShuffle: expand
// =========================================================================

struct ExpandKernel<'a> {
    input: &'a [u32],
    mask_bits: u64,
    out: &'a mut [u32],
}

impl WithSimd for ExpandKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let mask = s.first_n::<u32>(self.mask_bits as usize);
            let result = s.expand(v, mask);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_expand_u32() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 2 {
            continue;
        }
        let input: Vec<u32> =
            (0..lanes).map(|i| (i + 1) as u32 * 100).collect();
        // Mask: first half of lanes are true
        let mask_count = lanes / 2;
        let mut out = vec![0u32; lanes];
        dispatch_to(
            ExpandKernel {
                input: &input,
                mask_bits: mask_count as u64,
                out: &mut out,
            },
            target,
        );
        // Expand: lanes where mask is true get consecutive values from input,
        // lanes where mask is false get zero
        let mut src_idx = 0;
        for i in 0..lanes {
            if i < mask_count {
                assert_eq!(
                    out[i], input[src_idx],
                    "expand lane {i} (true) for {target:?}"
                );
                src_idx += 1;
            } else {
                assert_eq!(out[i], 0, "expand lane {i} (false) for {target:?}");
            }
        }
    }
}

// =========================================================================
// SimdReduce: sums_of_8_abs_diff
// =========================================================================

struct SumsOf8AbsDiffKernel<'a> {
    a: &'a [u8],
    b: &'a [u8],
    out: &'a mut [u64],
}

impl WithSimd for SumsOf8AbsDiffKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_u64 = s.lanes::<u64>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let result = s.sums_of_8_abs_diff(va, vb);
            for i in 0..lanes_u64.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_sums_of_8_abs_diff() {
    for target in available_targets() {
        let lanes_u8 = lanes_for::<u8>(target);
        let lanes_u64 = lanes_for::<u64>(target);
        let a: Vec<u8> = (0..lanes_u8).map(|i| (i * 3 + 5) as u8).collect();
        let b: Vec<u8> = (0..lanes_u8).map(|i| (i * 2 + 1) as u8).collect();
        let mut out = vec![0u64; lanes_u64];
        dispatch_to(
            SumsOf8AbsDiffKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        // Each u64 lane is the sum of 8 absolute differences
        for g in 0..lanes_u64 {
            let mut expected = 0u64;
            let base = g * 8;
            for j in 0..8.min(lanes_u8 - base) {
                expected += (a[base + j] as i16 - b[base + j] as i16)
                    .unsigned_abs() as u64;
            }
            assert_eq!(
                out[g], expected,
                "sums_of_8_abs_diff group {g} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdReduce: sums_of_2 (u16 -> u32)
// =========================================================================

struct SumsOf2U16Kernel<'a> {
    input: &'a [u16],
    out: &'a mut [u32],
}

impl WithSimd for SumsOf2U16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_u32 = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let result = s.sums_of_2(v);
            for i in 0..lanes_u32.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_sums_of_2_u16() {
    for target in available_targets() {
        let lanes_u16 = lanes_for::<u16>(target);
        let lanes_u32 = lanes_for::<u32>(target);
        if lanes_u16 < 2 {
            continue; // pairwise needs at least 2 narrow lanes
        }
        let input: Vec<u16> =
            (0..lanes_u16).map(|i| (i * 100 + 50) as u16).collect();
        let mut out = vec![0u32; lanes_u32];
        dispatch_to(
            SumsOf2U16Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes_u32 {
            let expected = input[2 * i] as u32 + input[2 * i + 1] as u32;
            assert_eq!(out[i], expected, "sums_of_2 lane {i} for {target:?}");
        }
    }
}

// =========================================================================
// SimdReduce: sums_of_2 (i16 -> i32)
// =========================================================================

struct SumsOf2I16Kernel<'a> {
    input: &'a [i16],
    out: &'a mut [i32],
}

impl WithSimd for SumsOf2I16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_i32 = s.lanes::<i32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let result = s.sums_of_2(v);
            for i in 0..lanes_i32.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_sums_of_2_i16() {
    for target in available_targets() {
        let lanes_i16 = lanes_for::<i16>(target);
        let lanes_i32 = lanes_for::<i32>(target);
        if lanes_i16 < 2 {
            continue;
        }
        let input: Vec<i16> =
            (0..lanes_i16).map(|i| i as i16 * 100 - 500).collect();
        let mut out = vec![0i32; lanes_i32];
        dispatch_to(
            SumsOf2I16Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes_i32 {
            let expected = input[2 * i] as i32 + input[2 * i + 1] as i32;
            assert_eq!(
                out[i], expected,
                "sums_of_2_i16 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdReduce: sums_of_2 (u8 -> u16)
// =========================================================================

struct SumsOf2U8Kernel<'a> {
    input: &'a [u8],
    out: &'a mut [u16],
}

impl WithSimd for SumsOf2U8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_u16 = s.lanes::<u16>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let result = s.sums_of_2(v);
            for i in 0..lanes_u16.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_sums_of_2_u8() {
    for target in available_targets() {
        let lanes_u8 = lanes_for::<u8>(target);
        let lanes_u16 = lanes_for::<u16>(target);
        if lanes_u8 < 2 {
            continue;
        }
        let input: Vec<u8> = (0..lanes_u8).map(|i| (i * 7 + 3) as u8).collect();
        let mut out = vec![0u16; lanes_u16];
        dispatch_to(
            SumsOf2U8Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes_u16 {
            let expected = input[2 * i] as u16 + input[2 * i + 1] as u16;
            assert_eq!(
                out[i], expected,
                "sums_of_2_u8 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// Cross-target consistency: all targets produce the same results
// =========================================================================

struct GatherConsistencyKernel<'a> {
    table: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for GatherConsistencyKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            // gather with iota indices: result should equal sequential load
            let idx = s.iota(0u32);
            let idx_i32: S::Vec<i32> = s.bitcast(idx);
            let result = s.gather_index(self.table.as_ptr(), idx_i32);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_gather_equals_sequential_load() {
    let table: Vec<u32> = (0..64).collect();
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let mut out = vec![0u32; lanes];
        dispatch_to(
            GatherConsistencyKernel {
                table: &table,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i], i as u32,
                "gather iota consistency lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// Scatter-gather roundtrip
// =========================================================================

struct ScatterGatherRoundtripKernel<'a> {
    values: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ScatterGatherRoundtripKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.values.as_ptr());
            // Scatter with iota indices, then gather back
            let idx = s.iota(0u32);
            let idx_i32: S::Vec<i32> = s.bitcast(idx);
            let mut buf = vec![0u32; lanes + 8]; // extra padding
            s.scatter_index(v, buf.as_mut_ptr(), idx_i32);
            let result = s.gather_index(buf.as_ptr(), idx_i32);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_scatter_gather_roundtrip() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let values: Vec<u32> = (0..lanes).map(|i| (i + 42) as u32).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            ScatterGatherRoundtripKernel {
                values: &values,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, values, "scatter-gather roundtrip for {target:?}");
    }
}

// =========================================================================
// slide_1_up and slide_1_down are inverses (mostly)
// =========================================================================

struct SlideUpDownKernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for SlideUpDownKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            // slide up then slide down: should recover original except lane 0 becomes 0
            let up = s.slide_1_up(v);
            let result = s.slide_1_down(up);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_slide_up_down_partial_inverse() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        if lanes < 2 {
            continue;
        }
        let input: Vec<u32> = (0..lanes).map(|i| (i + 1) as u32 * 10).collect();
        let mut out = vec![0u32; lanes];
        dispatch_to(
            SlideUpDownKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // After slide_up then slide_down: lanes 0..N-2 should recover original,
        // lane N-1 is zero (lost from slide_up)
        for i in 0..lanes - 1 {
            assert_eq!(
                out[i], input[i],
                "slide_up_down lane {i} for {target:?}"
            );
        }
        assert_eq!(out[lanes - 1], 0, "slide_up_down last lane for {target:?}");
    }
}

// =========================================================================
// ordered_demote_2_to: u32 -> u16 with large values
// =========================================================================

struct OrderedDemote2U32Kernel<'a> {
    lo: &'a [u32],
    hi: &'a [u32],
    out: &'a mut [u16],
}

impl WithSimd for OrderedDemote2U32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u32>();
        unsafe {
            let vlo = s.load_u(self.lo.as_ptr());
            let vhi = s.load_u(self.hi.as_ptr());
            let result: S::Vec<u16> = s.ordered_demote_2_to(vlo, vhi);
            for i in 0..(2 * lanes).min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_ordered_demote_2_u32_to_u16_saturation() {
    for target in available_targets() {
        let lanes = lanes_for::<u32>(target);
        let lanes_out = lanes_for::<u16>(target);
        if lanes < 2 {
            continue;
        }
        // Values above u16::MAX should saturate to 65535
        let mut lo: Vec<u32> = vec![0; lanes];
        lo[0] = 100;
        lo[1] = 0x80000000; // should saturate to 65535
        if lanes > 2 {
            lo[2] = 0xFFFFFFFF; // should saturate to 65535
        }
        if lanes > 3 {
            lo[3] = 65535; // exactly max, should stay 65535
        }
        let hi: Vec<u32> = vec![42; lanes];
        let mut out = vec![0u16; lanes_out];
        dispatch_to(
            OrderedDemote2U32Kernel {
                lo: &lo,
                hi: &hi,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out[0], 100, "u32->u16 lo[0] for {target:?}");
        assert_eq!(
            out[1], 65535,
            "u32->u16 lo[1]=0x80000000 should saturate for {target:?}"
        );
        if lanes > 2 {
            assert_eq!(
                out[2], 65535,
                "u32->u16 lo[2]=0xFFFFFFFF should saturate for {target:?}"
            );
        }
        if lanes > 3 {
            assert_eq!(out[3], 65535, "u32->u16 lo[3]=65535 for {target:?}");
        }
        for i in 0..lanes {
            assert_eq!(
                out[lanes + i],
                42,
                "u32->u16 hi lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// ordered_demote_2_to: u16 -> u8 with large values
// =========================================================================

struct OrderedDemote2U16Kernel<'a> {
    lo: &'a [u16],
    hi: &'a [u16],
    out: &'a mut [u8],
}

impl WithSimd for OrderedDemote2U16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u16>();
        unsafe {
            let vlo = s.load_u(self.lo.as_ptr());
            let vhi = s.load_u(self.hi.as_ptr());
            let result: S::Vec<u8> = s.ordered_demote_2_to(vlo, vhi);
            for i in 0..(2 * lanes).min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_ordered_demote_2_u16_to_u8_saturation() {
    for target in available_targets() {
        let lanes = lanes_for::<u16>(target);
        let lanes_out = lanes_for::<u8>(target);
        if lanes < 2 {
            continue;
        }
        let mut lo: Vec<u16> = vec![0; lanes];
        lo[0] = 50;
        lo[1] = 0x8000; // 32768, should saturate to 255
        if lanes > 2 {
            lo[2] = 0xFFFF; // should saturate to 255
        }
        if lanes > 3 {
            lo[3] = 255; // exactly max, should stay 255
        }
        let hi: Vec<u16> = vec![42; lanes];
        let mut out = vec![0u8; lanes_out];
        dispatch_to(
            OrderedDemote2U16Kernel {
                lo: &lo,
                hi: &hi,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out[0], 50, "u16->u8 lo[0] for {target:?}");
        assert_eq!(
            out[1], 255,
            "u16->u8 lo[1]=0x8000 should saturate for {target:?}"
        );
        if lanes > 2 {
            assert_eq!(
                out[2], 255,
                "u16->u8 lo[2]=0xFFFF should saturate for {target:?}"
            );
        }
        if lanes > 3 {
            assert_eq!(out[3], 255, "u16->u8 lo[3]=255 for {target:?}");
        }
        for i in 0..lanes {
            assert_eq!(
                out[lanes + i],
                42,
                "u16->u8 hi lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// ordered_demote_2_to: u64 -> u32 with saturation
// =========================================================================

struct OrderedDemote2U64Kernel<'a> {
    lo: &'a [u64],
    hi: &'a [u64],
    out: &'a mut [u32],
}

impl WithSimd for OrderedDemote2U64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u64>();
        unsafe {
            let vlo = s.load_u(self.lo.as_ptr());
            let vhi = s.load_u(self.hi.as_ptr());
            let result: S::Vec<u32> = s.ordered_demote_2_to(vlo, vhi);
            for i in 0..(2 * lanes).min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_ordered_demote_2_u64_to_u32_saturation() {
    for target in available_targets() {
        let lanes = lanes_for::<u64>(target);
        let lanes_out = lanes_for::<u32>(target);
        if lanes < 2 {
            continue;
        }
        let mut lo: Vec<u64> = vec![0; lanes];
        lo[0] = 100;
        lo[1] = 0x1_0000_0000; // > u32::MAX, should saturate to u32::MAX
        let mut hi: Vec<u64> = vec![0; lanes];
        hi[0] = 0xFFFF_FFFF_FFFF_FFFF; // should saturate to u32::MAX
        hi[1] = 42;
        let mut out = vec![0u32; lanes_out];
        dispatch_to(
            OrderedDemote2U64Kernel {
                lo: &lo,
                hi: &hi,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out[0], 100, "u64->u32 lo[0] for {target:?}");
        assert_eq!(
            out[1],
            u32::MAX,
            "u64->u32 lo[1] should saturate for {target:?}"
        );
        assert_eq!(
            out[lanes],
            u32::MAX,
            "u64->u32 hi[0] should saturate for {target:?}"
        );
        assert_eq!(out[lanes + 1], 42, "u64->u32 hi[1] for {target:?}");
    }
}

// =========================================================================
// ordered_demote_2_to: i64 -> i32 with saturation
// =========================================================================

struct OrderedDemote2I64Kernel<'a> {
    lo: &'a [i64],
    hi: &'a [i64],
    out: &'a mut [i32],
}

impl WithSimd for OrderedDemote2I64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<i64>();
        unsafe {
            let vlo = s.load_u(self.lo.as_ptr());
            let vhi = s.load_u(self.hi.as_ptr());
            let result: S::Vec<i32> = s.ordered_demote_2_to(vlo, vhi);
            for i in 0..(2 * lanes).min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_ordered_demote_2_i64_to_i32_saturation() {
    for target in available_targets() {
        let lanes = lanes_for::<i64>(target);
        let lanes_out = lanes_for::<i32>(target);
        if lanes < 2 {
            continue;
        }
        let mut lo: Vec<i64> = vec![0; lanes];
        lo[0] = 100;
        lo[1] = 3_000_000_000; // > i32::MAX, should saturate to i32::MAX
        let mut hi: Vec<i64> = vec![0; lanes];
        hi[0] = -3_000_000_000; // < i32::MIN, should saturate to i32::MIN
        hi[1] = -42;
        let mut out = vec![0i32; lanes_out];
        dispatch_to(
            OrderedDemote2I64Kernel {
                lo: &lo,
                hi: &hi,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out[0], 100, "i64->i32 lo[0] for {target:?}");
        assert_eq!(
            out[1],
            i32::MAX,
            "i64->i32 lo[1] should saturate to MAX for {target:?}"
        );
        assert_eq!(
            out[lanes],
            i32::MIN,
            "i64->i32 hi[0] should saturate to MIN for {target:?}"
        );
        assert_eq!(out[lanes + 1], -42, "i64->i32 hi[1] for {target:?}");
    }
}

// =========================================================================
// SimdConvert: nearest_int (f32 -> i32)
// =========================================================================

struct NearestIntF32Kernel<'a> {
    input: &'a [f32],
    out: &'a mut [i32],
}

impl WithSimd for NearestIntF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<f32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.nearest_int(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_nearest_int_f32() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        // Build input with various rounding cases
        let mut input = vec![0.0f32; lanes];
        let expected: Vec<i32>;
        if lanes >= 4 {
            input[0] = 1.4; // rounds to 1
            input[1] = 1.5; // rounds to 2 (ties-to-even)
            input[2] = 2.5; // rounds to 2 (ties-to-even)
            input[3] = -1.6; // rounds to -2
            expected = vec![1, 2, 2, -2];
        } else {
            input[0] = 1.5;
            expected = vec![2];
        }
        let mut out = vec![0i32; lanes];
        dispatch_to(
            NearestIntF32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..expected.len() {
            assert_eq!(
                out[i], expected[i],
                "nearest_int f32 lane {i} for {target:?}"
            );
        }
    }
}

#[test]
fn test_nearest_int_f32_overflow() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let mut input = vec![0.0f32; lanes];
        input[0] = 3.0e9; // > i32::MAX
        let mut out = vec![0i32; lanes];
        dispatch_to(
            NearestIntF32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out[0], i32::MAX, "nearest_int overflow for {target:?}");
    }
}

// =========================================================================
// SimdShuffle: combine_shift_right_bytes (SSE2 skip for scalar)
// =========================================================================

struct CombineShiftRightBytesKernel<'a> {
    hi: &'a [u8],
    lo: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for CombineShiftRightBytesKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<u8>();
        if lanes < 2 {
            return; // skip scalar
        }
        unsafe {
            let hi_v = s.load_u(self.hi.as_ptr());
            let lo_v = s.load_u(self.lo.as_ptr());
            let result: S::Vec<u8> =
                s.combine_shift_right_bytes::<u8, 4>(hi_v, lo_v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(result, i);
            }
        }
    }
}

#[test]
fn test_combine_shift_right_bytes() {
    for target in available_targets() {
        let lanes = lanes_for::<u8>(target);
        if lanes < 2 {
            continue; // skip scalar
        }
        // Fill lo and hi with sequential values per 128-bit block.
        let mut lo = vec![0u8; lanes];
        let mut hi = vec![0u8; lanes];
        for block in 0..(lanes / 16) {
            for i in 0..16 {
                lo[block * 16 + i] = i as u8; // 0..15
                hi[block * 16 + i] = (16 + i) as u8; // 16..31
            }
        }
        let mut out = vec![0u8; lanes];
        dispatch_to(
            CombineShiftRightBytesKernel {
                hi: &hi,
                lo: &lo,
                out: &mut out,
            },
            target,
        );
        // Shift right by 4: per 128-bit block, result[i] = concat(hi,lo)[i+4] for i < 16.
        // So result[0..12] = lo[4..16], result[12..16] = hi[0..4].
        for block in 0..(lanes / 16) {
            let base = block * 16;
            for i in 0..12 {
                assert_eq!(
                    out[base + i],
                    (i + 4) as u8,
                    "combine_shift_right lane {} block {} for {:?}",
                    i,
                    block,
                    target,
                );
            }
            for i in 12..16 {
                assert_eq!(
                    out[base + i],
                    (16 + i - 12) as u8,
                    "combine_shift_right lane {} block {} for {:?}",
                    i,
                    block,
                    target,
                );
            }
        }
    }
}

// =========================================================================
// SimdFloat: zero_if_negative
// =========================================================================

struct ZeroIfNegativeF32Kernel<'a> {
    input: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for ZeroIfNegativeF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<f32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.zero_if_negative(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_zero_if_negative_f32() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let mut input = vec![0.0f32; lanes];
        let mut expected = vec![0.0f32; lanes];
        if lanes >= 4 {
            input[0] = 5.0;
            expected[0] = 5.0;
            input[1] = -3.0;
            expected[1] = 0.0;
            input[2] = 0.0;
            expected[2] = 0.0;
            input[3] = -0.0;
            expected[3] = 0.0; // -0 has sign bit set
        } else {
            input[0] = -5.0;
            expected[0] = 0.0;
        }
        let mut out = vec![0.0f32; lanes];
        dispatch_to(
            ZeroIfNegativeF32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i].to_bits(),
                expected[i].to_bits(),
                "zero_if_negative lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdFloat: is_finite
// =========================================================================

struct IsFiniteF32Kernel<'a> {
    input: &'a [f32],
    out: &'a mut [bool],
}

impl WithSimd for IsFiniteF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<f32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let mask = s.is_finite(v);
            for i in 0..lanes.min(self.out.len()) {
                let bits = s.bits_from_mask(mask);
                self.out[i] = (bits >> i) & 1 != 0;
            }
        }
    }
}

#[test]
fn test_is_finite_f32() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let mut input = vec![1.0f32; lanes];
        let mut expected = vec![true; lanes];
        if lanes >= 4 {
            input[0] = 42.0;
            expected[0] = true;
            input[1] = f32::INFINITY;
            expected[1] = false;
            input[2] = f32::NAN;
            expected[2] = false;
            input[3] = f32::NEG_INFINITY;
            expected[3] = false;
        }
        let mut out = vec![false; lanes];
        dispatch_to(
            IsFiniteF32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i], expected[i],
                "is_finite f32 lane {i} for {target:?}: input={}",
                input[i]
            );
        }
    }
}

// =========================================================================
// SimdFloat: add_sub
// =========================================================================

struct AddSubF32Kernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for AddSubF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<f32>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.add_sub(va, vb);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_add_sub_f32() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let a = vec![10.0f32; lanes];
        let b = vec![3.0f32; lanes];
        let mut out = vec![0.0f32; lanes];
        dispatch_to(
            AddSubF32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = if i % 2 == 0 { 10.0 - 3.0 } else { 10.0 + 3.0 };
            assert_eq!(out[i], expected, "add_sub f32 lane {i} for {target:?}");
        }
    }
}

// =========================================================================
// SimdReduce: sums_of_4 (u8 -> u32)
// =========================================================================

struct SumsOf4U8Kernel<'a> {
    input: &'a [u8],
    out: &'a mut [u32],
}

impl WithSimd for SumsOf4U8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let _lanes_in = s.lanes::<u8>();
        let lanes_out = s.lanes::<u32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.sums_of_4(v);
            for i in 0..lanes_out.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_sums_of_4_u8() {
    for target in available_targets() {
        let lanes_in = lanes_for::<u8>(target);
        let lanes_out = lanes_for::<u32>(target);
        // Fill with sequential values
        let input: Vec<u8> =
            (0..lanes_in).map(|i| (i as u8).wrapping_add(1)).collect();
        let mut out = vec![0u32; lanes_out];
        dispatch_to(
            SumsOf4U8Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // Each output lane = sum of 4 adjacent input lanes
        for i in 0..lanes_out {
            let base = i * 4;
            if base + 3 < lanes_in {
                let expected = input[base] as u32
                    + input[base + 1] as u32
                    + input[base + 2] as u32
                    + input[base + 3] as u32;
                assert_eq!(
                    out[i], expected,
                    "sums_of_4 u8 lane {i} for {target:?}"
                );
            }
        }
    }
}

// =========================================================================
// SimdShuffle: compress_blended_store
// =========================================================================

struct CompressBlendedStoreKernel<'a> {
    input: &'a [i32],
    mask_bits: u64,
    buf: &'a mut [i32],
}

impl WithSimd for CompressBlendedStoreKernel<'_> {
    type Output = usize;
    fn with_simd<S: SimdOps>(self, s: S) -> usize {
        let lanes = s.lanes::<i32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            // Build mask from bits
            let mut mask_arr = vec![0i32; lanes];
            for i in 0..lanes {
                if (self.mask_bits >> i) & 1 != 0 {
                    mask_arr[i] = -1; // all-ones
                }
            }
            let mask_vec: S::Vec<i32> = s.load_u(mask_arr.as_ptr());
            let mask = s.mask_from_vec(mask_vec);
            s.compress_blended_store(v, mask, self.buf.as_mut_ptr())
        }
    }
}

#[test]
fn test_compress_blended_store() {
    for target in available_targets() {
        let lanes = lanes_for::<i32>(target);
        let input: Vec<i32> = (0..lanes).map(|i| (i * 10 + 1) as i32).collect();
        // Select even lanes
        let mask_bits: u64 = (0..lanes as u64)
            .filter(|i| i % 2 == 0)
            .fold(0u64, |acc, i| acc | (1 << i));
        // Pre-fill buffer with sentinel values
        let mut buf = vec![-999i32; lanes];
        let count = dispatch_to(
            CompressBlendedStoreKernel {
                input: &input,
                mask_bits,
                buf: &mut buf,
            },
            target,
        );
        let expected_count = (0..lanes).filter(|i| i % 2 == 0).count();
        assert_eq!(
            count, expected_count,
            "compress_blended_store count for {target:?}"
        );
        // First `count` elements should be the compressed values
        let mut j = 0;
        for i in 0..lanes {
            if i % 2 == 0 {
                assert_eq!(
                    buf[j], input[i],
                    "compress_blended_store lane {j} for {target:?}"
                );
                j += 1;
            }
        }
        // Remaining should be sentinel (untouched)
        for i in count..lanes {
            assert_eq!(
                buf[i], -999,
                "compress_blended_store should not touch lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdShuffle: odd_even_blocks
// =========================================================================

struct OddEvenBlocksKernel<'a> {
    odd: &'a [i32],
    even: &'a [i32],
    out: &'a mut [i32],
}

impl WithSimd for OddEvenBlocksKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<i32>();
        unsafe {
            let odd_v = s.load_u(self.odd.as_ptr());
            let even_v = s.load_u(self.even.as_ptr());
            let r = s.odd_even_blocks(odd_v, even_v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_odd_even_blocks() {
    for target in available_targets() {
        let lanes = lanes_for::<i32>(target);
        let odd: Vec<i32> = (0..lanes).map(|i| 100 + i as i32).collect();
        let even: Vec<i32> = (0..lanes).map(|i| 200 + i as i32).collect();
        let mut out = vec![0i32; lanes];
        dispatch_to(
            OddEvenBlocksKernel {
                odd: &odd,
                even: &even,
                out: &mut out,
            },
            target,
        );
        // i32 lanes per 128-bit block = 4
        let lanes_per_block = 4;
        for i in 0..lanes {
            let block = i / lanes_per_block;
            let expected = if block % 2 == 0 {
                even[i] // even block from `even`
            } else {
                odd[i] // odd block from `odd`
            };
            assert_eq!(
                out[i], expected,
                "odd_even_blocks lane {i} (block {block}) for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdShuffle: reverse_blocks
// =========================================================================

struct ReverseBlocksKernel<'a> {
    input: &'a [i32],
    out: &'a mut [i32],
}

impl WithSimd for ReverseBlocksKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<i32>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.reverse_blocks(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_reverse_blocks() {
    for target in available_targets() {
        let lanes = lanes_for::<i32>(target);
        let input: Vec<i32> = (0..lanes).map(|i| i as i32).collect();
        let mut out = vec![0i32; lanes];
        dispatch_to(
            ReverseBlocksKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let lanes_per_block = 4; // 128 bits / 32 bits
        let num_blocks = lanes / lanes_per_block;
        for block in 0..num_blocks {
            let reversed_block = num_blocks - 1 - block;
            for j in 0..lanes_per_block {
                let out_idx = block * lanes_per_block + j;
                let in_idx = reversed_block * lanes_per_block + j;
                assert_eq!(
                    out[out_idx], input[in_idx],
                    "reverse_blocks lane {out_idx} for {target:?}"
                );
            }
        }
    }
}

// =========================================================================
// SimdFloat: add_sub f64
// =========================================================================

struct AddSubF64Kernel<'a> {
    a: &'a [f64],
    b: &'a [f64],
    out: &'a mut [f64],
}

impl WithSimd for AddSubF64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<f64>();
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let r = s.add_sub(va, vb);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_add_sub_f64() {
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let a = vec![20.0f64; lanes];
        let b = vec![7.0f64; lanes];
        let mut out = vec![0.0f64; lanes];
        dispatch_to(
            AddSubF64Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            let expected = if i % 2 == 0 { 20.0 - 7.0 } else { 20.0 + 7.0 };
            assert_eq!(out[i], expected, "add_sub f64 lane {i} for {target:?}");
        }
    }
}

// =========================================================================
// SimdFloat: zero_if_negative f64
// =========================================================================

struct ZeroIfNegativeF64Kernel<'a> {
    input: &'a [f64],
    out: &'a mut [f64],
}

impl WithSimd for ZeroIfNegativeF64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<f64>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.zero_if_negative(v);
            for i in 0..lanes.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_zero_if_negative_f64() {
    for target in available_targets() {
        let lanes = lanes_for::<f64>(target);
        let mut input = vec![0.0f64; lanes];
        let mut expected = vec![0.0f64; lanes];
        if lanes >= 2 {
            input[0] = 5.0;
            expected[0] = 5.0;
            input[1] = -3.0;
            expected[1] = 0.0;
        } else {
            input[0] = -5.0;
            expected[0] = 0.0;
        }
        let mut out = vec![0.0f64; lanes];
        dispatch_to(
            ZeroIfNegativeF64Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i].to_bits(),
                expected[i].to_bits(),
                "zero_if_negative f64 lane {i} for {target:?}"
            );
        }
    }
}

// =========================================================================
// SimdReduce: sums_of_4 (i16 -> i64)
// =========================================================================

struct SumsOf4I16Kernel<'a> {
    input: &'a [i16],
    out: &'a mut [i64],
}

impl WithSimd for SumsOf4I16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes_out = s.lanes::<i64>();
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.sums_of_4(v);
            for i in 0..lanes_out.min(self.out.len()) {
                self.out[i] = s.extract_lane(r, i);
            }
        }
    }
}

#[test]
fn test_sums_of_4_i16() {
    for target in available_targets() {
        let lanes_in = lanes_for::<i16>(target);
        let lanes_out = lanes_for::<i64>(target);
        let input: Vec<i16> = (0..lanes_in).map(|i| (i as i16) - 5).collect();
        let mut out = vec![0i64; lanes_out];
        dispatch_to(
            SumsOf4I16Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes_out {
            let base = i * 4;
            if base + 3 < lanes_in {
                let expected = input[base] as i64
                    + input[base + 1] as i64
                    + input[base + 2] as i64
                    + input[base + 3] as i64;
                assert_eq!(
                    out[i], expected,
                    "sums_of_4 i16 lane {i} for {target:?}"
                );
            }
        }
    }
}

// =========================================================================
// Group 9: lower_half / upper_half / combine
// =========================================================================

struct LowerUpperCombineKernel;
impl WithSimd for LowerUpperCombineKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let mut data = vec![0u32; lanes];
            for i in 0..lanes {
                data[i] = i as u32;
            }
            let v = s.load_u(data.as_ptr());
            let lo = s.lower_half::<u32>(v);
            let hi = s.upper_half::<u32>(v);
            let combined = s.combine::<u32>(lo, hi);
            for i in 0..lanes {
                if s.extract_lane(combined, i) != i as u32 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_lower_upper_combine() {
    for target in available_targets() {
        assert!(
            dispatch_to(LowerUpperCombineKernel, target),
            "lower_half/upper_half/combine failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 9: insert_block / extract_block
// =========================================================================

struct InsertExtractBlockKernel;
impl WithSimd for InsertExtractBlockKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let mut data = vec![0u32; lanes];
            for i in 0..lanes {
                data[i] = i as u32;
            }
            let v = s.load_u(data.as_ptr());

            // extract_block<0> should match lower_half
            let blk0 = s.extract_block::<u32, 0>(v);
            let lo = s.lower_half::<u32>(v);
            let from_blk =
                s.combine::<u32>(blk0, s.lower_half(s.zero::<u32>()));
            let from_lo = s.combine::<u32>(lo, s.lower_half(s.zero::<u32>()));
            let half_lanes = (lanes / 2).max(1);
            for i in 0..half_lanes.min(lanes) {
                if s.extract_lane(from_blk, i) != s.extract_lane(from_lo, i) {
                    return false;
                }
            }

            // insert_block<0> with all-ones half
            let all_ones = s.splat::<u32>(0xFFFF_FFFF);
            let lo_half = s.lower_half::<u32>(all_ones);
            let inserted = s.insert_block::<u32, 0>(v, lo_half);
            for i in 0..half_lanes.min(lanes) {
                if s.extract_lane(inserted, i) != 0xFFFF_FFFF {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_insert_extract_block() {
    for target in available_targets() {
        assert!(
            dispatch_to(InsertExtractBlockKernel, target),
            "insert_block/extract_block failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 9: promote_lower_to / promote_upper_to
// =========================================================================

struct PromoteLowerUpperKernel;
impl WithSimd for PromoteLowerUpperKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes_narrow = s.lanes::<u16>();
            let mut data = vec![0u16; lanes_narrow];
            for i in 0..lanes_narrow {
                data[i] = (i + 1) as u16;
            }
            let v = s.load_u(data.as_ptr());
            let lanes_wide = s.lanes::<u32>();

            let promoted_lo = s.promote_lower_to::<u16>(v);
            for i in 0..lanes_wide {
                let val: u32 = s.extract_lane(promoted_lo, i);
                if val != (i + 1) as u32 {
                    return false;
                }
            }

            let promoted_hi = s.promote_upper_to::<u16>(v);
            for i in 0..lanes_wide {
                let expected = if lanes_narrow > lanes_wide {
                    (lanes_wide + i + 1) as u32
                } else {
                    0u32
                };
                let val: u32 = s.extract_lane(promoted_hi, i);
                if val != expected {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_promote_lower_upper() {
    for target in available_targets() {
        assert!(
            dispatch_to(PromoteLowerUpperKernel, target),
            "promote_lower_to/promote_upper_to failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 1: compress_not
// =========================================================================

struct CompressNotKernel;
impl WithSimd for CompressNotKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let mut data = vec![0u32; lanes];
            for i in 0..lanes {
                data[i] = (i + 1) as u32;
            }
            let v = s.load_u(data.as_ptr());

            // Mask: first lane true, rest false
            let mut mask_data = vec![0u32; lanes];
            mask_data[0] = 0xFFFF_FFFF;
            let mask_vec = s.load_u(mask_data.as_ptr());
            let mask = s.mask_from_vec(mask_vec);

            // compress_not keeps where mask is FALSE
            let result = s.compress_not(v, mask);
            for i in 0..lanes - 1 {
                let val: u32 = s.extract_lane(result, i);
                if val != (i + 2) as u32 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_compress_not() {
    for target in available_targets() {
        assert!(
            dispatch_to(CompressNotKernel, target),
            "compress_not failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 1: exclusive_neither (XNOR)
// =========================================================================

struct ExclusiveNeitherKernel;
impl WithSimd for ExclusiveNeitherKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let all_true = s.mask_from_vec(s.splat::<u32>(0xFFFF_FFFF));
            let all_false = s.mask_from_vec(s.zero::<u32>());

            // NOR semantics ("neither a nor b"):
            // Both true -> false
            if s.count_true(s.exclusive_neither(all_true, all_true)) != 0 {
                return false;
            }
            // Both false -> true
            if s.count_true(s.exclusive_neither(all_false, all_false)) != lanes
            {
                return false;
            }
            // Mixed -> false
            if s.count_true(s.exclusive_neither(all_true, all_false)) != 0 {
                return false;
            }
            // Per-lane: NOR true only where both lanes are false
            if lanes >= 2 {
                let mut ma = vec![0u32; lanes];
                let mut mb = vec![0u32; lanes];
                for i in 0..lanes {
                    ma[i] = if i % 2 == 0 { 0xFFFF_FFFF } else { 0 };
                    mb[i] = if i % 4 == 0 { 0xFFFF_FFFF } else { 0 };
                }
                let mka = s.mask_from_vec(s.load_u(ma.as_ptr()));
                let mkb = s.mask_from_vec(s.load_u(mb.as_ptr()));
                let vr = s.vec_from_mask::<u32>(s.exclusive_neither(mka, mkb));
                for i in 0..lanes {
                    let expected = (ma[i] == 0) && (mb[i] == 0);
                    let got: u32 = s.extract_lane(vr, i);
                    if (got != 0) != expected {
                        return false;
                    }
                }
            }
            true
        }
    }
}

#[test]
fn test_exclusive_neither() {
    for target in available_targets() {
        assert!(
            dispatch_to(ExclusiveNeitherKernel, target),
            "exclusive_neither failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 1: slide_mask_1_up / slide_mask_1_down
// =========================================================================

struct SlideMaskKernel;
impl WithSimd for SlideMaskKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let first_n = s.first_n::<u32>(1);

            let slid_up = s.slide_mask_1_up(first_n);
            if lanes > 1 {
                if s.count_true(slid_up) != 1 {
                    return false;
                }
                let vec = s.vec_from_mask::<u32>(slid_up);
                let lane0: u32 = s.extract_lane(vec, 0);
                let lane1: u32 = s.extract_lane(vec, 1);
                if lane0 != 0 || lane1 == 0 {
                    return false;
                }
            } else if s.count_true(slid_up) != 0 {
                return false;
            }

            // slide_mask_1_down with last lane set
            let last_n = s.first_n::<u32>(lanes);
            let almost_all = s.first_n::<u32>(lanes - 1);
            let last_only = s.xor_mask(last_n, almost_all);
            let slid_down = s.slide_mask_1_down(last_only);
            if lanes > 1 {
                if s.count_true(slid_down) != 1 {
                    return false;
                }
                let vec = s.vec_from_mask::<u32>(slid_down);
                let second_to_last: u32 = s.extract_lane(vec, lanes - 2);
                let last: u32 = s.extract_lane(vec, lanes - 1);
                if second_to_last == 0 || last != 0 {
                    return false;
                }
            } else if s.count_true(slid_down) != 0 {
                return false;
            }
            true
        }
    }
}

#[test]
fn test_slide_mask() {
    for target in available_targets() {
        assert!(
            dispatch_to(SlideMaskKernel, target),
            "slide_mask_1_up/down failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 2: broadcast_block
// =========================================================================

struct BroadcastBlockKernel;
impl WithSimd for BroadcastBlockKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let mut data = vec![0u32; lanes];
            for i in 0..lanes {
                data[i] = i as u32;
            }
            let v = s.load_u(data.as_ptr());
            let result = s.broadcast_block::<u32, 0>(v);
            let lanes_per_block = 4; // 128 bits / 32 bits
            for i in 0..lanes {
                let expected = (i % lanes_per_block) as u32;
                let val: u32 = s.extract_lane(result, i);
                if val != expected {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_broadcast_block() {
    for target in available_targets() {
        assert!(
            dispatch_to(BroadcastBlockKernel, target),
            "broadcast_block failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 2: compress_bits / compress_bits_store
// =========================================================================

struct CompressBitsKernel;
impl WithSimd for CompressBitsKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let mut data = vec![0u32; lanes];
            for i in 0..lanes {
                data[i] = (i + 1) as u32;
            }
            let v = s.load_u(data.as_ptr());

            // Select even-indexed lanes: 0x55 = 0b01010101
            let bits: [u8; 8] = [0x55; 8];
            let result = s.compress_bits(v, bits.as_ptr());
            let expected_count = (lanes + 1) / 2;
            for i in 0..expected_count {
                let val: u32 = s.extract_lane(result, i);
                if val != (2 * i + 1) as u32 {
                    return false;
                }
            }

            let mut output = vec![0u32; lanes];
            let count =
                s.compress_bits_store(v, bits.as_ptr(), output.as_mut_ptr());
            if count != expected_count {
                return false;
            }
            for i in 0..count {
                if output[i] != (2 * i + 1) as u32 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_compress_bits() {
    for target in available_targets() {
        assert!(
            dispatch_to(CompressBitsKernel, target),
            "compress_bits failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 2: load_expand
// =========================================================================

struct LoadExpandKernel;
impl WithSimd for LoadExpandKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let src: Vec<u32> = (1..=(lanes as u32)).collect();

            let mut mask_data = vec![0u32; lanes];
            for i in (0..lanes).step_by(2) {
                mask_data[i] = 0xFFFF_FFFF;
            }
            let mask_vec = s.load_u(mask_data.as_ptr());
            let mask = s.mask_from_vec(mask_vec);
            let result = s.load_expand(mask, src.as_ptr());

            let mut src_idx = 0;
            for i in 0..lanes {
                let val: u32 = s.extract_lane(result, i);
                if i % 2 == 0 {
                    if val != src[src_idx] {
                        return false;
                    }
                    src_idx += 1;
                } else if val != 0 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_load_expand() {
    for target in available_targets() {
        assert!(
            dispatch_to(LoadExpandKernel, target),
            "load_expand failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 3: min_number / max_number
// =========================================================================

struct MinMaxNumberKernel;
impl WithSimd for MinMaxNumberKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<f32>();
            let a = s.splat::<f32>(f32::NAN);
            let b = s.splat::<f32>(5.0f32);

            // NaN in a -> return b
            let min_result = s.min_number(a, b);
            let max_result = s.max_number(a, b);
            for i in 0..lanes {
                let v1: f32 = s.extract_lane(min_result, i);
                let v2: f32 = s.extract_lane(max_result, i);
                if v1 != 5.0 || v2 != 5.0 {
                    return false;
                }
            }

            // Normal: min(3,7)=3, max(3,7)=7
            let c = s.splat::<f32>(3.0f32);
            let d = s.splat::<f32>(7.0f32);
            let min_cd = s.min_number(c, d);
            let max_cd = s.max_number(c, d);
            for i in 0..lanes {
                let v1: f32 = s.extract_lane(min_cd, i);
                let v2: f32 = s.extract_lane(max_cd, i);
                if v1 != 3.0 || v2 != 7.0 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_min_max_number() {
    for target in available_targets() {
        assert!(
            dispatch_to(MinMaxNumberKernel, target),
            "min_number/max_number failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 3: min_magnitude / max_magnitude
// =========================================================================

struct MinMaxMagnitudeKernel;
impl WithSimd for MinMaxMagnitudeKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<f32>();
            let a = s.splat::<f32>(-3.0f32);
            let b = s.splat::<f32>(5.0f32);

            let min_mag = s.min_magnitude(a, b);
            let max_mag = s.max_magnitude(a, b);
            for i in 0..lanes {
                let v1: f32 = s.extract_lane(min_mag, i);
                let v2: f32 = s.extract_lane(max_mag, i);
                if v1 != -3.0 || v2 != 5.0 {
                    return false;
                }
            }

            // Equal magnitude: min(-4,4)=-4, max(-4,4)=4
            let c = s.splat::<f32>(-4.0f32);
            let d = s.splat::<f32>(4.0f32);
            let min_eq = s.min_magnitude(c, d);
            let max_eq = s.max_magnitude(c, d);
            for i in 0..lanes {
                let v1: f32 = s.extract_lane(min_eq, i);
                let v2: f32 = s.extract_lane(max_eq, i);
                if v1 != -4.0 || v2 != 4.0 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_min_max_magnitude() {
    for target in available_targets() {
        assert!(
            dispatch_to(MinMaxMagnitudeKernel, target),
            "min_magnitude/max_magnitude failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 4: interleave_even / interleave_odd
// =========================================================================

struct InterleaveEvenOddKernel;
impl WithSimd for InterleaveEvenOddKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            if lanes < 2 {
                return true;
            }
            let mut data_a = vec![0u32; lanes];
            let mut data_b = vec![0u32; lanes];
            for i in 0..lanes {
                data_a[i] = (i * 2) as u32;
                data_b[i] = (i * 2 + 1) as u32;
            }
            let a = s.load_u(data_a.as_ptr());
            let b = s.load_u(data_b.as_ptr());

            let ie = s.interleave_even(a, b);
            let io = s.interleave_odd(a, b);

            let half = lanes / 2;
            for i in 0..half {
                let va: u32 = s.extract_lane(ie, i * 2);
                let vb: u32 = s.extract_lane(ie, i * 2 + 1);
                if va != data_a[i * 2] || vb != data_b[i * 2] {
                    return false;
                }
            }
            for i in 0..half {
                let va: u32 = s.extract_lane(io, i * 2);
                let vb: u32 = s.extract_lane(io, i * 2 + 1);
                if va != data_a[i * 2 + 1] || vb != data_b[i * 2 + 1] {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_interleave_even_odd() {
    for target in available_targets() {
        assert!(
            dispatch_to(InterleaveEvenOddKernel, target),
            "interleave_even/odd failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 5: two_tables_lookup_lanes
// =========================================================================

struct TwoTablesLookupKernel;
impl WithSimd for TwoTablesLookupKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let mut data_a = vec![0u32; lanes];
            let mut data_b = vec![0u32; lanes];
            for i in 0..lanes {
                data_a[i] = 100 + i as u32;
                data_b[i] = 200 + i as u32;
            }
            let a = s.load_u(data_a.as_ptr());
            let b = s.load_u(data_b.as_ptr());

            // Indices: reverse from a for first half, reverse from b for second half
            let mut idx_data = vec![0u32; lanes];
            for i in 0..lanes {
                if i < lanes / 2 {
                    idx_data[i] = (lanes / 2 - 1 - i) as u32;
                } else {
                    idx_data[i] = (lanes + lanes - 1 - i) as u32;
                }
            }
            let idx = s.load_u(idx_data.as_ptr());
            let result = s.two_tables_lookup_lanes::<u32, u32>(a, b, idx);

            for i in 0..lanes {
                let val: u32 = s.extract_lane(result, i);
                let expected_idx = idx_data[i] as usize;
                let expected = if expected_idx < lanes {
                    data_a[expected_idx]
                } else {
                    data_b[expected_idx - lanes]
                };
                if val != expected {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_two_tables_lookup_lanes() {
    for target in available_targets() {
        assert!(
            dispatch_to(TwoTablesLookupKernel, target),
            "two_tables_lookup_lanes failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 5: table_lookup_lanes_or0
// =========================================================================

struct TableLookupOr0Kernel;
impl WithSimd for TableLookupOr0Kernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let mut data = vec![0u32; lanes];
            for i in 0..lanes {
                data[i] = (i + 1) as u32;
            }
            let v = s.load_u(data.as_ptr());

            let mut idx_data = vec![0u32; lanes];
            for i in 0..lanes {
                if i % 2 == 0 {
                    idx_data[i] = 0;
                } else {
                    idx_data[i] = 0x8000_0000;
                } // negative -> zero
            }
            let idx = s.load_u(idx_data.as_ptr());
            let result = s.table_lookup_lanes_or0::<u32, u32>(v, idx);

            for i in 0..lanes {
                let val: u32 = s.extract_lane(result, i);
                if i % 2 == 0 {
                    if val != 1 {
                        return false;
                    }
                } else if val != 0 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_table_lookup_lanes_or0() {
    for target in available_targets() {
        assert!(
            dispatch_to(TableLookupOr0Kernel, target),
            "table_lookup_lanes_or0 failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 6: reorder_demote_2_to
// =========================================================================

struct ReorderDemote2Kernel;
impl WithSimd for ReorderDemote2Kernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes_wide = s.lanes::<i32>();
            let lanes_narrow = s.lanes::<i16>();

            let mut data_a = vec![0i32; lanes_wide];
            let mut data_b = vec![0i32; lanes_wide];
            for i in 0..lanes_wide {
                data_a[i] = (i as i32) + 1;
                data_b[i] = (i as i32) + 100;
            }
            let a = s.load_u(data_a.as_ptr());
            let b = s.load_u(data_b.as_ptr());
            let result = s.reorder_demote_2_to::<i32>(a, b);

            let mut result_vals = vec![0i16; lanes_narrow];
            for i in 0..lanes_narrow {
                result_vals[i] = s.extract_lane(result, i);
            }

            // For scalar (lanes_narrow==1): only data_a[0] fits in the output
            // For SIMD: all 2*lanes_wide values from a and b must appear
            if lanes_narrow >= 2 * lanes_wide {
                let mut expected: Vec<i16> = Vec::new();
                for i in 0..lanes_wide {
                    expected.push(data_a[i] as i16);
                }
                for i in 0..lanes_wide {
                    expected.push(data_b[i] as i16);
                }
                for &e in &expected {
                    if !result_vals.contains(&e) {
                        return false;
                    }
                }
            } else {
                // Scalar: just verify the single output is data_a[0] demoted
                if result_vals[0] != data_a[0] as i16 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_reorder_demote_2_to() {
    for target in available_targets() {
        assert!(
            dispatch_to(ReorderDemote2Kernel, target),
            "reorder_demote_2_to failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 6: demote_in_range_to
// =========================================================================

struct DemoteInRangeKernel;
impl WithSimd for DemoteInRangeKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<i32>();
            let mut data = vec![0i32; lanes];
            for i in 0..lanes {
                data[i] = (i as i32) * 100 - 500;
            }
            let v = s.load_u(data.as_ptr());
            let result = s.demote_in_range_to::<i32>(v);
            let lanes_narrow = s.lanes::<i16>();
            let check_lanes = lanes.min(lanes_narrow);
            for i in 0..check_lanes {
                let val: i16 = s.extract_lane(result, i);
                if val != data[i] as i16 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_demote_in_range_to() {
    for target in available_targets() {
        assert!(
            dispatch_to(DemoteInRangeKernel, target),
            "demote_in_range_to failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 6: convert_in_range_to_int
// =========================================================================

struct ConvertInRangeKernel;
impl WithSimd for ConvertInRangeKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<f32>();
            let mut data = vec![0f32; lanes];
            for i in 0..lanes {
                data[i] = (i as f32) * 10.0 - 50.0;
            }
            let v = s.load_u(data.as_ptr());
            let result = s.convert_in_range_to_int(v);
            for i in 0..lanes {
                let val: i32 = s.extract_lane(result, i);
                if val != data[i] as i32 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_convert_in_range_to_int() {
    for target in available_targets() {
        assert!(
            dispatch_to(ConvertInRangeKernel, target),
            "convert_in_range_to_int failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Group 7: ror (variable rotate right)
// =========================================================================

struct RorKernel;
impl WithSimd for RorKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let data = vec![0x8000_0001u32; lanes];
            let mut shift = vec![0u32; lanes];
            for i in 0..lanes {
                shift[i] = (i as u32) % 32;
            }
            let v = s.load_u(data.as_ptr());
            let sh = s.load_u(shift.as_ptr());
            let result = s.ror(v, sh);
            for i in 0..lanes {
                let val: u32 = s.extract_lane(result, i);
                let expected = 0x8000_0001u32.rotate_right((i as u32) % 32);
                if val != expected {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_ror() {
    for target in available_targets() {
        assert!(dispatch_to(RorKernel, target), "ror failed on {:?}", target);
    }
}

// =========================================================================
// Group 8: aes_key_gen_assist
// =========================================================================

fn simd_targets() -> Vec<TargetId> {
    let mut targets = Vec::new();
    let best = highway::dispatch::detect_best_target();
    match best {
        TargetId::Avx512 => {
            targets.push(TargetId::Sse2);
            targets.push(TargetId::Avx2);
            targets.push(TargetId::Avx512);
        }
        TargetId::Avx2 => {
            targets.push(TargetId::Sse2);
            targets.push(TargetId::Avx2);
        }
        TargetId::Sse2 => {
            targets.push(TargetId::Sse2);
        }
        _ => {}
    }
    targets
}

unsafe fn test_aes_key_gen_assist_on<S: SimdOps + SimdCrypto>(s: S) -> bool {
    unsafe {
        let input = s.zero::<u8>();
        let result = s.aes_key_gen_assist::<0x01>(input);
        // S-box[0] = 0x63; RotWord(63,63,63,63) XOR (01,00,00,00) = (62,63,63,63)
        let expected: [u8; 16] = [
            0x63, 0x63, 0x63, 0x63, 0x62, 0x63, 0x63, 0x63, 0x63, 0x63, 0x63,
            0x63, 0x62, 0x63, 0x63, 0x63,
        ];
        for i in 0..16 {
            let val: u8 = s.extract_lane(result, i);
            if val != expected[i] {
                return false;
            }
        }
        true
    }
}

unsafe fn test_aes_inv_mix_columns_on<S: SimdOps + SimdCrypto>(s: S) -> bool {
    unsafe {
        let lanes = s.lanes::<u8>();
        // Known: MixColumns([db,13,53,45,...]) = [8e,4d,a1,bc,...]
        // So InvMixColumns([8e,4d,a1,bc,...]) = [db,13,53,45,...]
        let input_data: [u8; 16] = [
            0x8e, 0x4d, 0xa1, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00,
        ];
        let expected_col: [u8; 4] = [0xdb, 0x13, 0x53, 0x45];
        let mut full_input = vec![0u8; lanes];
        for block in 0..lanes / 16 {
            for i in 0..16 {
                full_input[block * 16 + i] = input_data[i];
            }
        }
        let v = s.load_u(full_input.as_ptr());
        let result = s.aes_inv_mix_columns(v);
        for i in 0..4 {
            let val: u8 = s.extract_lane(result, i);
            if val != expected_col[i] {
                return false;
            }
        }
        true
    }
}

#[test]
fn test_aes_key_gen_assist() {
    unsafe {
        for target in simd_targets() {
            let ok = match target {
                TargetId::Sse2 => test_aes_key_gen_assist_on(unsafe {
                    highway::backend::sse2::Sse2::new_unchecked()
                }),
                TargetId::Avx2 => test_aes_key_gen_assist_on(unsafe {
                    highway::backend::avx2::Avx2::new_unchecked()
                }),
                TargetId::Avx512 => test_aes_key_gen_assist_on(unsafe {
                    highway::backend::avx512::Avx512::new_unchecked()
                }),
                _ => true,
            };
            assert!(ok, "aes_key_gen_assist failed on {:?}", target);
        }
    }
}

#[test]
fn test_aes_inv_mix_columns() {
    unsafe {
        for target in simd_targets() {
            let ok = match target {
                TargetId::Sse2 => test_aes_inv_mix_columns_on(unsafe {
                    highway::backend::sse2::Sse2::new_unchecked()
                }),
                TargetId::Avx2 => test_aes_inv_mix_columns_on(unsafe {
                    highway::backend::avx2::Avx2::new_unchecked()
                }),
                TargetId::Avx512 => test_aes_inv_mix_columns_on(unsafe {
                    highway::backend::avx512::Avx512::new_unchecked()
                }),
                _ => true,
            };
            assert!(ok, "aes_inv_mix_columns failed on {:?}", target);
        }
    }
}

// =========================================================================
// Group 1: compress_blocks_not
// =========================================================================

struct CompressBlocksNotKernel;
impl WithSimd for CompressBlocksNotKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u64>();
            let mut data = vec![0u64; lanes];
            for i in 0..lanes {
                data[i] = (i + 1) as u64;
            }
            let v = s.load_u(data.as_ptr());

            // All-false mask -> compress_not keeps everything
            let all_false = s.mask_from_vec(s.zero::<u64>());
            let result = s.compress_blocks_not(v, all_false);
            for i in 0..lanes {
                let val: u64 = s.extract_lane(result, i);
                if val != (i + 1) as u64 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_compress_blocks_not() {
    for target in available_targets() {
        assert!(
            dispatch_to(CompressBlocksNotKernel, target),
            "compress_blocks_not failed on {:?}",
            target
        );
    }
}

// =========================================================================
// broadcast_sign_bit
// =========================================================================

struct BroadcastSignBitKernel;
impl WithSimd for BroadcastSignBitKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<i32>();
            let mut data = vec![0i32; lanes];
            for i in 0..lanes {
                data[i] = if i % 2 == 0 {
                    -(i as i32) - 1
                } else {
                    i as i32
                };
            }
            let v = s.load_u(data.as_ptr());
            let r = s.broadcast_sign_bit(v);
            for i in 0..lanes {
                let got: i32 = s.extract_lane(r, i);
                let expected = if data[i] < 0 { -1i32 } else { 0i32 };
                if got != expected {
                    return false;
                }
            }
            // i16 too
            let lanes16 = s.lanes::<i16>();
            let mut d16 = vec![0i16; lanes16];
            for i in 0..lanes16 {
                d16[i] = if i % 2 == 0 { -1 } else { 100 };
            }
            let v16 = s.load_u(d16.as_ptr());
            let r16 = s.broadcast_sign_bit(v16);
            for i in 0..lanes16 {
                let got: i16 = s.extract_lane(r16, i);
                let expected = if d16[i] < 0 { -1i16 } else { 0i16 };
                if got != expected {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_broadcast_sign_bit() {
    for target in available_targets() {
        assert!(
            dispatch_to(BroadcastSignBitKernel, target),
            "broadcast_sign_bit failed on {:?}",
            target
        );
    }
}

// =========================================================================
// rol / rotate_left
// =========================================================================

struct RolKernel;
impl WithSimd for RolKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            let data = vec![0x8000_0001u32; lanes];
            let mut shift = vec![0u32; lanes];
            for i in 0..lanes {
                shift[i] = (i as u32) % 32;
            }
            let v = s.load_u(data.as_ptr());
            let sh = s.load_u(shift.as_ptr());
            let r = s.rol(v, sh);
            for i in 0..lanes {
                let got: u32 = s.extract_lane(r, i);
                let expected = 0x8000_0001u32.rotate_left((i as u32) % 32);
                if got != expected {
                    return false;
                }
            }
            // rotate_left const by 4
            let r2 = s.rotate_left::<u32, 4>(v);
            for i in 0..lanes {
                let got: u32 = s.extract_lane(r2, i);
                if got != 0x8000_0001u32.rotate_left(4) {
                    return false;
                }
            }
            // u8 variable
            let lanes8 = s.lanes::<u8>();
            let d8 = vec![0b1000_0001u8; lanes8];
            let mut sh8 = vec![0u8; lanes8];
            for i in 0..lanes8 {
                sh8[i] = (i as u8) % 8;
            }
            let v8 = s.load_u(d8.as_ptr());
            let s8 = s.load_u(sh8.as_ptr());
            let r8 = s.rol(v8, s8);
            for i in 0..lanes8 {
                let got: u8 = s.extract_lane(r8, i);
                if got != 0b1000_0001u8.rotate_left((i as u32) % 8) {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_rol() {
    for target in available_targets() {
        assert!(
            dispatch_to(RolKernel, target),
            "rol/rotate_left failed on {:?}",
            target
        );
    }
}

// =========================================================================
// saturated_neg / saturated_abs
// =========================================================================

struct SaturatedNegAbsKernel;
impl WithSimd for SaturatedNegAbsKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<i32>();
            let mut data = vec![0i32; lanes];
            for i in 0..lanes {
                data[i] = match i % 3 {
                    0 => i32::MIN,
                    1 => -5,
                    _ => 7,
                };
            }
            let v = s.load_u(data.as_ptr());
            let neg = s.saturated_neg(v);
            let ab = s.saturated_abs(v);
            for i in 0..lanes {
                let gn: i32 = s.extract_lane(neg, i);
                let ga: i32 = s.extract_lane(ab, i);
                let en = data[i].saturating_neg();
                let ea = if data[i] == i32::MIN {
                    i32::MAX
                } else {
                    data[i].abs()
                };
                if gn != en || ga != ea {
                    return false;
                }
            }
            // i8
            let lanes8 = s.lanes::<i8>();
            let mut d8 = vec![0i8; lanes8];
            for i in 0..lanes8 {
                d8[i] = match i % 3 {
                    0 => i8::MIN,
                    1 => -3,
                    _ => 9,
                };
            }
            let v8 = s.load_u(d8.as_ptr());
            let neg8 = s.saturated_neg(v8);
            let ab8 = s.saturated_abs(v8);
            for i in 0..lanes8 {
                let gn: i8 = s.extract_lane(neg8, i);
                let ga: i8 = s.extract_lane(ab8, i);
                if gn != d8[i].saturating_neg() {
                    return false;
                }
                let ea = if d8[i] == i8::MIN {
                    i8::MAX
                } else {
                    d8[i].abs()
                };
                if ga != ea {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_saturated_neg_abs() {
    for target in available_targets() {
        assert!(
            dispatch_to(SaturatedNegAbsKernel, target),
            "saturated_neg/saturated_abs failed on {:?}",
            target
        );
    }
}

// =========================================================================
// masked_{min,max,add,sub,mul}_or
// =========================================================================

struct MaskedOrKernel;
impl WithSimd for MaskedOrKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<i32>();
            let mut da = vec![0i32; lanes];
            let mut db = vec![0i32; lanes];
            let mut dno = vec![0i32; lanes];
            for i in 0..lanes {
                da[i] = (i as i32) + 3;
                db[i] = (i as i32) * 2 + 1;
                dno[i] = -100 - i as i32;
            }
            let a = s.load_u(da.as_ptr());
            let b = s.load_u(db.as_ptr());
            let no = s.load_u(dno.as_ptr());
            // mask: even lanes true
            let mut md = vec![0i32; lanes];
            for i in (0..lanes).step_by(2) {
                md[i] = -1;
            }
            let mask = s.mask_from_vec(s.load_u(md.as_ptr()));

            let rmin = s.masked_min_or(no, mask, a, b);
            let rmax = s.masked_max_or(no, mask, a, b);
            let radd = s.masked_add_or(no, mask, a, b);
            let rsub = s.masked_sub_or(no, mask, a, b);
            let rmul = s.masked_mul_or(no, mask, a, b);
            for i in 0..lanes {
                let on = i % 2 == 0;
                let gmin: i32 = s.extract_lane(rmin, i);
                let gmax: i32 = s.extract_lane(rmax, i);
                let gadd: i32 = s.extract_lane(radd, i);
                let gsub: i32 = s.extract_lane(rsub, i);
                let gmul: i32 = s.extract_lane(rmul, i);
                let emin = if on { da[i].min(db[i]) } else { dno[i] };
                let emax = if on { da[i].max(db[i]) } else { dno[i] };
                let eadd = if on {
                    da[i].wrapping_add(db[i])
                } else {
                    dno[i]
                };
                let esub = if on {
                    da[i].wrapping_sub(db[i])
                } else {
                    dno[i]
                };
                let emul = if on {
                    da[i].wrapping_mul(db[i])
                } else {
                    dno[i]
                };
                if gmin != emin
                    || gmax != emax
                    || gadd != eadd
                    || gsub != esub
                    || gmul != emul
                {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_masked_or() {
    for target in available_targets() {
        assert!(
            dispatch_to(MaskedOrKernel, target),
            "masked_*_or failed on {:?}",
            target
        );
    }
}

// =========================================================================
// if_negative_then_else / _zero / zero_else
// =========================================================================

struct IfNegativeKernel;
impl WithSimd for IfNegativeKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            // integer i32
            let lanes = s.lanes::<i32>();
            let mut dv = vec![0i32; lanes];
            let mut dy = vec![0i32; lanes];
            let mut dn = vec![0i32; lanes];
            for i in 0..lanes {
                dv[i] = if i % 2 == 0 {
                    -(i as i32) - 1
                } else {
                    i as i32
                };
                dy[i] = 1000 + i as i32;
                dn[i] = 2000 + i as i32;
            }
            let v = s.load_u(dv.as_ptr());
            let yes = s.load_u(dy.as_ptr());
            let no = s.load_u(dn.as_ptr());
            let r1 = s.if_negative_then_else(v, yes, no);
            let r2 = s.if_negative_then_else_zero(v, yes);
            let r3 = s.if_negative_then_zero_else(v, no);
            for i in 0..lanes {
                let neg = dv[i] < 0;
                let g1: i32 = s.extract_lane(r1, i);
                let g2: i32 = s.extract_lane(r2, i);
                let g3: i32 = s.extract_lane(r3, i);
                if g1 != (if neg { dy[i] } else { dn[i] }) {
                    return false;
                }
                if g2 != (if neg { dy[i] } else { 0 }) {
                    return false;
                }
                if g3 != (if neg { 0 } else { dn[i] }) {
                    return false;
                }
            }

            // float f32 (sign bit, incl -0.0)
            let lf = s.lanes::<f32>();
            let mut fv = vec![0f32; lf];
            for i in 0..lf {
                fv[i] = if i % 2 == 0 { -1.5 } else { 2.5 };
            }
            let vf = s.load_u(fv.as_ptr());
            let yf = s.splat::<f32>(9.0);
            let nf = s.splat::<f32>(-9.0);
            let rf = s.if_negative_then_else(vf, yf, nf);
            for i in 0..lf {
                let g: f32 = s.extract_lane(rf, i);
                let e = if fv[i] < 0.0 { 9.0 } else { -9.0 };
                if g != e {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_if_negative() {
    for target in available_targets() {
        assert!(
            dispatch_to(IfNegativeKernel, target),
            "if_negative_then_* failed on {:?}",
            target
        );
    }
}

// =========================================================================
// ordered_truncate_2_to
// =========================================================================

struct OrderedTruncate2Kernel;
impl WithSimd for OrderedTruncate2Kernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes_wide = s.lanes::<u32>();
            let lanes_narrow = s.lanes::<u16>();
            let mut da = vec![0u32; lanes_wide];
            let mut db = vec![0u32; lanes_wide];
            for i in 0..lanes_wide {
                da[i] = 0x1234_0000 | (i as u32);
                db[i] = 0x5678_0000 | (100 + i as u32);
            }
            let a = s.load_u(da.as_ptr());
            let b = s.load_u(db.as_ptr());
            let r = s.ordered_truncate_2_to::<u32>(a, b);

            if lanes_narrow >= 2 * lanes_wide {
                // lower half = truncated a, upper half = truncated b
                for i in 0..lanes_wide {
                    let lo: u16 = s.extract_lane(r, i);
                    let hi: u16 = s.extract_lane(r, lanes_wide + i);
                    if lo != (da[i] & 0xFFFF) as u16 {
                        return false;
                    }
                    if hi != (db[i] & 0xFFFF) as u16 {
                        return false;
                    }
                }
            } else {
                // scalar: single lane = truncated a[0]
                let lo: u16 = s.extract_lane(r, 0);
                if lo != (da[0] & 0xFFFF) as u16 {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_ordered_truncate_2_to() {
    for target in available_targets() {
        assert!(
            dispatch_to(OrderedTruncate2Kernel, target),
            "ordered_truncate_2_to failed on {:?}",
            target
        );
    }
}

// =========================================================================
// is_either_nan
// =========================================================================

struct IsEitherNanKernel;
impl WithSimd for IsEitherNanKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<f32>();
            let mut da = vec![0f32; lanes];
            let mut db = vec![0f32; lanes];
            for i in 0..lanes {
                da[i] = if i % 3 == 0 { f32::NAN } else { i as f32 };
                db[i] = if i % 3 == 1 {
                    f32::NAN
                } else {
                    (i as f32) + 0.5
                };
            }
            let a = s.load_u(da.as_ptr());
            let b = s.load_u(db.as_ptr());
            let m = s.is_either_nan(a, b);
            let vm = s.vec_from_mask::<f32>(m);
            for i in 0..lanes {
                let bits: u32 = s.extract_lane(s.bitcast::<f32, u32>(vm), i);
                let expected = da[i].is_nan() || db[i].is_nan();
                if (bits != 0) != expected {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_is_either_nan() {
    for target in available_targets() {
        assert!(
            dispatch_to(IsEitherNanKernel, target),
            "is_either_nan failed on {:?}",
            target
        );
    }
}

// =========================================================================
// interleave_whole_lower / interleave_whole_upper (exact position check)
// =========================================================================

struct InterleaveWholePosKernel;
impl WithSimd for InterleaveWholePosKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            if lanes < 2 {
                return true;
            } // scalar: not meaningful
            let half = lanes / 2;
            let mut da = vec![0u32; lanes];
            let mut db = vec![0u32; lanes];
            for i in 0..lanes {
                da[i] = i as u32;
                db[i] = 100 + i as u32;
            }
            let a = s.load_u(da.as_ptr());
            let b = s.load_u(db.as_ptr());
            let lo = s.interleave_whole_lower(a, b);
            let up = s.interleave_whole_upper(a, b);
            for i in 0..half {
                // lower: result[2i]=a[i], result[2i+1]=b[i]
                if s.extract_lane::<u32>(lo, 2 * i) != da[i] {
                    return false;
                }
                if s.extract_lane::<u32>(lo, 2 * i + 1) != db[i] {
                    return false;
                }
                // upper: result[2i]=a[half+i], result[2i+1]=b[half+i]
                if s.extract_lane::<u32>(up, 2 * i) != da[half + i] {
                    return false;
                }
                if s.extract_lane::<u32>(up, 2 * i + 1) != db[half + i] {
                    return false;
                }
            }
            true
        }
    }
}

#[test]
fn test_interleave_whole_positions() {
    for target in available_targets() {
        assert!(
            dispatch_to(InterleaveWholePosKernel, target),
            "interleave_whole_lower/upper ordering failed on {:?}",
            target
        );
    }
}

// =========================================================================
// Coverage-gap tests (added during test-coverage audit)
// =========================================================================

// broadcast_sign_bit: i8 (cmpgt/movm path) + i64 (shuffle-trick / srai_epi64)
struct BroadcastSignBitWideKernel;
impl WithSimd for BroadcastSignBitWideKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let l8 = s.lanes::<i8>();
            let mut d8 = vec![0i8; l8];
            for i in 0..l8 {
                d8[i] = if i % 2 == 0 { -1 } else { 5 };
            }
            let r8 = s.broadcast_sign_bit(s.load_u(d8.as_ptr()));
            for i in 0..l8 {
                let g: i8 = s.extract_lane(r8, i);
                if g != (if d8[i] < 0 { -1 } else { 0 }) {
                    return false;
                }
            }
            let l64 = s.lanes::<i64>();
            let mut d64 = vec![0i64; l64];
            for i in 0..l64 {
                d64[i] = if i % 2 == 0 { -777 } else { 123 };
            }
            let r64 = s.broadcast_sign_bit(s.load_u(d64.as_ptr()));
            for i in 0..l64 {
                let g: i64 = s.extract_lane(r64, i);
                if g != (if d64[i] < 0 { -1 } else { 0 }) {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_broadcast_sign_bit_wide() {
    for t in available_targets() {
        assert!(
            dispatch_to(BroadcastSignBitWideKernel, t),
            "broadcast_sign_bit i8/i64 on {:?}",
            t
        );
    }
}

// load_expand f32 + f64 (AVX-512 expandloadu_ps/pd paths)
struct LoadExpandFloatKernel;
impl WithSimd for LoadExpandFloatKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            // f32
            let l = s.lanes::<f32>();
            let src: Vec<f32> = (1..=l as u32).map(|x| x as f32).collect();
            let mut md = vec![0u32; l];
            for i in (0..l).step_by(2) {
                md[i] = 0xFFFF_FFFF;
            }
            let mask =
                s.mask_from_vec(s.bitcast::<u32, f32>(s.load_u(md.as_ptr())));
            let r = s.load_expand(mask, src.as_ptr());
            let mut si = 0;
            for i in 0..l {
                let g: f32 = s.extract_lane(r, i);
                if i % 2 == 0 {
                    if g != src[si] {
                        return false;
                    }
                    si += 1;
                } else if g != 0.0 {
                    return false;
                }
            }
            // f64
            let l2 = s.lanes::<f64>();
            let src2: Vec<f64> = (1..=l2 as u32).map(|x| x as f64).collect();
            let mut md2 = vec![0u64; l2];
            for i in (0..l2).step_by(2) {
                md2[i] = 0xFFFF_FFFF_FFFF_FFFF;
            }
            let mask2 =
                s.mask_from_vec(s.bitcast::<u64, f64>(s.load_u(md2.as_ptr())));
            let r2 = s.load_expand(mask2, src2.as_ptr());
            let mut si2 = 0;
            for i in 0..l2 {
                let g: f64 = s.extract_lane(r2, i);
                if i % 2 == 0 {
                    if g != src2[si2] {
                        return false;
                    }
                    si2 += 1;
                } else if g != 0.0 {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_load_expand_float() {
    for t in available_targets() {
        assert!(
            dispatch_to(LoadExpandFloatKernel, t),
            "load_expand f32/f64 on {:?}",
            t
        );
    }
}

// ror: u64 (rorv_epi64) + u16 (scalar-loop path)
struct RorWideKernel;
impl WithSimd for RorWideKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let l = s.lanes::<u64>();
            let data = vec![0x8000_0000_0000_0001u64; l];
            let mut sh = vec![0u64; l];
            for i in 0..l {
                sh[i] = (i as u64) % 64;
            }
            let r = s.ror(s.load_u(data.as_ptr()), s.load_u(sh.as_ptr()));
            for i in 0..l {
                let g: u64 = s.extract_lane(r, i);
                if g != 0x8000_0000_0000_0001u64.rotate_right((i as u32) % 64) {
                    return false;
                }
            }
            let l16 = s.lanes::<u16>();
            let d16 = vec![0x8001u16; l16];
            let mut s16 = vec![0u16; l16];
            for i in 0..l16 {
                s16[i] = (i as u16) % 16;
            }
            let r16 = s.ror(s.load_u(d16.as_ptr()), s.load_u(s16.as_ptr()));
            for i in 0..l16 {
                let g: u16 = s.extract_lane(r16, i);
                if g != 0x8001u16.rotate_right((i as u32) % 16) {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_ror_wide() {
    for t in available_targets() {
        assert!(dispatch_to(RorWideKernel, t), "ror u64/u16 on {:?}", t);
    }
}

// f64 variants of min/max number & magnitude, is_either_nan
struct Float64NumMagNanKernel;
impl WithSimd for Float64NumMagNanKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let l = s.lanes::<f64>();
            let nan = s.splat::<f64>(f64::NAN);
            let five = s.splat::<f64>(5.0);
            // min/max_number: NaN in a -> b
            let mn = s.min_number(nan, five);
            let mx = s.max_number(nan, five);
            for i in 0..l {
                if s.extract_lane::<f64>(mn, i) != 5.0 {
                    return false;
                }
                if s.extract_lane::<f64>(mx, i) != 5.0 {
                    return false;
                }
            }
            // magnitude
            let a = s.splat::<f64>(-3.0);
            let b = s.splat::<f64>(5.0);
            let mmin = s.min_magnitude(a, b);
            let mmax = s.max_magnitude(a, b);
            for i in 0..l {
                if s.extract_lane::<f64>(mmin, i) != -3.0 {
                    return false;
                }
                if s.extract_lane::<f64>(mmax, i) != 5.0 {
                    return false;
                }
            }
            // is_either_nan
            let m = s.is_either_nan(nan, five);
            if s.count_true(m) != l {
                return false;
            }
            let m2 = s.is_either_nan(five, b);
            if s.count_true(m2) != 0 {
                return false;
            }
            true
        }
    }
}
#[test]
fn test_float64_num_mag_nan() {
    for t in available_targets() {
        assert!(
            dispatch_to(Float64NumMagNanKernel, t),
            "f64 number/magnitude/is_either_nan on {:?}",
            t
        );
    }
}

// convert_in_range_to_int f64 -> i64
struct ConvertInRangeF64Kernel;
impl WithSimd for ConvertInRangeF64Kernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let l = s.lanes::<f64>();
            let mut d = vec![0f64; l];
            for i in 0..l {
                d[i] = (i as f64) * 1000.0 - 3000.0 + 0.7 * (i as f64);
            }
            let r = s.convert_in_range_to_int(s.load_u(d.as_ptr()));
            for i in 0..l {
                let g: i64 = s.extract_lane(r, i);
                if g != d[i] as i64 {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_convert_in_range_f64() {
    for t in available_targets() {
        assert!(
            dispatch_to(ConvertInRangeF64Kernel, t),
            "convert_in_range_to_int f64 on {:?}",
            t
        );
    }
}

// broadcast_block / insert_block / extract_block with IDX=1 (multi-block only)
struct BlockIdx1Kernel;
impl WithSimd for BlockIdx1Kernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u32>();
            if lanes < 8 {
                return true;
            } // needs >=2 blocks (AVX2/AVX-512)
            let mut data = vec![0u32; lanes];
            for i in 0..lanes {
                data[i] = i as u32;
            }
            let v = s.load_u(data.as_ptr());

            // broadcast_block::<1>: replicate lanes 4..8 across all blocks
            let bb = s.broadcast_block::<u32, 1>(v);
            for i in 0..lanes {
                if s.extract_lane::<u32>(bb, i) != (4 + (i % 4)) as u32 {
                    return false;
                }
            }

            // extract_block/insert_block operate at VecHalf granularity (128-bit on
            // AVX2, 256-bit on AVX-512): IDX selects lower/upper HALF, not a 128-bit block.
            let half = lanes / 2;
            let blk = s.extract_block::<u32, 1>(v);
            let rebuilt = s.insert_block::<u32, 1>(s.zero::<u32>(), blk);
            for i in 0..lanes {
                let g: u32 = s.extract_lane(rebuilt, i);
                let expected = if i >= half { data[i] } else { 0 };
                if g != expected {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_block_idx1() {
    for t in available_targets() {
        assert!(
            dispatch_to(BlockIdx1Kernel, t),
            "block IDX=1 ops on {:?}",
            t
        );
    }
}

// compress_blocks_not: real block compression (>=2 blocks)
struct CompressBlocksNotRealKernel;
impl WithSimd for CompressBlocksNotRealKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let lanes = s.lanes::<u64>();
            if lanes < 4 {
                return true;
            } // need >=2 128-bit blocks (2 u64 each)
            let mut data = vec![0u64; lanes];
            for i in 0..lanes {
                data[i] = (i + 1) as u64;
            }
            let v = s.load_u(data.as_ptr());
            // Block-uniform mask: first 128-bit block (lanes 0,1) TRUE, rest FALSE.
            let mut md = vec![0u64; lanes];
            md[0] = u64::MAX;
            md[1] = u64::MAX;
            let mask = s.mask_from_vec(s.load_u(md.as_ptr()));
            // compress_blocks_not keeps blocks where mask is FALSE -> blocks 1.. come first.
            let r = s.compress_blocks_not(v, mask);
            // First (lanes-2) lanes should be the data from lanes 2..lanes (the false blocks).
            for i in 0..(lanes - 2) {
                let g: u64 = s.extract_lane(r, i);
                if g != data[i + 2] {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_compress_blocks_not_real() {
    for t in available_targets() {
        assert!(
            dispatch_to(CompressBlocksNotRealKernel, t),
            "compress_blocks_not real on {:?}",
            t
        );
    }
}

// =========================================================================
// Original-op coverage-gap tests (added during old-op test audit 2026-06-13)
// =========================================================================

// shl / shr for u8, u16, u64 (per-width branches; u8/i8 have emulated paths)
struct ShlShrWideKernel;
impl WithSimd for ShlShrWideKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            // u8
            let l8 = s.lanes::<u8>();
            let mut v8 = vec![0u8; l8];
            let mut sh8 = vec![0u8; l8];
            for i in 0..l8 {
                v8[i] = 0b1010_1101;
                sh8[i] = (i % 8) as u8;
            }
            let shl8 = s.shl(s.load_u(v8.as_ptr()), s.load_u(sh8.as_ptr()));
            let shr8 = s.shr(s.load_u(v8.as_ptr()), s.load_u(sh8.as_ptr()));
            for i in 0..l8 {
                if s.extract_lane::<u8>(shl8, i)
                    != v8[i].wrapping_shl(sh8[i] as u32)
                {
                    return false;
                }
                if s.extract_lane::<u8>(shr8, i)
                    != v8[i].wrapping_shr(sh8[i] as u32)
                {
                    return false;
                }
            }
            // u16
            let l16 = s.lanes::<u16>();
            let mut v16 = vec![0u16; l16];
            let mut sh16 = vec![0u16; l16];
            for i in 0..l16 {
                v16[i] = 0xABCD;
                sh16[i] = (i % 16) as u16;
            }
            let shl16 = s.shl(s.load_u(v16.as_ptr()), s.load_u(sh16.as_ptr()));
            let shr16 = s.shr(s.load_u(v16.as_ptr()), s.load_u(sh16.as_ptr()));
            for i in 0..l16 {
                if s.extract_lane::<u16>(shl16, i)
                    != v16[i].wrapping_shl(sh16[i] as u32)
                {
                    return false;
                }
                if s.extract_lane::<u16>(shr16, i)
                    != v16[i].wrapping_shr(sh16[i] as u32)
                {
                    return false;
                }
            }
            // u64
            let l64 = s.lanes::<u64>();
            let mut v64 = vec![0u64; l64];
            let mut sh64 = vec![0u64; l64];
            for i in 0..l64 {
                v64[i] = 0xDEAD_BEEF_CAFE_0001;
                sh64[i] = (i as u64 * 7) % 64;
            }
            let shl64 = s.shl(s.load_u(v64.as_ptr()), s.load_u(sh64.as_ptr()));
            let shr64 = s.shr(s.load_u(v64.as_ptr()), s.load_u(sh64.as_ptr()));
            for i in 0..l64 {
                if s.extract_lane::<u64>(shl64, i)
                    != v64[i].wrapping_shl(sh64[i] as u32)
                {
                    return false;
                }
                if s.extract_lane::<u64>(shr64, i)
                    != v64[i].wrapping_shr(sh64[i] as u32)
                {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_shl_shr_wide() {
    for t in available_targets() {
        assert!(
            dispatch_to(ShlShrWideKernel, t),
            "shl/shr u8/u16/u64 on {:?}",
            t
        );
    }
}

// leading_zero_count for u8, u16, u64 (incl. 0 and 1 edge values)
struct LzcntWideKernel;
impl WithSimd for LzcntWideKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let l8 = s.lanes::<u8>();
            let mut v8 = vec![0u8; l8];
            for i in 0..l8 {
                v8[i] = match i % 4 {
                    0 => 0,
                    1 => 1,
                    2 => 0x80,
                    _ => 0x0F,
                };
            }
            let r8 = s.leading_zero_count(s.load_u(v8.as_ptr()));
            for i in 0..l8 {
                if s.extract_lane::<u8>(r8, i) != v8[i].leading_zeros() as u8 {
                    return false;
                }
            }
            let l16 = s.lanes::<u16>();
            let mut v16 = vec![0u16; l16];
            for i in 0..l16 {
                v16[i] = match i % 4 {
                    0 => 0,
                    1 => 1,
                    2 => 0x8000,
                    _ => 0x00FF,
                };
            }
            let r16 = s.leading_zero_count(s.load_u(v16.as_ptr()));
            for i in 0..l16 {
                if s.extract_lane::<u16>(r16, i)
                    != v16[i].leading_zeros() as u16
                {
                    return false;
                }
            }
            let l64 = s.lanes::<u64>();
            let mut v64 = vec![0u64; l64];
            for i in 0..l64 {
                v64[i] = match i % 4 {
                    0 => 0,
                    1 => 1,
                    2 => 1u64 << 63,
                    _ => 0xFFFF,
                };
            }
            let r64 = s.leading_zero_count(s.load_u(v64.as_ptr()));
            for i in 0..l64 {
                if s.extract_lane::<u64>(r64, i)
                    != v64[i].leading_zeros() as u64
                {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_lzcnt_wide() {
    for t in available_targets() {
        assert!(
            dispatch_to(LzcntWideKernel, t),
            "leading_zero_count u8/u16/u64 on {:?}",
            t
        );
    }
}

// NOTE: mul_high (all int widths) and abs (all signed int widths incl. INT_MIN)
// are already covered by the macro-generated tests (`test_mul_high!`, `test_abs_int!`)
// in the macro section (~lines 5185, 5972), so no extra tests are added here.

// reverse_bits for u16 and u32 (only u8 was tested)
struct ReverseBitsWideKernel;
impl WithSimd for ReverseBitsWideKernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let l16 = s.lanes::<u16>();
            let mut v16 = vec![0u16; l16];
            for i in 0..l16 {
                v16[i] = 0x1234u16.wrapping_add((i as u16) * 17);
            }
            let r16 = s.reverse_bits(s.load_u(v16.as_ptr()));
            for i in 0..l16 {
                if s.extract_lane::<u16>(r16, i) != v16[i].reverse_bits() {
                    return false;
                }
            }
            let l32 = s.lanes::<u32>();
            let mut v32 = vec![0u32; l32];
            for i in 0..l32 {
                v32[i] = 0x1234_5678u32.wrapping_add((i as u32) * 0x1111);
            }
            let r32 = s.reverse_bits(s.load_u(v32.as_ptr()));
            for i in 0..l32 {
                if s.extract_lane::<u32>(r32, i) != v32[i].reverse_bits() {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_reverse_bits_wide() {
    for t in available_targets() {
        assert!(
            dispatch_to(ReverseBitsWideKernel, t),
            "reverse_bits u16/u32 on {:?}",
            t
        );
    }
}

// table_lookup_bytes: non-identity mapping + high-bit-index -> 0 (TableLookupBytesOr0)
struct TableLookupBytesOr0Kernel;
impl WithSimd for TableLookupBytesOr0Kernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            let l = s.lanes::<u8>();
            // table_lookup_bytes is a 16-byte-block op; the scalar Vec1<u8> (1 byte)
            // can't represent a 16-entry table, so skip backends with <16 byte lanes.
            if l < 16 {
                return true;
            }
            // Table: block-local values 0..15 in every 128-bit block.
            let mut tbl = vec![0u8; l];
            for i in 0..l {
                tbl[i] = (i % 16) as u8;
            }
            // Indices: reverse within block, with every 4th index high-bit set (->0).
            let mut idx = vec![0u8; l];
            for i in 0..l {
                idx[i] = if i % 4 == 3 {
                    0x80
                } else {
                    (15 - (i % 16)) as u8
                };
            }
            let r = s.table_lookup_bytes(
                s.load_u(tbl.as_ptr()),
                s.load_u(idx.as_ptr()),
            );
            for i in 0..l {
                let expected = if idx[i] & 0x80 != 0 {
                    0u8
                } else {
                    idx[i] & 0x0F
                };
                if s.extract_lane::<u8>(r, i) != expected {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_table_lookup_bytes_or0() {
    for t in available_targets() {
        assert!(
            dispatch_to(TableLookupBytesOr0Kernel, t),
            "table_lookup_bytes non-identity on {:?}",
            t
        );
    }
}

// =========================================================================
// Safe slice-based load/store wrappers (no `unsafe` in user code)
// =========================================================================

struct SliceLoadStoreKernel<'a> {
    input: &'a [f32],
    out: &'a mut [f32],
}
impl WithSimd for SliceLoadStoreKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        // Entirely safe: no `unsafe` block anywhere in this kernel.
        let lanes = s.lanes::<f32>();
        let v = s.load_slice(&self.input[..lanes]);
        let doubled = s.add(v, v);
        s.store_slice(doubled, &mut self.out[..lanes]);
    }
}

#[test]
fn test_load_store_slice_safe() {
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let input: Vec<f32> = (0..lanes).map(|i| i as f32 + 1.0).collect();
        let mut out = vec![0.0f32; lanes];
        dispatch_to(
            SliceLoadStoreKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i],
                (i as f32 + 1.0) * 2.0,
                "load/store_slice on {target:?}"
            );
        }
    }
}

struct AlignedSliceKernel<'a> {
    input: &'a [f32],
    out: &'a mut [f32],
}
impl WithSimd for AlignedSliceKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let lanes = s.lanes::<f32>();
        // Safe aligned load/store (bounds + alignment checked internally).
        let v = s.load_aligned_slice(&self.input[..lanes]);
        let tripled = s.add(s.add(v, v), v);
        s.store_aligned_slice(tripled, &mut self.out[..lanes]);
    }
}

#[test]
fn test_load_store_aligned_slice_safe() {
    use highway::{aligned_vec_from_slice, aligned_vec_with_capacity};
    for target in available_targets() {
        let lanes = lanes_for::<f32>(target);
        let src: Vec<f32> = (0..lanes).map(|i| i as f32 + 1.0).collect();
        let input = aligned_vec_from_slice(&src);
        let mut out = aligned_vec_with_capacity::<f32>(lanes);
        out.resize(lanes, 0.0);
        dispatch_to(
            AlignedSliceKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for i in 0..lanes {
            assert_eq!(
                out[i],
                (i as f32 + 1.0) * 3.0,
                "aligned load/store_slice on {target:?}"
            );
        }
    }
}

// if_negative_then_else on 8-byte lanes (exercises the i64/f64 sign-mask path)
struct IfNegative64Kernel;
impl WithSimd for IfNegative64Kernel {
    type Output = bool;
    fn with_simd<S: SimdOps>(self, s: S) -> bool {
        unsafe {
            // i64
            let li = s.lanes::<i64>();
            let mut dv = vec![0i64; li];
            for i in 0..li {
                dv[i] = if i % 2 == 0 {
                    -(i as i64) - 1
                } else {
                    i as i64
                };
            }
            let v = s.load_u(dv.as_ptr());
            let yes = s.splat::<i64>(7777);
            let no = s.splat::<i64>(-3333);
            let r = s.if_negative_then_else(v, yes, no);
            for i in 0..li {
                let g: i64 = s.extract_lane(r, i);
                if g != (if dv[i] < 0 { 7777 } else { -3333 }) {
                    return false;
                }
            }
            // f64 (incl. -0.0 must count as negative)
            let lf = s.lanes::<f64>();
            let mut fv = vec![0f64; lf];
            for i in 0..lf {
                fv[i] = if i % 2 == 0 { -1.5 } else { 2.5 };
            }
            let vf = s.load_u(fv.as_ptr());
            let rf = s.if_negative_then_else(
                vf,
                s.splat::<f64>(9.0),
                s.splat::<f64>(-9.0),
            );
            for i in 0..lf {
                let g: f64 = s.extract_lane(rf, i);
                if g != (if fv[i] < 0.0 { 9.0 } else { -9.0 }) {
                    return false;
                }
            }
            true
        }
    }
}
#[test]
fn test_if_negative_64bit() {
    for t in available_targets() {
        assert!(
            dispatch_to(IfNegative64Kernel, t),
            "if_negative 64-bit on {:?}",
            t
        );
    }
}
