#![allow(clippy::undocumented_unsafe_blocks, clippy::type_complexity)]
//! Cross-target consistency tests.
//!
//! Verifies that all available SIMD targets produce identical results
//! for the same inputs. Tests exercise dispatch and per-target correctness.

use highway::{dispatch, dispatch_to, simd_fn, SimdOps, TargetId, WithSimd};

#[cfg(feature = "alloc")]
use highway::{aligned_vec_from_slice, aligned_vec_with_capacity};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// All targets we can test on this platform.
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

// Helper: get lane count for u32 on a given target
struct U32Lanes;
impl WithSimd for U32Lanes {
    type Output = usize;
    fn with_simd<S: SimdOps>(self, s: S) -> usize {
        s.lanes::<u32>()
    }
}

// Helper: get lane count for i32 on a given target
struct I32Lanes;
impl WithSimd for I32Lanes {
    type Output = usize;
    fn with_simd<S: SimdOps>(self, s: S) -> usize {
        s.lanes::<i32>()
    }
}

// ---------------------------------------------------------------------------
// Addition kernel
// ---------------------------------------------------------------------------

struct AddI32Kernel<'a> {
    a: &'a [i32],
    b: &'a [i32],
    out: &'a mut [i32],
}

impl WithSimd for AddI32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let vc = s.add(va, vb);
                s.store_u(vc, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        // Scalar tail
        while i < n {
            self.out[i] = self.a[i] + self.b[i];
            i += 1;
        }
    }
}

#[test]
fn test_add_i32_dispatch() {
    let a: Vec<i32> = (0..64).collect();
    let b: Vec<i32> = (100..164).collect();
    let expected: Vec<i32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    for target in available_targets() {
        let mut out = vec![0i32; 64];
        dispatch_to(
            AddI32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "failed for target {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Float mul+add kernel
// ---------------------------------------------------------------------------

struct FmaF32Kernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    c: &'a [f32],
    out: &'a mut [f32],
}

impl WithSimd for FmaF32Kernel<'_> {
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
                let vr = s.mul_add(va, vb, vc);
                s.store_u(vr, self.out.as_mut_ptr().add(i));
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
fn test_fma_f32_cross_target() {
    let n = 32;
    let a: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
    let b: Vec<f32> = (0..n).map(|i| i as f32 * 0.25 + 1.0).collect();
    let c: Vec<f32> = (0..n).map(|i| -(i as f32)).collect();

    let expected: Vec<f32> = a
        .iter()
        .zip(&b)
        .zip(&c)
        .map(|((x, y), z)| x * y + z)
        .collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            FmaF32Kernel {
                a: &a,
                b: &b,
                c: &c,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-4,
                "FMA mismatch at index {j} for target {target:?}: got {}, expected {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Bitwise kernel
// ---------------------------------------------------------------------------

struct BitwiseKernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    and_out: &'a mut [u32],
    or_out: &'a mut [u32],
    xor_out: &'a mut [u32],
}

impl WithSimd for BitwiseKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                s.store_u(s.and(va, vb), self.and_out.as_mut_ptr().add(i));
                s.store_u(s.or(va, vb), self.or_out.as_mut_ptr().add(i));
                s.store_u(s.xor(va, vb), self.xor_out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.and_out[i] = self.a[i] & self.b[i];
            self.or_out[i] = self.a[i] | self.b[i];
            self.xor_out[i] = self.a[i] ^ self.b[i];
            i += 1;
        }
    }
}

#[test]
fn test_bitwise_cross_target() {
    let n = 32;
    let a: Vec<u32> = (0..n as u32).map(|i| 0xDEAD_0000 | i).collect();
    let b: Vec<u32> = (0..n as u32).map(|i| 0x0000_BEEF ^ (i << 16)).collect();

    let exp_and: Vec<u32> = a.iter().zip(&b).map(|(x, y)| x & y).collect();
    let exp_or: Vec<u32> = a.iter().zip(&b).map(|(x, y)| x | y).collect();
    let exp_xor: Vec<u32> = a.iter().zip(&b).map(|(x, y)| x ^ y).collect();

    for target in available_targets() {
        let mut and_out = vec![0u32; n];
        let mut or_out = vec![0u32; n];
        let mut xor_out = vec![0u32; n];
        dispatch_to(
            BitwiseKernel {
                a: &a,
                b: &b,
                and_out: &mut and_out,
                or_out: &mut or_out,
                xor_out: &mut xor_out,
            },
            target,
        );
        assert_eq!(and_out, exp_and, "AND failed for {target:?}");
        assert_eq!(or_out, exp_or, "OR failed for {target:?}");
        assert_eq!(xor_out, exp_xor, "XOR failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Comparison / select kernel
// ---------------------------------------------------------------------------

struct MinMaxKernel<'a> {
    a: &'a [i32],
    b: &'a [i32],
    min_out: &'a mut [i32],
    max_out: &'a mut [i32],
}

impl WithSimd for MinMaxKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                s.store_u(s.min(va, vb), self.min_out.as_mut_ptr().add(i));
                s.store_u(s.max(va, vb), self.max_out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.min_out[i] = self.a[i].min(self.b[i]);
            self.max_out[i] = self.a[i].max(self.b[i]);
            i += 1;
        }
    }
}

#[test]
fn test_min_max_cross_target() {
    let n = 32;
    let a: Vec<i32> = (0..n as i32).map(|i| i * 3 - 20).collect();
    let b: Vec<i32> = (0..n as i32).map(|i| 30 - i * 2).collect();

    let exp_min: Vec<i32> = a.iter().zip(&b).map(|(x, y)| *x.min(y)).collect();
    let exp_max: Vec<i32> = a.iter().zip(&b).map(|(x, y)| *x.max(y)).collect();

    for target in available_targets() {
        let mut min_out = vec![0i32; n];
        let mut max_out = vec![0i32; n];
        dispatch_to(
            MinMaxKernel {
                a: &a,
                b: &b,
                min_out: &mut min_out,
                max_out: &mut max_out,
            },
            target,
        );
        assert_eq!(min_out, exp_min, "min failed for {target:?}");
        assert_eq!(max_out, exp_max, "max failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Shift kernel
// ---------------------------------------------------------------------------

struct ShiftKernel<'a> {
    input: &'a [u32],
    shl_out: &'a mut [u32],
    shr_out: &'a mut [u32],
}

impl WithSimd for ShiftKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let shl: S::Vec<u32> = s.shift_left::<u32, 4>(v);
                let shr: S::Vec<u32> = s.shift_right::<u32, 4>(v);
                s.store_u(shl, self.shl_out.as_mut_ptr().add(i));
                s.store_u(shr, self.shr_out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.shl_out[i] = self.input[i] << 4;
            self.shr_out[i] = self.input[i] >> 4;
            i += 1;
        }
    }
}

#[test]
fn test_shift_cross_target() {
    let n = 32;
    let input: Vec<u32> = (0..n as u32).map(|i| 0xABCD_0000 | (i * 17)).collect();
    let exp_shl: Vec<u32> = input.iter().map(|x| x << 4).collect();
    let exp_shr: Vec<u32> = input.iter().map(|x| x >> 4).collect();

    for target in available_targets() {
        let mut shl_out = vec![0u32; n];
        let mut shr_out = vec![0u32; n];
        dispatch_to(
            ShiftKernel {
                input: &input,
                shl_out: &mut shl_out,
                shr_out: &mut shr_out,
            },
            target,
        );
        assert_eq!(shl_out, exp_shl, "shift_left failed for {target:?}");
        assert_eq!(shr_out, exp_shr, "shift_right failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Saturating arithmetic kernel
// ---------------------------------------------------------------------------

struct SatAddU8Kernel<'a> {
    a: &'a [u8],
    b: &'a [u8],
    out: &'a mut [u8],
}

impl WithSimd for SatAddU8Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u8>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let vc = s.saturated_add(va, vb);
                s.store_u(vc, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i].saturating_add(self.b[i]);
            i += 1;
        }
    }
}

#[test]
fn test_saturating_add_u8_cross_target() {
    let n = 64;
    let a: Vec<u8> = (0..n as u8).map(|i| 200u8.wrapping_add(i)).collect();
    let b: Vec<u8> = (0..n as u8).map(|i| 100u8.wrapping_add(i * 3)).collect();
    let expected: Vec<u8> = a.iter().zip(&b).map(|(x, y)| x.saturating_add(*y)).collect();

    for target in available_targets() {
        let mut out = vec![0u8; n];
        dispatch_to(
            SatAddU8Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "sat_add u8 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Float sqrt kernel
// ---------------------------------------------------------------------------

struct SqrtF64Kernel<'a> {
    input: &'a [f64],
    out: &'a mut [f64],
}

impl WithSimd for SqrtF64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f64>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.sqrt(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].sqrt();
            i += 1;
        }
    }
}

#[test]
fn test_sqrt_f64_cross_target() {
    let n = 16;
    let input: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let expected: Vec<f64> = input.iter().map(|x| x.sqrt()).collect();

    for target in available_targets() {
        let mut out = vec![0f64; n];
        dispatch_to(
            SqrtF64Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-10,
                "sqrt mismatch at {j} for {target:?}: {} vs {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Compare + if_then_else (clamp) kernel
// ---------------------------------------------------------------------------

struct ClampF32Kernel<'a> {
    input: &'a [f32],
    lo: f32,
    hi: f32,
    out: &'a mut [f32],
}

impl WithSimd for ClampF32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let vlo = s.splat(self.lo);
                let vhi = s.splat(self.hi);
                // max(lo, min(hi, v))
                let clamped = s.max(vlo, s.min(vhi, v));
                s.store_u(clamped, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].max(self.lo).min(self.hi);
            i += 1;
        }
    }
}

#[test]
fn test_clamp_f32_cross_target() {
    let n = 32;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5 - 5.0).collect();
    let lo = -2.0f32;
    let hi = 8.0f32;
    let expected: Vec<f32> = input.iter().map(|x| x.max(lo).min(hi)).collect();

    for target in available_targets() {
        let mut out = vec![0f32; n];
        dispatch_to(
            ClampF32Kernel {
                input: &input,
                lo,
                hi,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-6,
                "clamp mismatch at {j} for {target:?}: {} vs {}",
                out[j],
                expected[j]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Reduction (sum) kernel
// ---------------------------------------------------------------------------

struct SumI32Kernel<'a> {
    data: &'a [i32],
}

impl WithSimd for SumI32Kernel<'_> {
    type Output = i32;
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        let n = self.data.len();
        let mut acc = unsafe { s.zero::<i32>() };
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.data.as_ptr().add(i));
                acc = s.add(acc, v);
            }
            i += lanes;
        }
        let mut total = unsafe { s.sum_of_lanes(acc) };
        // Scalar tail
        while i < n {
            total += self.data[i];
            i += 1;
        }
        total
    }
}

#[test]
fn test_sum_i32_cross_target() {
    let data: Vec<i32> = (1..=100).collect();
    let expected: i32 = data.iter().sum();

    for target in available_targets() {
        let result = dispatch_to(SumI32Kernel { data: &data }, target);
        assert_eq!(result, expected, "sum failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// dispatch() uses the best target
// ---------------------------------------------------------------------------

struct TargetReporter;

impl WithSimd for TargetReporter {
    type Output = usize;
    fn with_simd<S: SimdOps>(self, _s: S) -> Self::Output {
        S::VECTOR_BYTES
    }
}

#[test]
fn test_dispatch_picks_best() {
    let vector_bytes = dispatch(TargetReporter);
    let best = highway::dispatch::detect_best_target();
    let expected = match best {
        TargetId::Avx512 => 64,
        TargetId::Avx2 => 32,
        TargetId::Sse2 => 16,
        _ => 1, // Scalar
    };
    assert_eq!(vector_bytes, expected);
}

// ---------------------------------------------------------------------------
// 16-bit integer operations
// ---------------------------------------------------------------------------

struct MulI16Kernel<'a> {
    a: &'a [i16],
    b: &'a [i16],
    out: &'a mut [i16],
}

impl WithSimd for MulI16Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i16>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let vc = s.mul(va, vb);
                s.store_u(vc, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i].wrapping_mul(self.b[i]);
            i += 1;
        }
    }
}

#[test]
fn test_mul_i16_cross_target() {
    let n = 32;
    let a: Vec<i16> = (0..n as i16).map(|i| i - 10).collect();
    let b: Vec<i16> = (0..n as i16).map(|i| i * 2 + 1).collect();
    let expected: Vec<i16> = a.iter().zip(&b).map(|(x, y)| x.wrapping_mul(*y)).collect();

    for target in available_targets() {
        let mut out = vec![0i16; n];
        dispatch_to(
            MulI16Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "mul i16 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// 64-bit integer add
// ---------------------------------------------------------------------------

struct AddU64Kernel<'a> {
    a: &'a [u64],
    b: &'a [u64],
    out: &'a mut [u64],
}

impl WithSimd for AddU64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u64>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let vc = s.add(va, vb);
                s.store_u(vc, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i].wrapping_add(self.b[i]);
            i += 1;
        }
    }
}

#[test]
fn test_add_u64_cross_target() {
    let n = 16;
    let a: Vec<u64> = (0..n).map(|i| i as u64 * 1_000_000_007).collect();
    let b: Vec<u64> = (0..n).map(|i| i as u64 * 998_244_353 + 1).collect();
    let expected: Vec<u64> = a.iter().zip(&b).map(|(x, y)| x.wrapping_add(*y)).collect();

    for target in available_targets() {
        let mut out = vec![0u64; n];
        dispatch_to(
            AddU64Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "add u64 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Sub + abs pattern (f64)
// ---------------------------------------------------------------------------

struct AbsDiffF64Kernel<'a> {
    a: &'a [f64],
    b: &'a [f64],
    out: &'a mut [f64],
}

impl WithSimd for AbsDiffF64Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f64>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let diff = s.sub(va, vb);
                let abs = s.abs(diff);
                s.store_u(abs, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = (self.a[i] - self.b[i]).abs();
            i += 1;
        }
    }
}

#[test]
fn test_abs_diff_f64_cross_target() {
    let n = 16;
    let a: Vec<f64> = (0..n).map(|i| i as f64 * 1.5 - 10.0).collect();
    let b: Vec<f64> = (0..n).map(|i| i as f64 * 0.5 + 2.0).collect();
    let expected: Vec<f64> = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).collect();

    for target in available_targets() {
        let mut out = vec![0f64; n];
        dispatch_to(
            AbsDiffF64Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        for j in 0..n {
            assert!(
                (out[j] - expected[j]).abs() < 1e-10,
                "abs_diff mismatch at {j} for {target:?}",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Aligned load/store via AlignedVec
// ---------------------------------------------------------------------------

/// Test that AlignedVec allows aligned loads (not just unaligned).
#[cfg(feature = "alloc")]
struct AlignedAddKernel<'a> {
    a: &'a [f32],
    b: &'a [f32],
    out: &'a mut [f32],
}

#[cfg(feature = "alloc")]
impl WithSimd for AlignedAddKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<f32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            // SAFETY: AlignedVec guarantees 128-byte alignment, which exceeds
            // the requirement of any SIMD target (max 64 bytes for AVX-512).
            unsafe {
                let va = s.load(self.a.as_ptr().add(i));
                let vb = s.load(self.b.as_ptr().add(i));
                let vc = s.add(va, vb);
                s.store(vc, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i] + self.b[i];
            i += 1;
        }
    }
}

#[cfg(feature = "alloc")]
#[test]
fn test_aligned_load_store() {
    let a = aligned_vec_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let b = aligned_vec_from_slice(&[10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
    let mut out = aligned_vec_with_capacity::<f32>(8);
    out.resize(8, 0.0);

    // Verify alignment
    assert_eq!(a.as_ptr() as usize % 128, 0, "a should be 128-byte aligned");
    assert_eq!(b.as_ptr() as usize % 128, 0, "b should be 128-byte aligned");
    assert_eq!(
        out.as_ptr() as usize % 128,
        0,
        "out should be 128-byte aligned"
    );

    for target in available_targets() {
        out.fill(0.0);
        dispatch_to(
            AlignedAddKernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
        assert_eq!(&out[..], &expected[..], "aligned add failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// AbsDiff kernel (u32)
// ---------------------------------------------------------------------------

struct AbsDiffU32Kernel<'a> {
    a: &'a [u32],
    b: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for AbsDiffU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.a.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let va = s.load_u(self.a.as_ptr().add(i));
                let vb = s.load_u(self.b.as_ptr().add(i));
                let vr = s.abs_diff(va, vb);
                s.store_u(vr, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.a[i].abs_diff(self.b[i]);
            i += 1;
        }
    }
}

#[test]
fn test_abs_diff_u32_cross_target() {
    let n = 32;
    let a: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
    let b: Vec<u32> = (0..n as u32).map(|i| 50u32.wrapping_sub(i * 2)).collect();
    let expected: Vec<u32> = a
        .iter()
        .zip(&b)
        .map(|(x, y)| x.abs_diff(*y))
        .collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            AbsDiffU32Kernel {
                a: &a,
                b: &b,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "abs_diff u32 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Clamp kernel (i32)
// ---------------------------------------------------------------------------

struct ClampI32Kernel<'a> {
    input: &'a [i32],
    lo: i32,
    hi: i32,
    out: &'a mut [i32],
}

impl WithSimd for ClampI32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let vlo = s.splat(self.lo);
                let vhi = s.splat(self.hi);
                let clamped = s.clamp(v, vlo, vhi);
                s.store_u(clamped, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].max(self.lo).min(self.hi);
            i += 1;
        }
    }
}

#[test]
fn test_clamp_i32_cross_target() {
    let n = 32;
    let input: Vec<i32> = (0..n as i32).map(|i| i * 5 - 60).collect();
    let lo = -20i32;
    let hi = 50i32;
    let expected: Vec<i32> = input.iter().map(|x| *x.max(&lo).min(&hi)).collect();

    for target in available_targets() {
        let mut out = vec![0i32; n];
        dispatch_to(
            ClampI32Kernel {
                input: &input,
                lo,
                hi,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "clamp i32 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// PopulationCount kernel (u32)
// ---------------------------------------------------------------------------

struct PopcountU32Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for PopcountU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.population_count(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].count_ones();
            i += 1;
        }
    }
}

#[test]
fn test_popcount_u32_cross_target() {
    let n = 32;
    let input: Vec<u32> = (0..n as u32)
        .map(|i| 0xDEAD_BEEFu32 ^ (i.wrapping_mul(0x1234_5678)))
        .collect();
    let expected: Vec<u32> = input.iter().map(|x| x.count_ones()).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            PopcountU32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "popcount u32 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// LeadingZeroCount kernel (u32)
// ---------------------------------------------------------------------------

struct LzcntU32Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for LzcntU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.leading_zero_count(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].leading_zeros();
            i += 1;
        }
    }
}

#[test]
fn test_lzcnt_u32_cross_target() {
    let n = 32;
    let input: Vec<u32> = (0..n as u32).map(|i| 1u32 << (i % 32)).collect();
    let expected: Vec<u32> = input.iter().map(|x| x.leading_zeros()).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            LzcntU32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "lzcnt u32 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// BitsFromMask + FindLastTrue kernel
// ---------------------------------------------------------------------------

struct MaskOpsKernel<'a> {
    input: &'a [i32],
    threshold: i32,
}

impl WithSimd for MaskOpsKernel<'_> {
    type Output = (u64, Option<usize>, Option<usize>, usize);
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        if lanes > self.input.len() {
            return (0, None, None, 0);
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let threshold = s.splat(self.threshold);
            let mask = s.gt(v, threshold);
            let bits = s.bits_from_mask(mask);
            let first = s.find_first_true(mask);
            let last = s.find_last_true(mask);
            let count = s.count_true(mask);
            (bits, first, last, count)
        }
    }
}

#[test]
fn test_mask_ops_cross_target() {
    // Input: [0, 10, 20, 30, 5, 15, 25, 35, ...]
    let n = 16;
    let input: Vec<i32> = (0..n).map(|i: i32| i * 10 - 5 * (i % 3)).collect();
    let threshold = 15i32;

    // Expected for scalar (1 lane):
    // Lane 0: input[0] = 0, 0 > 15 = false -> bits=0, first=None, last=None, count=0
    // But for multi-lane targets, more lanes are involved.
    // We just verify consistency across all targets with the same lane count group.

    let targets = available_targets();
    // Group by vector width and verify consistency within each group
    let mut results: Vec<(TargetId, (u64, Option<usize>, Option<usize>, usize))> = Vec::new();
    for target in &targets {
        let r = dispatch_to(
            MaskOpsKernel {
                input: &input,
                threshold,
            },
            *target,
        );
        results.push((*target, r));
    }

    // Verify first_true and find_last_true are consistent with bits_from_mask
    for (target, (bits, first, last, count)) in &results {
        assert_eq!(
            *count,
            bits.count_ones() as usize,
            "count_true != popcount(bits_from_mask) for {target:?}"
        );
        if *count > 0 {
            assert!(first.is_some(), "first should be Some for {target:?}");
            assert!(last.is_some(), "last should be Some for {target:?}");
            let first_idx = first.unwrap();
            let last_idx = last.unwrap();
            assert!(
                (bits >> first_idx) & 1 == 1,
                "first_true bit not set in bits_from_mask for {target:?}"
            );
            assert!(
                (bits >> last_idx) & 1 == 1,
                "find_last_true bit not set in bits_from_mask for {target:?}"
            );
            assert!(first_idx <= last_idx, "first > last for {target:?}");
        } else {
            assert!(first.is_none(), "first should be None for {target:?}");
            assert!(last.is_none(), "last should be None for {target:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// Reverse2 kernel (u32)
// ---------------------------------------------------------------------------

struct Reverse2U32Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for Reverse2U32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.reverse2(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        // Scalar tail: swap pairs
        while i + 1 < n {
            self.out[i] = self.input[i + 1];
            self.out[i + 1] = self.input[i];
            i += 2;
        }
        if i < n {
            self.out[i] = self.input[i];
        }
    }
}

#[test]
fn test_reverse2_u32_cross_target() {
    let n = 16;
    let input: Vec<u32> = (0..n as u32).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            Reverse2U32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // Verify round-trip: reverse2(reverse2(x)) == x
        let mut roundtrip = vec![0u32; n];
        dispatch_to(
            Reverse2U32Kernel {
                input: &out,
                out: &mut roundtrip,
            },
            target,
        );
        assert_eq!(
            roundtrip, input,
            "reverse2 round-trip failed for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// Compress kernel (u32)
// ---------------------------------------------------------------------------

struct CompressU32Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for CompressU32Kernel<'_> {
    type Output = usize;
    fn with_simd<S: SimdOps>(self, s: S) -> usize {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return 0;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            // Keep only even values (v % 2 == 0)
            let one = s.splat(1u32);
            // v & 1 == 0 means even
            let remainder = s.and(v, one);
            let zero = s.zero::<u32>();
            let mask = s.eq(remainder, zero);
            s.compress_store(v, mask, self.out.as_mut_ptr())
        }
    }
}

#[test]
fn test_compress_u32_cross_target() {
    let n = 16;
    let input: Vec<u32> = (0..n as u32).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        let count = dispatch_to(
            CompressU32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // Verify: the first `count` elements should be the even values
        let lanes = dispatch_to(U32Lanes, target);
        // Only the first `lanes` values from input are processed
        let expected_evens: Vec<u32> = input[..lanes].iter().copied().filter(|x| x % 2 == 0).collect();
        assert_eq!(count, expected_evens.len(), "compress count wrong for {target:?}");
        assert_eq!(
            &out[..count],
            &expected_evens[..],
            "compress values wrong for {target:?}"
        );
    }
}

// ---------------------------------------------------------------------------
// MaskedLoad + BlendedStore kernel
// ---------------------------------------------------------------------------

struct MaskedLoadStoreKernel<'a> {
    input: &'a [i32],
    out: &'a mut [i32],
}

impl WithSimd for MaskedLoadStoreKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<i32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            // Load only where value > 0
            let threshold = s.splat(0i32);
            let all_data = s.load_u(self.input.as_ptr());
            let mask = s.gt(all_data, threshold);
            let loaded = s.masked_load(mask, self.input.as_ptr());
            // Store only the positive values
            s.blended_store(loaded, mask, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_masked_load_store_cross_target() {
    let n = 16;
    let input: Vec<i32> = (0..n as i32).map(|i| i - 5).collect();
    // input: [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    for target in available_targets() {
        let mut out = vec![-999i32; n];
        dispatch_to(
            MaskedLoadStoreKernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        let lanes = dispatch_to(I32Lanes, target);
        // Verify: positive input values should be written, negative ones left as -999
        for i in 0..lanes.min(n) {
            if input[i] > 0 {
                assert_eq!(
                    out[i], input[i],
                    "masked_load+blended_store: positive value wrong at {i} for {target:?}"
                );
            } else {
                assert_eq!(
                    out[i], -999,
                    "masked_load+blended_store: negative value should be untouched at {i} for {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ReverseLaneBytes kernel (u32)
// ---------------------------------------------------------------------------

struct ReverseLaneBytesU32Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for ReverseLaneBytesU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.input.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let v = s.load_u(self.input.as_ptr().add(i));
                let r = s.reverse_lane_bytes(v);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        while i < n {
            self.out[i] = self.input[i].swap_bytes();
            i += 1;
        }
    }
}

#[test]
fn test_reverse_lane_bytes_u32_cross_target() {
    let n = 16;
    let input: Vec<u32> = (0..n as u32).map(|i| 0x01_02_03_04u32.wrapping_add(i * 0x11_11_11_11)).collect();
    let expected: Vec<u32> = input.iter().map(|x| x.swap_bytes()).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            ReverseLaneBytesU32Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        assert_eq!(out, expected, "reverse_lane_bytes u32 failed for {target:?}");
    }
}

// ---------------------------------------------------------------------------
// XorMask kernel
// ---------------------------------------------------------------------------

struct XorMaskKernel<'a> {
    a: &'a [i32],
    b: &'a [i32],
}

impl WithSimd for XorMaskKernel<'_> {
    type Output = Vec<bool>;
    fn with_simd<S: SimdOps>(self, s: S) -> Vec<bool> {
        let lanes = s.lanes::<i32>();
        let n = lanes.min(self.a.len());
        let mut result = Vec::new();
        if n == 0 {
            return result;
        }
        unsafe {
            let va = s.load_u(self.a.as_ptr());
            let vb = s.load_u(self.b.as_ptr());
            let zero = s.zero::<i32>();
            let ma = s.gt(va, zero); // a > 0
            let mb = s.gt(vb, zero); // b > 0
            let xored = s.xor_mask(ma, mb);
            for i in 0..n {
                let v = s.vec_from_mask(xored);
                let lane_val = s.extract_lane(v, i);
                result.push(lane_val != 0);
            }
        }
        result
    }
}

#[test]
fn test_xor_mask_cross_target() {
    let n = 16;
    let a: Vec<i32> = (0..n).map(|i: i32| i - 3).collect();
    let b: Vec<i32> = (0..n).map(|i: i32| 5 - i).collect();

    for target in available_targets() {
        let result = dispatch_to(
            XorMaskKernel { a: &a, b: &b },
            target,
        );
        // Verify XOR semantics: true iff exactly one of (a>0, b>0) is true
        for (i, &xor_val) in result.iter().enumerate() {
            let a_pos = a[i] > 0;
            let b_pos = b[i] > 0;
            let expected = a_pos ^ b_pos;
            assert_eq!(
                xor_val, expected,
                "xor_mask wrong at {i}: a={}, b={}, a>0={}, b>0={}, expected xor={}, got {} for {target:?}",
                a[i], b[i], a_pos, b_pos, expected, xor_val
            );
        }
    }
}

// ---------------------------------------------------------------------------
// OddEven kernel (u32)
// ---------------------------------------------------------------------------

struct OddEvenU32Kernel<'a> {
    odd_src: &'a [u32],
    even_src: &'a [u32],
    out: &'a mut [u32],
}

impl WithSimd for OddEvenU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let n = self.odd_src.len();
        let mut i = 0;
        while i + lanes <= n {
            unsafe {
                let vodd = s.load_u(self.odd_src.as_ptr().add(i));
                let veven = s.load_u(self.even_src.as_ptr().add(i));
                let r = s.odd_even(vodd, veven);
                s.store_u(r, self.out.as_mut_ptr().add(i));
            }
            i += lanes;
        }
    }
}

#[test]
fn test_odd_even_u32_cross_target() {
    let n = 16;
    let odd_src: Vec<u32> = (0..n as u32).map(|i| 100 + i).collect();
    let even_src: Vec<u32> = (0..n as u32).map(|i| 200 + i).collect();

    for target in available_targets() {
        let mut out = vec![0u32; n];
        dispatch_to(
            OddEvenU32Kernel {
                odd_src: &odd_src,
                even_src: &even_src,
                out: &mut out,
            },
            target,
        );
        let lanes = dispatch_to(U32Lanes, target);
        // Verify: even lanes from even_src, odd lanes from odd_src
        for i in 0..lanes.min(n) {
            if i % 2 == 0 {
                assert_eq!(
                    out[i], even_src[i],
                    "odd_even: even lane {i} wrong for {target:?}"
                );
            } else {
                assert_eq!(
                    out[i], odd_src[i],
                    "odd_even: odd lane {i} wrong for {target:?}"
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// LoadDup128 kernel (u32)
// ---------------------------------------------------------------------------

struct LoadDup128Kernel<'a> {
    input: &'a [u32],  // at least 4 elements (128 bits)
    out: &'a mut [u32],
}

impl WithSimd for LoadDup128Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        unsafe {
            let v = s.load_dup128(self.input.as_ptr());
            s.store_u(v, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_load_dup128_u32_cross_target() {
    let input: Vec<u32> = vec![10, 20, 30, 40];

    for target in available_targets() {
        let lanes = dispatch_to(U32Lanes, target);
        let mut out = vec![0u32; lanes];
        dispatch_to(
            LoadDup128Kernel {
                input: &input,
                out: &mut out,
            },
            target,
        );
        // The 128-bit pattern [10, 20, 30, 40] should repeat across the vector
        for i in 0..lanes {
            assert_eq!(
                out[i],
                input[i % 4],
                "load_dup128: lane {i} wrong for {target:?}, got {}, expected {}",
                out[i],
                input[i % 4]
            );
        }
    }
}

// ---------------------------------------------------------------------------
// simd_fn! macro tests
// ---------------------------------------------------------------------------

#[test]
fn test_simd_fn_no_captures() {
    let result: i32 = simd_fn!(|s| -> i32 {
        unsafe {
            let v = s.splat(7i32);
            s.extract_lane(v, 0)
        }
    });
    assert_eq!(result, 7);
}

#[test]
fn test_simd_fn_with_captures() {
    let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let ptr = data.as_ptr();
    let len = data.len();

    let sum: f32 = simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
        let lanes = s.lanes::<f32>();
        let mut acc = unsafe { s.zero::<f32>() };
        let mut i = 0;
        while i + lanes <= len {
            unsafe {
                let v = s.load_u(ptr.add(i));
                acc = s.add(acc, v);
            }
            i += lanes;
        }
        let mut total = unsafe { s.sum_of_lanes(acc) };
        while i < len {
            total += unsafe { *ptr.add(i) };
            i += 1;
        }
        total
    });
    assert_eq!(sum, 36.0);
}

// ---------------------------------------------------------------------------
// simd_fn! as struct methods — peak, sum, find
// ---------------------------------------------------------------------------

struct AudioBuffer {
    samples: Vec<f32>,
}

impl AudioBuffer {
    fn peak(&self) -> f32 {
        let ptr = self.samples.as_ptr();
        let len = self.samples.len();
        simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
            let lanes = s.lanes::<f32>();
            let mut best = unsafe { s.splat(0.0f32) };
            let mut i = 0;
            while i + lanes <= len {
                unsafe {
                    let v = s.load_u(ptr.add(i));
                    let a = s.abs(v);
                    best = s.max(best, a);
                }
                i += lanes;
            }
            let mut peak = unsafe { s.max_of_lanes(best) };
            while i < len {
                let a = unsafe { *ptr.add(i) }.abs();
                if a > peak { peak = a; }
                i += 1;
            }
            peak
        })
    }

    fn sum(&self) -> f32 {
        let ptr = self.samples.as_ptr();
        let len = self.samples.len();
        simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
            let lanes = s.lanes::<f32>();
            let mut acc = unsafe { s.zero::<f32>() };
            let mut i = 0;
            while i + lanes <= len {
                unsafe {
                    let v = s.load_u(ptr.add(i));
                    acc = s.add(acc, v);
                }
                i += lanes;
            }
            let mut total = unsafe { s.sum_of_lanes(acc) };
            while i < len {
                total += unsafe { *ptr.add(i) };
                i += 1;
            }
            total
        })
    }

    fn find(&self, needle: f32) -> Option<usize> {
        let ptr = self.samples.as_ptr();
        let len = self.samples.len();
        simd_fn!([ptr: *const f32, len: usize, needle: f32] |s| -> Option<usize> {
            let lanes = s.lanes::<f32>();
            let needle_v = unsafe { s.splat(needle) };
            let mut i = 0;
            while i + lanes <= len {
                unsafe {
                    let v = s.load_u(ptr.add(i));
                    let mask = s.eq(v, needle_v);
                    if let Some(idx) = s.find_first_true(mask) {
                        return Some(i + idx);
                    }
                }
                i += lanes;
            }
            while i < len {
                if unsafe { *ptr.add(i) } == needle {
                    return Some(i);
                }
                i += 1;
            }
            None
        })
    }
}

#[test]
fn test_struct_methods_peak() {
    let buf = AudioBuffer {
        samples: vec![0.1, -0.5, 0.3, 0.8, -0.2, 0.5, 0.0, 0.4],
    };
    assert_eq!(buf.peak(), 0.8);
}

#[test]
fn test_struct_methods_sum() {
    let buf = AudioBuffer {
        samples: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    };
    assert_eq!(buf.sum(), 36.0);
}

#[test]
fn test_struct_methods_find() {
    let buf = AudioBuffer {
        samples: vec![0.1, -0.5, 0.3, 0.8, -0.2, 0.5, 0.0, 0.4],
    };
    assert_eq!(buf.find(0.3), Some(2));
    assert_eq!(buf.find(0.8), Some(3));
    assert_eq!(buf.find(9.9), None);
}

// ---------------------------------------------------------------------------
// SlideDownLanes kernel (u32)
// ---------------------------------------------------------------------------

struct SlideDownU32Kernel<'a> {
    input: &'a [u32],
    out: &'a mut [u32],
    n_slide: usize,
}

impl WithSimd for SlideDownU32Kernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        if lanes > self.input.len() {
            return;
        }
        unsafe {
            let v = s.load_u(self.input.as_ptr());
            let r = s.slide_down_lanes(v, self.n_slide);
            s.store_u(r, self.out.as_mut_ptr());
        }
    }
}

#[test]
fn test_slide_down_u32_cross_target() {
    let n = 16;
    let input: Vec<u32> = (1..=n as u32).collect();

    for target in available_targets() {
        let lanes = dispatch_to(U32Lanes, target);
        let slide_n = 1;
        let mut out = vec![0u32; lanes];
        dispatch_to(
            SlideDownU32Kernel {
                input: &input,
                out: &mut out,
                n_slide: slide_n,
            },
            target,
        );
        // After slide_down by 1: lane[i] = input[i+1], last lane = 0
        for i in 0..lanes {
            let expected = if i + slide_n < lanes {
                input[i + slide_n]
            } else {
                0
            };
            assert_eq!(
                out[i], expected,
                "slide_down: lane {i} wrong for {target:?}"
            );
        }
    }
}
