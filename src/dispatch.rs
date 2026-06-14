//! Runtime CPU feature detection and target dispatch.
//!
//! Detects the best available SIMD target at runtime and dispatches
//! user-provided kernels through `#[target_feature]`-annotated trampolines.

use core::sync::atomic::{AtomicU64, Ordering};

use crate::backend::scalar::Scalar;
use crate::ops::SimdOps;

// ---------------------------------------------------------------------------
// Target identifiers (stored in the atomic cache)
// ---------------------------------------------------------------------------

/// Numeric identifiers for each detected target, stored in [`CHOSEN`].
///
/// The dispatch system caches the detected target as a `u64` in an atomic.
/// This enum provides a typed view of those values.
/// Value 0 is reserved for "not yet detected".
#[repr(u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetId {
    /// Detection has not yet been performed.
    Uninitialized = 0,
    /// Portable scalar fallback (all platforms).
    Scalar = 1,
    /// x86_64 SSE2 (128-bit vectors).
    Sse2 = 2,
    /// x86_64 AVX2 + FMA (256-bit vectors).
    Avx2 = 3,
    /// x86_64 AVX-512F/BW/CD/DQ/VL (512-bit vectors).
    Avx512 = 4,
    /// aarch64 NEON (128-bit vectors, future).
    Neon = 5,
}

impl TargetId {
    fn from_u64(v: u64) -> Self {
        match v {
            1 => Self::Scalar,
            2 => Self::Sse2,
            3 => Self::Avx2,
            4 => Self::Avx512,
            5 => Self::Neon,
            _ => Self::Uninitialized,
        }
    }

    /// Returns `true` if this target is a SIMD-accelerated target (not scalar).
    pub fn is_accelerated(self) -> bool {
        !matches!(self, Self::Scalar | Self::Uninitialized)
    }
}

/// Cached best target. 0 = not yet detected.
static CHOSEN: AtomicU64 = AtomicU64::new(0);

/// Detect the best available SIMD target for the current CPU.
///
/// Results are cached in an atomic for subsequent calls.
/// This function is safe to call from any thread.
pub fn detect_best_target() -> TargetId {
    let cached = CHOSEN.load(Ordering::Relaxed);
    if cached != 0 {
        return TargetId::from_u64(cached);
    }

    let best = detect_impl();
    CHOSEN.store(best as u64, Ordering::Relaxed);
    best
}

/// Reset the cached target detection (useful for testing).
///
/// After calling this, the next call to [`detect_best_target`] will re-detect.
#[doc(hidden)]
pub fn reset_detection() {
    CHOSEN.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Platform-specific detection
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512bw")
        && is_x86_feature_detected!("avx512cd")
        && is_x86_feature_detected!("avx512dq")
        && is_x86_feature_detected!("avx512vl")
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_avx2() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(target_arch = "x86_64")]
#[inline]
fn has_sse2() -> bool {
    is_x86_feature_detected!("sse2")
}

#[cfg(target_arch = "x86_64")]
fn detect_impl() -> TargetId {
    if has_avx512() {
        return TargetId::Avx512;
    }
    if has_avx2() {
        return TargetId::Avx2;
    }
    if has_sse2() {
        return TargetId::Sse2;
    }
    TargetId::Scalar
}

#[cfg(target_arch = "aarch64")]
fn detect_impl() -> TargetId {
    // NEON is mandatory on aarch64, but we don't have a backend yet.
    // When the neon backend is added, this will return TargetId::Neon
    TargetId::Scalar
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn detect_impl() -> TargetId {
    TargetId::Scalar
}

// ---------------------------------------------------------------------------
// WithSimd trait — user-defined kernels
// ---------------------------------------------------------------------------

/// Trait for user-defined SIMD kernels that can be dispatched across targets.
///
/// Implement this trait to write portable code that gets compiled and run
/// on the best available target. The same kernel type can run on any platform;
/// the dispatch system selects the optimal backend at runtime.
///
/// # Example
/// ```
/// use highway::{WithSimd, SimdOps, dispatch};
///
/// struct AddKernel<'a> {
///     a: &'a [f32],
///     b: &'a [f32],
///     out: &'a mut [f32],
/// }
///
/// impl WithSimd for AddKernel<'_> {
///     type Output = ();
///     fn with_simd<S: SimdOps>(self, _s: S) -> Self::Output {
///         // Portable SIMD code here
///         for i in 0..self.a.len() {
///             self.out[i] = self.a[i] + self.b[i];
///         }
///     }
/// }
/// ```
pub trait WithSimd {
    /// The return type of the kernel.
    type Output;

    /// Execute the kernel with the given SIMD target.
    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output;
}

// ---------------------------------------------------------------------------
// dispatch() — runtime target selection
// ---------------------------------------------------------------------------

/// Dispatch a kernel to the best available SIMD target.
///
/// Detects CPU features (cached after first call) and invokes
/// `kernel.with_simd(target)` where `target` is the best supported.
///
/// Each branch is compiled with the appropriate `#[target_feature]` annotations
/// so the compiler can generate optimized code for that target.
///
/// This function has the same signature on all platforms — on non-x86_64 it
/// currently always dispatches to the scalar backend.
pub fn dispatch<K: WithSimd>(kernel: K) -> K::Output {
    let target = detect_best_target();
    dispatch_to(kernel, target)
}

/// Dispatch a kernel to a specific target (useful for testing or forced selection).
///
/// The requested target's CPU features are verified at runtime before use. If
/// the running CPU does not support the requested target (or the target is not
/// compiled for this platform), the kernel falls back to the scalar backend.
/// This makes the function sound to call with any [`TargetId`] on any CPU.
pub fn dispatch_to<K: WithSimd>(kernel: K, target: TargetId) -> K::Output {
    #[cfg(target_arch = "x86_64")]
    {
        match target {
            // Each arm re-checks the required features at runtime. The guard is
            // the safety precondition for calling the `#[target_feature]` trampoline
            TargetId::Avx512 if has_avx512() => {
                // SAFETY: has_avx512() confirmed the required AVX-512 features at runtime
                return unsafe { dispatch_avx512(kernel) };
            }
            TargetId::Avx2 if has_avx2() => {
                // SAFETY: has_avx2() confirmed AVX2+FMA at runtime
                return unsafe { dispatch_avx2(kernel) };
            }
            TargetId::Sse2 if has_sse2() => {
                // SAFETY: has_sse2() confirmed SSE2 at runtime
                return unsafe { dispatch_sse2(kernel) };
            }
            // Requested target unsupported on this CPU -> fall through to scalar
            _ => {}
        }
    }
    let _ = target;
    kernel.with_simd(Scalar)
}

// ---------------------------------------------------------------------------
// simd_fn! — macro-based ergonomic dispatch
// ---------------------------------------------------------------------------

/// Dispatch a SIMD kernel inline without defining a separate struct.
///
/// Rust closures cannot be generic over `S: SimdOps`, so a trait-object
/// closure API is impossible. This macro generates the [`WithSimd`] struct
/// and impl, then calls [`dispatch`].
///
/// # Forms
///
/// | Syntax | When to use |
/// |--------|------------|
/// | `simd_fn!(\|s\| -> T { ... })` | No external data needed |
/// | `simd_fn!([a: A, b: B] \|s\| -> T { ... })` | Captures with types |
///
/// # Examples
///
/// **No captures:**
/// ```
/// use highway::simd_fn;
/// let result: i32 = simd_fn!(|s| -> i32 { 42 });
/// assert_eq!(result, 42);
/// ```
///
/// **With captures:**
/// ```
/// use highway::simd_fn;
///
/// let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
/// let ptr = data.as_ptr();
/// let len = data.len();
///
/// let sum: f32 = simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
///     let lanes = s.lanes::<f32>();
///     let mut acc = unsafe { s.zero::<f32>() };
///     let mut i = 0;
///     while i + lanes <= len {
///         unsafe {
///             let v = s.load_u(ptr.add(i));
///             acc = s.add(acc, v);
///         }
///         i += lanes;
///     }
///     let mut total = unsafe { s.sum_of_lanes(acc) };
///     while i < len {
///         total += unsafe { *ptr.add(i) };
///         i += 1;
///     }
///     total
/// });
/// assert_eq!(sum, 36.0);
/// ```
///
/// # As struct methods
///
/// Extract `self` fields into locals, then capture them — each method gets
/// its own SIMD kernel without a separate top-level struct:
///
/// ```
/// use highway::{simd_fn, SimdOps};
///
/// struct AudioBuffer {
///     samples: Vec<f32>,
/// }
///
/// impl AudioBuffer {
///     fn peak(&self) -> f32 {
///         let ptr = self.samples.as_ptr();
///         let len = self.samples.len();
///         simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
///             let lanes = s.lanes::<f32>();
///             let mut best = unsafe { s.splat(0.0f32) };
///             let mut i = 0;
///             while i + lanes <= len {
///                 unsafe {
///                     let v = s.load_u(ptr.add(i));
///                     let a = s.abs(v);
///                     best = s.max(best, a);
///                 }
///                 i += lanes;
///             }
///             let mut peak = unsafe { s.max_of_lanes(best) };
///             while i < len {
///                 let a = unsafe { *ptr.add(i) }.abs();
///                 if a > peak { peak = a; }
///                 i += 1;
///             }
///             peak
///         })
///     }
///
///     fn find(&self, needle: f32) -> Option<usize> {
///         let ptr = self.samples.as_ptr();
///         let len = self.samples.len();
///         simd_fn!([ptr: *const f32, len: usize, needle: f32] |s| -> Option<usize> {
///             let lanes = s.lanes::<f32>();
///             let needle_v = unsafe { s.splat(needle) };
///             let mut i = 0;
///             while i + lanes <= len {
///                 unsafe {
///                     let v = s.load_u(ptr.add(i));
///                     let mask = s.eq(v, needle_v);
///                     if let Some(idx) = s.find_first_true(mask) {
///                         return Some(i + idx);
///                     }
///                 }
///                 i += lanes;
///             }
///             while i < len {
///                 if unsafe { *ptr.add(i) } == needle {
///                     return Some(i);
///                 }
///                 i += 1;
///             }
///             None
///         })
///     }
/// }
///
/// let buf = AudioBuffer { samples: vec![0.1, -0.5, 0.3, 0.8, -0.2, 0.5, 0.0, 0.4] };
/// assert_eq!(buf.peak(), 0.8);
/// assert_eq!(buf.find(0.3), Some(2));
/// assert_eq!(buf.find(9.9), None);
/// ```
///
/// For kernels with lifetimes or generics,
/// define a struct and implement [`WithSimd`] manually.
#[macro_export]
macro_rules! simd_fn {
    // No captures
    (|$s:ident| -> $ret:ty $body:block) => {{
        struct __K;
        impl $crate::WithSimd for __K {
            type Output = $ret;
            #[inline(always)]
            fn with_simd<__S: $crate::SimdOps>(self, $s: __S) -> $ret $body
        }
        $crate::dispatch(__K)
    }};

    // Typed captures: [name: Type, ...]
    ([$($cap:ident : $cap_ty:ty),* $(,)?] |$s:ident| -> $ret:ty $body:block) => {{
        struct __K { $($cap: $cap_ty,)* }
        impl $crate::WithSimd for __K {
            type Output = $ret;
            #[inline(always)]
            fn with_simd<__S: $crate::SimdOps>(self, $s: __S) -> $ret {
                #[allow(unused_variables)]
                let __K { $($cap,)* } = self;
                $body
            }
        }
        $crate::dispatch(__K { $($cap,)* })
    }};
}

// ---------------------------------------------------------------------------
// Target-feature trampolines (x86_64)
// ---------------------------------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline(never)]
unsafe fn dispatch_sse2<K: WithSimd>(kernel: K) -> K::Output {
    kernel.with_simd(crate::backend::sse2::Sse2::new())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[inline(never)]
unsafe fn dispatch_avx2<K: WithSimd>(kernel: K) -> K::Output {
    kernel.with_simd(crate::backend::avx2::Avx2::new())
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw,avx512cd,avx512dq,avx512vl")]
#[inline(never)]
unsafe fn dispatch_avx512<K: WithSimd>(kernel: K) -> K::Output {
    kernel.with_simd(crate::backend::avx512::Avx512::new())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct ScalarAddKernel {
        a: i32,
        b: i32,
    }

    impl WithSimd for ScalarAddKernel {
        type Output = i32;
        fn with_simd<S: SimdOps>(self, _s: S) -> i32 {
            self.a + self.b
        }
    }

    #[test]
    fn test_dispatch_returns_correct_result() {
        let result = dispatch(ScalarAddKernel { a: 10, b: 32 });
        assert_eq!(result, 42);
    }

    #[test]
    fn test_detect_target_not_uninitialized() {
        let target = detect_best_target();
        assert_ne!(target, TargetId::Uninitialized);
    }

    #[test]
    fn test_dispatch_to_scalar() {
        let result = dispatch_to(ScalarAddKernel { a: 5, b: 7 }, TargetId::Scalar);
        assert_eq!(result, 12);
    }

    #[test]
    fn test_target_is_accelerated() {
        assert!(!TargetId::Scalar.is_accelerated());
        assert!(!TargetId::Uninitialized.is_accelerated());
        assert!(TargetId::Sse2.is_accelerated());
        assert!(TargetId::Avx2.is_accelerated());
        assert!(TargetId::Avx512.is_accelerated());
        assert!(TargetId::Neon.is_accelerated());
    }

    /// Verifies that dispatch works identically through all available targets.
    #[test]
    fn test_dispatch_consistency_across_targets() {
        struct MulKernel(i32, i32);
        impl WithSimd for MulKernel {
            type Output = i32;
            fn with_simd<S: SimdOps>(self, _s: S) -> i32 {
                self.0 * self.1
            }
        }

        let expected = 6 * 7;
        // Scalar always available
        assert_eq!(dispatch_to(MulKernel(6, 7), TargetId::Scalar), expected);
        // Whatever the best target is
        assert_eq!(dispatch(MulKernel(6, 7)), expected);
    }

    /// Soundness: `dispatch_to` must be safe to call with *any* `TargetId`,
    /// even one the running CPU does not support. Forcing an unsupported target
    /// must fall back to scalar (returning the correct result) rather than
    /// executing unsupported instructions (UB / SIGILL).
    #[test]
    fn test_dispatch_to_unsupported_target_falls_back() {
        struct MulKernel(i32, i32);
        impl WithSimd for MulKernel {
            type Output = i32;
            fn with_simd<S: SimdOps>(self, _s: S) -> i32 {
                self.0 * self.1
            }
        }

        // Every variant, regardless of CPU support, yields the correct result.
        // On a CPU lacking a given target this exercises the runtime-feature
        // fallback path; on a capable CPU it runs natively. Either way: no UB.
        for target in [
            TargetId::Scalar,
            TargetId::Sse2,
            TargetId::Avx2,
            TargetId::Avx512,
            TargetId::Neon,
        ] {
            assert_eq!(dispatch_to(MulKernel(6, 7), target), 42);
        }
    }
}
