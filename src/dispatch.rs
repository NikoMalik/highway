//! Runtime CPU feature detection and target dispatch.
//!
//! Detects the best available SIMD target at runtime and dispatches
//! user-provided kernels through `#[target_feature]`-annotated trampolines.

use core::sync::atomic::{AtomicU8, Ordering};

use crate::backend::scalar::Scalar;
use crate::ops::{SimdCrypto, SimdOps};

// ---------------------------------------------------------------------------
// Target identifiers
// ---------------------------------------------------------------------------

/// The backend a [`Tier`] runs on.
///
/// Value 0 means "not yet detected".
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
    /// Returns `true` if this target is a SIMD-accelerated target (not scalar).
    pub fn is_accelerated(self) -> bool {
        !matches!(self, Self::Scalar | Self::Uninitialized)
    }
}

/// Cached [`Tier`] of this CPU as `u8`, 0 = not yet detected.
static CHOSEN: AtomicU8 = AtomicU8::new(0);

/// Detect the best available SIMD target for the current CPU.
///
/// The full [`Cpu`] detection runs once and is cached; later calls are one
/// relaxed atomic load.
pub fn detect_best_target() -> TargetId {
    Cpu::cached().target()
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
fn detect_tier() -> Tier {
    use crate::features::{
        AES, BITALG, PCLMUL, SSE41, SSSE3, VBMI, VBMI2, VPOPCNTDQ,
    };
    let mut ext = 0;
    for (on, flag) in [
        (is_x86_feature_detected!("ssse3"), SSSE3),
        (is_x86_feature_detected!("sse4.1"), SSE41),
        (is_x86_feature_detected!("aes"), AES),
        (is_x86_feature_detected!("pclmulqdq"), PCLMUL),
        (is_x86_feature_detected!("avx512vbmi"), VBMI),
        (is_x86_feature_detected!("avx512vbmi2"), VBMI2),
        (is_x86_feature_detected!("avx512bitalg"), BITALG),
        (is_x86_feature_detected!("avx512vpopcntdq"), VPOPCNTDQ),
    ] {
        if on {
            ext |= flag;
        }
    }
    let base = if has_avx512() {
        3
    } else if has_avx2() {
        2
    } else if has_sse2() {
        1
    } else {
        0
    };
    Tier::ALL
        .into_iter()
        .rev()
        .find(|t| t.base() <= base && t.extensions() & !ext == 0)
        .unwrap_or(Tier::Scalar)
}

#[cfg(not(target_arch = "x86_64"))]
fn detect_tier() -> Tier {
    // NEON is mandatory on aarch64, but there is no NEON backend yet.
    Tier::Scalar
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
/// Uses the [`Cpu`] detected once and cached, see [`Cpu::dispatch`].
pub fn dispatch<K: WithSimd>(kernel: K) -> K::Output {
    Cpu::cached().dispatch(kernel)
}

/// Dispatch a kernel to a specific target (useful for testing or forced selection).
///
/// When the running CPU lacks `target`, the kernel runs on the best supported
/// tier below it (scalar at worst). Sound to call with any [`TargetId`].
pub fn dispatch_to<K: WithSimd>(kernel: K, target: TargetId) -> K::Output {
    let top = match target {
        TargetId::Avx512 => Tier::Avx512Ext,
        TargetId::Avx2 => Tier::Avx2Crypto,
        TargetId::Sse2 => Tier::Sse2Crypto,
        TargetId::Scalar | TargetId::Neon | TargetId::Uninitialized => {
            Tier::Scalar
        }
    };
    Cpu::cached().capped(top).dispatch(kernel)
}

// ---------------------------------------------------------------------------
// Tier and Cpu — detect once at startup, no checks per operation
// ---------------------------------------------------------------------------

/// A backend plus the optional extensions its token carries.
///
/// Each tier maps to one token type (`Sse2<EXT>`, `Avx2<EXT>`, `Avx512<EXT>`)
/// and one `#[target_feature]` trampoline, so operations test their extensions
/// at compile time.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Portable scalar fallback.
    Scalar = 1,
    /// SSE2 only.
    Sse2 = 2,
    /// SSE2 + SSSE3 + SSE4.1 + AES + PCLMUL.
    Sse2Crypto = 3,
    /// AVX2 + FMA.
    Avx2 = 4,
    /// AVX2 + FMA + AES + PCLMUL.
    Avx2Crypto = 5,
    /// AVX-512F/BW/CD/DQ/VL.
    Avx512 = 6,
    /// AVX-512 + AES + PCLMUL.
    Avx512Crypto = 7,
    /// AVX-512 + AES + PCLMUL + VBMI + VBMI2 + BITALG + VPOPCNTDQ.
    Avx512Ext = 8,
}

impl Tier {
    /// Every tier, lowest first.
    pub const ALL: [Self; 8] = [
        Self::Scalar,
        Self::Sse2,
        Self::Sse2Crypto,
        Self::Avx2,
        Self::Avx2Crypto,
        Self::Avx512,
        Self::Avx512Crypto,
        Self::Avx512Ext,
    ];

    /// The backend of this tier.
    pub fn target(self) -> TargetId {
        match self {
            Self::Scalar => TargetId::Scalar,
            Self::Sse2 | Self::Sse2Crypto => TargetId::Sse2,
            Self::Avx2 | Self::Avx2Crypto => TargetId::Avx2,
            Self::Avx512 | Self::Avx512Crypto | Self::Avx512Ext => {
                TargetId::Avx512
            }
        }
    }

    /// The [`features`](crate::features) flags the tier's token carries.
    pub fn extensions(self) -> u32 {
        use crate::features::{
            AES, BITALG, PCLMUL, SSE41, SSSE3, VBMI, VBMI2, VPOPCNTDQ,
        };
        match self {
            Self::Scalar | Self::Sse2 => 0,
            Self::Avx2 | Self::Avx512 => SSSE3 | SSE41,
            Self::Sse2Crypto | Self::Avx2Crypto | Self::Avx512Crypto => {
                SSSE3 | SSE41 | AES | PCLMUL
            }
            Self::Avx512Ext => {
                SSSE3 | SSE41 | AES | PCLMUL | VBMI | VBMI2 | BITALG | VPOPCNTDQ
            }
        }
    }

    fn base(self) -> u8 {
        match self.target() {
            TargetId::Avx512 => 3,
            TargetId::Avx2 => 2,
            TargetId::Sse2 => 1,
            TargetId::Scalar | TargetId::Neon | TargetId::Uninitialized => 0,
        }
    }

    fn covers(self, other: Self) -> bool {
        other.base() <= self.base()
            && other.extensions() & !self.extensions() == 0
    }

    fn from_u8(value: u8) -> Option<Self> {
        Self::ALL.into_iter().find(|t| *t as u8 == value)
    }
}

/// CPU capabilities detected once, at startup.
///
/// A `Cpu` can only be created by [`Cpu::detect`] (or lowered with
/// [`Cpu::capped`]), so holding one proves its [`Tier`] runs on this machine.
/// Dispatching through it goes straight to the tier's trampoline: no feature
/// checks per call and none inside the operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cpu {
    tier: Tier,
}

impl Cpu {
    /// Run every feature check once and remember the result.
    pub fn detect() -> Self {
        let tier = detect_tier();
        CHOSEN.store(tier as u8, Ordering::Relaxed);
        Self { tier }
    }

    pub(crate) fn cached() -> Self {
        match Tier::from_u8(CHOSEN.load(Ordering::Relaxed)) {
            Some(tier) => Self { tier },
            None => Self::detect(),
        }
    }

    /// The tier this CPU runs.
    pub fn tier(self) -> Tier {
        self.tier
    }

    /// The backend of [`tier`](Self::tier).
    pub fn target(self) -> TargetId {
        self.tier.target()
    }

    /// `true` when every flag in `extensions` (see [`features`](crate::features)) is on.
    pub fn has(self, extensions: u32) -> bool {
        self.tier.extensions() & extensions == extensions
    }

    /// The best tier at or below `max` that this CPU also supports.
    pub fn capped(self, max: Tier) -> Self {
        let tier = Tier::ALL
            .into_iter()
            .rev()
            .find(|t| *t <= max && self.tier.covers(*t))
            .unwrap_or(Tier::Scalar);
        Self { tier }
    }

    /// Run `kernel` on this tier's token.
    #[inline]
    pub fn dispatch<K: WithSimd>(self, kernel: K) -> K::Output {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: a Cpu only holds a tier that detect() found on this CPU
            unsafe {
                match self.tier {
                    Tier::Scalar => kernel.with_simd(Scalar),
                    Tier::Sse2 => tier_sse2::simd(kernel),
                    Tier::Sse2Crypto => tier_sse2_crypto::simd(kernel),
                    Tier::Avx2 => tier_avx2::simd(kernel),
                    Tier::Avx2Crypto => tier_avx2_crypto::simd(kernel),
                    Tier::Avx512 => tier_avx512::simd(kernel),
                    Tier::Avx512Crypto => tier_avx512_crypto::simd(kernel),
                    Tier::Avx512Ext => tier_avx512_ext::simd(kernel),
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        kernel.with_simd(Scalar)
    }

    /// Run a kernel that uses [`SimdCrypto`] on this tier's token, or its
    /// scalar path on [`Tier::Scalar`].
    #[inline]
    pub fn dispatch_crypto<K: WithSimdCrypto>(self, kernel: K) -> K::Output {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: a Cpu only holds a tier that detect() found on this CPU
            unsafe {
                match self.tier {
                    Tier::Scalar => kernel.with_scalar(),
                    Tier::Sse2 => tier_sse2::crypto(kernel),
                    Tier::Sse2Crypto => tier_sse2_crypto::crypto(kernel),
                    Tier::Avx2 => tier_avx2::crypto(kernel),
                    Tier::Avx2Crypto => tier_avx2_crypto::crypto(kernel),
                    Tier::Avx512 => tier_avx512::crypto(kernel),
                    Tier::Avx512Crypto => tier_avx512_crypto::crypto(kernel),
                    Tier::Avx512Ext => tier_avx512_ext::crypto(kernel),
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        kernel.with_scalar()
    }

    /// Like [`dispatch_crypto`](Self::dispatch_crypto), but the kernel always
    /// gets a 128-bit `Sse2` token, compiled with this tier's features (so an
    /// AVX-512 tier still gets EVEX encodings and fused ops on 128-bit data).
    #[inline]
    pub fn dispatch_crypto_128<K: WithSimdCrypto>(
        self,
        kernel: K,
    ) -> K::Output {
        #[cfg(target_arch = "x86_64")]
        {
            // SAFETY: a Cpu only holds a tier that detect() found on this CPU
            unsafe {
                match self.tier {
                    Tier::Scalar => kernel.with_scalar(),
                    Tier::Sse2 => tier_sse2::crypto_128(kernel),
                    Tier::Sse2Crypto => tier_sse2_crypto::crypto_128(kernel),
                    Tier::Avx2 => tier_avx2::crypto_128(kernel),
                    Tier::Avx2Crypto => tier_avx2_crypto::crypto_128(kernel),
                    Tier::Avx512 => tier_avx512::crypto_128(kernel),
                    Tier::Avx512Crypto => {
                        tier_avx512_crypto::crypto_128(kernel)
                    }
                    Tier::Avx512Ext => tier_avx512_ext::crypto_128(kernel),
                }
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        kernel.with_scalar()
    }
}

/// A kernel entry point picked once for one [`Cpu`].
///
/// Built only by [`Cpu::resolve`], [`Cpu::resolve_crypto`] and
/// [`Cpu::resolve_crypto_128`], so [`call`](Self::call) always lands on a
/// trampoline this CPU supports: one indirect call, no tier match per call.
pub struct Resolved<K, R> {
    entry: unsafe fn(K) -> R,
}

impl<K, R> Clone for Resolved<K, R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K, R> Copy for Resolved<K, R> {}

impl<K, R> core::fmt::Debug for Resolved<K, R> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Resolved").finish_non_exhaustive()
    }
}

impl<K, R> Resolved<K, R> {
    /// Run `kernel` on the entry point picked at resolve time.
    #[inline(always)]
    pub fn call(self, kernel: K) -> R {
        // SAFETY: entry is a trampoline for a tier that Cpu::detect() found
        unsafe { (self.entry)(kernel) }
    }
}

impl Cpu {
    /// Pick the [`Cpu::dispatch`] entry point for `K` once.
    pub fn resolve<K: WithSimd>(self) -> Resolved<K, K::Output> {
        #[cfg(target_arch = "x86_64")]
        let entry: unsafe fn(K) -> K::Output = match self.tier {
            Tier::Scalar => scalar_simd::<K>,
            Tier::Sse2 => tier_sse2::simd::<K>,
            Tier::Sse2Crypto => tier_sse2_crypto::simd::<K>,
            Tier::Avx2 => tier_avx2::simd::<K>,
            Tier::Avx2Crypto => tier_avx2_crypto::simd::<K>,
            Tier::Avx512 => tier_avx512::simd::<K>,
            Tier::Avx512Crypto => tier_avx512_crypto::simd::<K>,
            Tier::Avx512Ext => tier_avx512_ext::simd::<K>,
        };
        #[cfg(not(target_arch = "x86_64"))]
        let entry: unsafe fn(K) -> K::Output = scalar_simd::<K>;
        Resolved { entry }
    }

    /// Pick the [`Cpu::dispatch_crypto`] entry point for `K` once.
    pub fn resolve_crypto<K: WithSimdCrypto>(self) -> Resolved<K, K::Output> {
        #[cfg(target_arch = "x86_64")]
        let entry: unsafe fn(K) -> K::Output = match self.tier {
            Tier::Scalar => scalar_crypto::<K>,
            Tier::Sse2 => tier_sse2::crypto::<K>,
            Tier::Sse2Crypto => tier_sse2_crypto::crypto::<K>,
            Tier::Avx2 => tier_avx2::crypto::<K>,
            Tier::Avx2Crypto => tier_avx2_crypto::crypto::<K>,
            Tier::Avx512 => tier_avx512::crypto::<K>,
            Tier::Avx512Crypto => tier_avx512_crypto::crypto::<K>,
            Tier::Avx512Ext => tier_avx512_ext::crypto::<K>,
        };
        #[cfg(not(target_arch = "x86_64"))]
        let entry: unsafe fn(K) -> K::Output = scalar_crypto::<K>;
        Resolved { entry }
    }

    /// Pick the [`Cpu::dispatch_crypto_128`] entry point for `K` once.
    pub fn resolve_crypto_128<K: WithSimdCrypto>(
        self,
    ) -> Resolved<K, K::Output> {
        #[cfg(target_arch = "x86_64")]
        let entry: unsafe fn(K) -> K::Output = match self.tier {
            Tier::Scalar => scalar_crypto::<K>,
            Tier::Sse2 => tier_sse2::crypto_128::<K>,
            Tier::Sse2Crypto => tier_sse2_crypto::crypto_128::<K>,
            Tier::Avx2 => tier_avx2::crypto_128::<K>,
            Tier::Avx2Crypto => tier_avx2_crypto::crypto_128::<K>,
            Tier::Avx512 => tier_avx512::crypto_128::<K>,
            Tier::Avx512Crypto => tier_avx512_crypto::crypto_128::<K>,
            Tier::Avx512Ext => tier_avx512_ext::crypto_128::<K>,
        };
        #[cfg(not(target_arch = "x86_64"))]
        let entry: unsafe fn(K) -> K::Output = scalar_crypto::<K>;
        Resolved { entry }
    }
}

unsafe fn scalar_simd<K: WithSimd>(kernel: K) -> K::Output {
    kernel.with_simd(Scalar)
}

unsafe fn scalar_crypto<K: WithSimdCrypto>(kernel: K) -> K::Output {
    kernel.with_scalar()
}

/// Kernel that can use [`SimdCrypto`], dispatched by [`Cpu::dispatch_crypto`]
/// and [`Cpu::dispatch_crypto_128`].
///
/// Check `S::HAS_CLMUL` / `S::HAS_AES` (compile-time constants) to pick the
/// hardware path; without them the crypto ops fall back to software.
pub trait WithSimdCrypto {
    /// The return type of the kernel.
    type Output;

    /// Execute the kernel on a SIMD token.
    fn with_simd<S: SimdOps + SimdCrypto>(self, s: S) -> Self::Output;

    /// Execute the kernel without SIMD (scalar tier).
    fn with_scalar(self) -> Self::Output;
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

    // Startup-detected Cpu, no captures: `simd_fn!(cpu => |s| -> T { .. })`
    ($cpu:expr => |$s:ident| -> $ret:ty $body:block) => {{
        struct __K;
        impl $crate::WithSimd for __K {
            type Output = $ret;
            #[inline(always)]
            fn with_simd<__S: $crate::SimdOps>(self, $s: __S) -> $ret $body
        }
        $crate::Cpu::dispatch($cpu, __K)
    }};

    // Startup-detected Cpu with typed captures
    ($cpu:expr => [$($cap:ident : $cap_ty:ty),* $(,)?] |$s:ident| -> $ret:ty $body:block) => {{
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
        $crate::Cpu::dispatch($cpu, __K { $($cap,)* })
    }};
}

// ---------------------------------------------------------------------------
// Target-feature trampolines (x86_64), one module per tier
// ---------------------------------------------------------------------------

macro_rules! tier_trampolines {
    ($($module:ident: $token:ty, $sse_token:ty, $features:literal;)*) => {$(
        #[cfg(target_arch = "x86_64")]
        mod $module {
            use super::{WithSimd, WithSimdCrypto};

            #[target_feature(enable = $features)]
            #[inline(never)]
            pub(super) unsafe fn simd<K: WithSimd>(kernel: K) -> K::Output {
                kernel.with_simd(<$token>::token())
            }

            #[target_feature(enable = $features)]
            #[inline(never)]
            pub(super) unsafe fn crypto<K: WithSimdCrypto>(kernel: K) -> K::Output {
                kernel.with_simd(<$token>::token())
            }

            #[target_feature(enable = $features)]
            #[inline(never)]
            pub(super) unsafe fn crypto_128<K: WithSimdCrypto>(kernel: K) -> K::Output {
                kernel.with_simd(<$sse_token>::token())
            }
        }
    )*};
}

tier_trampolines! {
    tier_sse2:
        crate::backend::sse2::Sse2<0>,
        crate::backend::sse2::Sse2<0>,
        "sse2";
    tier_sse2_crypto:
        crate::backend::sse2::Sse2<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL }>,
        crate::backend::sse2::Sse2<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL }>,
        "sse2,ssse3,sse4.1,aes,pclmulqdq";
    tier_avx2:
        crate::backend::avx2::Avx2<{ crate::features::SSSE3 | crate::features::SSE41 }>,
        crate::backend::sse2::Sse2<{ crate::features::SSSE3 | crate::features::SSE41 }>,
        "avx2,fma";
    tier_avx2_crypto:
        crate::backend::avx2::Avx2<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL }>,
        crate::backend::sse2::Sse2<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL }>,
        "avx2,fma,aes,pclmulqdq";
    tier_avx512:
        crate::backend::avx512::Avx512<{ crate::features::SSSE3 | crate::features::SSE41 }>,
        crate::backend::sse2::Sse2<{ crate::features::SSSE3 | crate::features::SSE41 }>,
        "avx512f,avx512bw,avx512cd,avx512dq,avx512vl";
    tier_avx512_crypto:
        crate::backend::avx512::Avx512<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL }>,
        crate::backend::sse2::Sse2<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL }>,
        "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,aes,pclmulqdq";
    tier_avx512_ext:
        crate::backend::avx512::Avx512<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL | crate::features::VBMI | crate::features::VBMI2 | crate::features::BITALG | crate::features::VPOPCNTDQ }>,
        crate::backend::sse2::Sse2<{ crate::features::SSSE3 | crate::features::SSE41 | crate::features::AES | crate::features::PCLMUL }>,
        "avx512f,avx512bw,avx512cd,avx512dq,avx512vl,aes,pclmulqdq,avx512vbmi,avx512vbmi2,avx512bitalg,avx512vpopcntdq";
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
        let result =
            dispatch_to(ScalarAddKernel { a: 5, b: 7 }, TargetId::Scalar);
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
