//! Highway: A Rust port of Google's Highway SIMD library.
//!
//! Provides portable SIMD operations with runtime dispatch to the best
//! available target (Scalar, SSE2, AVX2, AVX-512).
//!
//! # Architecture
//!
//! - **Lane types** ([`lane`]): Scalar element types (`u8`..`f64`).
//! - **Tags** ([`tag`]): Zero-sized types selecting vector width.
//! - **Simd trait** ([`simd`]): GAT-based trait mapping targets to vector/mask types.
//! - **Operations** ([`ops`]): Sub-traits for arithmetic, memory, comparison, etc.
//! - **Backends** ([`backend`]): Concrete implementations per CPU target.
//! - **Dispatch** ([`dispatch`]): Runtime CPU detection and target selection.
//! - **Allocator** ([`alloc`]): Aligned memory allocation (behind `alloc` feature).
//!
//! # Platform Support
//!
//! The public API (`WithSimd`, `dispatch()`, `Simd` trait, operation traits) is
//! platform-independent. On non-x86_64 platforms, dispatch falls back to the
//! scalar backend. Architecture-specific backends are conditionally compiled.

/// Concrete SIMD backend implementations.
pub mod backend;
/// Runtime CPU detection and dispatch.
pub mod dispatch;
/// Lane types — scalar elements that occupy a single SIMD vector lane.
pub mod lane;
/// Operation sub-traits (arithmetic, memory, bitwise, comparison, etc.).
pub mod ops;
/// The core `Simd` trait with GATs for vector and mask types.
pub mod simd;
/// Tags — zero-sized types encoding vector width for type-level dispatch.
pub mod tag;

/// Aligned memory allocator for SIMD-friendly allocation.
#[cfg(feature = "alloc")]
pub mod alloc;

/// Trait for number types
pub trait SimdElement {}

macro_rules! impl_simd_element {
    ($($t:ty),*) => {
        $(
            impl SimdElement for $t {}
        )*
    };
}

impl_simd_element!(u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64);

/// A 64-byte aligned array of SIMD elements.
#[repr(align(64))]
pub struct AlignedT<const N: usize, T: SimdElement>([T; N]);

// Re-exports for convenience.
pub use backend::scalar::Scalar;
pub use dispatch::{TargetId, WithSimd, dispatch, dispatch_to};
pub use lane::Lane;
pub use ops::SimdOps;
pub use simd::Simd;
// Re-export all op sub-traits so user code can bound on them individually.
pub use ops::{
    SimdArith, SimdBitwise, SimdCompare, SimdConvert, SimdCore, SimdCrypto, SimdFloat, SimdMask,
    SimdMemory, SimdReduce, SimdShuffle,
};
// Re-export alignment types for aligned arrays.
pub use ops::{A1, A2, A4, A8, A16, A32, A64, A128, Aligned, Alignment};

#[cfg(feature = "alloc")]
pub use alloc::{AlignedVec, aligned_vec, aligned_vec_from_slice, aligned_vec_with_capacity};

#[cfg(target_arch = "x86_64")]
pub use backend::sse2::Sse2;

#[cfg(target_arch = "x86_64")]
pub use backend::avx2::Avx2;

#[cfg(target_arch = "x86_64")]
pub use backend::avx512::Avx512;
