//! SIMD backend implementations.
//!
//! Each submodule provides a concrete target type implementing the `Simd` trait
//! and all operation sub-traits.

/// Software fallback for AES / carry-less multiply (used by SIMD backends).
#[cfg(target_arch = "x86_64")]
pub(crate) mod crypto_soft;

/// Portable scalar backend (all platforms).
pub mod scalar;

/// SSE2 backend (x86_64, 128-bit vectors).
#[cfg(target_arch = "x86_64")]
pub mod sse2;

/// AVX2 backend (x86_64, 256-bit vectors).
#[cfg(target_arch = "x86_64")]
pub mod avx2;

/// AVX-512 backend (x86_64, 512-bit vectors).
#[cfg(target_arch = "x86_64")]
pub mod avx512;
