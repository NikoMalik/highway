/// Core SIMD trait defining the vector and mask types for each target.
///
/// Each backend (Scalar, Sse2, Avx2, Avx512) implements this trait, binding
/// its concrete vector and mask types via GATs.
use crate::lane::Lane;

/// A SIMD target providing vector and mask types for all lane types.
///
/// # Safety
/// Implementations must ensure:
/// - `Vec<T>` is a valid SIMD vector holding `VECTOR_BYTES / T::BYTES` lanes of type `T`.
/// - `Mask<T>` correctly represents per-lane boolean results for the target.
/// - `VECTOR_BYTES` matches the actual hardware vector width.
/// - `VecHalf<T>` is the half-width vector type (or identity for min-width targets).
/// - `MaskHalf<T>` is the half-width mask type (or identity for min-width targets).
pub unsafe trait Simd: Copy + Sized + 'static {
    /// A SIMD vector holding lanes of type `T`.
    type Vec<T: Lane>: Copy;

    /// A mask (per-lane boolean) corresponding to vectors of type `T`.
    type Mask<T: Lane>: Copy;

    /// A half-width SIMD vector. For Scalar/SSE2, same as Vec (can't go below).
    /// For AVX2, this is V128. For AVX-512, this is V256.
    type VecHalf<T: Lane>: Copy;

    /// A half-width mask. For Scalar/SSE2, same as Mask.
    /// For AVX2, this is M128. For AVX-512, this is M256.
    type MaskHalf<T: Lane>: Copy;

    /// Total bytes in a vector register for this target.
    const VECTOR_BYTES: usize;

    /// Number of lanes of type `T` that fit in one vector.
    #[inline(always)]
    fn lanes<T: Lane>(self) -> usize {
        let l = Self::VECTOR_BYTES / T::BYTES;
        if l > 0 { l } else { 1 }
    }
}

/// Number of lanes of type `T` that fit in one vector (free function form).
#[inline(always)]
pub const fn lanes<T: Lane, G: Simd>() -> usize {
    let lanes = G::VECTOR_BYTES / T::BYTES;

    if lanes > 0 { lanes } else { 1 }
}
