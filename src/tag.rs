/// Tag types for selecting SIMD vector configurations.
///
/// Tags are zero-sized types that encode the SIMD target (`S`) and lane type (`T`).
/// They mirror Highway's `Simd<T, N, kPow2>` tag, used to select operations
/// and determine vector widths at compile time.
use core::marker::PhantomData;

use crate::lane::Lane;
use crate::simd::Simd;

/// A full-width tag: uses all lanes the target supports for type `T`.
///
/// `Tag<S, T>` requests a vector holding `S::VECTOR_BYTES / T::BYTES` lanes.
#[derive(Clone, Copy, Debug)]
pub struct Tag<S: Simd, T: Lane> {
    _simd: PhantomData<S>,
    _lane: PhantomData<T>,
}

impl<S: Simd, T: Lane> Tag<S, T> {
    /// Create a new full-width tag.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _simd: PhantomData,
            _lane: PhantomData,
        }
    }

    /// Number of lanes in the full vector for this tag.
    #[inline(always)]
    pub fn lanes(self) -> usize {
        (S::VECTOR_BYTES / T::BYTES).max(1)
    }
}

impl<S: Simd, T: Lane> Default for Tag<S, T> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// A capped tag: at most `N` lanes, but never more than the target supports.
///
/// Useful when the caller knows it will only process up to `N` elements and
/// wants the compiler to use a narrower vector if beneficial.
#[derive(Clone, Copy, Debug)]
pub struct CappedTag<S: Simd, T: Lane, const N: usize> {
    _simd: PhantomData<S>,
    _lane: PhantomData<T>,
}

impl<S: Simd, T: Lane, const N: usize> CappedTag<S, T, N> {
    /// Create a new capped tag.
    #[inline(always)]
    pub fn new() -> Self {
        Self {
            _simd: PhantomData,
            _lane: PhantomData,
        }
    }

    /// Number of lanes: `min(N, full vector lanes)`.
    #[inline(always)]
    pub fn lanes(self) -> usize {
        let full = (S::VECTOR_BYTES / T::BYTES).max(1);
        if N < full { N } else { full }
    }
}

impl<S: Simd, T: Lane, const N: usize> Default for CappedTag<S, T, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

/// A fixed-width tag: exactly `N` lanes, compile-time assertion that the target
/// supports at least that many.
///
/// Panics at compile time (via const assertion) if `N > S::VECTOR_BYTES / T::BYTES`.
#[derive(Clone, Copy, Debug)]
pub struct FixedTag<S: Simd, T: Lane, const N: usize> {
    _simd: PhantomData<S>,
    _lane: PhantomData<T>,
}

impl<S: Simd, T: Lane, const N: usize> FixedTag<S, T, N> {
    /// Create a new fixed-width tag.
    ///
    /// # Panics
    /// At compile time if `N` exceeds the target's capacity for `T`.
    #[inline(always)]
    pub fn new() -> Self {
        assert!(
            N <= (S::VECTOR_BYTES / T::BYTES).max(1),
            "FixedTag: N exceeds target capacity"
        );
        Self {
            _simd: PhantomData,
            _lane: PhantomData,
        }
    }

    /// Number of lanes: always `N`.
    #[inline(always)]
    pub fn lanes(self) -> usize {
        N
    }
}

impl<S: Simd, T: Lane, const N: usize> Default for FixedTag<S, T, N> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::scalar::Scalar;

    #[test]
    fn tag_lanes() {
        // Scalar has VECTOR_BYTES = 1 (one lane always)
        let tag = Tag::<Scalar, u8>::new();
        assert_eq!(tag.lanes(), 1);

        let tag = Tag::<Scalar, u32>::new();
        assert_eq!(tag.lanes(), 1);

        let tag = Tag::<Scalar, u64>::new();
        assert_eq!(tag.lanes(), 1);
    }

    #[test]
    fn capped_tag_lanes() {
        // min(4, 1) = 1 for u8 on Scalar
        let tag = CappedTag::<Scalar, u8, 4>::new();
        assert_eq!(tag.lanes(), 1);

        // min(100, 1) = 1 for u8 on Scalar
        let tag = CappedTag::<Scalar, u8, 100>::new();
        assert_eq!(tag.lanes(), 1);
    }

    #[test]
    fn fixed_tag_lanes() {
        let tag = FixedTag::<Scalar, u32, 1>::new();
        assert_eq!(tag.lanes(), 1);
    }

    #[test]
    #[should_panic(expected = "FixedTag: N exceeds target capacity")]
    fn fixed_tag_too_many_lanes() {
        // Scalar can only hold 1 lane of any type
        let _ = FixedTag::<Scalar, u8, 2>::new();
    }
}
