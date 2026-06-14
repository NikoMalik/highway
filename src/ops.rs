use std::borrow::{Borrow, BorrowMut};
use std::ops::{Deref, DerefMut};

/// SIMD operation traits and free-function wrappers.
///
/// Operations are organized into sub-traits by category. Each backend
/// implements these traits for its target type. The `SimdOps` supertrait
/// combines all sub-traits, making it easy to bound generic code on a
/// single trait.
///
/// Value operations (arithmetic, compare, shuffle, mask, convert, float, ...)
/// are safe: the `Self` target token can only be obtained through dispatch after
/// a runtime feature check, so it proves the required CPU features are enabled.
/// Only operations taking a raw pointer (loads/stores) or a runtime lane index
/// (`extract_lane`/`insert_lane`) are `unsafe`; safe slice-based wrappers
/// (`load_slice`/`store_slice`/`load_aligned_slice`/`store_aligned_slice`) cover
/// the common cases.
use crate::lane::{
    FloatLane, IntegerLane, Lane, NarrowLane, UnsignedLane, WideLane,
};
use crate::simd::Simd;

// ---------------------------------------------------------------------------
// SimdCore — fundamental vector construction and element access
// ---------------------------------------------------------------------------

/// Core vector construction and element access.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdCore: Simd {
    /// Returns a vector with all lanes set to zero.
    fn zero<T: Lane>(self) -> Self::Vec<T>;

    /// Returns a vector with all lanes set to `value`.
    fn splat<T: Lane>(self, value: T) -> Self::Vec<T>;

    /// Returns an uninitialized vector. In practice, returns zero.
    fn undefined<T: Lane>(self) -> Self::Vec<T>;

    /// Reinterprets the bits of `v` as a vector of type `U`.
    fn bitcast<T: Lane, U: Lane>(self, v: Self::Vec<T>) -> Self::Vec<U>;

    /// Extracts the lane at `index`.
    ///
    /// # Safety
    /// `index` must be less than the number of lanes.
    unsafe fn extract_lane<T: Lane>(self, v: Self::Vec<T>, index: usize) -> T;

    /// Returns a copy of `v` with the lane at `index` replaced by `value`.
    ///
    /// # Safety
    /// `index` must be less than the number of lanes.
    unsafe fn insert_lane<T: Lane>(
        self,
        v: Self::Vec<T>,
        index: usize,
        value: T,
    ) -> Self::Vec<T>;

    /// Returns a vector where lane `i` contains `base + i` (converted to `T`).
    fn iota<T: Lane>(self, base: T) -> Self::Vec<T>;

    /// Extracts lane 0 (shorthand for `extract_lane(v, 0)`).
    #[inline(always)]
    fn get_lane<T: Lane>(self, v: Self::Vec<T>) -> T {
        // SAFETY: Caller ensures CPU features; lane 0 always exists.
        unsafe { self.extract_lane(v, 0) }
    }
}

// ---------------------------------------------------------------------------
// SimdMemory — load and store
// ---------------------------------------------------------------------------

/// Aligned and unaligned memory operations.
///
/// # Safety
/// Callers must ensure pointers are valid and point to at least `lanes` elements.
pub unsafe trait SimdMemory: Simd {
    /// Load a full vector from an aligned pointer.
    unsafe fn load<T: Lane>(self, ptr: *const T) -> Self::Vec<T>;

    /// Load a full vector from an unaligned pointer.
    unsafe fn load_u<T: Lane>(self, ptr: *const T) -> Self::Vec<T>;

    /// Store a full vector to an aligned pointer.
    unsafe fn store<T: Lane>(self, v: Self::Vec<T>, ptr: *mut T);

    /// Store a full vector to an unaligned pointer.
    unsafe fn store_u<T: Lane>(self, v: Self::Vec<T>, ptr: *mut T);

    /// Non-temporal (streaming) store. The pointer must be aligned.
    unsafe fn stream<T: Lane>(self, v: Self::Vec<T>, ptr: *mut T);

    /// Load 128 bits from `ptr` and duplicate across all 128-bit blocks.
    /// For 128-bit targets, same as `load`. For 256-bit, duplicates to both halves.
    unsafe fn load_dup128<T: Lane>(self, ptr: *const T) -> Self::Vec<T>;

    /// Load from `ptr` where mask is true; zero where false.
    unsafe fn masked_load<T: Lane>(
        self,
        mask: Self::Mask<T>,
        ptr: *const T,
    ) -> Self::Vec<T>;

    /// Store lanes of `v` to `ptr` where mask is true; leave other memory unchanged.
    unsafe fn blended_store<T: Lane>(
        self,
        v: Self::Vec<T>,
        mask: Self::Mask<T>,
        ptr: *mut T,
    );

    /// Gather: load non-contiguous elements using per-lane i32 indices.
    /// `result[i] = base[idx[i]]`
    unsafe fn gather_index<T: Lane>(
        self,
        base: *const T,
        idx: Self::Vec<i32>,
    ) -> Self::Vec<T>;

    /// Scatter: store elements to non-contiguous locations using per-lane i32 indices.
    /// `base[idx[i]] = v[i]`
    unsafe fn scatter_index<T: Lane>(
        self,
        v: Self::Vec<T>,
        base: *mut T,
        idx: Self::Vec<i32>,
    );

    /// Load interleaved 2: deinterleave 2-channel data.
    /// Reads `2 * lanes` elements from `ptr` and returns two vectors.
    unsafe fn load_interleaved_2<T: Lane>(
        self,
        ptr: *const T,
    ) -> (Self::Vec<T>, Self::Vec<T>);

    /// Load interleaved 3: deinterleave 3-channel data.
    unsafe fn load_interleaved_3<T: Lane>(
        self,
        ptr: *const T,
    ) -> (Self::Vec<T>, Self::Vec<T>, Self::Vec<T>);

    /// Load interleaved 4: deinterleave 4-channel data.
    #[allow(clippy::type_complexity)]
    unsafe fn load_interleaved_4<T: Lane>(
        self,
        ptr: *const T,
    ) -> (Self::Vec<T>, Self::Vec<T>, Self::Vec<T>, Self::Vec<T>);

    /// Store interleaved 2: interleave and store 2-channel data.
    unsafe fn store_interleaved_2<T: Lane>(
        self,
        v0: Self::Vec<T>,
        v1: Self::Vec<T>,
        ptr: *mut T,
    );

    /// Store interleaved 3: interleave and store 3-channel data.
    unsafe fn store_interleaved_3<T: Lane>(
        self,
        v0: Self::Vec<T>,
        v1: Self::Vec<T>,
        v2: Self::Vec<T>,
        ptr: *mut T,
    );

    /// Store interleaved 4: interleave and store 4-channel data.
    unsafe fn store_interleaved_4<T: Lane>(
        self,
        v0: Self::Vec<T>,
        v1: Self::Vec<T>,
        v2: Self::Vec<T>,
        v3: Self::Vec<T>,
        ptr: *mut T,
    );

    /// Load from compressed stream and expand to masked positions.
    /// Inverse of compress_store.
    unsafe fn load_expand<T: Lane>(
        self,
        mask: Self::Mask<T>,
        ptr: *const T,
    ) -> Self::Vec<T>;

    /// Safe, bounds-checked load of one full vector from the start of `src`.
    ///
    /// Reads exactly `lanes::<T>()` elements; panics if `src` is shorter.
    /// This is the safe alternative to the raw-pointer [`load_u`](Self::load_u).
    #[inline(always)]
    fn load_slice<T: Lane>(self, src: &[T]) -> Self::Vec<T> {
        let lanes = self.lanes::<T>();
        assert!(
            src.len() >= lanes,
            "load_slice: need at least {lanes} elements, got {}",
            src.len()
        );
        // SAFETY: the bounds check above guarantees `src` has `lanes` readable
        // elements at `src.as_ptr()`, and `self` proves the CPU features.
        unsafe { self.load_u(src.as_ptr()) }
    }

    /// Safe, bounds-checked store of one full vector to the start of `dst`.
    ///
    /// Writes exactly `lanes::<T>()` elements; panics if `dst` is shorter.
    /// This is the safe alternative to the raw-pointer [`store_u`](Self::store_u).
    #[inline(always)]
    fn store_slice<T: Lane>(self, v: Self::Vec<T>, dst: &mut [T]) {
        let lanes = self.lanes::<T>();
        assert!(
            dst.len() >= lanes,
            "store_slice: need at least {lanes} elements, got {}",
            dst.len()
        );
        // SAFETY: the bounds check above guarantees `dst` has `lanes` writable
        // elements at `dst.as_mut_ptr()`, and `self` proves the CPU features.
        unsafe { self.store_u(v, dst.as_mut_ptr()) }
    }

    /// Safe, bounds- and alignment-checked aligned load of one full vector from
    /// the start of `src` (uses the faster aligned [`load`](Self::load)).
    ///
    /// Panics if `src` has fewer than `lanes::<T>()` elements, or if its start
    /// is not aligned to the vector width (`VECTOR_BYTES`). Use with `AlignedVec`.
    #[inline(always)]
    fn load_aligned_slice<T: Lane>(self, src: &[T]) -> Self::Vec<T> {
        let lanes = self.lanes::<T>();
        assert!(
            src.len() >= lanes,
            "load_aligned_slice: need at least {lanes} elements, got {}",
            src.len()
        );
        assert!(
            (src.as_ptr() as usize).is_multiple_of(Self::VECTOR_BYTES),
            "load_aligned_slice: pointer not aligned to {} bytes",
            Self::VECTOR_BYTES
        );
        // SAFETY: length and alignment are both checked above, and `self` proves
        // the CPU features.
        unsafe { self.load(src.as_ptr()) }
    }

    /// Safe, bounds- and alignment-checked aligned store of one full vector to
    /// the start of `dst` (uses the faster aligned [`store`](Self::store)).
    ///
    /// Panics if `dst` has fewer than `lanes::<T>()` elements, or if its start
    /// is not aligned to the vector width (`VECTOR_BYTES`). Use with `AlignedVec`.
    #[inline(always)]
    fn store_aligned_slice<T: Lane>(self, v: Self::Vec<T>, dst: &mut [T]) {
        let lanes = self.lanes::<T>();
        assert!(
            dst.len() >= lanes,
            "store_aligned_slice: need at least {lanes} elements, got {}",
            dst.len()
        );
        assert!(
            (dst.as_ptr() as usize).is_multiple_of(Self::VECTOR_BYTES),
            "store_aligned_slice: pointer not aligned to {} bytes",
            Self::VECTOR_BYTES
        );
        // SAFETY: length and alignment are both checked above, and `self` proves
        // the CPU features.
        unsafe { self.store(v, dst.as_mut_ptr()) }
    }
}
/// mode for declare new a types
pub mod sealed {
    /// trait for declare new a types
    pub trait Sealed {}

    impl Sealed for super::A1 {}
    impl Sealed for super::A2 {}
    impl Sealed for super::A4 {}
    impl Sealed for super::A8 {}
    impl Sealed for super::A16 {}
    impl Sealed for super::A32 {}
    impl Sealed for super::A64 {}
    impl Sealed for super::A128 {}
}

/// 1-byte alignment
#[derive(Clone, Copy)]
#[repr(align(1))]
pub struct A1;

/// 2-byte alignment
#[derive(Clone, Copy)]
#[repr(align(2))]
pub struct A2;

/// 4-byte alignment
#[derive(Clone, Copy)]
#[repr(align(4))]
pub struct A4;

/// 8-byte alignment
#[derive(Clone, Copy)]
#[repr(align(8))]
pub struct A8;

/// 16-byte alignment
#[derive(Clone, Copy)]
#[repr(align(16))]
pub struct A16;

/// 32-byte alignment
#[derive(Clone, Copy)]
#[repr(align(32))]
pub struct A32;

/// 64-byte alignment
#[derive(Clone, Copy)]
#[repr(align(64))]
pub struct A64;

/// 128-byte alignment
#[derive(Clone, Copy)]
#[repr(align(128))]
pub struct A128;

/// A marker trait for an alignment value.
pub trait Alignment: Copy + sealed::Sealed {
    /// The alignment in bytes.
    const ALIGN: usize;
}

impl Alignment for A1 {
    const ALIGN: usize = 1;
}
impl Alignment for A2 {
    const ALIGN: usize = 2;
}
impl Alignment for A4 {
    const ALIGN: usize = 4;
}
impl Alignment for A8 {
    const ALIGN: usize = 8;
}
impl Alignment for A16 {
    const ALIGN: usize = 16;
}
impl Alignment for A32 {
    const ALIGN: usize = 32;
}
impl Alignment for A64 {
    const ALIGN: usize = 64;
}

impl Alignment for A128 {
    const ALIGN: usize = 128;
}

/// A newtype with alignment of at least `A` bytes
#[repr(C)]
pub struct Aligned<A, T>
where
    T: ?Sized,
{
    _alignment: [A; 0],
    /// Raw Value
    pub value: T,
}

impl<A, T> Aligned<A, T>
where
    A: Alignment,
{
    /// Changes the alignment of value to be at least A bytes
    pub const fn new(value: T) -> Self {
        Aligned {
            _alignment: [],
            value,
        }
    }
}

impl<A, T> AsRef<T> for Aligned<A, T>
where
    A: Alignment,
    T: ?Sized,
{
    /// return ref pointer to raw value
    fn as_ref(&self) -> &T {
        &self.value
    }
}

impl<A, T> AsMut<T> for Aligned<A, T>
where
    A: Alignment,
    T: ?Sized,
{
    /// return mut pointer to raw value
    fn as_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<A, T> Deref for Aligned<A, T>
where
    A: Alignment,
    T: ?Sized,
{
    type Target = T;

    fn deref(&self) -> &T {
        &self.value
    }
}

impl<A, T> Borrow<T> for Aligned<A, T>
where
    A: Alignment,
{
    fn borrow(&self) -> &T {
        &self.value
    }
}

impl<A, T> DerefMut for Aligned<A, T>
where
    A: Alignment,
    T: ?Sized,
{
    fn deref_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<A, T> BorrowMut<T> for Aligned<A, T>
where
    A: Alignment,
{
    fn borrow_mut(&mut self) -> &mut T {
        &mut self.value
    }
}

impl<A, T> Clone for Aligned<A, T>
where
    A: Alignment,
    T: Clone,
{
    fn clone(&self) -> Self {
        Self {
            _alignment: [],
            value: self.value.clone(),
        }
    }
}

impl<A, T> Default for Aligned<A, T>
where
    A: Alignment,
    T: Default,
{
    fn default() -> Self {
        Self {
            _alignment: [],
            value: Default::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// SimdArith — arithmetic operations
// ---------------------------------------------------------------------------

/// Arithmetic operations on SIMD vectors.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdArith: Simd {
    /// Lane-wise addition.
    fn add<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise subtraction.
    fn sub<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise multiplication (for integer types, returns the low bits).
    fn mul<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise division (float only).
    fn div<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Saturating addition for integer lanes.
    fn saturated_add<T: IntegerLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Saturating subtraction for integer lanes.
    fn saturated_sub<T: IntegerLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Absolute value (signed integers and floats).
    fn abs<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Negation.
    fn neg<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise minimum.
    fn min<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise maximum.
    fn max<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// High half of multiplication for integer lanes.
    fn mul_high<T: IntegerLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Averaging (rounding) addition: `(a + b + 1) >> 1`.
    fn average_round<T: UnsignedLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Absolute difference: `|a - b|` for unsigned types, or `max(a,b) - min(a,b)`.
    fn abs_diff<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Clamp each lane to `[lo, hi]`: `min(max(v, lo), hi)`.
    fn clamp<T: Lane>(
        self,
        v: Self::Vec<T>,
        lo: Self::Vec<T>,
        hi: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Multiply even-indexed lanes (0, 2, ...) producing double-width results.
    /// Input is `T` (narrow), output is `T::Wide`.
    fn mul_even<T: NarrowLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T::Wide>
    where
        T::Wide: Lane;

    /// Multiply odd-indexed lanes (1, 3, ...) producing double-width results.
    fn mul_odd<T: NarrowLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T::Wide>
    where
        T::Wide: Lane;

    /// Widening multiply and pairwise add for i16 lanes.
    /// `result[i] = a[2i]*b[2i] + a[2i+1]*b[2i+1]`
    fn widen_mul_pairwise_add_i16(
        self,
        a: Self::Vec<i16>,
        b: Self::Vec<i16>,
    ) -> Self::Vec<i32>;

    /// Saturating widening multiply and pairwise add: u8 * i8 -> i16.
    /// `result[i] = sat_i16(a[2i]*b[2i] + a[2i+1]*b[2i+1])`
    fn sat_widen_mul_pairwise_add(
        self,
        a: Self::Vec<u8>,
        b: Self::Vec<i8>,
    ) -> Self::Vec<i16>;

    /// Fixed-point multiplication: `((a * b) + (1 << 14)) >> 15` for i16 lanes.
    fn mul_fixed_point_15(
        self,
        a: Self::Vec<i16>,
        b: Self::Vec<i16>,
    ) -> Self::Vec<i16>;

    /// Widening multiply and accumulate (reordered): `sum + widen_mul_pairwise_add(a, b)`.
    fn reorder_widen_mul_accumulate(
        self,
        a: Self::Vec<i16>,
        b: Self::Vec<i16>,
        sum: Self::Vec<i32>,
    ) -> Self::Vec<i32>;

    /// Saturating negation for signed integer lanes: `saturated_sub(0, v)`.
    /// For the most-negative value, returns the most-positive value.
    fn saturated_neg<T: IntegerLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Saturating absolute value for signed integer lanes: `max(v, saturated_neg(v))`.
    /// For the most-negative value, returns the most-positive value.
    fn saturated_abs<T: IntegerLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Masked minimum with fallback: `mask ? min(a, b) : no`.
    fn masked_min_or<T: Lane>(
        self,
        no: Self::Vec<T>,
        mask: Self::Mask<T>,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Masked maximum with fallback: `mask ? max(a, b) : no`.
    fn masked_max_or<T: Lane>(
        self,
        no: Self::Vec<T>,
        mask: Self::Mask<T>,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Masked addition with fallback: `mask ? a + b : no`.
    fn masked_add_or<T: Lane>(
        self,
        no: Self::Vec<T>,
        mask: Self::Mask<T>,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Masked subtraction with fallback: `mask ? a - b : no`.
    fn masked_sub_or<T: Lane>(
        self,
        no: Self::Vec<T>,
        mask: Self::Mask<T>,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Masked multiplication with fallback: `mask ? a * b : no`.
    fn masked_mul_or<T: Lane>(
        self,
        no: Self::Vec<T>,
        mask: Self::Mask<T>,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;
}

// ---------------------------------------------------------------------------
// SimdBitwise — bitwise and shift operations
// ---------------------------------------------------------------------------

/// Bitwise and shift operations.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdBitwise: Simd {
    /// Lane-wise bitwise AND.
    fn and<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise bitwise OR.
    fn or<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise bitwise XOR.
    fn xor<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Vec<T>;

    /// Lane-wise bitwise NOT.
    fn not<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Bitwise AND-NOT: `!a & b`.
    fn and_not<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>)
    -> Self::Vec<T>;

    /// Shift each lane left by a compile-time constant.
    fn shift_left<T: IntegerLane, const BITS: u32>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Shift each lane right by a compile-time constant.
    /// Arithmetic for signed types, logical for unsigned.
    fn shift_right<T: IntegerLane, const BITS: u32>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Rotate each lane right by a compile-time constant.
    fn rotate_right<T: IntegerLane, const BITS: u32>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Shift each lane left by a runtime amount (same shift for all lanes).
    fn shift_left_same<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
        bits: u32,
    ) -> Self::Vec<T>;

    /// Shift each lane right by a runtime amount (same shift for all lanes).
    fn shift_right_same<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
        bits: u32,
    ) -> Self::Vec<T>;

    /// Shift the entire vector left by `BYTES` bytes (within each 128-bit block).
    /// New bytes are zero-filled from the right.
    fn shift_left_bytes<T: Lane, const BYTES: usize>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Shift the entire vector right by `BYTES` bytes (within each 128-bit block).
    /// New bytes are zero-filled from the left.
    fn shift_right_bytes<T: Lane, const BYTES: usize>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Count the number of set bits in each lane.
    fn population_count<T: IntegerLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Count leading zeros in each lane.
    fn leading_zero_count<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Count trailing zeros in each lane.
    fn trailing_zero_count<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Reverse the byte order within each lane (byte-swap / endian swap).
    fn reverse_lane_bytes<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Reverse the bits within each lane.
    fn reverse_bits<T: IntegerLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Per-lane variable left shift: each lane shifted by the corresponding lane in `bits`.
    fn shl<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
        bits: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane variable right shift: each lane shifted by the corresponding lane in `bits`.
    /// Arithmetic for signed types, logical for unsigned.
    fn shr<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
        bits: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane variable rotate right.
    fn ror<T: IntegerLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane variable rotate left.
    fn rol<T: IntegerLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Rotate each lane left by a compile-time constant.
    fn rotate_left<T: IntegerLane, const BITS: u32>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Broadcast the sign bit of each lane: all-ones if negative, all-zeros otherwise.
    /// Equivalent to an arithmetic right shift by `bits - 1`.
    fn broadcast_sign_bit<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;
}

// ---------------------------------------------------------------------------
// SimdCompare — comparison operations producing masks
// ---------------------------------------------------------------------------

/// Comparison operations that produce per-lane masks.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdCompare: Simd {
    /// Lane-wise equality.
    fn eq<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Mask<T>;

    /// Lane-wise inequality.
    fn ne<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Mask<T>;

    /// Lane-wise less-than.
    fn lt<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Mask<T>;

    /// Lane-wise less-or-equal.
    fn le<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Mask<T>;

    /// Lane-wise greater-than.
    fn gt<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Mask<T>;

    /// Lane-wise greater-or-equal.
    fn ge<T: Lane>(self, a: Self::Vec<T>, b: Self::Vec<T>) -> Self::Mask<T>;

    /// Test if the given bit position is set in each lane.
    fn test_bit<T: IntegerLane>(
        self,
        v: Self::Vec<T>,
        bit: Self::Vec<T>,
    ) -> Self::Mask<T>;
}

// ---------------------------------------------------------------------------
// SimdMask — mask construction and query
// ---------------------------------------------------------------------------

/// Mask creation, conversion, and query operations.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdMask: Simd {
    /// Convert a vector to a mask: lane != 0 -> true.
    fn mask_from_vec<T: Lane>(self, v: Self::Vec<T>) -> Self::Mask<T>;

    /// Convert a mask back to a vector: true -> all-ones, false -> zero.
    fn vec_from_mask<T: Lane>(self, m: Self::Mask<T>) -> Self::Vec<T>;

    /// Returns a mask with the first `n` lanes set to true.
    fn first_n<T: Lane>(self, n: usize) -> Self::Mask<T>;

    /// Count the number of true lanes.
    fn count_true<T: Lane>(self, m: Self::Mask<T>) -> usize;

    /// True if all lanes are true.
    fn all_true<T: Lane>(self, m: Self::Mask<T>) -> bool;

    /// True if all lanes are false.
    fn all_false<T: Lane>(self, m: Self::Mask<T>) -> bool;

    /// Index of the first true lane, or `None`.
    fn find_first_true<T: Lane>(self, m: Self::Mask<T>) -> Option<usize>;

    /// Per-lane select: `if mask then yes else no`.
    fn if_then_else<T: Lane>(
        self,
        mask: Self::Mask<T>,
        yes: Self::Vec<T>,
        no: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane select: `if mask then yes else 0`.
    fn if_then_else_zero<T: Lane>(
        self,
        mask: Self::Mask<T>,
        yes: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane select: `if mask then 0 else no`.
    fn if_then_zero_else<T: Lane>(
        self,
        mask: Self::Mask<T>,
        no: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Logical AND of two masks.
    fn and_mask<T: Lane>(
        self,
        a: Self::Mask<T>,
        b: Self::Mask<T>,
    ) -> Self::Mask<T>;

    /// Logical OR of two masks.
    fn or_mask<T: Lane>(
        self,
        a: Self::Mask<T>,
        b: Self::Mask<T>,
    ) -> Self::Mask<T>;

    /// Logical NOT of a mask.
    fn not_mask<T: Lane>(self, m: Self::Mask<T>) -> Self::Mask<T>;

    /// Logical XOR of two masks.
    fn xor_mask<T: Lane>(
        self,
        a: Self::Mask<T>,
        b: Self::Mask<T>,
    ) -> Self::Mask<T>;

    /// Index of the last true lane, or `None`.
    fn find_last_true<T: Lane>(self, m: Self::Mask<T>) -> Option<usize>;

    /// Convert a mask to a bitmask `u64`: bit `i` is set if lane `i` is true.
    fn bits_from_mask<T: Lane>(self, m: Self::Mask<T>) -> u64;

    /// XNOR for masks: true where both are true OR both are false.
    fn exclusive_neither<T: Lane>(
        self,
        a: Self::Mask<T>,
        b: Self::Mask<T>,
    ) -> Self::Mask<T>;

    /// Shift mask up by one position, filling lane 0 with false.
    fn slide_mask_1_up<T: Lane>(self, mask: Self::Mask<T>) -> Self::Mask<T>;

    /// Shift mask down by one position, filling the last lane with false.
    fn slide_mask_1_down<T: Lane>(self, mask: Self::Mask<T>) -> Self::Mask<T>;

    /// Per-lane select on sign bit: `if v < 0 then yes else no`.
    fn if_negative_then_else<T: Lane>(
        self,
        v: Self::Vec<T>,
        yes: Self::Vec<T>,
        no: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane select on sign bit: `if v < 0 then yes else 0`.
    fn if_negative_then_else_zero<T: Lane>(
        self,
        v: Self::Vec<T>,
        yes: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane select on sign bit: `if v < 0 then 0 else no`.
    fn if_negative_then_zero_else<T: Lane>(
        self,
        v: Self::Vec<T>,
        no: Self::Vec<T>,
    ) -> Self::Vec<T>;
}

// ---------------------------------------------------------------------------
// SimdConvert — type promotion, demotion, and conversion
// ---------------------------------------------------------------------------

/// Type conversion operations between lane types.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdConvert: Simd {
    /// Promote narrow lanes to wide lanes (e.g. u8 -> u16).
    fn promote_to<N: NarrowLane>(self, v: Self::Vec<N>) -> Self::Vec<N::Wide>
    where
        N::Wide: Lane;

    /// Demote wide lanes to narrow lanes (e.g. i32 -> i16), saturating.
    fn demote_to<W: WideLane>(self, v: Self::Vec<W>) -> Self::Vec<W::Narrow>
    where
        W::Narrow: Lane;

    /// Convert between same-width float/int (e.g. f32 -> i32, i32 -> f32).
    fn convert_to_int<F: FloatLane>(self, v: Self::Vec<F>)
    -> Self::Vec<F::Int>;

    /// Convert from int to float.
    fn convert_to_float<F: FloatLane>(
        self,
        v: Self::Vec<F::Int>,
    ) -> Self::Vec<F>;

    /// Truncate wide lanes to narrow lanes (keep low bits, no saturation).
    fn truncate_to<W: WideLane>(self, v: Self::Vec<W>) -> Self::Vec<W::Narrow>
    where
        W::Narrow: Lane;

    /// Demote two wide vectors into one narrow vector, preserving order.
    /// The `lo` vector contributes the lower half, `hi` the upper half.
    fn ordered_demote_2_to<W: WideLane>(
        self,
        lo: Self::Vec<W>,
        hi: Self::Vec<W>,
    ) -> Self::Vec<W::Narrow>
    where
        W::Narrow: Lane;

    /// Convert float to same-width signed integer with round-to-nearest-even.
    /// Unlike `convert_to_int` which truncates toward zero.
    /// Values >= 2^(bits-1) are clamped to INT_MAX.
    fn nearest_int<F: FloatLane>(self, v: Self::Vec<F>) -> Self::Vec<F::Int>;

    /// Demote two wide vectors into one narrow (unordered packing).
    /// May reorder elements compared to `ordered_demote_2_to`.
    fn reorder_demote_2_to<W: WideLane>(
        self,
        a: Self::Vec<W>,
        b: Self::Vec<W>,
    ) -> Self::Vec<W::Narrow>
    where
        W::Narrow: Lane;

    /// Demote assuming values are in range (no saturation check).
    fn demote_in_range_to<W: WideLane>(
        self,
        v: Self::Vec<W>,
    ) -> Self::Vec<W::Narrow>
    where
        W::Narrow: Lane;

    /// Convert float to int same-width, assuming values are in range (no overflow check).
    fn convert_in_range_to_int<F: FloatLane>(
        self,
        v: Self::Vec<F>,
    ) -> Self::Vec<F::Int>;

    /// Promote the lower half lanes from narrow to wide type.
    fn promote_lower_to<N: NarrowLane>(
        self,
        v: Self::Vec<N>,
    ) -> Self::Vec<N::Wide>
    where
        N::Wide: Lane;

    /// Promote the upper half lanes from narrow to wide type.
    fn promote_upper_to<N: NarrowLane>(
        self,
        v: Self::Vec<N>,
    ) -> Self::Vec<N::Wide>
    where
        N::Wide: Lane;

    /// Truncate two wide vectors into one narrow vector, preserving order.
    /// Like `ordered_demote_2_to` but keeps low bits (no saturation).
    /// The `lo` vector contributes the lower half, `hi` the upper half.
    fn ordered_truncate_2_to<W: WideLane>(
        self,
        lo: Self::Vec<W>,
        hi: Self::Vec<W>,
    ) -> Self::Vec<W::Narrow>
    where
        W::Narrow: Lane;
}

// ---------------------------------------------------------------------------
// SimdShuffle — lane rearrangement
// ---------------------------------------------------------------------------

/// Lane reordering and shuffle operations.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdShuffle: Simd {
    /// Reverse the order of all lanes.
    fn reverse<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Broadcast the lane at compile-time index `IDX` to all lanes.
    fn broadcast_lane<T: Lane, const IDX: usize>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Interleave the lower halves of two vectors.
    fn interleave_lower<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Interleave the upper halves of two vectors.
    fn interleave_upper<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Zip lower halves: `[a0, b0, a1, b1, ...]`.
    fn zip_lower<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Zip upper halves.
    fn zip_upper<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Byte-level table lookup. Each byte in `idx` selects a byte from `table`.
    fn table_lookup_bytes<T: Lane>(
        self,
        table: Self::Vec<T>,
        idx: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Lane-level table lookup. Each lane in `idx` (as integer) selects a lane
    /// from `v`. `I` is the integer lane type used for indices.
    fn table_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        v: Self::Vec<T>,
        idx: Self::Vec<I>,
    ) -> Self::Vec<T>;

    /// Reverse pairs of adjacent lanes: `[0,1,2,3]` -> `[1,0,3,2]`.
    fn reverse2<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Reverse groups of 4 lanes: `[0,1,2,3,4,5,6,7]` -> `[3,2,1,0,7,6,5,4]`.
    fn reverse4<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Reverse groups of 8 lanes.
    fn reverse8<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Upper half of `hi`, lower half of `lo`.
    fn concat_upper_lower<T: Lane>(
        self,
        hi: Self::Vec<T>,
        lo: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Lower half of `hi`, upper half of `lo`.
    fn concat_lower_upper<T: Lane>(
        self,
        hi: Self::Vec<T>,
        lo: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Even-indexed lanes from `a` (lower) and `b` (upper).
    fn concat_even<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Odd-indexed lanes from `a` (lower) and `b` (upper).
    fn concat_odd<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Odd lanes from `odd`, even lanes from `even`.
    fn odd_even<T: Lane>(
        self,
        odd: Self::Vec<T>,
        even: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Shift all lanes up (toward higher indices) by `N`, filling low lanes with zero.
    fn slide_up_lanes<T: Lane>(self, v: Self::Vec<T>, n: usize)
    -> Self::Vec<T>;

    /// Shift all lanes down (toward lower indices) by `N`, filling high lanes with zero.
    fn slide_down_lanes<T: Lane>(
        self,
        v: Self::Vec<T>,
        n: usize,
    ) -> Self::Vec<T>;

    /// Compress: pack lanes where mask is true to the low end.
    /// Inactive lane values are implementation-defined.
    fn compress<T: Lane>(
        self,
        v: Self::Vec<T>,
        mask: Self::Mask<T>,
    ) -> Self::Vec<T>;

    /// Compress and store to `ptr`. Returns the number of lanes written.
    unsafe fn compress_store<T: Lane>(
        self,
        v: Self::Vec<T>,
        mask: Self::Mask<T>,
        ptr: *mut T,
    ) -> usize;

    /// Duplicate even lanes: `[a0, a0, a2, a2, ...]`.
    fn dup_even<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Duplicate odd lanes: `[a1, a1, a3, a3, ...]`.
    fn dup_odd<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Concatenate the lower halves of `hi` and `lo`: lower half of `lo` in low,
    /// lower half of `hi` in high.
    fn concat_lower_lower<T: Lane>(
        self,
        hi: Self::Vec<T>,
        lo: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Concatenate the upper halves of `hi` and `lo`: upper half of `lo` in low,
    /// upper half of `hi` in high.
    fn concat_upper_upper<T: Lane>(
        self,
        hi: Self::Vec<T>,
        lo: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Shift all lanes up by 1, filling lane 0 with zero.
    fn slide_1_up<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Shift all lanes down by 1, filling the last lane with zero.
    fn slide_1_down<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Expand (inverse of compress): scatter low lanes to mask-true positions,
    /// zero where mask is false.
    fn expand<T: Lane>(
        self,
        v: Self::Vec<T>,
        mask: Self::Mask<T>,
    ) -> Self::Vec<T>;

    /// Concatenate hi:lo and extract a window of vector-size starting at byte BYTES.
    /// Per-128-bit-block operation (PALIGNR semantics). BYTES must be 1..=15.
    fn combine_shift_right_bytes<T: Lane, const BYTES: usize>(
        self,
        hi: Self::Vec<T>,
        lo: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Compress + blended store: compressed lanes are written to memory,
    /// leaving other positions unchanged. Returns the number of true lanes.
    unsafe fn compress_blended_store<T: Lane>(
        self,
        v: Self::Vec<T>,
        mask: Self::Mask<T>,
        ptr: *mut T,
    ) -> usize;

    /// Alternate 128-bit blocks: even blocks from `even`, odd blocks from `odd`.
    /// For 128-bit vectors returns `even`. For 256-bit: lower 128 from even, upper from odd.
    fn odd_even_blocks<T: Lane>(
        self,
        odd: Self::Vec<T>,
        even: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Reverse the order of 128-bit blocks within the vector.
    /// For 128-bit vectors returns `v` unchanged.
    fn reverse_blocks<T: Lane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Compress keeping elements where mask is FALSE.
    fn compress_not<T: Lane>(
        self,
        v: Self::Vec<T>,
        mask: Self::Mask<T>,
    ) -> Self::Vec<T>;

    /// Compress 128-bit blocks where mask is FALSE. Operates on u64 only.
    fn compress_blocks_not(
        self,
        v: Self::Vec<u64>,
        mask: Self::Mask<u64>,
    ) -> Self::Vec<u64>;

    /// Duplicate 128-bit block IDX across the entire vector.
    fn broadcast_block<T: Lane, const IDX: usize>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Compress using a raw bit array instead of mask.
    unsafe fn compress_bits<T: Lane>(
        self,
        v: Self::Vec<T>,
        bits: *const u8,
    ) -> Self::Vec<T>;

    /// Compress from bit array and store. Returns number of lanes written.
    unsafe fn compress_bits_store<T: Lane>(
        self,
        v: Self::Vec<T>,
        bits: *const u8,
        ptr: *mut T,
    ) -> usize;

    /// Extract the lower half of a vector as a half-width vector.
    fn lower_half<T: Lane>(self, v: Self::Vec<T>) -> Self::VecHalf<T>;

    /// Extract the upper half of a vector as a half-width vector.
    fn upper_half<T: Lane>(self, v: Self::Vec<T>) -> Self::VecHalf<T>;

    /// Combine two half-width vectors into a full-width vector.
    fn combine<T: Lane>(
        self,
        lo: Self::VecHalf<T>,
        hi: Self::VecHalf<T>,
    ) -> Self::Vec<T>;

    /// Replace 128-bit block at index IDX.
    fn insert_block<T: Lane, const IDX: usize>(
        self,
        v: Self::Vec<T>,
        blk: Self::VecHalf<T>,
    ) -> Self::Vec<T>;

    /// Extract 128-bit block at index IDX.
    fn extract_block<T: Lane, const IDX: usize>(
        self,
        v: Self::Vec<T>,
    ) -> Self::VecHalf<T>;

    /// Cross-block interleave of lower halves.
    fn interleave_whole_lower<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Cross-block interleave of upper halves.
    fn interleave_whole_upper<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Interleave even-indexed lanes: `[a0, b0, a2, b2, ...]`.
    fn interleave_even<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Interleave odd-indexed lanes: `[a1, b1, a3, b3, ...]`.
    fn interleave_odd<T: Lane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Lane-level lookup across two source vectors.
    /// Index values 0..N-1 select from `a`, N..2N-1 select from `b`.
    fn two_tables_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
        idx: Self::Vec<I>,
    ) -> Self::Vec<T>;

    /// Lane lookup with zero for out-of-range indices (high bit set).
    fn table_lookup_lanes_or0<T: Lane, I: IntegerLane>(
        self,
        v: Self::Vec<T>,
        idx: Self::Vec<I>,
    ) -> Self::Vec<T>;
}

// ---------------------------------------------------------------------------
// SimdReduce — horizontal reductions
// ---------------------------------------------------------------------------

/// Horizontal (cross-lane) reduction operations.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdReduce: Simd {
    /// Sum of all lanes.
    fn sum_of_lanes<T: Lane>(self, v: Self::Vec<T>) -> T;

    /// Minimum of all lanes.
    fn min_of_lanes<T: Lane>(self, v: Self::Vec<T>) -> T;

    /// Maximum of all lanes.
    fn max_of_lanes<T: Lane>(self, v: Self::Vec<T>) -> T;

    /// Sum of 8 absolute differences per u64 group.
    /// Each u64 result lane = sum of |a[i]-b[i]| for 8 consecutive u8 lanes.
    fn sums_of_8_abs_diff(
        self,
        a: Self::Vec<u8>,
        b: Self::Vec<u8>,
    ) -> Self::Vec<u64>;

    /// Pairwise widening addition: `result[i] = v[2i] + v[2i+1]`.
    /// Promotes each pair of narrow lanes, adds, and returns the wide result.
    fn sums_of_2<T: NarrowLane>(self, v: Self::Vec<T>) -> Self::Vec<T::Wide>
    where
        T::Wide: Lane;

    /// Sum groups of 4 adjacent lanes with double-widening.
    /// e.g. u8->u32, i8->i32, u16->u64, i16->i64.
    /// Equivalent to `sums_of_2(sums_of_2(v))`.
    fn sums_of_4<T: NarrowLane>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<<T::Wide as NarrowLane>::Wide>
    where
        T::Wide: NarrowLane + Lane,
        <T::Wide as NarrowLane>::Wide: Lane;
}

// ---------------------------------------------------------------------------
// SimdFloat — floating-point specific operations
// ---------------------------------------------------------------------------

/// Floating-point specific operations.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled.
pub unsafe trait SimdFloat: Simd {
    /// Square root.
    fn sqrt<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Approximate reciprocal (1/x).
    fn approx_reciprocal<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Approximate reciprocal square root (1/sqrt(x)).
    fn approx_reciprocal_sqrt<T: FloatLane>(
        self,
        v: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Round to nearest integer (ties to even).
    fn round<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Truncate toward zero.
    fn trunc<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Round toward positive infinity.
    fn ceil<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Round toward negative infinity.
    fn floor<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Fused multiply-add: `a * b + c`.
    fn mul_add<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
        c: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Negated fused multiply-add: `-(a * b) + c`.
    fn neg_mul_add<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
        c: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Fused multiply-subtract: `a * b - c`.
    fn mul_sub<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
        c: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Negated fused multiply-subtract: `-(a * b) - c`.
    fn neg_mul_sub<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
        c: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Copy the sign of `sign` to the magnitude of `mag`.
    fn copy_sign<T: FloatLane>(
        self,
        mag: Self::Vec<T>,
        sign: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane NaN test.
    fn is_nan<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Mask<T>;

    /// Per-lane infinity test.
    fn is_inf<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Mask<T>;

    /// Zero lanes with a negative sign bit. Positive values (and +0) are unchanged.
    fn zero_if_negative<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Vec<T>;

    /// Per-lane: true if the value is finite (not NaN and not Inf).
    fn is_finite<T: FloatLane>(self, v: Self::Vec<T>) -> Self::Mask<T>;

    /// Alternating subtract/add: `[a0-b0, a1+b1, a2-b2, a3+b3, ...]`.
    /// Even lanes subtract, odd lanes add.
    fn add_sub<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// NaN-propagating minimum: if one operand is NaN, returns the non-NaN operand.
    fn min_number<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// NaN-propagating maximum: if one operand is NaN, returns the non-NaN operand.
    fn max_number<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Minimum by absolute value. Break ties via Min.
    fn min_magnitude<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Maximum by absolute value. Break ties via Max.
    fn max_magnitude<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Vec<T>;

    /// Per-lane: true if either operand is NaN.
    fn is_either_nan<T: FloatLane>(
        self,
        a: Self::Vec<T>,
        b: Self::Vec<T>,
    ) -> Self::Mask<T>;
}

// ---------------------------------------------------------------------------
// SimdCrypto — optional AES and carry-less multiply operations
// ---------------------------------------------------------------------------

/// Cryptographic SIMD operations (AES, carry-less multiply).
///
/// This is a separate, optional trait — not included in `SimdOps`.
/// Backend support depends on the availability of AES-NI / VAES and PCLMULQDQ / VPCLMULQDQ.
///
/// # Safety
/// Callers must ensure the target's CPU features are enabled, including
/// the appropriate AES/PCLMUL feature flags.
pub unsafe trait SimdCrypto: Simd {
    /// One AES encryption round: `aesenc(state, round_key)`.
    fn aes_round(
        self,
        state: Self::Vec<u8>,
        round_key: Self::Vec<u8>,
    ) -> Self::Vec<u8>;

    /// Last AES encryption round: `aesenclast(state, round_key)`.
    fn aes_last_round(
        self,
        state: Self::Vec<u8>,
        round_key: Self::Vec<u8>,
    ) -> Self::Vec<u8>;

    /// One AES decryption round: `aesdec(state, round_key)`.
    fn aes_round_inv(
        self,
        state: Self::Vec<u8>,
        round_key: Self::Vec<u8>,
    ) -> Self::Vec<u8>;

    /// Last AES decryption round: `aesdeclast(state, round_key)`.
    fn aes_last_round_inv(
        self,
        state: Self::Vec<u8>,
        round_key: Self::Vec<u8>,
    ) -> Self::Vec<u8>;

    /// Carry-less multiply of the lower 64-bit halves of each 128-bit block.
    fn cl_mul_lower(
        self,
        a: Self::Vec<u64>,
        b: Self::Vec<u64>,
    ) -> Self::Vec<u64>;

    /// Carry-less multiply of the upper 64-bit halves of each 128-bit block.
    fn cl_mul_upper(
        self,
        a: Self::Vec<u64>,
        b: Self::Vec<u64>,
    ) -> Self::Vec<u64>;

    /// AES key schedule assist per 128-bit block.
    fn aes_key_gen_assist<const RCON: i32>(
        self,
        v: Self::Vec<u8>,
    ) -> Self::Vec<u8>;

    /// AES inverse MixColumns for decryption key expansion per 128-bit block.
    fn aes_inv_mix_columns(self, v: Self::Vec<u8>) -> Self::Vec<u8>;
}

// ---------------------------------------------------------------------------
// SimdOps — supertrait combining everything
// ---------------------------------------------------------------------------

/// Supertrait combining all SIMD operation categories.
///
/// Bounding on `SimdOps` gives access to the full operation set.
pub unsafe trait SimdOps:
    SimdCore
    + SimdMemory
    + SimdArith
    + SimdBitwise
    + SimdCompare
    + SimdMask
    + SimdConvert
    + SimdShuffle
    + SimdReduce
    + SimdFloat
{
}

// Blanket implementation: any type implementing all sub-traits is SimdOps.
// SAFETY: Delegates entirely to the sub-trait implementations.
unsafe impl<S> SimdOps for S where
    S: SimdCore
        + SimdMemory
        + SimdArith
        + SimdBitwise
        + SimdCompare
        + SimdMask
        + SimdConvert
        + SimdShuffle
        + SimdReduce
        + SimdFloat
{
}

// ---------------------------------------------------------------------------
// Free-function wrappers (ergonomic API matching Highway's style)
// ---------------------------------------------------------------------------

/// Returns a vector with all lanes set to zero.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn zero<S: SimdCore, T: Lane>(s: S) -> S::Vec<T> {
    // SAFETY: Caller guarantees CPU features.
    s.zero()
}

/// Returns a vector with all lanes set to `value`.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn splat<S: SimdCore, T: Lane>(s: S, value: T) -> S::Vec<T> {
    // SAFETY: Caller guarantees CPU features.
    s.splat(value)
}

/// Load a full vector from an aligned pointer.
///
/// # Safety
/// `ptr` must be aligned and point to at least `s.lanes::<T>()` valid elements.
#[inline(always)]
pub unsafe fn load<S: SimdMemory, T: Lane>(s: S, ptr: *const T) -> S::Vec<T> {
    // SAFETY: Caller guarantees pointer validity.
    unsafe { s.load(ptr) }
}

/// Store a full vector to an aligned pointer.
///
/// # Safety
/// `ptr` must be aligned and point to at least `s.lanes::<T>()` valid elements.
#[inline(always)]
pub unsafe fn store<S: SimdMemory, T: Lane>(s: S, v: S::Vec<T>, ptr: *mut T) {
    // SAFETY: Caller guarantees pointer validity.
    unsafe { s.store(v, ptr) }
}

/// Lane-wise addition.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn add<S: SimdArith, T: Lane>(
    s: S,
    a: S::Vec<T>,
    b: S::Vec<T>,
) -> S::Vec<T> {
    // SAFETY: Caller guarantees CPU features.
    s.add(a, b)
}

/// Lane-wise subtraction.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn sub<S: SimdArith, T: Lane>(
    s: S,
    a: S::Vec<T>,
    b: S::Vec<T>,
) -> S::Vec<T> {
    // SAFETY: Caller guarantees CPU features.
    s.sub(a, b)
}

/// Lane-wise multiplication.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn mul<S: SimdArith, T: Lane>(
    s: S,
    a: S::Vec<T>,
    b: S::Vec<T>,
) -> S::Vec<T> {
    // SAFETY: Caller guarantees CPU features.
    s.mul(a, b)
}

/// Lane-wise equality comparison.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn eq<S: SimdCompare, T: Lane>(
    s: S,
    a: S::Vec<T>,
    b: S::Vec<T>,
) -> S::Mask<T> {
    // SAFETY: Caller guarantees CPU features.
    s.eq(a, b)
}

/// Per-lane select: `if mask then yes else no`.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn if_then_else<S: SimdMask, T: Lane>(
    s: S,
    mask: S::Mask<T>,
    yes: S::Vec<T>,
    no: S::Vec<T>,
) -> S::Vec<T> {
    // SAFETY: Caller guarantees CPU features.
    s.if_then_else(mask, yes, no)
}

/// Sum of all lanes.
///
/// # Safety
/// The target's CPU features must be enabled.
#[inline(always)]
pub fn sum_of_lanes<S: SimdReduce, T: Lane>(s: S, v: S::Vec<T>) -> T {
    // SAFETY: Caller guarantees CPU features.
    s.sum_of_lanes(v)
}
