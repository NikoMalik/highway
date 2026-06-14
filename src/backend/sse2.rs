// All unsafe blocks in this module wrap SSE2 intrinsics or transmute_copy
// for type-punning. Safety invariants are documented on the outer `unsafe impl`
// blocks; individual intrinsic calls are safe when inputs are valid __m128i.
#![allow(clippy::undocumented_unsafe_blocks)]

/// SSE2 backend.
///
/// Provides 128-bit SIMD operations via `core::arch::x86_64::_mm_*` intrinsics.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::marker::PhantomData;

use crate::lane::{FloatLane, IntegerLane, Lane, NarrowLane, UnsignedLane, WideLane};
use crate::ops::{
    SimdArith, SimdBitwise, SimdCompare, SimdConvert, SimdCore, SimdFloat, SimdMask, SimdMemory,
    SimdReduce, SimdShuffle,
};
use crate::simd::{self, Simd};
use crate::{A16, Aligned};

// ---------------------------------------------------------------------------
// Target type
// ---------------------------------------------------------------------------

/// The SSE2 SIMD target (128-bit vectors).
///
/// This token is a *proof* that SSE2 is available on the running CPU: it cannot
/// be constructed from safe code (the inner field is private). It is handed to
/// kernels only by the dispatch machinery after a runtime feature check, which
/// is why all SSE2 vector operations can have safe signatures.
#[derive(Clone, Copy, Debug)]
pub struct Sse2(());

impl Sse2 {
    /// Construct an SSE2 token without checking CPU support.
    ///
    /// # Safety
    /// The caller must ensure SSE2 is available on the running CPU. Prefer
    /// obtaining a token through `dispatch`/`dispatch_to`, which checks at runtime.
    #[inline(always)]
    pub unsafe fn new_unchecked() -> Self {
        Sse2(())
    }

    #[inline(always)]
    pub(crate) fn new() -> Self {
        Sse2(())
    }
}

// ---------------------------------------------------------------------------
// Vector and Mask types
// ---------------------------------------------------------------------------

/// A 128-bit SIMD vector holding lanes of type `T`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct V128<T: Lane> {
    raw: __m128i,
    _marker: PhantomData<T>,
}

impl<T: Lane> core::fmt::Debug for V128<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("V128").finish_non_exhaustive()
    }
}

/// A 128-bit mask corresponding to vectors of type `T`.
/// For SSE2, masks are full vectors with all-ones/all-zeros per lane.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct M128<T: Lane> {
    raw: __m128i,
    _marker: PhantomData<T>,
}

impl<T: Lane> core::fmt::Debug for M128<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("M128").finish_non_exhaustive()
    }
}

impl<T: Lane> V128<T> {
    #[inline(always)]
    pub(crate) fn from_raw(raw: __m128i) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Access the raw intrinsic type.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m128i {
        self.raw
    }
}

impl<T: Lane> M128<T> {
    #[inline(always)]
    pub(crate) fn from_raw(raw: __m128i) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }
}

// SAFETY: SSE2 vectors are 128 bits = 16 bytes.
unsafe impl Simd for Sse2 {
    type Vec<T: Lane> = V128<T>;
    type Mask<T: Lane> = M128<T>;
    // Half-width is identity for SSE2 (can't go below 128-bit).
    type VecHalf<T: Lane> = V128<T>;
    type MaskHalf<T: Lane> = M128<T>;
    const VECTOR_BYTES: usize = 16;
}

// ---------------------------------------------------------------------------
// SimdCore
// ---------------------------------------------------------------------------

// dispatch trampoline. Callers through dispatch() always have SSE2 available.
unsafe impl SimdCore for Sse2 {
    #[inline(always)]
    fn zero<T: Lane>(self) -> V128<T> {
        V128::from_raw(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    fn splat<T: Lane>(self, value: T) -> V128<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    let b: u8 = core::mem::transmute_copy(&value);
                    V128::from_raw(_mm_set1_epi8(b as i8))
                }
                2 => {
                    let h: u16 = core::mem::transmute_copy(&value);
                    V128::from_raw(_mm_set1_epi16(h as i16))
                }
                4 => {
                    let w: u32 = core::mem::transmute_copy(&value);
                    V128::from_raw(_mm_set1_epi32(w as i32))
                }
                8 => {
                    let d: u64 = core::mem::transmute_copy(&value);
                    V128::from_raw(_mm_set1_epi64x(d as i64))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn undefined<T: Lane>(self) -> V128<T> {
        // SAFETY: Same as zero for safety; undefined allows optimizer freedom.
        V128::from_raw(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    fn bitcast<T: Lane, U: Lane>(self, v: V128<T>) -> V128<U> {
        V128::from_raw(v.raw)
    }

    #[inline(always)]
    unsafe fn extract_lane<T: Lane>(self, v: V128<T>, index: usize) -> T {
        unsafe {
            let mut arr: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            _mm_store_si128(arr.as_mut_ptr().cast(), v.raw);
            let offset = index * T::BYTES;
            let mut result = T::default();
            core::ptr::copy_nonoverlapping(
                arr.as_ptr().add(offset),
                core::ptr::from_mut(&mut result).cast::<u8>(),
                T::BYTES,
            );
            result
        }
    }

    #[inline(always)]
    unsafe fn insert_lane<T: Lane>(self, v: V128<T>, index: usize, value: T) -> V128<T> {
        unsafe {
            let mut arr: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            _mm_store_si128(arr.as_mut_ptr().cast(), v.raw);
            let offset = index * T::BYTES;
            core::ptr::copy_nonoverlapping(
                core::ptr::from_ref(&value).cast::<u8>(),
                arr.as_mut_ptr().add(offset),
                T::BYTES,
            );
            V128::from_raw(_mm_load_si128(arr.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn iota<T: Lane>(self, base: T) -> V128<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    let b: i8 = core::mem::transmute_copy(&base);
                    let iota = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                    V128::from_raw(_mm_add_epi8(iota, _mm_set1_epi8(b)))
                }
                2 => {
                    let b: i16 = core::mem::transmute_copy(&base);
                    let iota = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
                    V128::from_raw(_mm_add_epi16(iota, _mm_set1_epi16(b)))
                }
                4 => {
                    if is_type::<T, f32>() {
                        let b: f32 = core::mem::transmute_copy(&base);
                        let iota = _mm_set_ps(3.0, 2.0, 1.0, 0.0);
                        V128::from_raw(_mm_castps_si128(_mm_add_ps(iota, _mm_set1_ps(b))))
                    } else {
                        let b: i32 = core::mem::transmute_copy(&base);
                        let iota = _mm_set_epi32(3, 2, 1, 0);
                        V128::from_raw(_mm_add_epi32(iota, _mm_set1_epi32(b)))
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        let b: f64 = core::mem::transmute_copy(&base);
                        let iota = _mm_set_pd(1.0, 0.0);
                        V128::from_raw(_mm_castpd_si128(_mm_add_pd(iota, _mm_set1_pd(b))))
                    } else {
                        let b: i64 = core::mem::transmute_copy(&base);
                        let iota = _mm_set_epi64x(1, 0);
                        V128::from_raw(_mm_add_epi64(iota, _mm_set_epi64x(b, b)))
                    }
                }
                _ => unreachable!(),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimdMemory
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require SSE2, guaranteed by dispatch trampoline.
unsafe impl SimdMemory for Sse2 {
    #[inline(always)]
    unsafe fn load<T: Lane>(self, ptr: *const T) -> V128<T> {
        V128::from_raw(unsafe { _mm_load_si128(ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn load_u<T: Lane>(self, ptr: *const T) -> V128<T> {
        V128::from_raw(unsafe { _mm_loadu_si128(ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn store<T: Lane>(self, v: V128<T>, ptr: *mut T) {
        unsafe { _mm_store_si128(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn store_u<T: Lane>(self, v: V128<T>, ptr: *mut T) {
        unsafe { _mm_storeu_si128(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn stream<T: Lane>(self, v: V128<T>, ptr: *mut T) {
        unsafe { _mm_stream_si128(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn load_dup128<T: Lane>(self, ptr: *const T) -> V128<T> {
        // For 128-bit SSE2, load_dup128 is the same as an unaligned load
        // (the pointer may not be 16-byte aligned).
        unsafe { self.load_u(ptr) }
    }

    #[inline(always)]
    unsafe fn masked_load<T: Lane>(self, mask: M128<T>, ptr: *const T) -> V128<T> {
        // Load all lanes (unaligned), then zero out lanes where mask is false.
        unsafe {
            let loaded = self.load_u(ptr);
            self.if_then_else_zero(mask, loaded)
        }
    }

    #[inline(always)]
    unsafe fn blended_store<T: Lane>(self, v: V128<T>, mask: M128<T>, ptr: *mut T) {
        // Load existing data, blend new values where mask is true, store back.
        unsafe {
            let existing = self.load_u(ptr);
            let blended = self.if_then_else(mask, v, existing);
            self.store_u(blended, ptr);
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn gather_index<T: Lane>(
        self,
        base: *const T,
        idx: V128<i32>,
    ) -> V128<T> {
        unsafe {
            let mut idx_arr = [0i32; 4];
            _mm_storeu_si128(idx_arr.as_mut_ptr().cast(), idx.raw);
            // Number of usable indices: min(lanes, 4) since idx has 4 i32 slots.
            // For T::BYTES >= 4, lanes <= 4 so all indices are used.
            // For smaller types, only the first 4 indices are available.
            let lanes = (16 / T::BYTES).min(4);
            let mut result = [0u8; 16];
            for i in 0..lanes {
                let offset = idx_arr[i] as isize;
                let src = base.offset(offset);
                core::ptr::copy_nonoverlapping(
                    src.cast::<u8>(),
                    result.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn scatter_index<T: Lane>(
        self,
        v: V128<T>,
        base: *mut T,
        idx: V128<i32>,
    ) {
        unsafe {
            let mut idx_arr = [0i32; 4];
            _mm_storeu_si128(idx_arr.as_mut_ptr().cast(), idx.raw);
            let mut v_arr = [0u8; 16];
            _mm_storeu_si128(v_arr.as_mut_ptr().cast(), v.raw);
            // Only use as many indices as available (4 i32 slots)
            let lanes = (16 / T::BYTES).min(4);
            for i in 0..lanes {
                let offset = idx_arr[i] as isize;
                let dst = base.offset(offset);
                core::ptr::copy_nonoverlapping(
                    v_arr.as_ptr().add(i * T::BYTES),
                    dst.cast::<u8>(),
                    T::BYTES,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_2<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V128<T>, V128<T>) {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr0 = [0u8; 16];
            let mut arr1 = [0u8; 16];
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 2).cast::<u8>(),
                    arr0.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 2 + 1).cast::<u8>(),
                    arr1.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V128::from_raw(_mm_loadu_si128(arr0.as_ptr().cast())),
                V128::from_raw(_mm_loadu_si128(arr1.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_3<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V128<T>, V128<T>, V128<T>) {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr0 = [0u8; 16];
            let mut arr1 = [0u8; 16];
            let mut arr2 = [0u8; 16];
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 3).cast::<u8>(),
                    arr0.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 3 + 1).cast::<u8>(),
                    arr1.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 3 + 2).cast::<u8>(),
                    arr2.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V128::from_raw(_mm_loadu_si128(arr0.as_ptr().cast())),
                V128::from_raw(_mm_loadu_si128(arr1.as_ptr().cast())),
                V128::from_raw(_mm_loadu_si128(arr2.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_4<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V128<T>, V128<T>, V128<T>, V128<T>) {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr0 = [0u8; 16];
            let mut arr1 = [0u8; 16];
            let mut arr2 = [0u8; 16];
            let mut arr3 = [0u8; 16];
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 4).cast::<u8>(),
                    arr0.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 4 + 1).cast::<u8>(),
                    arr1.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 4 + 2).cast::<u8>(),
                    arr2.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    ptr.add(i * 4 + 3).cast::<u8>(),
                    arr3.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V128::from_raw(_mm_loadu_si128(arr0.as_ptr().cast())),
                V128::from_raw(_mm_loadu_si128(arr1.as_ptr().cast())),
                V128::from_raw(_mm_loadu_si128(arr2.as_ptr().cast())),
                V128::from_raw(_mm_loadu_si128(arr3.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_2<T: Lane>(
        self,
        v0: V128<T>,
        v1: V128<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr0 = [0u8; 16];
            let mut arr1 = [0u8; 16];
            _mm_storeu_si128(arr0.as_mut_ptr().cast(), v0.raw);
            _mm_storeu_si128(arr1.as_mut_ptr().cast(), v1.raw);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    arr0.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 2).cast::<u8>(),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    arr1.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 2 + 1).cast::<u8>(),
                    T::BYTES,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_3<T: Lane>(
        self,
        v0: V128<T>,
        v1: V128<T>,
        v2: V128<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr0 = [0u8; 16];
            let mut arr1 = [0u8; 16];
            let mut arr2 = [0u8; 16];
            _mm_storeu_si128(arr0.as_mut_ptr().cast(), v0.raw);
            _mm_storeu_si128(arr1.as_mut_ptr().cast(), v1.raw);
            _mm_storeu_si128(arr2.as_mut_ptr().cast(), v2.raw);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    arr0.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 3).cast::<u8>(),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    arr1.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 3 + 1).cast::<u8>(),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    arr2.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 3 + 2).cast::<u8>(),
                    T::BYTES,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_4<T: Lane>(
        self,
        v0: V128<T>,
        v1: V128<T>,
        v2: V128<T>,
        v3: V128<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr0 = [0u8; 16];
            let mut arr1 = [0u8; 16];
            let mut arr2 = [0u8; 16];
            let mut arr3 = [0u8; 16];
            _mm_storeu_si128(arr0.as_mut_ptr().cast(), v0.raw);
            _mm_storeu_si128(arr1.as_mut_ptr().cast(), v1.raw);
            _mm_storeu_si128(arr2.as_mut_ptr().cast(), v2.raw);
            _mm_storeu_si128(arr3.as_mut_ptr().cast(), v3.raw);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    arr0.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 4).cast::<u8>(),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    arr1.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 4 + 1).cast::<u8>(),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    arr2.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 4 + 2).cast::<u8>(),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    arr3.as_ptr().add(i * T::BYTES),
                    ptr.add(i * 4 + 3).cast::<u8>(),
                    T::BYTES,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn load_expand<T: Lane>(self, mask: M128<T>, ptr: *const T) -> V128<T> {
        // Load and then expand
        unsafe {
            let loaded = self.load_u(ptr);
            self.expand(loaded, mask)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdArith
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require SSE2, guaranteed by dispatch trampoline.
unsafe impl SimdArith for Sse2 {
    #[inline(always)]
    fn add<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_add_epi8(a.raw, b.raw),
                2 => _mm_add_epi16(a.raw, b.raw),
                4 => {
                    // Could be i32/u32 or f32
                    if core::mem::align_of::<T>() == core::mem::align_of::<f32>()
                        && core::mem::size_of::<T>() == 4
                        && is_type::<T, f32>()
                    {
                        let fa = _mm_castsi128_ps(a.raw);
                        let fb = _mm_castsi128_ps(b.raw);
                        _mm_castps_si128(_mm_add_ps(fa, fb))
                    } else {
                        _mm_add_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        let fa = _mm_castsi128_pd(a.raw);
                        let fb = _mm_castsi128_pd(b.raw);
                        _mm_castpd_si128(_mm_add_pd(fa, fb))
                    } else {
                        _mm_add_epi64(a.raw, b.raw)
                    }
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn sub<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_sub_epi8(a.raw, b.raw),
                2 => _mm_sub_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm_castps_si128(_mm_sub_ps(
                            _mm_castsi128_ps(a.raw),
                            _mm_castsi128_ps(b.raw),
                        ))
                    } else {
                        _mm_sub_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm_castpd_si128(_mm_sub_pd(
                            _mm_castsi128_pd(a.raw),
                            _mm_castsi128_pd(b.raw),
                        ))
                    } else {
                        _mm_sub_epi64(a.raw, b.raw)
                    }
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn mul<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // No _mm_mullo_epi8; use 16-bit multiply with mask
                    let mask = _mm_set1_epi16(0x00FF);
                    let a_lo = _mm_and_si128(a.raw, mask);
                    let b_lo = _mm_and_si128(b.raw, mask);
                    let mul_lo = _mm_and_si128(_mm_mullo_epi16(a_lo, b_lo), mask);

                    let a_hi = _mm_srli_epi16(a.raw, 8);
                    let b_hi = _mm_srli_epi16(b.raw, 8);
                    let mul_hi = _mm_slli_epi16(_mm_mullo_epi16(a_hi, b_hi), 8);

                    _mm_or_si128(mul_lo, mul_hi)
                }
                2 => _mm_mullo_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm_castps_si128(_mm_mul_ps(
                            _mm_castsi128_ps(a.raw),
                            _mm_castsi128_ps(b.raw),
                        ))
                    } else {
                        // SSE2 doesn't have _mm_mullo_epi32; emulate
                        // mullo_epi32: multiply pairs and pack
                        let a13 = _mm_shuffle_epi32(a.raw, 0xF5); // a[1], a[3], ...
                        let b13 = _mm_shuffle_epi32(b.raw, 0xF5);
                        let prod02 = _mm_mul_epu32(a.raw, b.raw);
                        let prod13 = _mm_mul_epu32(a13, b13);
                        let prod02_lo = _mm_shuffle_epi32(prod02, 0x08); // [prod0_lo, prod2_lo, 0, 0]
                        let prod13_lo = _mm_shuffle_epi32(prod13, 0x08);
                        _mm_unpacklo_epi32(prod02_lo, prod13_lo)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm_castpd_si128(_mm_mul_pd(
                            _mm_castsi128_pd(a.raw),
                            _mm_castsi128_pd(b.raw),
                        ))
                    } else {
                        // 64-bit integer mul: emulate with 32-bit ops
                        // (a_lo * b_lo) + ((a_lo * b_hi + a_hi * b_lo) << 32)
                        let a_hi = _mm_srli_epi64(a.raw, 32);
                        let b_hi = _mm_srli_epi64(b.raw, 32);
                        let mul_ll = _mm_mul_epu32(a.raw, b.raw);
                        let mul_lh = _mm_mul_epu32(a.raw, b_hi);
                        let mul_hl = _mm_mul_epu32(a_hi, b.raw);
                        let cross = _mm_add_epi64(mul_lh, mul_hl);
                        _mm_add_epi64(mul_ll, _mm_slli_epi64(cross, 32))
                    }
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn div<T: FloatLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm_castps_si128(_mm_div_ps(_mm_castsi128_ps(a.raw), _mm_castsi128_ps(b.raw)))
            } else {
                _mm_castpd_si128(_mm_div_pd(_mm_castsi128_pd(a.raw), _mm_castsi128_pd(b.raw)))
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn saturated_add<T: IntegerLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    if is_type::<T, u8>() {
                        _mm_adds_epu8(a.raw, b.raw)
                    } else {
                        _mm_adds_epi8(a.raw, b.raw)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm_adds_epu16(a.raw, b.raw)
                    } else {
                        _mm_adds_epi16(a.raw, b.raw)
                    }
                }
                // SSE2 doesn't have 32/64-bit saturating add; vectorized emulation
                _ => {
                    if is_type::<T, u32>() {
                        // Unsigned u32: sum = a + b; if sum < a, overflow -> MAX
                        let sum = _mm_add_epi32(a.raw, b.raw);
                        // Unsigned compare: XOR sign bits to convert to signed compare
                        let sign = _mm_set1_epi32(i32::MIN);
                        let overflow =
                            _mm_cmplt_epi32(_mm_xor_si128(sum, sign), _mm_xor_si128(a.raw, sign));
                        _mm_or_si128(sum, overflow) // overflow lanes become 0xFFFFFFFF
                    } else if is_type::<T, i32>() {
                        // Signed i32: overflow if signs(a)==signs(b) but signs(sum)!=signs(a)
                        let sum = _mm_add_epi32(a.raw, b.raw);
                        let sign_a = _mm_srai_epi32(a.raw, 31);
                        // same_sign = ~(a ^ b), signs_differ = a ^ sum
                        let same_sign =
                            _mm_andnot_si128(_mm_xor_si128(a.raw, b.raw), _mm_set1_epi32(-1));
                        let overflow = _mm_and_si128(same_sign, _mm_xor_si128(a.raw, sum));
                        let overflow_mask = _mm_srai_epi32(overflow, 31);
                        // Saturated value: if a >= 0, MAX; if a < 0, MIN
                        let sat_val = _mm_xor_si128(sign_a, _mm_set1_epi32(i32::MAX));
                        // Blend: if overflow, use sat_val; else use sum
                        _mm_or_si128(
                            _mm_and_si128(overflow_mask, sat_val),
                            _mm_andnot_si128(overflow_mask, sum),
                        )
                    } else if is_type::<T, u64>() {
                        // SaturatedAdd u64: Add(a, Min(b, Not(a)))
                        let not_a: V128<u64> = self.not(V128::from_raw(a.raw));
                        let b_u64: V128<u64> = V128::from_raw(b.raw);
                        let clamped: V128<u64> = self.min(b_u64, not_a);
                        _mm_add_epi64(a.raw, clamped.raw)
                    } else {
                        // i64: SaturatedAdd
                        let sum = _mm_add_epi64(a.raw, b.raw);
                        // overflow_mask = andnot(xor(a, b), xor(a, sum)) — overflow if same signs, result differs
                        let overflow_mask = _mm_andnot_si128(
                            _mm_xor_si128(a.raw, b.raw),
                            _mm_xor_si128(a.raw, sum),
                        );
                        // BroadcastSignBit i64 on SSE2: srai_epi32 by 31, then shuffle to broadcast high i32
                        let sign_a = _mm_srai_epi32(a.raw, 31);
                        let sign_a = _mm_shuffle_epi32(sign_a, 0xF5); // broadcast bit 31 of each i64
                        // overflow_result = xor(sign_a, MAX) -> positive->MAX, negative->MIN
                        let overflow_result = _mm_xor_si128(sign_a, _mm_set1_epi64x(i64::MAX));
                        // if_negative_then_else: use sign bit of overflow_mask as blend mask
                        let mask = _mm_srai_epi32(overflow_mask, 31);
                        let mask = _mm_shuffle_epi32(mask, 0xF5);
                        _mm_or_si128(
                            _mm_and_si128(mask, overflow_result),
                            _mm_andnot_si128(mask, sum),
                        )
                    }
                }
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn saturated_sub<T: IntegerLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    if is_type::<T, u8>() {
                        _mm_subs_epu8(a.raw, b.raw)
                    } else {
                        _mm_subs_epi8(a.raw, b.raw)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm_subs_epu16(a.raw, b.raw)
                    } else {
                        _mm_subs_epi16(a.raw, b.raw)
                    }
                }
                _ => {
                    if is_type::<T, u32>() {
                        // Unsigned u32: if b > a, result = 0; else a - b
                        let diff = _mm_sub_epi32(a.raw, b.raw);
                        // Unsigned compare b > a: XOR sign bits
                        let sign = _mm_set1_epi32(i32::MIN);
                        let underflow =
                            _mm_cmpgt_epi32(_mm_xor_si128(b.raw, sign), _mm_xor_si128(a.raw, sign));
                        _mm_andnot_si128(underflow, diff) // zero out underflow lanes
                    } else if is_type::<T, i32>() {
                        // Signed i32: overflow if signs(a)!=signs(b) and signs(result)!=signs(a)
                        let diff = _mm_sub_epi32(a.raw, b.raw);
                        let sign_a = _mm_srai_epi32(a.raw, 31);
                        // diff_sign = a ^ b (different signs makes overflow possible)
                        let diff_signs = _mm_xor_si128(a.raw, b.raw);
                        let overflow = _mm_and_si128(diff_signs, _mm_xor_si128(a.raw, diff));
                        let overflow_mask = _mm_srai_epi32(overflow, 31);
                        let sat_val = _mm_xor_si128(sign_a, _mm_set1_epi32(i32::MAX));
                        _mm_or_si128(
                            _mm_and_si128(overflow_mask, sat_val),
                            _mm_andnot_si128(overflow_mask, diff),
                        )
                    } else if is_type::<T, u64>() {
                        // SaturatedSub u64: Sub(a, Min(a, b))
                        let a_u64: V128<u64> = V128::from_raw(a.raw);
                        let b_u64: V128<u64> = V128::from_raw(b.raw);
                        let clamped: V128<u64> = self.min(a_u64, b_u64);
                        _mm_sub_epi64(a.raw, clamped.raw)
                    } else {
                        // i64: SaturatedSub
                        let diff = _mm_sub_epi64(a.raw, b.raw);
                        // overflow_mask = and(xor(a, b), xor(a, diff)) — overflow if different signs, result sign differs from a
                        let overflow_mask = _mm_and_si128(
                            _mm_xor_si128(a.raw, b.raw),
                            _mm_xor_si128(a.raw, diff),
                        );
                        // BroadcastSignBit i64 on SSE2
                        let sign_a = _mm_srai_epi32(a.raw, 31);
                        let sign_a = _mm_shuffle_epi32(sign_a, 0xF5);
                        // overflow_result = xor(sign_a, MAX) -> positive->MAX, negative->MIN
                        let overflow_result = _mm_xor_si128(sign_a, _mm_set1_epi64x(i64::MAX));
                        // if_negative_then_else: use sign bit of overflow_mask as blend mask
                        let mask = _mm_srai_epi32(overflow_mask, 31);
                        let mask = _mm_shuffle_epi32(mask, 0xF5);
                        _mm_or_si128(
                            _mm_and_si128(mask, overflow_result),
                            _mm_andnot_si128(mask, diff),
                        )
                    }
                }
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn abs<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            // For floats: clear sign bit. For signed ints: negate if negative.
            if is_type::<T, f32>() {
                let mask = _mm_set1_epi32(0x7FFF_FFFFu32 as i32);
                V128::from_raw(_mm_and_si128(v.raw, mask))
            } else if is_type::<T, f64>() {
                let mask = _mm_set_epi64x(
                    0x7FFF_FFFF_FFFF_FFFFu64 as i64,
                    0x7FFF_FFFF_FFFF_FFFFu64 as i64,
                );
                V128::from_raw(_mm_and_si128(v.raw, mask))
            } else {
                // Integer abs — SSE2 doesn't have pabs; emulate for signed
                match T::BYTES {
                    1 => {
                        if is_signed::<T>() {
                            // abs(x) = (x ^ (x >> 7)) - (x >> 7)
                            let shift = _mm_cmpgt_epi8(_mm_setzero_si128(), v.raw);
                            V128::from_raw(_mm_sub_epi8(_mm_xor_si128(v.raw, shift), shift))
                        } else {
                            v
                        }
                    }
                    2 => {
                        if is_signed::<T>() {
                            let shift = _mm_srai_epi16(v.raw, 15);
                            V128::from_raw(_mm_sub_epi16(_mm_xor_si128(v.raw, shift), shift))
                        } else {
                            v
                        }
                    }
                    4 => {
                        if is_signed::<T>() {
                            let shift = _mm_srai_epi32(v.raw, 31);
                            V128::from_raw(_mm_sub_epi32(_mm_xor_si128(v.raw, shift), shift))
                        } else {
                            v
                        }
                    }
                    8 => {
                        if is_signed::<T>() {
                            // No _mm_srai_epi64 in SSE2; use comparison
                            let sign = _mm_cmpgt_epi32(_mm_setzero_si128(), v.raw);
                            let sign = _mm_shuffle_epi32(sign, 0xF5); // replicate high 32 bits
                            V128::from_raw(_mm_sub_epi64(_mm_xor_si128(v.raw, sign), sign))
                        } else {
                            v
                        }
                    }
                    _ => unreachable!(),
                }
            }
        }
    }

    #[inline(always)]
    fn neg<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            if is_type::<T, f32>() {
                let sign_bit = _mm_set1_epi32(0x8000_0000u32 as i32);
                V128::from_raw(_mm_xor_si128(v.raw, sign_bit))
            } else if is_type::<T, f64>() {
                let sign_bit = _mm_set_epi64x(
                    0x8000_0000_0000_0000u64 as i64,
                    0x8000_0000_0000_0000u64 as i64,
                );
                V128::from_raw(_mm_xor_si128(v.raw, sign_bit))
            } else {
                // Integer negate: 0 - v
                let z = _mm_setzero_si128();
                let raw = match T::BYTES {
                    1 => _mm_sub_epi8(z, v.raw),
                    2 => _mm_sub_epi16(z, v.raw),
                    4 => _mm_sub_epi32(z, v.raw),
                    8 => _mm_sub_epi64(z, v.raw),
                    _ => unreachable!(),
                };
                V128::from_raw(raw)
            }
        }
    }

    #[inline(always)]
    fn min<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            if is_type::<T, f32>() {
                V128::from_raw(_mm_castps_si128(_mm_min_ps(
                    _mm_castsi128_ps(a.raw),
                    _mm_castsi128_ps(b.raw),
                )))
            } else if is_type::<T, f64>() {
                V128::from_raw(_mm_castpd_si128(_mm_min_pd(
                    _mm_castsi128_pd(a.raw),
                    _mm_castsi128_pd(b.raw),
                )))
            } else if is_type::<T, u8>() {
                V128::from_raw(_mm_min_epu8(a.raw, b.raw))
            } else if is_type::<T, i16>() {
                V128::from_raw(_mm_min_epi16(a.raw, b.raw))
            } else {
                // Emulate for other integer types: if a < b then a else b
                let lt_mask = self.lt::<T>(a, b).raw;
                V128::from_raw(_mm_or_si128(
                    _mm_and_si128(lt_mask, a.raw),
                    _mm_andnot_si128(lt_mask, b.raw),
                ))
            }
        }
    }

    #[inline(always)]
    fn max<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            if is_type::<T, f32>() {
                V128::from_raw(_mm_castps_si128(_mm_max_ps(
                    _mm_castsi128_ps(a.raw),
                    _mm_castsi128_ps(b.raw),
                )))
            } else if is_type::<T, f64>() {
                V128::from_raw(_mm_castpd_si128(_mm_max_pd(
                    _mm_castsi128_pd(a.raw),
                    _mm_castsi128_pd(b.raw),
                )))
            } else if is_type::<T, u8>() {
                V128::from_raw(_mm_max_epu8(a.raw, b.raw))
            } else if is_type::<T, i16>() {
                V128::from_raw(_mm_max_epi16(a.raw, b.raw))
            } else {
                let gt_mask = self.gt::<T>(a, b).raw;
                V128::from_raw(_mm_or_si128(
                    _mm_and_si128(gt_mask, a.raw),
                    _mm_andnot_si128(gt_mask, b.raw),
                ))
            }
        }
    }

    #[inline(always)]
    fn mul_high<T: IntegerLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // u8/i8: high byte of each 16-bit product
                    let mask_lo = _mm_set1_epi16(0x00FF);
                    let mask_hi = _mm_set1_epi16(0xFF00u16 as i16);
                    if is_type::<T, u8>() {
                        // Even bytes: zero-extend
                        let a_even = _mm_and_si128(a.raw, mask_lo);
                        let b_even = _mm_and_si128(b.raw, mask_lo);
                        let prod_even = _mm_mullo_epi16(a_even, b_even);
                        let hi_even = _mm_srli_epi16(prod_even, 8);
                        // Odd bytes: shift right by 8
                        let a_odd = _mm_srli_epi16(a.raw, 8);
                        let b_odd = _mm_srli_epi16(b.raw, 8);
                        let prod_odd = _mm_mullo_epi16(a_odd, b_odd);
                        let hi_odd = _mm_and_si128(prod_odd, mask_hi);
                        _mm_or_si128(hi_even, hi_odd)
                    } else {
                        // i8: sign-extend even bytes via slli+srai
                        let a_even = _mm_srai_epi16(_mm_slli_epi16(a.raw, 8), 8);
                        let b_even = _mm_srai_epi16(_mm_slli_epi16(b.raw, 8), 8);
                        let prod_even = _mm_mullo_epi16(a_even, b_even);
                        let hi_even = _mm_srli_epi16(prod_even, 8);
                        // Odd bytes: arithmetic shift right by 8
                        let a_odd = _mm_srai_epi16(a.raw, 8);
                        let b_odd = _mm_srai_epi16(b.raw, 8);
                        let prod_odd = _mm_mullo_epi16(a_odd, b_odd);
                        let hi_odd = _mm_and_si128(prod_odd, mask_hi);
                        _mm_or_si128(hi_even, hi_odd)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm_mulhi_epu16(a.raw, b.raw)
                    } else {
                        _mm_mulhi_epi16(a.raw, b.raw)
                    }
                }
                _ => {
                    if is_type::<T, u32>() {
                        // u32: high 32 bits of each 64-bit product
                        let p_even = _mm_mul_epu32(a.raw, b.raw);
                        let a_odd = _mm_srli_epi64(a.raw, 32);
                        let b_odd = _mm_srli_epi64(b.raw, 32);
                        let p_odd = _mm_mul_epu32(a_odd, b_odd);
                        let hi_even = _mm_srli_epi64(p_even, 32);
                        let mask_hi32 = _mm_set_epi32(-1, 0, -1, 0);
                        let hi_odd = _mm_and_si128(p_odd, mask_hi32);
                        _mm_or_si128(hi_even, hi_odd)
                    } else {
                        // i32: unsigned MulHigh + sign correction
                        // signed_hi(a,b) = unsigned_hi(a,b) - (sign(a) ? b : 0) - (sign(b) ? a : 0)
                        let p_even = _mm_mul_epu32(a.raw, b.raw);
                        let a_odd = _mm_srli_epi64(a.raw, 32);
                        let b_odd = _mm_srli_epi64(b.raw, 32);
                        let p_odd = _mm_mul_epu32(a_odd, b_odd);
                        let hi_even = _mm_srli_epi64(p_even, 32);
                        let mask_hi32 = _mm_set_epi32(-1, 0, -1, 0);
                        let hi_odd = _mm_and_si128(p_odd, mask_hi32);
                        let unsigned_hi = _mm_or_si128(hi_even, hi_odd);
                        // Sign correction: broadcast sign bit of each i32
                        let sign_a = _mm_srai_epi32(a.raw, 31);
                        let sign_b = _mm_srai_epi32(b.raw, 31);
                        // sign_a is all 1s for negative lanes, 0 for positive
                        let correction_a = _mm_and_si128(sign_a, b.raw);
                        let correction_b = _mm_and_si128(sign_b, a.raw);
                        _mm_sub_epi32(_mm_sub_epi32(unsigned_hi, correction_a), correction_b)
                    }
                }
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn average_round<T: UnsignedLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_avg_epu8(a.raw, b.raw),
                2 => _mm_avg_epu16(a.raw, b.raw),
                _ => {
                    // avg(a,b) = (a >> 1) + (b >> 1) + ((a | b) & 1)
                    // Works for u32 and u64 without overflow.
                    if T::BYTES == 4 {
                        let one = _mm_set1_epi32(1);
                        let a_half = _mm_srli_epi32(a.raw, 1);
                        let b_half = _mm_srli_epi32(b.raw, 1);
                        let carry = _mm_and_si128(_mm_or_si128(a.raw, b.raw), one);
                        _mm_add_epi32(_mm_add_epi32(a_half, b_half), carry)
                    } else {
                        // u64
                        let one = _mm_set_epi64x(1, 1);
                        let a_half = _mm_srli_epi64(a.raw, 1);
                        let b_half = _mm_srli_epi64(b.raw, 1);
                        let carry = _mm_and_si128(_mm_or_si128(a.raw, b.raw), one);
                        _mm_add_epi64(_mm_add_epi64(a_half, b_half), carry)
                    }
                }
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn abs_diff<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // |a - b| = max(a, b) - min(a, b), which is always non-negative.
        {
            let hi = self.max(a, b);
            let lo = self.min(a, b);
            self.sub(hi, lo)
        }
    }

    #[inline(always)]
    fn clamp<T: Lane>(self, v: V128<T>, lo: V128<T>, hi: V128<T>) -> V128<T> {
        self.min(self.max(v, lo), hi)
    }

    #[inline(always)]
    fn mul_even<T: NarrowLane>(self, a: V128<T>, b: V128<T>) -> V128<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        // f32 -> f64: shuffle even lanes (0,2) to low, cvt, mul
                        let a_ps = _mm_castsi128_ps(a.raw);
                        let b_ps = _mm_castsi128_ps(b.raw);
                        let a_even = _mm_shuffle_ps(a_ps, a_ps, 0x08); // [a0, a2, ...]
                        let b_even = _mm_shuffle_ps(b_ps, b_ps, 0x08);
                        let a_pd = _mm_cvtps_pd(a_even);
                        let b_pd = _mm_cvtps_pd(b_even);
                        V128::from_raw(_mm_castpd_si128(_mm_mul_pd(a_pd, b_pd)))
                    } else if is_type::<T, u32>() {
                        // _mm_mul_epu32: multiplies lanes 0 and 2 (even 32-bit lanes)
                        // producing two 64-bit results.
                        V128::from_raw(_mm_mul_epu32(a.raw, b.raw))
                    } else if is_x86_feature_detected!("sse4.1") {
                        V128::from_raw(_mm_mul_epi32(a.raw, b.raw))
                    } else {
                        // i32 -> i64: scalar fallback (no _mm_mul_epi32 in SSE2)
                        let mut arr_a: Aligned<A16, [i32; 4]> = Aligned::new([0i32; 4]);
                        let mut arr_b: Aligned<A16, [i32; 4]> = Aligned::new([0i32; 4]);
                        _mm_store_si128(arr_a.as_mut_ptr().cast(), a.raw);
                        _mm_store_si128(arr_b.as_mut_ptr().cast(), b.raw);
                        let r0 = arr_a[0] as i64 * arr_b[0] as i64;
                        let r1 = arr_a[2] as i64 * arr_b[2] as i64;
                        let out: Aligned<A16, [i64; 2]> = Aligned::new([r0, r1]);
                        V128::from_raw(_mm_load_si128(out.as_ptr().cast()))
                    }
                }
                2 => {
                    // u16/i16 -> u32/i32: extract even 16-bit lanes, widen to 32-bit, multiply
                    let (a32, b32) = if is_signed::<T>() {
                        // Sign-extend even i16 to i32
                        (
                            _mm_srai_epi32(_mm_slli_epi32(a.raw, 16), 16),
                            _mm_srai_epi32(_mm_slli_epi32(b.raw, 16), 16),
                        )
                    } else {
                        // Zero-extend even u16 to u32
                        let mask = _mm_set1_epi32(0x0000FFFFu32 as i32);
                        (_mm_and_si128(a.raw, mask), _mm_and_si128(b.raw, mask))
                    };
                    // SSE2 lacks mullo_epi32; use two mul_epu32 calls + pack
                    let prod_02 = _mm_mul_epu32(a32, b32);
                    let a32_hi = _mm_srli_epi64(a32, 32);
                    let b32_hi = _mm_srli_epi64(b32, 32);
                    let prod_13 = _mm_mul_epu32(a32_hi, b32_hi);
                    let p02 = _mm_shuffle_epi32(prod_02, 0x08);
                    let p13 = _mm_shuffle_epi32(prod_13, 0x08);
                    V128::from_raw(_mm_unpacklo_epi32(p02, p13))
                }
                1 => {
                    // u8/i8 -> u16/i16: reinterpret as u16, isolate even bytes, multiply.
                    if is_signed::<T>() {
                        // i8 -> i16: sign-extend even bytes via shift left 8 + arithmetic shift right 8
                        let a16 = _mm_srai_epi16(_mm_slli_epi16(a.raw, 8), 8);
                        let b16 = _mm_srai_epi16(_mm_slli_epi16(b.raw, 8), 8);
                        V128::from_raw(_mm_mullo_epi16(a16, b16))
                    } else {
                        // u8 -> u16: mask to keep even bytes (low byte of each u16 lane)
                        let mask = _mm_set1_epi16(0x00FF);
                        let a16 = _mm_and_si128(a.raw, mask);
                        let b16 = _mm_and_si128(b.raw, mask);
                        V128::from_raw(_mm_mullo_epi16(a16, b16))
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn mul_odd<T: NarrowLane>(self, a: V128<T>, b: V128<T>) -> V128<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        // f32 -> f64: shuffle odd lanes (1,3) to low, cvt, mul
                        let a_ps = _mm_castsi128_ps(a.raw);
                        let b_ps = _mm_castsi128_ps(b.raw);
                        let a_odd = _mm_shuffle_ps(a_ps, a_ps, 0x0D); // [a1, a3, ...]
                        let b_odd = _mm_shuffle_ps(b_ps, b_ps, 0x0D);
                        let a_pd = _mm_cvtps_pd(a_odd);
                        let b_pd = _mm_cvtps_pd(b_odd);
                        V128::from_raw(_mm_castpd_si128(_mm_mul_pd(a_pd, b_pd)))
                    } else if is_type::<T, u32>() {
                        // Shift right by 32 bits to move odd lanes (1, 3) to even positions (0, 2),
                        // then use _mm_mul_epu32.
                        let a_odd = _mm_srli_epi64(a.raw, 32);
                        let b_odd = _mm_srli_epi64(b.raw, 32);
                        V128::from_raw(_mm_mul_epu32(a_odd, b_odd))
                    } else {
                        // i32 -> i64: scalar fallback
                        let mut arr_a: Aligned<A16, [i32; 4]> = Aligned::new([0i32; 4]);
                        let mut arr_b: Aligned<A16, [i32; 4]> = Aligned::new([0i32; 4]);
                        _mm_store_si128(arr_a.as_mut_ptr().cast(), a.raw);
                        _mm_store_si128(arr_b.as_mut_ptr().cast(), b.raw);
                        let r0 = arr_a[1] as i64 * arr_b[1] as i64;
                        let r1 = arr_a[3] as i64 * arr_b[3] as i64;
                        let out: Aligned<A16, [i64; 2]> = Aligned::new([r0, r1]);
                        V128::from_raw(_mm_load_si128(out.as_ptr().cast()))
                    }
                }
                2 => {
                    // u16/i16 -> u32/i32: extract odd 16-bit lanes, widen to 32-bit, multiply
                    let (a32, b32) = if is_signed::<T>() {
                        // Arithmetic shift right 16 sign-extends odd i16 into i32
                        (_mm_srai_epi32(a.raw, 16), _mm_srai_epi32(b.raw, 16))
                    } else {
                        // Logical shift right 16 zero-extends odd u16 into u32
                        (_mm_srli_epi32(a.raw, 16), _mm_srli_epi32(b.raw, 16))
                    };
                    // SSE2 lacks mullo_epi32; use two mul_epu32 calls + pack
                    let prod_02 = _mm_mul_epu32(a32, b32);
                    let a32_hi = _mm_srli_epi64(a32, 32);
                    let b32_hi = _mm_srli_epi64(b32, 32);
                    let prod_13 = _mm_mul_epu32(a32_hi, b32_hi);
                    let p02 = _mm_shuffle_epi32(prod_02, 0x08);
                    let p13 = _mm_shuffle_epi32(prod_13, 0x08);
                    V128::from_raw(_mm_unpacklo_epi32(p02, p13))
                }
                1 => {
                    // u8/i8 -> u16/i16: reinterpret as u16, extract odd bytes, multiply.
                    if is_signed::<T>() {
                        // i8 -> i16: arithmetic shift right 8 sign-extends odd bytes
                        let a16 = _mm_srai_epi16(a.raw, 8);
                        let b16 = _mm_srai_epi16(b.raw, 8);
                        V128::from_raw(_mm_mullo_epi16(a16, b16))
                    } else {
                        // u8 -> u16: logical shift right 8 zero-extends odd bytes
                        let a16 = _mm_srli_epi16(a.raw, 8);
                        let b16 = _mm_srli_epi16(b.raw, 8);
                        V128::from_raw(_mm_mullo_epi16(a16, b16))
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn widen_mul_pairwise_add_i16(
        self,
        a: V128<i16>,
        b: V128<i16>,
    ) -> V128<i32> {
        V128::from_raw(unsafe { _mm_madd_epi16(a.raw, b.raw) })
    }

    #[inline(always)]
    fn sat_widen_mul_pairwise_add(
        self,
        a: V128<u8>,
        b: V128<i8>,
    ) -> V128<i16> {
        unsafe {
            if is_x86_feature_detected!("ssse3") {
                V128::from_raw(_mm_maddubs_epi16(a.raw, b.raw))
            } else {
                // Scalar fallback: pairs of u8*i8 summed with saturation to i16
                let mut arr_a = [0u8; 16];
                let mut arr_b = [0i8; 16];
                _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
                _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
                let mut result = [0i16; 8];
                for i in 0..8 {
                    let prod0 = (arr_a[2 * i] as i16) * (arr_b[2 * i] as i16);
                    let prod1 = (arr_a[2 * i + 1] as i16) * (arr_b[2 * i + 1] as i16);
                    let sum = (prod0 as i32) + (prod1 as i32);
                    result[i] = sum.clamp(-32768, 32767) as i16;
                }
                V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn mul_fixed_point_15(
        self,
        a: V128<i16>,
        b: V128<i16>,
    ) -> V128<i16> {
        unsafe {
            if is_x86_feature_detected!("ssse3") {
                V128::from_raw(_mm_mulhrs_epi16(a.raw, b.raw))
            } else {
                // Emulate: ((a*b) + (1<<14)) >> 15
                let mut arr_a = [0i16; 8];
                let mut arr_b = [0i16; 8];
                _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
                _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
                let mut result = [0i16; 8];
                for i in 0..8 {
                    let prod = (arr_a[i] as i32) * (arr_b[i] as i32);
                    result[i] = ((prod + 16384) >> 15) as i16;
                }
                V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn reorder_widen_mul_accumulate(
        self,
        a: V128<i16>,
        b: V128<i16>,
        sum: V128<i32>,
    ) -> V128<i32> {
        V128::from_raw(unsafe { _mm_add_epi32(sum.raw, _mm_madd_epi16(a.raw, b.raw)) })
    }

    #[inline(always)]
    fn saturated_neg<T: IntegerLane>(self, v: V128<T>) -> V128<T> {
        self.saturated_sub(self.zero::<T>(), v)
    }

    #[inline(always)]
    fn saturated_abs<T: IntegerLane>(self, v: V128<T>) -> V128<T> {
        self.max(v, self.saturated_neg(v))
    }

    #[inline(always)]
    fn masked_min_or<T: Lane>(self, no: V128<T>, mask: M128<T>, a: V128<T>, b: V128<T>) -> V128<T> {
        self.if_then_else(mask, self.min(a, b), no)
    }

    #[inline(always)]
    fn masked_max_or<T: Lane>(self, no: V128<T>, mask: M128<T>, a: V128<T>, b: V128<T>) -> V128<T> {
        self.if_then_else(mask, self.max(a, b), no)
    }

    #[inline(always)]
    fn masked_add_or<T: Lane>(self, no: V128<T>, mask: M128<T>, a: V128<T>, b: V128<T>) -> V128<T> {
        self.if_then_else(mask, self.add(a, b), no)
    }

    #[inline(always)]
    fn masked_sub_or<T: Lane>(self, no: V128<T>, mask: M128<T>, a: V128<T>, b: V128<T>) -> V128<T> {
        self.if_then_else(mask, self.sub(a, b), no)
    }

    #[inline(always)]
    fn masked_mul_or<T: Lane>(self, no: V128<T>, mask: M128<T>, a: V128<T>, b: V128<T>) -> V128<T> {
        self.if_then_else(mask, self.mul(a, b), no)
    }
}

// ---------------------------------------------------------------------------
// SimdBitwise
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require SSE2.
unsafe impl SimdBitwise for Sse2 {
    #[inline(always)]
    fn and<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        V128::from_raw(unsafe { _mm_and_si128(a.raw, b.raw) })
    }

    #[inline(always)]
    fn or<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        V128::from_raw(unsafe { _mm_or_si128(a.raw, b.raw) })
    }

    #[inline(always)]
    fn xor<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        V128::from_raw(unsafe { _mm_xor_si128(a.raw, b.raw) })
    }

    #[inline(always)]
    fn not<T: Lane>(self, v: V128<T>) -> V128<T> {
        let all_ones = unsafe { _mm_set1_epi8(!0) };
        V128::from_raw(unsafe { _mm_xor_si128(v.raw, all_ones) })
    }

    #[inline(always)]
    fn and_not<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // _mm_andnot_si128 computes ~a & b
        V128::from_raw(unsafe { _mm_andnot_si128(a.raw, b.raw) })
    }

    #[inline(always)]
    fn shift_left<T: IntegerLane, const BITS: u32>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(BITS as i64);
            let raw = match T::BYTES {
                2 => _mm_sll_epi16(v.raw, count),
                4 => _mm_sll_epi32(v.raw, count),
                8 => _mm_sll_epi64(v.raw, count),
                // 1-byte shift: emulate with 16-bit shift + mask
                1 => {
                    let shifted = _mm_sll_epi16(v.raw, count);
                    let mask = _mm_set1_epi8((0xFFu8.wrapping_shl(BITS)) as i8);
                    _mm_and_si128(shifted, mask)
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_right<T: IntegerLane, const BITS: u32>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(BITS as i64);
            let raw = match T::BYTES {
                2 => {
                    if is_signed::<T>() {
                        _mm_sra_epi16(v.raw, count)
                    } else {
                        _mm_srl_epi16(v.raw, count)
                    }
                }
                4 => {
                    if is_signed::<T>() {
                        _mm_sra_epi32(v.raw, count)
                    } else {
                        _mm_srl_epi32(v.raw, count)
                    }
                }
                8 => {
                    if is_signed::<T>() {
                        // No _mm_sra_epi64 in SSE2; emulate
                        let logical = _mm_srl_epi64(v.raw, count);
                        let sign = _mm_srai_epi32(v.raw, 31);
                        let sign = _mm_shuffle_epi32(sign, 0xF5); // broadcast high dword
                        let left_count = _mm_cvtsi64_si128((64 - BITS) as i64);
                        let sign_ext = _mm_sll_epi64(sign, left_count);
                        _mm_or_si128(logical, sign_ext)
                    } else {
                        _mm_srl_epi64(v.raw, count)
                    }
                }
                1 => {
                    if is_signed::<T>() {
                        // Emulate arithmetic shift right for i8
                        let count8 = _mm_cvtsi64_si128(8i64);
                        let count_plus_8 = _mm_cvtsi64_si128((BITS + 8) as i64);
                        let shifted = _mm_sra_epi16(_mm_sll_epi16(v.raw, count8), count_plus_8);
                        let mask = _mm_set1_epi16(0x00FF);
                        let lo = _mm_and_si128(shifted, mask);
                        let hi_shifted = _mm_sra_epi16(v.raw, count);
                        let hi = _mm_andnot_si128(mask, hi_shifted);
                        _mm_or_si128(lo, hi)
                    } else {
                        let shifted = _mm_srl_epi16(v.raw, count);
                        let mask = _mm_set1_epi8((0xFFu8.wrapping_shr(BITS)) as i8);
                        _mm_and_si128(shifted, mask)
                    }
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn rotate_right<T: IntegerLane, const BITS: u32>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let type_bits = (T::BYTES * 8) as u32;
            let right = BITS % type_bits;
            if right == 0 {
                return v;
            }
            let left = type_bits - right;
            let count_r = _mm_cvtsi64_si128(right as i64);
            let count_l = _mm_cvtsi64_si128(left as i64);
            // Always use logical (unsigned) shifts for rotation
            let raw = match T::BYTES {
                1 => {
                    let shr = _mm_and_si128(
                        _mm_srl_epi16(v.raw, count_r),
                        _mm_set1_epi8((0xFFu8.wrapping_shr(right)) as i8),
                    );
                    let shl = _mm_and_si128(
                        _mm_sll_epi16(v.raw, count_l),
                        _mm_set1_epi8((0xFFu8.wrapping_shl(left)) as i8),
                    );
                    _mm_or_si128(shr, shl)
                }
                2 => {
                    let shr = _mm_srl_epi16(v.raw, count_r);
                    let shl = _mm_sll_epi16(v.raw, count_l);
                    _mm_or_si128(shr, shl)
                }
                4 => {
                    let shr = _mm_srl_epi32(v.raw, count_r);
                    let shl = _mm_sll_epi32(v.raw, count_l);
                    _mm_or_si128(shr, shl)
                }
                8 => {
                    let shr = _mm_srl_epi64(v.raw, count_r);
                    let shl = _mm_sll_epi64(v.raw, count_l);
                    _mm_or_si128(shr, shl)
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_left_same<T: IntegerLane>(self, v: V128<T>, bits: u32) -> V128<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(bits as i64);
            let raw = match T::BYTES {
                2 => _mm_sll_epi16(v.raw, count),
                4 => _mm_sll_epi32(v.raw, count),
                8 => _mm_sll_epi64(v.raw, count),
                1 => {
                    let shifted = _mm_sll_epi16(v.raw, count);
                    let mask = _mm_set1_epi8((0xFFu8.wrapping_shl(bits)) as i8);
                    _mm_and_si128(shifted, mask)
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_right_same<T: IntegerLane>(self, v: V128<T>, bits: u32) -> V128<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(bits as i64);
            let raw = match T::BYTES {
                2 => {
                    if is_signed::<T>() {
                        _mm_sra_epi16(v.raw, count)
                    } else {
                        _mm_srl_epi16(v.raw, count)
                    }
                }
                4 => {
                    if is_signed::<T>() {
                        _mm_sra_epi32(v.raw, count)
                    } else {
                        _mm_srl_epi32(v.raw, count)
                    }
                }
                8 => {
                    if is_signed::<T>() {
                        // No _mm_sra_epi64 in SSE2; emulate
                        let logical = _mm_srl_epi64(v.raw, count);
                        let sign = _mm_srai_epi32(v.raw, 31);
                        let sign64 = _mm_shuffle_epi32(sign, 0xF5);
                        let left_count = _mm_cvtsi64_si128(64i64 - bits as i64);
                        let sign_ext = _mm_sll_epi64(sign64, left_count);
                        _mm_or_si128(logical, sign_ext)
                    } else {
                        _mm_srl_epi64(v.raw, count)
                    }
                }
                1 => {
                    if is_signed::<T>() {
                        // Emulate arithmetic shift right for i8
                        let count8 = _mm_cvtsi64_si128(8i64);
                        let count_plus_8 = _mm_cvtsi64_si128((bits + 8) as i64);
                        let shifted = _mm_sra_epi16(_mm_sll_epi16(v.raw, count8), count_plus_8);
                        let mask = _mm_set1_epi16(0x00FF);
                        let lo = _mm_and_si128(shifted, mask);
                        let hi_shifted = _mm_sra_epi16(v.raw, count);
                        let hi = _mm_andnot_si128(mask, hi_shifted);
                        _mm_or_si128(lo, hi)
                    } else {
                        let shifted = _mm_srl_epi16(v.raw, count);
                        let mask = _mm_set1_epi8((0xFFu8.wrapping_shr(bits)) as i8);
                        _mm_and_si128(shifted, mask)
                    }
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_left_bytes<T: Lane, const BYTES: usize>(self, v: V128<T>) -> V128<T> {
        // Shift the entire 128-bit register left by BYTES bytes, zero-filling from the right.
        // _mm_slli_si128 requires const i32, so we dispatch on BYTES value.
        unsafe {
            let raw = match BYTES {
                0 => v.raw,
                1 => _mm_slli_si128::<1>(v.raw),
                2 => _mm_slli_si128::<2>(v.raw),
                3 => _mm_slli_si128::<3>(v.raw),
                4 => _mm_slli_si128::<4>(v.raw),
                5 => _mm_slli_si128::<5>(v.raw),
                6 => _mm_slli_si128::<6>(v.raw),
                7 => _mm_slli_si128::<7>(v.raw),
                8 => _mm_slli_si128::<8>(v.raw),
                9 => _mm_slli_si128::<9>(v.raw),
                10 => _mm_slli_si128::<10>(v.raw),
                11 => _mm_slli_si128::<11>(v.raw),
                12 => _mm_slli_si128::<12>(v.raw),
                13 => _mm_slli_si128::<13>(v.raw),
                14 => _mm_slli_si128::<14>(v.raw),
                15 => _mm_slli_si128::<15>(v.raw),
                _ => _mm_setzero_si128(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_right_bytes<T: Lane, const BYTES: usize>(self, v: V128<T>) -> V128<T> {
        // Shift the entire 128-bit register right by BYTES bytes, zero-filling from the left.
        // _mm_srli_si128 requires const i32, so we dispatch on BYTES value.
        unsafe {
            let raw = match BYTES {
                0 => v.raw,
                1 => _mm_srli_si128::<1>(v.raw),
                2 => _mm_srli_si128::<2>(v.raw),
                3 => _mm_srli_si128::<3>(v.raw),
                4 => _mm_srli_si128::<4>(v.raw),
                5 => _mm_srli_si128::<5>(v.raw),
                6 => _mm_srli_si128::<6>(v.raw),
                7 => _mm_srli_si128::<7>(v.raw),
                8 => _mm_srli_si128::<8>(v.raw),
                9 => _mm_srli_si128::<9>(v.raw),
                10 => _mm_srli_si128::<10>(v.raw),
                11 => _mm_srli_si128::<11>(v.raw),
                12 => _mm_srli_si128::<12>(v.raw),
                13 => _mm_srli_si128::<13>(v.raw),
                14 => _mm_srli_si128::<14>(v.raw),
                15 => _mm_srli_si128::<15>(v.raw),
                _ => _mm_setzero_si128(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn population_count<T: IntegerLane>(self, v: V128<T>) -> V128<T> {
        // SSE2 bit-manipulation popcount (no pshufb needed).
        // See https://arxiv.org/pdf/1611.07612.pdf, Figure 3
        unsafe {
            // Step 1: u8 popcount using bit manipulation.
            // Uses 16-bit shifts since SSE2 has no 8-bit shifts;
            // the AND masks handle cross-byte contamination.
            let k55 = _mm_set1_epi8(0x55u8 as i8);
            let k33 = _mm_set1_epi8(0x33u8 as i8);
            let k0f = _mm_set1_epi8(0x0Fu8 as i8);

            let mut pop = v.raw;
            pop = _mm_sub_epi8(pop, _mm_and_si128(_mm_srli_epi16(pop, 1), k55));
            pop = _mm_add_epi8(
                _mm_and_si128(_mm_srli_epi16(pop, 2), k33),
                _mm_and_si128(pop, k33),
            );
            pop = _mm_and_si128(_mm_add_epi8(pop, _mm_srli_epi16(pop, 4)), k0f);

            // Step 2: reduce u8 popcount to wider lane types.
            if T::BYTES == 1 {
                // Already u8 popcount.
            } else if T::BYTES == 2 {
                // u16: sum adjacent u8 popcounts within each 16-bit lane.
                pop = _mm_add_epi16(
                    _mm_srli_epi16(pop, 8),
                    _mm_and_si128(pop, _mm_set1_epi16(0xFF)),
                );
            } else if T::BYTES == 4 {
                // u32: reduce u16 popcount, which is reduce of u8.
                // First reduce u8->u16
                let pop16 = _mm_add_epi16(
                    _mm_srli_epi16(pop, 8),
                    _mm_and_si128(pop, _mm_set1_epi16(0xFF)),
                );
                // Then reduce u16->u32
                pop = _mm_add_epi32(
                    _mm_srli_epi32(pop16, 16),
                    _mm_and_si128(pop16, _mm_set1_epi32(0xFF)),
                );
            } else {
                // u64: reduce u8->u16->u32->u64.
                let pop16 = _mm_add_epi16(
                    _mm_srli_epi16(pop, 8),
                    _mm_and_si128(pop, _mm_set1_epi16(0xFF)),
                );
                let pop32 = _mm_add_epi32(
                    _mm_srli_epi32(pop16, 16),
                    _mm_and_si128(pop16, _mm_set1_epi32(0xFF)),
                );
                pop = _mm_add_epi64(
                    _mm_srli_epi64(pop32, 32),
                    _mm_and_si128(pop32, _mm_set1_epi64x(0xFF)),
                );
            }

            V128::from_raw(pop)
        }
    }

    #[inline(always)]
    fn leading_zero_count<T: IntegerLane>(self, v: V128<T>) -> V128<T> {
        // SSE2 LZC via float conversion trick: convert to f32, extract biased
        // exponent to determine position of highest set bit.
        unsafe {
            if T::BYTES == 4 {
                V128::from_raw(lzc_u32(v.raw))
            } else if T::BYTES == 2 {
                // Promote each u16 to u32, compute lzc_u32, subtract 16, pack.
                let zero = _mm_setzero_si128();
                let lo = _mm_unpacklo_epi16(v.raw, zero);
                let hi = _mm_unpackhi_epi16(v.raw, zero);
                let lzc_lo = _mm_sub_epi32(lzc_u32(lo), _mm_set1_epi32(16));
                let lzc_hi = _mm_sub_epi32(lzc_u32(hi), _mm_set1_epi32(16));
                V128::from_raw(_mm_packs_epi32(lzc_lo, lzc_hi))
            } else if T::BYTES == 1 {
                // Promote each u8 to u32 (four groups), compute lzc, subtract 24,
                // then pack u32->i16->i8.
                let zero = _mm_setzero_si128();
                let lo16 = _mm_unpacklo_epi8(v.raw, zero);
                let hi16 = _mm_unpackhi_epi8(v.raw, zero);
                let v0 = _mm_unpacklo_epi16(lo16, zero);
                let v1 = _mm_unpackhi_epi16(lo16, zero);
                let v2 = _mm_unpacklo_epi16(hi16, zero);
                let v3 = _mm_unpackhi_epi16(hi16, zero);
                let adj = _mm_set1_epi32(24);
                let lzc0 = _mm_sub_epi32(lzc_u32(v0), adj);
                let lzc1 = _mm_sub_epi32(lzc_u32(v1), adj);
                let lzc2 = _mm_sub_epi32(lzc_u32(v2), adj);
                let lzc3 = _mm_sub_epi32(lzc_u32(v3), adj);
                let lo_i16 = _mm_packs_epi32(lzc0, lzc1);
                let hi_i16 = _mm_packs_epi32(lzc2, lzc3);
                V128::from_raw(_mm_packs_epi16(lo_i16, hi_i16))
            } else {
                // u64: process each 32-bit half, combine.
                // For each u64 lane: lzc = if hi != 0 { lzc_u32(hi) } else { 32 + lzc_u32(lo) }
                let zero = _mm_setzero_si128();
                // Reinterpret as u32 and compute lzc for all 4 lanes
                let lzc32 = lzc_u32(v.raw);
                // We need: for each 64-bit lane [lo32, hi32]:
                //   if hi32 != 0: result = lzc_u32(hi32)
                //   else:         result = 32 + lzc_u32(lo32)
                // lzc32 = [lzc(lo0), lzc(hi0), lzc(lo1), lzc(hi1)]
                let hi_zero =
                    _mm_cmpeq_epi32(_mm_and_si128(v.raw, _mm_set_epi32(-1, 0, -1, 0)), zero);
                // hi_zero has all-1s in the hi32 slot if hi32 was 0
                // We want: where hi32==0, lzc_hi=32 (so total = 32 + lzc_lo)
                //          where hi32!=0, lzc_hi stays as is
                // Adjusted: add 32 to lo32 lzc
                let k32 = _mm_set1_epi32(32);
                let lzc_lo_adj = _mm_add_epi32(lzc32, k32);
                // For each 64-bit lane, pick hi32's lzc if hi32!=0, else lo32's adjusted lzc
                // Create mask: hi_is_zero per 64-bit lane
                // Shuffle hi_zero to broadcast hi32's mask to lo32 position
                let hi_mask = _mm_shuffle_epi32(hi_zero, 0xF5); // [hi0,hi0,hi1,hi1]
                // Where hi was zero, take lo_adj; else take lzc32 from hi position
                // Build result: [lo_adj_or_hi_lzc, ?, lo_adj_or_hi_lzc, ?]
                // Actually, just extract per-lane:
                // result_lo = hi_mask ? lzc_lo_adj : lzc_hi (in lo slot)
                // But positions are interleaved. Let me just build directly:
                // Move hi32 lzc to lo32 position: shuffle [1,X,3,X]
                let hi_lzc = _mm_shuffle_epi32(lzc32, 0xF5); // [lzc_hi0, lzc_hi0, lzc_hi1, lzc_hi1]
                // Select: if hi was zero -> lzc_lo_adj, else -> hi_lzc
                let selected = _mm_or_si128(
                    _mm_and_si128(hi_mask, lzc_lo_adj),
                    _mm_andnot_si128(hi_mask, hi_lzc),
                );
                // Zero out the upper 32-bit halves of each 64-bit lane
                let result = _mm_and_si128(selected, _mm_set_epi32(0, -1, 0, -1));
                V128::from_raw(result)
            }
        }
    }

    #[inline(always)]
    fn trailing_zero_count<T: IntegerLane>(self, v: V128<T>) -> V128<T> {
        // tzcnt(v) = popcount((v - 1) & ~v)
        // This leverages the SIMD population_count we already have.
        unsafe {
            let v_minus_1 = match T::BYTES {
                1 => _mm_sub_epi8(v.raw, _mm_set1_epi8(1)),
                2 => _mm_sub_epi16(v.raw, _mm_set1_epi16(1)),
                4 => _mm_sub_epi32(v.raw, _mm_set1_epi32(1)),
                _ => _mm_sub_epi64(v.raw, _mm_set1_epi64x(1)),
            };
            // ~v & (v - 1): isolates all bits below the lowest set bit
            let masked = _mm_andnot_si128(v.raw, v_minus_1);
            self.population_count::<T>(V128::from_raw(masked))
        }
    }

    #[inline(always)]
    fn reverse_lane_bytes<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    // Single byte lanes: no-op.
                    v
                }
                2 => {
                    // Swap bytes within each 16-bit lane.
                    let hi = _mm_srli_epi16(v.raw, 8);
                    let lo = _mm_slli_epi16(v.raw, 8);
                    V128::from_raw(_mm_or_si128(hi, lo))
                }
                4 => {
                    // Byte-swap within each 32-bit lane using shifts + masks (bswap32).
                    // [b0, b1, b2, b3] -> [b3, b2, b1, b0]
                    let x = v.raw;
                    let a = _mm_srli_epi32(x, 24);
                    let b = _mm_and_si128(_mm_srli_epi32(x, 8), _mm_set1_epi32(0x0000FF00));
                    let c = _mm_and_si128(_mm_slli_epi32(x, 8), _mm_set1_epi32(0x00FF0000));
                    let d = _mm_slli_epi32(x, 24);
                    V128::from_raw(_mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d)))
                }
                _ => {
                    // 64-bit: bswap32 each dword, then swap dword halves within each 64-bit lane.
                    let x = v.raw;
                    let a = _mm_srli_epi32(x, 24);
                    let b = _mm_and_si128(_mm_srli_epi32(x, 8), _mm_set1_epi32(0x0000FF00));
                    let c = _mm_and_si128(_mm_slli_epi32(x, 8), _mm_set1_epi32(0x00FF0000));
                    let d = _mm_slli_epi32(x, 24);
                    let bswap32 = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
                    // Swap 32-bit halves within each 64-bit lane: [d0,d1,d2,d3] -> [d1,d0,d3,d2]
                    V128::from_raw(_mm_shuffle_epi32(bswap32, 0xB1))
                }
            }
        }
    }

    #[inline(always)]
    fn reverse_bits<T: IntegerLane>(self, v: V128<T>) -> V128<T> {
        // SSE2 bit-reversal: three swap steps for u8, then reverse lane bytes
        // for wider types. Uses 16-bit shifts since SSE2 has no 8-bit shifts.
        unsafe {
            let raw = v.raw;

            // Step 1: swap adjacent single bits
            // Or(And(shr(v,1), 0x55), AndNot(0x55, shl(v,1)))
            let k55 = _mm_set1_epi8(0x55u8 as i8);
            let shr1 = _mm_srli_epi16(raw, 1);
            let shl1 = _mm_slli_epi16(raw, 1);
            let step1 = _mm_or_si128(_mm_and_si128(shr1, k55), _mm_andnot_si128(k55, shl1));

            // Step 2: swap adjacent 2-bit groups
            let k33 = _mm_set1_epi8(0x33u8 as i8);
            let shr2 = _mm_srli_epi16(step1, 2);
            let shl2 = _mm_slli_epi16(step1, 2);
            let step2 = _mm_or_si128(_mm_and_si128(shr2, k33), _mm_andnot_si128(k33, shl2));

            // Step 3: swap nibbles
            let k0f = _mm_set1_epi8(0x0Fu8 as i8);
            let shr4 = _mm_srli_epi16(step2, 4);
            let shl4 = _mm_slli_epi16(step2, 4);
            let reversed_u8 = _mm_or_si128(_mm_and_si128(shr4, k0f), _mm_andnot_si128(k0f, shl4));

            if T::BYTES == 1 {
                // u8: bit-reversed bytes are the final result.
                V128::from_raw(reversed_u8)
            } else {
                // u16/u32/u64: reverse bits within each byte, then reverse
                // the byte order within each lane.
                self.reverse_lane_bytes::<T>(V128::from_raw(reversed_u8))
            }
        }
    }

    #[inline(always)]
    fn shl<T: IntegerLane>(
        self,
        v: V128<T>,
        bits: V128<T>,
    ) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr_v = [0u8; 16];
            let mut arr_b = [0u8; 16];
            _mm_storeu_si128(arr_v.as_mut_ptr().cast(), v.raw);
            _mm_storeu_si128(arr_b.as_mut_ptr().cast(), bits.raw);
            let mut result = [0u8; 16];
            for i in 0..lanes {
                let off = i * T::BYTES;
                match T::BYTES {
                    1 => {
                        let val = arr_v[off];
                        let shift = arr_b[off];
                        result[off] = if shift >= 8 { 0 } else { val << shift };
                    }
                    2 => {
                        let val = u16::from_le_bytes([arr_v[off], arr_v[off + 1]]);
                        let shift = u16::from_le_bytes([arr_b[off], arr_b[off + 1]]);
                        let r = if shift >= 16 { 0 } else { val << shift };
                        let bytes = r.to_le_bytes();
                        result[off] = bytes[0];
                        result[off + 1] = bytes[1];
                    }
                    4 => {
                        let val = u32::from_le_bytes([
                            arr_v[off], arr_v[off + 1], arr_v[off + 2], arr_v[off + 3],
                        ]);
                        let shift = u32::from_le_bytes([
                            arr_b[off], arr_b[off + 1], arr_b[off + 2], arr_b[off + 3],
                        ]);
                        let r = if shift >= 32 { 0 } else { val << shift };
                        let bytes = r.to_le_bytes();
                        result[off..off + 4].copy_from_slice(&bytes);
                    }
                    8 => {
                        let val = u64::from_le_bytes([
                            arr_v[off], arr_v[off + 1], arr_v[off + 2], arr_v[off + 3],
                            arr_v[off + 4], arr_v[off + 5], arr_v[off + 6], arr_v[off + 7],
                        ]);
                        let shift = u64::from_le_bytes([
                            arr_b[off], arr_b[off + 1], arr_b[off + 2], arr_b[off + 3],
                            arr_b[off + 4], arr_b[off + 5], arr_b[off + 6], arr_b[off + 7],
                        ]);
                        let r = if shift >= 64 { 0 } else { val << shift };
                        let bytes = r.to_le_bytes();
                        result[off..off + 8].copy_from_slice(&bytes);
                    }
                    _ => unreachable!(),
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn shr<T: IntegerLane>(
        self,
        v: V128<T>,
        bits: V128<T>,
    ) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr_v = [0u8; 16];
            let mut arr_b = [0u8; 16];
            _mm_storeu_si128(arr_v.as_mut_ptr().cast(), v.raw);
            _mm_storeu_si128(arr_b.as_mut_ptr().cast(), bits.raw);
            let mut result = [0u8; 16];
            for i in 0..lanes {
                let off = i * T::BYTES;
                match T::BYTES {
                    1 => {
                        let shift = arr_b[off];
                        if is_signed::<T>() {
                            let val = arr_v[off] as i8;
                            let r = if shift >= 8 { val >> 7 } else { val >> shift };
                            result[off] = r as u8;
                        } else {
                            let val = arr_v[off];
                            result[off] = if shift >= 8 { 0 } else { val >> shift };
                        }
                    }
                    2 => {
                        let val_bytes = [arr_v[off], arr_v[off + 1]];
                        let shift = u16::from_le_bytes([arr_b[off], arr_b[off + 1]]);
                        let r = if is_signed::<T>() {
                            let val = i16::from_le_bytes(val_bytes);
                            let r = if shift >= 16 { val >> 15 } else { val >> shift };
                            (r as u16).to_le_bytes()
                        } else {
                            let val = u16::from_le_bytes(val_bytes);
                            let r = if shift >= 16 { 0 } else { val >> shift };
                            r.to_le_bytes()
                        };
                        result[off] = r[0];
                        result[off + 1] = r[1];
                    }
                    4 => {
                        let val_bytes = [
                            arr_v[off], arr_v[off + 1], arr_v[off + 2], arr_v[off + 3],
                        ];
                        let shift = u32::from_le_bytes([
                            arr_b[off], arr_b[off + 1], arr_b[off + 2], arr_b[off + 3],
                        ]);
                        let r = if is_signed::<T>() {
                            let val = i32::from_le_bytes(val_bytes);
                            let r = if shift >= 32 { val >> 31 } else { val >> shift };
                            (r as u32).to_le_bytes()
                        } else {
                            let val = u32::from_le_bytes(val_bytes);
                            let r = if shift >= 32 { 0 } else { val >> shift };
                            r.to_le_bytes()
                        };
                        result[off..off + 4].copy_from_slice(&r);
                    }
                    8 => {
                        let val_bytes = [
                            arr_v[off], arr_v[off + 1], arr_v[off + 2], arr_v[off + 3],
                            arr_v[off + 4], arr_v[off + 5], arr_v[off + 6], arr_v[off + 7],
                        ];
                        let shift = u64::from_le_bytes([
                            arr_b[off], arr_b[off + 1], arr_b[off + 2], arr_b[off + 3],
                            arr_b[off + 4], arr_b[off + 5], arr_b[off + 6], arr_b[off + 7],
                        ]);
                        let r = if is_signed::<T>() {
                            let val = i64::from_le_bytes(val_bytes);
                            let r = if shift >= 64 { val >> 63 } else { val >> shift };
                            (r as u64).to_le_bytes()
                        } else {
                            let val = u64::from_le_bytes(val_bytes);
                            let r = if shift >= 64 { 0 } else { val >> shift };
                            r.to_le_bytes()
                        };
                        result[off..off + 8].copy_from_slice(&r);
                    }
                    _ => unreachable!(),
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn ror<T: IntegerLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut result = [0u8; 16];
            let mut arr_a = [0u8; 16];
            let mut arr_b = [0u8; 16];
            _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
            _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
            for i in 0..lanes {
                let offset = i * T::BYTES;
                match T::BYTES {
                    1 => {
                        let va = arr_a[offset];
                        let vb = arr_b[offset] & 7;
                        let r = va.rotate_right(vb as u32);
                        result[offset] = r;
                    }
                    2 => {
                        let va = u16::from_le_bytes([arr_a[offset], arr_a[offset + 1]]);
                        let vb = u16::from_le_bytes([arr_b[offset], arr_b[offset + 1]]) & 15;
                        let r = va.rotate_right(vb as u32);
                        let rb = r.to_le_bytes();
                        result[offset] = rb[0];
                        result[offset + 1] = rb[1];
                    }
                    4 => {
                        let va = u32::from_le_bytes([arr_a[offset], arr_a[offset+1], arr_a[offset+2], arr_a[offset+3]]);
                        let vb = u32::from_le_bytes([arr_b[offset], arr_b[offset+1], arr_b[offset+2], arr_b[offset+3]]) & 31;
                        let r = va.rotate_right(vb);
                        let rb = r.to_le_bytes();
                        result[offset..offset+4].copy_from_slice(&rb);
                    }
                    8 => {
                        let va = u64::from_le_bytes([arr_a[offset], arr_a[offset+1], arr_a[offset+2], arr_a[offset+3], arr_a[offset+4], arr_a[offset+5], arr_a[offset+6], arr_a[offset+7]]);
                        let vb = u64::from_le_bytes([arr_b[offset], arr_b[offset+1], arr_b[offset+2], arr_b[offset+3], arr_b[offset+4], arr_b[offset+5], arr_b[offset+6], arr_b[offset+7]]) & 63;
                        let r = va.rotate_right(vb as u32);
                        let rb = r.to_le_bytes();
                        result[offset..offset+8].copy_from_slice(&rb);
                    }
                    _ => unreachable!(),
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn rol<T: IntegerLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut result = [0u8; 16];
            let mut arr_a = [0u8; 16];
            let mut arr_b = [0u8; 16];
            _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
            _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
            for i in 0..lanes {
                let offset = i * T::BYTES;
                match T::BYTES {
                    1 => {
                        let va = arr_a[offset];
                        let vb = arr_b[offset] & 7;
                        result[offset] = va.rotate_left(vb as u32);
                    }
                    2 => {
                        let va = u16::from_le_bytes([arr_a[offset], arr_a[offset + 1]]);
                        let vb = u16::from_le_bytes([arr_b[offset], arr_b[offset + 1]]) & 15;
                        let rb = va.rotate_left(vb as u32).to_le_bytes();
                        result[offset] = rb[0];
                        result[offset + 1] = rb[1];
                    }
                    4 => {
                        let va = u32::from_le_bytes([arr_a[offset], arr_a[offset+1], arr_a[offset+2], arr_a[offset+3]]);
                        let vb = u32::from_le_bytes([arr_b[offset], arr_b[offset+1], arr_b[offset+2], arr_b[offset+3]]) & 31;
                        result[offset..offset+4].copy_from_slice(&va.rotate_left(vb).to_le_bytes());
                    }
                    8 => {
                        let va = u64::from_le_bytes([arr_a[offset], arr_a[offset+1], arr_a[offset+2], arr_a[offset+3], arr_a[offset+4], arr_a[offset+5], arr_a[offset+6], arr_a[offset+7]]);
                        let vb = u64::from_le_bytes([arr_b[offset], arr_b[offset+1], arr_b[offset+2], arr_b[offset+3], arr_b[offset+4], arr_b[offset+5], arr_b[offset+6], arr_b[offset+7]]) & 63;
                        result[offset..offset+8].copy_from_slice(&va.rotate_left(vb as u32).to_le_bytes());
                    }
                    _ => unreachable!(),
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn rotate_left<T: IntegerLane, const BITS: u32>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut result = [0u8; 16];
            let mut arr = [0u8; 16];
            _mm_storeu_si128(arr.as_mut_ptr().cast(), v.raw);
            for i in 0..lanes {
                let offset = i * T::BYTES;
                match T::BYTES {
                    1 => result[offset] = arr[offset].rotate_left(BITS),
                    2 => {
                        let va = u16::from_le_bytes([arr[offset], arr[offset + 1]]);
                        let rb = va.rotate_left(BITS).to_le_bytes();
                        result[offset] = rb[0];
                        result[offset + 1] = rb[1];
                    }
                    4 => {
                        let va = u32::from_le_bytes([arr[offset], arr[offset+1], arr[offset+2], arr[offset+3]]);
                        result[offset..offset+4].copy_from_slice(&va.rotate_left(BITS).to_le_bytes());
                    }
                    8 => {
                        let va = u64::from_le_bytes([arr[offset], arr[offset+1], arr[offset+2], arr[offset+3], arr[offset+4], arr[offset+5], arr[offset+6], arr[offset+7]]);
                        result[offset..offset+8].copy_from_slice(&va.rotate_left(BITS).to_le_bytes());
                    }
                    _ => unreachable!(),
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn broadcast_sign_bit<T: IntegerLane>(self, v: V128<T>) -> V128<T> {
        // All-ones if the MSB (sign bit) is set, else all-zeros. Matches C++ Highway.
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_cmpgt_epi8(_mm_setzero_si128(), v.raw),
                2 => _mm_srai_epi16(v.raw, 15),
                4 => _mm_srai_epi32(v.raw, 31),
                // i64: SSE2 has no _mm_srai_epi64; broadcast the high dword's sign.
                _ => {
                    let sign = _mm_srai_epi32(v.raw, 31);
                    _mm_shuffle_epi32(sign, 0xF5) // _MM_SHUFFLE(3, 3, 1, 1)
                }
            };
            V128::from_raw(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdCompare
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require SSE2.
unsafe impl SimdCompare for Sse2 {
    #[inline(always)]
    fn eq<T: Lane>(self, a: V128<T>, b: V128<T>) -> M128<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm_castps_si128(_mm_cmpeq_ps(
                    _mm_castsi128_ps(a.raw),
                    _mm_castsi128_ps(b.raw),
                ))
            } else if is_type::<T, f64>() {
                _mm_castpd_si128(_mm_cmpeq_pd(
                    _mm_castsi128_pd(a.raw),
                    _mm_castsi128_pd(b.raw),
                ))
            } else {
                match T::BYTES {
                    1 => _mm_cmpeq_epi8(a.raw, b.raw),
                    2 => _mm_cmpeq_epi16(a.raw, b.raw),
                    4 => _mm_cmpeq_epi32(a.raw, b.raw),
                    8 => {
                        // No _mm_cmpeq_epi64 in SSE2; emulate
                        let eq32 = _mm_cmpeq_epi32(a.raw, b.raw);
                        let eq32_hi = _mm_shuffle_epi32(eq32, 0xB1);
                        _mm_and_si128(eq32, eq32_hi)
                    }
                    _ => unreachable!(),
                }
            };
            M128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn ne<T: Lane>(self, a: V128<T>, b: V128<T>) -> M128<T> {
        unsafe {
            let eq = self.eq::<T>(a, b);
            let all_ones = _mm_set1_epi8(!0);
            M128::from_raw(_mm_xor_si128(eq.raw, all_ones))
        }
    }

    #[inline(always)]
    fn lt<T: Lane>(self, a: V128<T>, b: V128<T>) -> M128<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm_castps_si128(_mm_cmplt_ps(
                    _mm_castsi128_ps(a.raw),
                    _mm_castsi128_ps(b.raw),
                ))
            } else if is_type::<T, f64>() {
                _mm_castpd_si128(_mm_cmplt_pd(
                    _mm_castsi128_pd(a.raw),
                    _mm_castsi128_pd(b.raw),
                ))
            } else if is_signed::<T>() {
                match T::BYTES {
                    1 => _mm_cmplt_epi8(a.raw, b.raw),
                    2 => _mm_cmplt_epi16(a.raw, b.raw),
                    4 => _mm_cmplt_epi32(a.raw, b.raw),
                    8 => {
                        // Emulate 64-bit signed compare
                        // a < b iff (a - b) has sign bit set AND a != b
                        let sub = _mm_sub_epi64(a.raw, b.raw);
                        // Check overflow: if signs differ, use sign of a
                        let xor_ab = _mm_xor_si128(a.raw, b.raw);
                        let diff_signs = _mm_srai_epi32(xor_ab, 31);
                        let diff_signs = _mm_shuffle_epi32(diff_signs, 0xF5);
                        let sub_sign = _mm_srai_epi32(sub, 31);
                        let sub_sign = _mm_shuffle_epi32(sub_sign, 0xF5);
                        let a_sign = _mm_srai_epi32(a.raw, 31);
                        let a_sign = _mm_shuffle_epi32(a_sign, 0xF5);
                        // If signs differ: a_sign; else: sub_sign
                        _mm_or_si128(
                            _mm_and_si128(diff_signs, a_sign),
                            _mm_andnot_si128(diff_signs, sub_sign),
                        )
                    }
                    _ => unreachable!(),
                }
            } else {
                // Unsigned comparison: flip sign bits, then use signed compare
                let sign_flip = match T::BYTES {
                    1 => _mm_set1_epi8(i8::MIN),
                    2 => _mm_set1_epi16(i16::MIN),
                    4 => _mm_set1_epi32(i32::MIN),
                    8 => _mm_set1_epi64x(i64::MIN),
                    _ => unreachable!(),
                };
                let a_flipped = _mm_xor_si128(a.raw, sign_flip);
                let b_flipped = _mm_xor_si128(b.raw, sign_flip);
                match T::BYTES {
                    1 => _mm_cmplt_epi8(a_flipped, b_flipped),
                    2 => _mm_cmplt_epi16(a_flipped, b_flipped),
                    4 => _mm_cmplt_epi32(a_flipped, b_flipped),
                    8 => {
                        let sub = _mm_sub_epi64(a_flipped, b_flipped);
                        let xor_ab = _mm_xor_si128(a_flipped, b_flipped);
                        let diff_signs = _mm_srai_epi32(xor_ab, 31);
                        let diff_signs = _mm_shuffle_epi32(diff_signs, 0xF5);
                        let sub_sign = _mm_srai_epi32(sub, 31);
                        let sub_sign = _mm_shuffle_epi32(sub_sign, 0xF5);
                        let a_sign = _mm_srai_epi32(a_flipped, 31);
                        let a_sign = _mm_shuffle_epi32(a_sign, 0xF5);
                        _mm_or_si128(
                            _mm_and_si128(diff_signs, a_sign),
                            _mm_andnot_si128(diff_signs, sub_sign),
                        )
                    }
                    _ => unreachable!(),
                }
            };
            M128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn le<T: Lane>(self, a: V128<T>, b: V128<T>) -> M128<T> {
        unsafe {
            if is_type::<T, f32>() {
                M128::from_raw(_mm_castps_si128(_mm_cmple_ps(
                    _mm_castsi128_ps(a.raw),
                    _mm_castsi128_ps(b.raw),
                )))
            } else if is_type::<T, f64>() {
                M128::from_raw(_mm_castpd_si128(_mm_cmple_pd(
                    _mm_castsi128_pd(a.raw),
                    _mm_castsi128_pd(b.raw),
                )))
            } else {
                // le = eq | lt
                let eq_m = self.eq::<T>(a, b);
                let lt_m = self.lt::<T>(a, b);
                M128::from_raw(_mm_or_si128(eq_m.raw, lt_m.raw))
            }
        }
    }

    #[inline(always)]
    fn gt<T: Lane>(self, a: V128<T>, b: V128<T>) -> M128<T> {
        // gt(a, b) = lt(b, a)
        self.lt(b, a)
    }

    #[inline(always)]
    fn ge<T: Lane>(self, a: V128<T>, b: V128<T>) -> M128<T> {
        // ge(a, b) = le(b, a)
        self.le(b, a)
    }

    #[inline(always)]
    fn test_bit<T: IntegerLane>(self, v: V128<T>, bit: V128<T>) -> M128<T> {
        unsafe {
            let anded = _mm_and_si128(v.raw, bit.raw);
            let zero = _mm_setzero_si128();
            // test_bit: (v & bit) != 0
            let eq_zero = match T::BYTES {
                1 => _mm_cmpeq_epi8(anded, zero),
                2 => _mm_cmpeq_epi16(anded, zero),
                4 => _mm_cmpeq_epi32(anded, zero),
                8 => {
                    let eq32 = _mm_cmpeq_epi32(anded, zero);
                    let eq32_hi = _mm_shuffle_epi32(eq32, 0xB1);
                    _mm_and_si128(eq32, eq32_hi)
                }
                _ => unreachable!(),
            };
            // Invert: not equal to zero = bit is set
            M128::from_raw(_mm_xor_si128(eq_zero, _mm_set1_epi8(!0)))
        }
    }
}

// ---------------------------------------------------------------------------
// SimdMask
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require SSE2.
unsafe impl SimdMask for Sse2 {
    #[inline(always)]
    fn mask_from_vec<T: Lane>(self, v: V128<T>) -> M128<T> {
        unsafe {
            // Non-zero -> all-ones mask per lane
            let zero = _mm_setzero_si128();
            let is_zero = match T::BYTES {
                1 => _mm_cmpeq_epi8(v.raw, zero),
                2 => _mm_cmpeq_epi16(v.raw, zero),
                4 => _mm_cmpeq_epi32(v.raw, zero),
                8 => {
                    let eq32 = _mm_cmpeq_epi32(v.raw, zero);
                    let eq32_hi = _mm_shuffle_epi32(eq32, 0xB1);
                    _mm_and_si128(eq32, eq32_hi)
                }
                _ => unreachable!(),
            };
            // Invert
            M128::from_raw(_mm_xor_si128(is_zero, _mm_set1_epi8(!0)))
        }
    }

    #[inline(always)]
    fn vec_from_mask<T: Lane>(self, m: M128<T>) -> V128<T> {
        V128::from_raw(m.raw)
    }

    #[inline(always)]
    fn first_n<T: Lane>(self, n: usize) -> M128<T> {
        unsafe {
            // Create iota [0, 1, 2, ..., lanes-1] and compare < n.
            // Since all values are small positive integers, signed comparison works.
            match T::BYTES {
                1 => {
                    let iota = _mm_set_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
                    let threshold = _mm_set1_epi8(n as i8);
                    // iota < threshold <-> cmpgt(threshold, iota)
                    M128::from_raw(_mm_cmpgt_epi8(threshold, iota))
                }
                2 => {
                    let iota = _mm_set_epi16(7, 6, 5, 4, 3, 2, 1, 0);
                    let threshold = _mm_set1_epi16(n as i16);
                    M128::from_raw(_mm_cmpgt_epi16(threshold, iota))
                }
                4 => {
                    let iota = _mm_set_epi32(3, 2, 1, 0);
                    let threshold = _mm_set1_epi32(n as i32);
                    M128::from_raw(_mm_cmpgt_epi32(threshold, iota))
                }
                8 => {
                    // SSE2 lacks _mm_cmpgt_epi64; use manual approach for 2 lanes
                    let mask: i64 = if n >= 2 { !0 } else { 0 };
                    let mask0: i64 = if n >= 1 { !0 } else { 0 };
                    M128::from_raw(_mm_set_epi64x(mask, mask0))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn count_true<T: Lane>(self, m: M128<T>) -> usize {
        unsafe {
            // Count lanes with all bits set. Use movemask.
            match T::BYTES {
                1 => _mm_movemask_epi8(m.raw).count_ones() as usize,
                2 => (_mm_movemask_epi8(m.raw).count_ones() as usize) / 2,
                4 => _mm_movemask_ps(_mm_castsi128_ps(m.raw)).count_ones() as usize,
                8 => _mm_movemask_pd(_mm_castsi128_pd(m.raw)).count_ones() as usize,
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn all_true<T: Lane>(self, m: M128<T>) -> bool {
        unsafe { _mm_movemask_epi8(m.raw) == 0xFFFF }
    }

    #[inline(always)]
    fn all_false<T: Lane>(self, m: M128<T>) -> bool {
        unsafe { _mm_movemask_epi8(m.raw) == 0 }
    }

    #[inline(always)]
    fn find_first_true<T: Lane>(self, m: M128<T>) -> Option<usize> {
        unsafe {
            let bits = match T::BYTES {
                1 => _mm_movemask_epi8(m.raw) as u32,
                2 => {
                    let b = _mm_movemask_epi8(m.raw) as u32;
                    // Each lane is 2 bytes, so bit pairs
                    b & 0x5555 // keep only even bits? No, compress to lane index
                }
                4 => _mm_movemask_ps(_mm_castsi128_ps(m.raw)) as u32,
                8 => _mm_movemask_pd(_mm_castsi128_pd(m.raw)) as u32,
                _ => unreachable!(),
            };
            if T::BYTES == 2 {
                // For 16-bit lanes, movemask_epi8 gives 2 bits per lane
                let byte_mask = _mm_movemask_epi8(m.raw) as u32;
                if byte_mask == 0 {
                    return None;
                }
                // First set bit / 2 = lane index
                Some((byte_mask.trailing_zeros() / 2) as usize)
            } else if bits == 0 {
                None
            } else {
                Some(bits.trailing_zeros() as usize)
            }
        }
    }

    #[inline(always)]
    fn if_then_else<T: Lane>(self, mask: M128<T>, yes: V128<T>, no: V128<T>) -> V128<T> {
        unsafe {
            // (mask & yes) | (~mask & no)
            V128::from_raw(_mm_or_si128(
                _mm_and_si128(mask.raw, yes.raw),
                _mm_andnot_si128(mask.raw, no.raw),
            ))
        }
    }

    #[inline(always)]
    fn if_then_else_zero<T: Lane>(self, mask: M128<T>, yes: V128<T>) -> V128<T> {
        V128::from_raw(unsafe { _mm_and_si128(mask.raw, yes.raw) })
    }

    #[inline(always)]
    fn if_then_zero_else<T: Lane>(self, mask: M128<T>, no: V128<T>) -> V128<T> {
        V128::from_raw(unsafe { _mm_andnot_si128(mask.raw, no.raw) })
    }

    #[inline(always)]
    fn and_mask<T: Lane>(self, a: M128<T>, b: M128<T>) -> M128<T> {
        M128::from_raw(unsafe { _mm_and_si128(a.raw, b.raw) })
    }

    #[inline(always)]
    fn or_mask<T: Lane>(self, a: M128<T>, b: M128<T>) -> M128<T> {
        M128::from_raw(unsafe { _mm_or_si128(a.raw, b.raw) })
    }

    #[inline(always)]
    fn not_mask<T: Lane>(self, m: M128<T>) -> M128<T> {
        M128::from_raw(unsafe { _mm_xor_si128(m.raw, _mm_set1_epi8(!0)) })
    }

    #[inline(always)]
    fn xor_mask<T: Lane>(self, a: M128<T>, b: M128<T>) -> M128<T> {
        M128::from_raw(unsafe { _mm_xor_si128(a.raw, b.raw) })
    }

    #[inline(always)]
    fn find_last_true<T: Lane>(self, m: M128<T>) -> Option<usize> {
        unsafe {
            match T::BYTES {
                1 => {
                    let bits = _mm_movemask_epi8(m.raw) as u32;
                    if bits == 0 {
                        None
                    } else {
                        Some(31 - bits.leading_zeros() as usize)
                    }
                }
                2 => {
                    let byte_mask = _mm_movemask_epi8(m.raw) as u32;
                    if byte_mask == 0 {
                        None
                    } else {
                        // Highest set bit / 2 = lane index
                        Some((31 - byte_mask.leading_zeros() as usize) / 2)
                    }
                }
                4 => {
                    let bits = _mm_movemask_ps(_mm_castsi128_ps(m.raw)) as u32;
                    if bits == 0 {
                        None
                    } else {
                        Some(31 - bits.leading_zeros() as usize)
                    }
                }
                8 => {
                    let bits = _mm_movemask_pd(_mm_castsi128_pd(m.raw)) as u32;
                    if bits == 0 {
                        None
                    } else {
                        Some(31 - bits.leading_zeros() as usize)
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn bits_from_mask<T: Lane>(self, m: M128<T>) -> u64 {
        unsafe {
            match T::BYTES {
                1 => {
                    // movemask_epi8 gives one bit per byte lane — exactly what we need.
                    _mm_movemask_epi8(m.raw) as u16 as u64
                }
                2 => {
                    // movemask_epi8 gives 16 bits (2 per 16-bit lane). Extract odd-position bits.
                    let byte_mask = _mm_movemask_epi8(m.raw) as u64;
                    // Pack bits at positions 1,3,5,7,9,11,13,15 into positions 0..7
                    let x = (byte_mask >> 1) & 0x5555;
                    let x = (x | (x >> 1)) & 0x3333;
                    let x = (x | (x >> 2)) & 0x0F0F;
                    (x | (x >> 4)) & 0x00FF
                }
                4 => _mm_movemask_ps(_mm_castsi128_ps(m.raw)) as u64,
                8 => _mm_movemask_pd(_mm_castsi128_pd(m.raw)) as u64,
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn exclusive_neither<T: Lane>(self, a: M128<T>, b: M128<T>) -> M128<T> {
        // NOR: true only where neither a nor b is set (C++ ExclusiveNeither).
        // ~(a | b) = andnot(a, ~b) = (~a) & (~b).
        unsafe {
            let ones = _mm_cmpeq_epi8(_mm_setzero_si128(), _mm_setzero_si128());
            let not_b = _mm_xor_si128(b.raw, ones);
            M128::from_raw(_mm_andnot_si128(a.raw, not_b))
        }
    }

    #[inline(always)]
    fn slide_mask_1_up<T: Lane>(self, mask: M128<T>) -> M128<T> {
        {
            let v = self.vec_from_mask::<T>(mask);
            let slid = self.slide_1_up(v);
            self.mask_from_vec(slid)
        }
    }

    #[inline(always)]
    fn slide_mask_1_down<T: Lane>(self, mask: M128<T>) -> M128<T> {
        {
            let v = self.vec_from_mask::<T>(mask);
            let slid = self.slide_1_down(v);
            self.mask_from_vec(slid)
        }
    }

    #[inline(always)]
    fn if_negative_then_else<T: Lane>(self, v: V128<T>, yes: V128<T>, no: V128<T>) -> V128<T> {
        unsafe {
            let sign = sse2_sign_mask::<T>(v.raw);
            let r = _mm_or_si128(_mm_and_si128(sign, yes.raw), _mm_andnot_si128(sign, no.raw));
            V128::from_raw(r)
        }
    }

    #[inline(always)]
    fn if_negative_then_else_zero<T: Lane>(self, v: V128<T>, yes: V128<T>) -> V128<T> {
        unsafe {
            let sign = sse2_sign_mask::<T>(v.raw);
            V128::from_raw(_mm_and_si128(sign, yes.raw))
        }
    }

    #[inline(always)]
    fn if_negative_then_zero_else<T: Lane>(self, v: V128<T>, no: V128<T>) -> V128<T> {
        unsafe {
            let sign = sse2_sign_mask::<T>(v.raw);
            V128::from_raw(_mm_andnot_si128(sign, no.raw))
        }
    }
}

/// Build an all-ones/all-zeros sign-broadcast mask from raw bits, per lane width.
/// All-ones where the lane's MSB (sign bit) is set. Works for ints and floats.
#[inline(always)]
unsafe fn sse2_sign_mask<T: Lane>(raw: __m128i) -> __m128i {
    unsafe {
        match T::BYTES {
            1 => _mm_cmpgt_epi8(_mm_setzero_si128(), raw),
            2 => _mm_srai_epi16(raw, 15),
            4 => _mm_srai_epi32(raw, 31),
            _ => {
                let s = _mm_srai_epi32(raw, 31);
                _mm_shuffle_epi32(s, 0xF5) // _MM_SHUFFLE(3, 3, 1, 1)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimdConvert
// ---------------------------------------------------------------------------

// SAFETY: All operations use SSE2 intrinsics.
unsafe impl SimdConvert for Sse2 {
    #[inline(always)]
    fn promote_to<N: NarrowLane>(self, v: V128<N>) -> V128<N::Wide>
    where
        N::Wide: Lane,
    {
        unsafe {
            // Extract lower half and widen
            let raw = match N::BYTES {
                1 => {
                    if is_signed::<N>() {
                        // Sign-extend i8 -> i16: unpack with sign extension
                        let sign = _mm_cmpgt_epi8(_mm_setzero_si128(), v.raw);
                        _mm_unpacklo_epi8(v.raw, sign)
                    } else {
                        _mm_unpacklo_epi8(v.raw, _mm_setzero_si128())
                    }
                }
                2 => {
                    if is_signed::<N>() {
                        let sign = _mm_srai_epi16(v.raw, 15);
                        _mm_unpacklo_epi16(v.raw, sign)
                    } else {
                        _mm_unpacklo_epi16(v.raw, _mm_setzero_si128())
                    }
                }
                4 => {
                    if is_type::<N, f32>() {
                        // f32 -> f64: use cvt
                        _mm_castpd_si128(_mm_cvtps_pd(_mm_castsi128_ps(v.raw)))
                    } else if is_signed::<N>() {
                        let sign = _mm_srai_epi32(v.raw, 31);
                        _mm_unpacklo_epi32(v.raw, sign)
                    } else {
                        _mm_unpacklo_epi32(v.raw, _mm_setzero_si128())
                    }
                }
                _ => v.raw, // u64 has no wider type
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn demote_to<W: WideLane>(self, v: V128<W>) -> V128<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = match W::BYTES {
                2 => {
                    if is_signed::<W>() {
                        // i16 -> i8 saturating
                        _mm_packs_epi16(v.raw, _mm_setzero_si128())
                    } else {
                        // u16 -> u8 saturating
                        // _mm_packus_epi16 treats input as signed i16, so u16
                        // values >= 32768 would become 0. Clamp to 255 first.
                        let max_val = _mm_set1_epi16(0xFF);
                        let excess = _mm_subs_epu16(v.raw, max_val);
                        let clamped = _mm_sub_epi16(v.raw, excess);
                        _mm_packus_epi16(clamped, _mm_setzero_si128())
                    }
                }
                4 => {
                    if is_signed::<W>() {
                        // i32 -> i16 saturating
                        _mm_packs_epi32(v.raw, _mm_setzero_si128())
                    } else {
                        // u32 -> u16 saturating: SSE2 lacks _mm_packus_epi32.
                        // Clamp to 0x7FFFFFFF using byte-level min so bias trick
                        // works correctly for all u32 values.
                        let max_i32 = _mm_set1_epi32(0x7FFFFFFFu32 as i32);
                        let clamped = _mm_min_epu8(v.raw, max_i32);
                        let bias32 = _mm_set1_epi32(0x8000u32 as i32);
                        let biased = _mm_sub_epi32(clamped, bias32);
                        let packed = _mm_packs_epi32(biased, _mm_setzero_si128());
                        let bias16 = _mm_set1_epi16(0x8000u16 as i16);
                        _mm_add_epi16(packed, bias16)
                    }
                }
                8 => {
                    if is_type::<W, f64>() {
                        // f64 -> f32
                        _mm_castps_si128(_mm_cvtpd_ps(_mm_castsi128_pd(v.raw)))
                    } else if is_signed::<W>() {
                        let min_val = V128::<W>::from_raw(_mm_set1_epi64x(i32::MIN as i64));
                        let max_val = V128::<W>::from_raw(_mm_set1_epi64x(i32::MAX as i64));
                        let clamped = self.min(self.max(v, min_val), max_val);
                        // Pack low 32-bit of each 64-bit lane: shuffle(0x08) -> [lo0, lo1, ?, ?]
                        let packed = _mm_shuffle_epi32(clamped.raw, 0x08);
                        _mm_unpacklo_epi64(packed, _mm_setzero_si128())
                    } else {
                        let max_val = V128::<W>::from_raw(_mm_set1_epi64x(u32::MAX as i64));
                        let clamped = self.min(v, max_val);
                        let packed = _mm_shuffle_epi32(clamped.raw, 0x08);
                        _mm_unpacklo_epi64(packed, _mm_setzero_si128())
                    }
                }
                _ => v.raw,
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn convert_to_int<F: FloatLane>(self, v: V128<F>) -> V128<F::Int> {
        unsafe {
            let raw = if F::BYTES == 4 {
                _mm_cvttps_epi32(_mm_castsi128_ps(v.raw))
            } else {
                // f64 -> i64: extract each f64 lane and convert with saturation.
                // cvttsd_si64 returns i64::MIN for both positive and negative
                // overflow. C++ saturates positive overflow to i64::MAX.
                let v_pd = _mm_castsi128_pd(v.raw);
                let overflow = _mm_castpd_si128(_mm_cmpge_pd(
                    v_pd,
                    _mm_set1_pd(9.223372036854776e18), // >= i64::MAX as f64
                ));
                let i0 = _mm_cvttsd_si64(v_pd);
                let i1 = _mm_cvttsd_si64(_mm_castsi128_pd(_mm_srli_si128(v.raw, 8)));
                let converted = _mm_set_epi64x(i1, i0);
                // Where overflow: i64::MAX; else: converted
                let max_val = _mm_set1_epi64x(i64::MAX);
                _mm_or_si128(
                    _mm_and_si128(overflow, max_val),
                    _mm_andnot_si128(overflow, converted),
                )
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn convert_to_float<F: FloatLane>(self, v: V128<F::Int>) -> V128<F> {
        unsafe {
            let raw = if F::BYTES == 4 {
                _mm_castps_si128(_mm_cvtepi32_ps(v.raw))
            } else {
                // i64 -> f64: Wim trick (no direct intrinsic in SSE2)
                // Magic constant: upper f64 bits for 2^84 + 2^63
                let k84_63 = _mm_set1_epi64x(0x4530000080000000u64 as i64);
                // Extract upper 32 bits of each i64, XOR with magic to get f64 representation
                let v_upper = _mm_castpd_si128(_mm_sub_pd(
                    _mm_castsi128_pd(_mm_xor_si128(_mm_srli_epi64(v.raw, 32), k84_63)),
                    _mm_castsi128_pd(_mm_set1_epi64x(0x4530000080100000u64 as i64)),
                ));
                // Build lower f64: OddEven(k52, bitcast_u32(v))
                // k52 in odd u32 positions (high u32 of each u64), v's low u32 in even positions
                let k52 = _mm_set1_epi64x(0x4330000000000000u64 as i64);
                // OddEven for u32 on SSE2: take odd-indexed u32 from k52, even-indexed from v
                let mask = _mm_set_epi32(-1, 0, -1, 0); // odd u32 positions
                let odd_even = _mm_or_si128(
                    _mm_and_si128(mask, k52),
                    _mm_andnot_si128(mask, v.raw),
                );
                // Result = v_upper + v_lower_dbl
                // v_upper contains -2^52 bias from subtraction constant;
                // odd_even as f64 = 2^52 + lower_bits, so they cancel.
                _mm_castpd_si128(_mm_add_pd(
                    _mm_castsi128_pd(v_upper),
                    _mm_castsi128_pd(odd_even),
                ))
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn truncate_to<W: WideLane>(self, v: V128<W>) -> V128<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = if W::BYTES == 2 {
                // u16/i16 -> u8/i8: mask low byte of each 16-bit lane, pack
                let mask = _mm_set1_epi16(0x00FF);
                let masked = _mm_and_si128(v.raw, mask);
                _mm_packus_epi16(masked, _mm_setzero_si128())
            } else if W::BYTES == 4 {
                // u32/i32 -> u16/i16: extract low 16 bits of each 32-bit lane
                // SSE2 lacks _mm_shuffle_epi8, use scalar approach
                let mut arr = [0u8; 16];
                _mm_storeu_si128(arr.as_mut_ptr().cast(), v.raw);
                let mut result = [0u8; 16];
                for i in 0..4 {
                    result[i * 2] = arr[i * 4];
                    result[i * 2 + 1] = arr[i * 4 + 1];
                }
                _mm_loadu_si128(result.as_ptr().cast())
            } else if W::BYTES == 8 {
                // u64/i64 -> u32/i32: extract low 32 bits of each 64-bit lane
                let mut arr = [0u8; 16];
                _mm_storeu_si128(arr.as_mut_ptr().cast(), v.raw);
                let mut result = [0u8; 16];
                core::ptr::copy_nonoverlapping(arr.as_ptr(), result.as_mut_ptr(), 4);
                core::ptr::copy_nonoverlapping(
                    arr.as_ptr().add(8),
                    result.as_mut_ptr().add(4),
                    4,
                );
                _mm_loadu_si128(result.as_ptr().cast())
            } else {
                v.raw
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn ordered_demote_2_to<W: WideLane>(
        self,
        lo: V128<W>,
        hi: V128<W>,
    ) -> V128<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = if W::BYTES == 2 {
                if is_type::<W, u16>() {
                    // u16 -> u8: packus_epi16 treats inputs as signed i16.
                    // Clamp u16 values to 0x7FFF so packus sees them as positive.
                    // Use byte-level min (SSE2) since _mm_min_epu16 requires SSE4.1.
                    let max_i16_bytes = _mm_set1_epi16(0x7FFFu16 as i16);
                    let lo_clamped = _mm_min_epu8(lo.raw, max_i16_bytes);
                    let hi_clamped = _mm_min_epu8(hi.raw, max_i16_bytes);
                    _mm_packus_epi16(lo_clamped, hi_clamped)
                } else {
                    // i16 -> i8: saturating pack
                    _mm_packs_epi16(lo.raw, hi.raw)
                }
            } else if W::BYTES == 4 {
                if is_type::<W, i32>() {
                    // i32 -> i16: saturating pack
                    _mm_packs_epi32(lo.raw, hi.raw)
                } else {
                    // u32 -> u16: clamp to 0x7FFFFFFF using byte-level min (SSE2),
                    // then use bias trick since SSE2 lacks _mm_packus_epi32.
                    let max_i32 = _mm_set1_epi32(0x7FFFFFFFu32 as i32);
                    let lo_clamped = _mm_min_epu8(lo.raw, max_i32);
                    let hi_clamped = _mm_min_epu8(hi.raw, max_i32);
                    let bias32 = _mm_set1_epi32(0x8000u32 as i32);
                    let lo_biased = _mm_sub_epi32(lo_clamped, bias32);
                    let hi_biased = _mm_sub_epi32(hi_clamped, bias32);
                    let packed = _mm_packs_epi32(lo_biased, hi_biased);
                    let bias16 = _mm_set1_epi16(0x8000u16 as i16);
                    _mm_add_epi16(packed, bias16)
                }
            } else if W::BYTES == 8 {
                // u64/i64 -> u32/i32: saturating demote
                let mut arr_lo = [0i64; 2];
                let mut arr_hi = [0i64; 2];
                _mm_storeu_si128(arr_lo.as_mut_ptr().cast(), lo.raw);
                _mm_storeu_si128(arr_hi.as_mut_ptr().cast(), hi.raw);
                let mut result = [0u32; 4];
                if is_type::<W, u64>() {
                    let vals = [
                        arr_lo[0] as u64, arr_lo[1] as u64,
                        arr_hi[0] as u64, arr_hi[1] as u64,
                    ];
                    for i in 0..4 {
                        result[i] = if vals[i] > u32::MAX as u64 { u32::MAX } else { vals[i] as u32 };
                    }
                } else if is_type::<W, i64>() {
                    for (i, &val) in [arr_lo[0], arr_lo[1], arr_hi[0], arr_hi[1]].iter().enumerate() {
                        result[i] = if val > i32::MAX as i64 {
                            i32::MAX as u32
                        } else if val < i32::MIN as i64 {
                            i32::MIN as u32
                        } else {
                            val as u32
                        };
                    }
                }
                _mm_loadu_si128(result.as_ptr().cast())
            } else {
                lo.raw
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn nearest_int<F: FloatLane>(self, v: V128<F>) -> V128<F::Int> {
        unsafe {
            let raw = if F::BYTES == 4 {
                // _mm_cvtps_epi32 uses current rounding mode (default = nearest-even).
                // Clamp >= 2^31 to i32::MAX first to match C++ Highway.
                let ps = _mm_castsi128_ps(v.raw);
                let max_f = _mm_set1_ps(2147483520.0f32); // largest f32 < 2^31
                let overflow = _mm_castps_si128(_mm_cmpge_ps(ps, _mm_set1_ps(2147483648.0f32)));
                let clamped = _mm_castps_si128(_mm_min_ps(ps, max_f));
                let converted = _mm_cvtps_epi32(_mm_castsi128_ps(clamped));
                // Where overflow: i32::MAX; else: converted
                _mm_or_si128(
                    _mm_and_si128(overflow, _mm_set1_epi32(i32::MAX)),
                    _mm_andnot_si128(overflow, converted),
                )
            } else {
                // f64 -> i64: scalar fallback with round-to-nearest-even.
                let v_pd = _mm_castsi128_pd(v.raw);
                let overflow = _mm_castpd_si128(_mm_cmpge_pd(
                    v_pd,
                    _mm_set1_pd(9.223372036854776e18),
                ));
                // Round to nearest-even first, then truncate-convert.
                let rounded = _mm_castpd_si128(_mm_round_pd(
                    v_pd,
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                ));
                let i0 = _mm_cvttsd_si64(_mm_castsi128_pd(rounded));
                let i1 = _mm_cvttsd_si64(_mm_castsi128_pd(_mm_srli_si128(rounded, 8)));
                let converted = _mm_set_epi64x(i1, i0);
                let max_val = _mm_set1_epi64x(i64::MAX);
                _mm_or_si128(
                    _mm_and_si128(overflow, max_val),
                    _mm_andnot_si128(overflow, converted),
                )
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn reorder_demote_2_to<W: WideLane>(
        self,
        a: V128<W>,
        b: V128<W>,
    ) -> V128<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // SSE2: pack instructions already produce correct order within 128 bits
        // so reorder_demote_2_to is the same as ordered_demote_2_to
        self.ordered_demote_2_to(a, b)
    }

    #[inline(always)]
    fn demote_in_range_to<W: WideLane>(self, v: V128<W>) -> V128<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // Same as demote_to but without clamping (assumes in-range)
        self.demote_to(v)
    }

    #[inline(always)]
    fn convert_in_range_to_int<F: FloatLane>(self, v: V128<F>) -> V128<F::Int> {
        // Same as convert_to_int (truncation toward zero), without overflow check
        self.convert_to_int(v)
    }

    #[inline(always)]
    fn promote_lower_to<N: NarrowLane>(self, v: V128<N>) -> V128<N::Wide>
    where
        N::Wide: Lane,
    {
        // For SSE2, half=full, so promote_lower_to is the same as promote_to
        self.promote_to(v)
    }

    #[inline(always)]
    fn promote_upper_to<N: NarrowLane>(self, v: V128<N>) -> V128<N::Wide>
    where
        N::Wide: Lane,
    {
        // For SSE2, "upper half" doesn't exist (half=full), so we extract the
        // upper portion of the lanes. E.g., for u8->u16 promotion, the upper 8 u8 lanes
        // get promoted. We shift right by half the lanes.
        {
            let lanes = 16 / N::BYTES;
            let half_lanes = lanes / 2;
            // Shift the vector down by half_lanes to bring upper lanes to lower position
            let shifted = self.slide_down_lanes(v, half_lanes);
            self.promote_to(shifted)
        }
    }

    #[inline(always)]
    fn ordered_truncate_2_to<W: WideLane>(
        self,
        lo: V128<W>,
        hi: V128<W>,
    ) -> V128<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // OrderedTruncate2To = ConcatEven of the narrow-reinterpreted vectors.
        // Even narrow lanes of a bitcast wide vector are the low (truncated) halves.
        {
            let lo_n = self.bitcast::<W, W::Narrow>(lo);
            let hi_n = self.bitcast::<W, W::Narrow>(hi);
            self.concat_even(lo_n, hi_n)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdShuffle
// ---------------------------------------------------------------------------

// SAFETY: All operations use SSE2 intrinsics.
unsafe impl SimdShuffle for Sse2 {
    #[inline(always)]
    fn reverse<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Reverse 16 bytes using shuffles
                    let swapped = _mm_shuffle_epi32(v.raw, 0x1B); // reverse dwords
                    let swapped = _mm_shufflelo_epi16(swapped, 0xB1);
                    let swapped = _mm_shufflehi_epi16(swapped, 0xB1);
                    // Now swap bytes within each 16-bit word
                    let hi = _mm_srli_epi16(swapped, 8);
                    let lo = _mm_slli_epi16(swapped, 8);
                    _mm_or_si128(hi, lo)
                }
                2 => {
                    let swapped = _mm_shuffle_epi32(v.raw, 0x1B);
                    let swapped = _mm_shufflelo_epi16(swapped, 0xB1);
                    _mm_shufflehi_epi16(swapped, 0xB1)
                }
                4 => _mm_shuffle_epi32(v.raw, 0x1B), // 3,2,1,0
                8 => _mm_shuffle_epi32(v.raw, 0x4E), // swap 64-bit halves
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn broadcast_lane<T: Lane, const IDX: usize>(self, v: V128<T>) -> V128<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    // _mm_shuffle_epi32 with all lanes set to IDX
                    // Can't use runtime imm8, use the known-IDX match
                    match IDX & 3 {
                        0 => V128::from_raw(_mm_shuffle_epi32(v.raw, 0x00)),
                        1 => V128::from_raw(_mm_shuffle_epi32(v.raw, 0x55)),
                        2 => V128::from_raw(_mm_shuffle_epi32(v.raw, 0xAA)),
                        3 => V128::from_raw(_mm_shuffle_epi32(v.raw, 0xFF)),
                        _ => unreachable!(),
                    }
                }
                8 => {
                    // 2 lanes: shuffle 32-bit pairs
                    match IDX & 1 {
                        0 => V128::from_raw(_mm_shuffle_epi32(v.raw, 0x44)), // [0,1,0,1]
                        1 => V128::from_raw(_mm_shuffle_epi32(v.raw, 0xEE)), // [2,3,2,3]
                        _ => unreachable!(),
                    }
                }
                2 => {
                    // 8 lanes: use shufflelo/shufflehi + shuffle_epi32
                    // Broadcast within low qword, then broadcast across qwords
                    let lo = match IDX & 3 {
                        0 => _mm_shufflelo_epi16(v.raw, 0x00),
                        1 => _mm_shufflelo_epi16(v.raw, 0x55),
                        2 => _mm_shufflelo_epi16(v.raw, 0xAA),
                        3 => _mm_shufflelo_epi16(v.raw, 0xFF),
                        _ => unreachable!(),
                    };
                    if IDX < 4 {
                        // Lane is in the low half, broadcast low qword to both
                        V128::from_raw(_mm_shuffle_epi32(lo, 0x44))
                    } else {
                        // Lane is in the high half, shuffle high part first
                        let hi = match IDX & 3 {
                            0 => _mm_shufflehi_epi16(v.raw, 0x00),
                            1 => _mm_shufflehi_epi16(v.raw, 0x55),
                            2 => _mm_shufflehi_epi16(v.raw, 0xAA),
                            3 => _mm_shufflehi_epi16(v.raw, 0xFF),
                            _ => unreachable!(),
                        };
                        V128::from_raw(_mm_shuffle_epi32(hi, 0xEE))
                    }
                }
                _ => {
                    // 1-byte: no efficient SSE2 single-byte broadcast, use extract+splat
                    let val: T = self.extract_lane(v, IDX);
                    self.splat(val)
                }
            }
        }
    }

    #[inline(always)]
    fn interleave_lower<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_unpacklo_epi8(a.raw, b.raw),
                2 => _mm_unpacklo_epi16(a.raw, b.raw),
                4 => _mm_unpacklo_epi32(a.raw, b.raw),
                8 => _mm_unpacklo_epi64(a.raw, b.raw),
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn interleave_upper<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_unpackhi_epi8(a.raw, b.raw),
                2 => _mm_unpackhi_epi16(a.raw, b.raw),
                4 => _mm_unpackhi_epi32(a.raw, b.raw),
                8 => _mm_unpackhi_epi64(a.raw, b.raw),
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn zip_lower<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // zip_lower is the same as interleave_lower for SSE
        self.interleave_lower(a, b)
    }

    #[inline(always)]
    fn zip_upper<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        self.interleave_upper(a, b)
    }

    #[inline(always)]
    fn table_lookup_bytes<T: Lane>(self, table: V128<T>, idx: V128<T>) -> V128<T> {
        unsafe {
            // SSE2 doesn't have pshufb; emulate byte-level table lookup
            let mut tbl: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            let mut indices: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            _mm_store_si128(tbl.as_mut_ptr().cast(), table.raw);
            _mm_store_si128(indices.as_mut_ptr().cast(), idx.raw);
            let mut result: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            for i in 0..16 {
                let index = (indices[i] & 0x0F) as usize;
                result[i] = if indices[i] & 0x80 != 0 {
                    0
                } else {
                    tbl[index]
                };
            }
            V128::from_raw(_mm_load_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn table_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        v: V128<T>,
        idx: V128<I>,
    ) -> V128<T> {
        unsafe {
            let lanes = simd::lanes::<T, Sse2>();
            let mut data: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            let mut indices: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            _mm_store_si128(data.as_mut_ptr().cast(), v.raw);
            _mm_store_si128(indices.as_mut_ptr().cast(), idx.raw);
            let mut result: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            for i in 0..lanes {
                let idx_offset = i * I::BYTES;
                let lane_idx = read_lane_bits(&indices, idx_offset, I::BYTES) as usize % lanes;
                let src_offset = lane_idx * T::BYTES;
                let dst_offset = i * T::BYTES;
                result[dst_offset..dst_offset + T::BYTES]
                    .copy_from_slice(&data[src_offset..src_offset + T::BYTES]);
            }
            V128::from_raw(_mm_load_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn reverse2<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Swap adjacent bytes: use 16-bit byte swap
                    let hi = _mm_srli_epi16(v.raw, 8);
                    let lo = _mm_slli_epi16(v.raw, 8);
                    _mm_or_si128(hi, lo)
                }
                2 => {
                    // Swap adjacent 16-bit lanes within each 32-bit pair
                    let swapped_lo = _mm_shufflelo_epi16(v.raw, 0b10_11_00_01);
                    _mm_shufflehi_epi16(swapped_lo, 0b10_11_00_01)
                }
                4 => {
                    // Swap adjacent 32-bit lanes: [1,0,3,2]
                    _mm_shuffle_epi32(v.raw, 0b10_11_00_01)
                }
                8 => {
                    // Swap the two 64-bit lanes
                    _mm_shuffle_epi32(v.raw, 0b01_00_11_10)
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn reverse4<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Reverse groups of 4 bytes within each 32-bit word = bswap32.
                    // Same as reverse_lane_bytes for u32, using shift+mask.
                    let x = v.raw;
                    let a = _mm_srli_epi32(x, 24);
                    let b = _mm_and_si128(_mm_srli_epi32(x, 8), _mm_set1_epi32(0x0000FF00));
                    let c = _mm_and_si128(_mm_slli_epi32(x, 8), _mm_set1_epi32(0x00FF0000));
                    let d = _mm_slli_epi32(x, 24);
                    _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d))
                }
                2 => {
                    // Reverse groups of 4 u16 lanes.
                    // Low 4 lanes reversed, high 4 lanes reversed.
                    let swapped_lo = _mm_shufflelo_epi16(v.raw, 0b00_01_10_11);
                    _mm_shufflehi_epi16(swapped_lo, 0b00_01_10_11)
                }
                4 => {
                    // Reverse all 4 u32 lanes: [3,2,1,0]
                    _mm_shuffle_epi32(v.raw, 0b00_01_10_11)
                }
                8 => {
                    // Only 2 lanes for 64-bit; reverse4 doesn't apply in a meaningful way.
                    // Treat as full reverse (same as reverse2 for 2 lanes).
                    _mm_shuffle_epi32(v.raw, 0b01_00_11_10)
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn reverse8<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    // Reverse groups of 8 bytes within each 64-bit half.
                    // Step 1: bswap32 within each dword
                    let x = v.raw;
                    let a = _mm_srli_epi32(x, 24);
                    let b = _mm_and_si128(_mm_srli_epi32(x, 8), _mm_set1_epi32(0x0000FF00));
                    let c = _mm_and_si128(_mm_slli_epi32(x, 8), _mm_set1_epi32(0x00FF0000));
                    let d = _mm_slli_epi32(x, 24);
                    let bswap32 = _mm_or_si128(_mm_or_si128(a, b), _mm_or_si128(c, d));
                    // Step 2: swap 32-bit halves within each 64-bit lane
                    V128::from_raw(_mm_shuffle_epi32(bswap32, 0xB1))
                }
                2 => {
                    // 8 u16 lanes total — reverse all 8 = full reverse.
                    self.reverse(v)
                }
                _ => {
                    // For 4-byte (4 lanes) and 8-byte (2 lanes), reverse8 is the same as
                    // full reverse since there aren't enough lanes to form a group of 8.
                    self.reverse(v)
                }
            }
        }
    }

    #[inline(always)]
    fn concat_upper_lower<T: Lane>(self, hi: V128<T>, lo: V128<T>) -> V128<T> {
        // Upper 64 bits of hi, lower 64 bits of lo.
        // _mm_move_sd(a, b) returns [b[0], a[1]], i.e. lower 64 of b, upper 64 of a.
        // We want upper of hi, lower of lo => _mm_move_sd(hi_pd, lo_pd)
        unsafe {
            let hi_pd = _mm_castsi128_pd(hi.raw);
            let lo_pd = _mm_castsi128_pd(lo.raw);
            V128::from_raw(_mm_castpd_si128(_mm_move_sd(hi_pd, lo_pd)))
        }
    }

    #[inline(always)]
    fn concat_lower_upper<T: Lane>(self, hi: V128<T>, lo: V128<T>) -> V128<T> {
        // Lower 64 bits of hi, upper 64 bits of lo.
        // _mm_move_sd(a, b) returns [b[0], a[1]].
        // _mm_move_sd(lo_pd, hi_pd) = [hi_lower, lo_upper]. That's what we want.
        unsafe {
            let hi_pd = _mm_castsi128_pd(hi.raw);
            let lo_pd = _mm_castsi128_pd(lo.raw);
            V128::from_raw(_mm_castpd_si128(_mm_move_sd(lo_pd, hi_pd)))
        }
    }

    #[inline(always)]
    fn concat_even<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // Even-indexed lanes from a (lower half), even-indexed lanes from b (upper half).
        unsafe {
            match T::BYTES {
                4 => {
                    // a has lanes [a0, a1, a2, a3], b has [b0, b1, b2, b3]
                    // Result: [a0, a2, b0, b2]
                    let a_ps = _mm_castsi128_ps(a.raw);
                    let b_ps = _mm_castsi128_ps(b.raw);
                    // _mm_shuffle_ps(a, b, imm8): selects 2 from a (low), 2 from b (high)
                    // imm8 = 0b10_00_10_00 = 0x88: a[0], a[2], b[0], b[2]
                    V128::from_raw(_mm_castps_si128(_mm_shuffle_ps(a_ps, b_ps, 0x88)))
                }
                8 => {
                    // 2 lanes each. Even = lane 0. Result: [a0, b0]
                    V128::from_raw(_mm_unpacklo_epi64(a.raw, b.raw))
                }
                2 => {
                    // Even u16 lanes: positions 0, 2, 4, 6 from each vector.
                    // Pack even lanes using shufflelo/shufflehi + shuffle_epi32.
                    // shufflelo selects from low 4 u16, shufflehi from high 4 u16.
                    // imm8 = 0b_10_00_10_00: positions [0, 2, 0, 2]
                    let a_s = _mm_shufflelo_epi16(a.raw, 0b_10_00_10_00);
                    let a_s = _mm_shufflehi_epi16(a_s, 0b_10_00_10_00);
                    // Now [a0,a2,a0,a2, a4,a6,a4,a6] -> shuffle_epi32 to pack
                    let a_packed = _mm_shuffle_epi32(a_s, 0b_10_00_10_00);
                    let b_s = _mm_shufflelo_epi16(b.raw, 0b_10_00_10_00);
                    let b_s = _mm_shufflehi_epi16(b_s, 0b_10_00_10_00);
                    let b_packed = _mm_shuffle_epi32(b_s, 0b_10_00_10_00);
                    // Lower 64 bits: [a0,a2,a4,a6]; lower 64 bits of b: [b0,b2,b4,b6]
                    V128::from_raw(_mm_unpacklo_epi64(a_packed, b_packed))
                }
                _ => {
                    // u8: even bytes at positions 0,2,4,...,14
                    // Mask to keep only even bytes (low byte of each u16 word)
                    let mask = _mm_set1_epi16(0x00FF);
                    let a_even = _mm_and_si128(a.raw, mask);
                    let b_even = _mm_and_si128(b.raw, mask);
                    // _mm_packus_epi16 packs 8 u16 values -> 8 u8 values per operand
                    V128::from_raw(_mm_packus_epi16(a_even, b_even))
                }
            }
        }
    }

    #[inline(always)]
    fn concat_odd<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // Odd-indexed lanes from a (lower half), odd-indexed lanes from b (upper half).
        unsafe {
            match T::BYTES {
                4 => {
                    // Result: [a1, a3, b1, b3]
                    let a_ps = _mm_castsi128_ps(a.raw);
                    let b_ps = _mm_castsi128_ps(b.raw);
                    // imm8 = 0b11_01_11_01 = 0xDD: a[1], a[3], b[1], b[3]
                    V128::from_raw(_mm_castps_si128(_mm_shuffle_ps(a_ps, b_ps, 0xDD)))
                }
                8 => {
                    // 2 lanes each. Odd = lane 1. Result: [a1, b1]
                    V128::from_raw(_mm_unpackhi_epi64(a.raw, b.raw))
                }
                2 => {
                    // Odd u16 lanes: positions 1, 3, 5, 7 from each vector.
                    // imm8 = 0b_11_01_11_01: positions [1, 3, 1, 3]
                    let a_s = _mm_shufflelo_epi16(a.raw, 0b_11_01_11_01);
                    let a_s = _mm_shufflehi_epi16(a_s, 0b_11_01_11_01);
                    let a_packed = _mm_shuffle_epi32(a_s, 0b_10_00_10_00);
                    let b_s = _mm_shufflelo_epi16(b.raw, 0b_11_01_11_01);
                    let b_s = _mm_shufflehi_epi16(b_s, 0b_11_01_11_01);
                    let b_packed = _mm_shuffle_epi32(b_s, 0b_10_00_10_00);
                    V128::from_raw(_mm_unpacklo_epi64(a_packed, b_packed))
                }
                _ => {
                    // u8: odd bytes at positions 1,3,5,...,15
                    // Shift right by 8 within each u16 to move odd bytes to even positions
                    let a_odd = _mm_srli_epi16(a.raw, 8);
                    let b_odd = _mm_srli_epi16(b.raw, 8);
                    V128::from_raw(_mm_packus_epi16(a_odd, b_odd))
                }
            }
        }
    }

    #[inline(always)]
    fn odd_even<T: Lane>(self, odd: V128<T>, even: V128<T>) -> V128<T> {
        // Take odd-indexed lanes from `odd`, even-indexed lanes from `even`.
        // Build a mask that is all-ones for odd lanes, all-zeros for even lanes.
        unsafe {
            match T::BYTES {
                4 => {
                    // Lanes: [even0, odd1, even2, odd3]
                    // Use _mm_shuffle_ps to interleave: pick even[0], odd[1], even[2], odd[3]
                    // We need lane 0 from even, lane 1 from odd, lane 2 from even, lane 3 from odd.
                    // Use blend via mask.
                    let mask = _mm_castsi128_ps(_mm_set_epi32(!0, 0, !0, 0));
                    // mask: lane 0 = 0 (even), lane 1 = all-ones (odd), lane 2 = 0, lane 3 = all-ones
                    let blended = _mm_or_si128(
                        _mm_and_si128(_mm_castps_si128(mask), odd.raw),
                        _mm_andnot_si128(_mm_castps_si128(mask), even.raw),
                    );
                    V128::from_raw(blended)
                }
                8 => {
                    // 2 lanes: lane 0 from even, lane 1 from odd.
                    // Use _mm_move_sd: takes lower 64 of second arg, upper 64 of first.
                    // _mm_move_sd(odd_pd, even_pd) = [even[0], odd[1]]
                    let o_pd = _mm_castsi128_pd(odd.raw);
                    let e_pd = _mm_castsi128_pd(even.raw);
                    V128::from_raw(_mm_castpd_si128(_mm_move_sd(o_pd, e_pd)))
                }
                2 => {
                    // Odd 16-bit lanes at positions 1, 3, 5, 7
                    // Mask: 0xFFFF at odd positions, 0x0000 at even positions
                    let mask = _mm_set1_epi32(!0i32 << 16); // 0xFFFF0000 per dword
                    V128::from_raw(_mm_or_si128(
                        _mm_and_si128(mask, odd.raw),
                        _mm_andnot_si128(mask, even.raw),
                    ))
                }
                _ => {
                    // u8: odd bytes at positions 1, 3, 5, ..., 15
                    // Mask: 0xFF at odd byte positions = 0xFF00 per u16 word
                    let mask = _mm_set1_epi16(!0i16 << 8); // 0xFF00 per word
                    V128::from_raw(_mm_or_si128(
                        _mm_and_si128(mask, odd.raw),
                        _mm_andnot_si128(mask, even.raw),
                    ))
                }
            }
        }
    }

    #[inline(always)]
    fn slide_up_lanes<T: Lane>(self, v: V128<T>, n: usize) -> V128<T> {
        // Shift lanes up (toward higher indices) = shift byte representation LEFT.
        unsafe {
            let byte_shift = n * T::BYTES;
            let raw = match byte_shift {
                0 => v.raw,
                1 => _mm_slli_si128::<1>(v.raw),
                2 => _mm_slli_si128::<2>(v.raw),
                3 => _mm_slli_si128::<3>(v.raw),
                4 => _mm_slli_si128::<4>(v.raw),
                5 => _mm_slli_si128::<5>(v.raw),
                6 => _mm_slli_si128::<6>(v.raw),
                7 => _mm_slli_si128::<7>(v.raw),
                8 => _mm_slli_si128::<8>(v.raw),
                9 => _mm_slli_si128::<9>(v.raw),
                10 => _mm_slli_si128::<10>(v.raw),
                11 => _mm_slli_si128::<11>(v.raw),
                12 => _mm_slli_si128::<12>(v.raw),
                13 => _mm_slli_si128::<13>(v.raw),
                14 => _mm_slli_si128::<14>(v.raw),
                15 => _mm_slli_si128::<15>(v.raw),
                _ => _mm_setzero_si128(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn slide_down_lanes<T: Lane>(self, v: V128<T>, n: usize) -> V128<T> {
        // Shift lanes down (toward lower indices) = shift byte representation RIGHT.
        unsafe {
            let byte_shift = n * T::BYTES;
            let raw = match byte_shift {
                0 => v.raw,
                1 => _mm_srli_si128::<1>(v.raw),
                2 => _mm_srli_si128::<2>(v.raw),
                3 => _mm_srli_si128::<3>(v.raw),
                4 => _mm_srli_si128::<4>(v.raw),
                5 => _mm_srli_si128::<5>(v.raw),
                6 => _mm_srli_si128::<6>(v.raw),
                7 => _mm_srli_si128::<7>(v.raw),
                8 => _mm_srli_si128::<8>(v.raw),
                9 => _mm_srli_si128::<9>(v.raw),
                10 => _mm_srli_si128::<10>(v.raw),
                11 => _mm_srli_si128::<11>(v.raw),
                12 => _mm_srli_si128::<12>(v.raw),
                13 => _mm_srli_si128::<13>(v.raw),
                14 => _mm_srli_si128::<14>(v.raw),
                15 => _mm_srli_si128::<15>(v.raw),
                _ => _mm_setzero_si128(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn compress<T: Lane>(self, v: V128<T>, mask: M128<T>) -> V128<T> {
        unsafe {
            // 64-bit: only 2 lanes, handle directly.
            if T::BYTES == 8 {
                let k = _mm_movemask_pd(_mm_castsi128_pd(mask.raw));
                return if k == 0b10 {
                    // Only high lane active -> swap 64-bit halves.
                    V128::from_raw(_mm_shuffle_epi32(v.raw, 0x4E))
                } else {
                    // 0b00, 0b01, 0b11: low lane stays in place.
                    v
                };
            }
            // For 4-byte types (u32/i32/f32), use shuffle_epi32 with 16 mask patterns.
            if T::BYTES == 4 {
                let mask_bits = _mm_movemask_ps(_mm_castsi128_ps(mask.raw)) as u8;
                return V128::from_raw(match mask_bits & 0xF {
                    0b0000 => _mm_setzero_si128(),
                    0b0001 => v.raw,                                   // [e0]
                    0b0010 => _mm_shuffle_epi32(v.raw, 0x01),          // [e1]
                    0b0011 => v.raw,                                   // [e0, e1]
                    0b0100 => _mm_shuffle_epi32(v.raw, 0x02),          // [e2]
                    0b0101 => _mm_shuffle_epi32(v.raw, 0x08),          // [e0, e2]
                    0b0110 => _mm_shuffle_epi32(v.raw, 0x09),          // [e1, e2]
                    0b0111 => v.raw,                                   // [e0, e1, e2]
                    0b1000 => _mm_shuffle_epi32(v.raw, 0x03),          // [e3]
                    0b1001 => _mm_shuffle_epi32(v.raw, 0x0C),          // [e0, e3]
                    0b1010 => _mm_shuffle_epi32(v.raw, 0x0D),          // [e1, e3]
                    0b1011 => _mm_shuffle_epi32(v.raw, 0x34),          // [e0, e1, e3]
                    0b1100 => _mm_shuffle_epi32(v.raw, 0x0E),          // [e2, e3]
                    0b1101 => _mm_shuffle_epi32(v.raw, 0x38),          // [e0, e2, e3]
                    0b1110 => _mm_shuffle_epi32(v.raw, 0x39),          // [e1, e2, e3]
                    _      => v.raw,                                   // [e0, e1, e2, e3]
                });
            }
            // Scalar fallback for u8/u16/i8/i16 without AVX-512VL.
            let lanes = simd::lanes::<T, Sse2>();
            let mut src: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            let mut mask_arr: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            _mm_store_si128(src.as_mut_ptr().cast(), v.raw);
            _mm_store_si128(mask_arr.as_mut_ptr().cast(), mask.raw);
            let mut result: Aligned<A16, [u8; 16]> = Aligned::new([0u8; 16]);
            let mut dst_idx = 0usize;
            for i in 0..lanes {
                let offset = i * T::BYTES;
                let is_true = mask_arr[offset + T::BYTES - 1] == 0xFF;
                if is_true {
                    let dst_offset = dst_idx * T::BYTES;
                    result[dst_offset..dst_offset + T::BYTES]
                        .copy_from_slice(&src[offset..offset + T::BYTES]);
                    dst_idx += 1;
                }
            }
            V128::from_raw(_mm_load_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn compress_store<T: Lane>(self, v: V128<T>, mask: M128<T>, ptr: *mut T) -> usize {
        unsafe {
            let compressed = self.compress(v, mask);
            let count = self.count_true::<T>(mask);
            _mm_storeu_si128(ptr.cast(), compressed.raw);
            count
        }
    }

    #[inline(always)]
    fn dup_even<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Duplicate even bytes: [v0,v0,v2,v2,...,v14,v14]
                    let even = _mm_and_si128(v.raw, _mm_set1_epi16(0x00FF));
                    _mm_or_si128(even, _mm_slli_epi16(even, 8))
                }
                2 => {
                    // Duplicate even u16 lanes: [v0,v0,v2,v2,v4,v4,v6,v6]
                    let even = _mm_and_si128(v.raw, _mm_set1_epi32(0x0000FFFFu32 as i32));
                    _mm_or_si128(even, _mm_slli_epi32(even, 16))
                }
                4 => {
                    // [v0,v0,v2,v2]
                    // 0b10_10_00_00 = 0xA0
                    _mm_shuffle_epi32(v.raw, 0xA0)
                }
                8 => {
                    // 2 lanes: dup_even = [v0,v0]
                    _mm_unpacklo_epi64(v.raw, v.raw)
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn dup_odd<T: Lane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Duplicate odd bytes: [v1,v1,v3,v3,...,v15,v15]
                    let odd = _mm_srli_epi16(v.raw, 8);
                    _mm_or_si128(odd, _mm_slli_epi16(odd, 8))
                }
                2 => {
                    // Duplicate odd u16 lanes: [v1,v1,v3,v3,v5,v5,v7,v7]
                    let odd = _mm_srli_epi32(v.raw, 16);
                    _mm_or_si128(odd, _mm_slli_epi32(odd, 16))
                }
                4 => {
                    // [v1,v1,v3,v3]
                    // 0b11_11_01_01 = 0xF5
                    _mm_shuffle_epi32(v.raw, 0xF5)
                }
                8 => {
                    // 2 lanes: dup_odd = [v1,v1]
                    _mm_unpackhi_epi64(v.raw, v.raw)
                }
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn concat_lower_lower<T: Lane>(
        self,
        hi: V128<T>,
        lo: V128<T>,
    ) -> V128<T> {
        // Lower 64 bits of lo in low half, lower 64 bits of hi in high half.
        V128::from_raw(unsafe { _mm_unpacklo_epi64(lo.raw, hi.raw) })
    }

    #[inline(always)]
    fn concat_upper_upper<T: Lane>(
        self,
        hi: V128<T>,
        lo: V128<T>,
    ) -> V128<T> {
        // Upper 64 bits of lo in low half, upper 64 bits of hi in high half.
        V128::from_raw(unsafe { _mm_unpackhi_epi64(lo.raw, hi.raw) })
    }

    #[inline(always)]
    fn slide_1_up<T: Lane>(self, v: V128<T>) -> V128<T> {
        // Shift lanes up by 1 = shift bytes left by T::BYTES.
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_slli_si128::<1>(v.raw),
                2 => _mm_slli_si128::<2>(v.raw),
                4 => _mm_slli_si128::<4>(v.raw),
                8 => _mm_slli_si128::<8>(v.raw),
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn slide_1_down<T: Lane>(self, v: V128<T>) -> V128<T> {
        // Shift lanes down by 1 = shift bytes right by T::BYTES.
        unsafe {
            let raw = match T::BYTES {
                1 => _mm_srli_si128::<1>(v.raw),
                2 => _mm_srli_si128::<2>(v.raw),
                4 => _mm_srli_si128::<4>(v.raw),
                8 => _mm_srli_si128::<8>(v.raw),
                _ => unreachable!(),
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn expand<T: Lane>(self, v: V128<T>, mask: M128<T>) -> V128<T> {
        // Inverse of compress: scatter low lanes to mask-true positions,
        // zero where mask is false.
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut v_arr = [0u8; 16];
            let mut m_arr = [0u8; 16];
            _mm_storeu_si128(v_arr.as_mut_ptr().cast(), v.raw);
            _mm_storeu_si128(m_arr.as_mut_ptr().cast(), mask.raw);
            let mut result = [0u8; 16];
            let mut src = 0usize;
            for dst in 0..lanes {
                let off = dst * T::BYTES;
                // For masks, all bytes in a lane are 0xFF (true) or 0x00 (false)
                let mask_byte = m_arr[off + T::BYTES - 1];
                if mask_byte != 0 {
                    if src < lanes {
                        core::ptr::copy_nonoverlapping(
                            v_arr.as_ptr().add(src * T::BYTES),
                            result.as_mut_ptr().add(off),
                            T::BYTES,
                        );
                    }
                    src += 1;
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn combine_shift_right_bytes<T: Lane, const BYTES: usize>(
        self,
        hi: V128<T>,
        lo: V128<T>,
    ) -> V128<T> {
        // SSE2 has no PALIGNR (that's SSSE3). Emulate with shifts + OR.
        // Result = (lo >> (BYTES*8)) | (hi << ((16-BYTES)*8))
        // Per-128-bit-block operation.
        unsafe {
            let lo_shifted = match BYTES {
                0 => lo.raw,
                1 => _mm_srli_si128::<1>(lo.raw),
                2 => _mm_srli_si128::<2>(lo.raw),
                3 => _mm_srli_si128::<3>(lo.raw),
                4 => _mm_srli_si128::<4>(lo.raw),
                5 => _mm_srli_si128::<5>(lo.raw),
                6 => _mm_srli_si128::<6>(lo.raw),
                7 => _mm_srli_si128::<7>(lo.raw),
                8 => _mm_srli_si128::<8>(lo.raw),
                9 => _mm_srli_si128::<9>(lo.raw),
                10 => _mm_srli_si128::<10>(lo.raw),
                11 => _mm_srli_si128::<11>(lo.raw),
                12 => _mm_srli_si128::<12>(lo.raw),
                13 => _mm_srli_si128::<13>(lo.raw),
                14 => _mm_srli_si128::<14>(lo.raw),
                15 => _mm_srli_si128::<15>(lo.raw),
                _ => _mm_setzero_si128(),
            };
            let hi_shifted = match 16 - BYTES {
                0 => _mm_setzero_si128(),
                1 => _mm_slli_si128::<1>(hi.raw),
                2 => _mm_slli_si128::<2>(hi.raw),
                3 => _mm_slli_si128::<3>(hi.raw),
                4 => _mm_slli_si128::<4>(hi.raw),
                5 => _mm_slli_si128::<5>(hi.raw),
                6 => _mm_slli_si128::<6>(hi.raw),
                7 => _mm_slli_si128::<7>(hi.raw),
                8 => _mm_slli_si128::<8>(hi.raw),
                9 => _mm_slli_si128::<9>(hi.raw),
                10 => _mm_slli_si128::<10>(hi.raw),
                11 => _mm_slli_si128::<11>(hi.raw),
                12 => _mm_slli_si128::<12>(hi.raw),
                13 => _mm_slli_si128::<13>(hi.raw),
                14 => _mm_slli_si128::<14>(hi.raw),
                15 => _mm_slli_si128::<15>(hi.raw),
                _ => hi.raw,
            };
            V128::from_raw(_mm_or_si128(lo_shifted, hi_shifted))
        }
    }

    #[inline(always)]
    unsafe fn compress_blended_store<T: Lane>(
        self,
        v: V128<T>,
        mask: M128<T>,
        ptr: *mut T,
    ) -> usize {
        unsafe {
            let compressed = self.compress(v, mask);
            let count = self.count_true(mask);
            let store_mask = self.first_n::<T>(count);
            self.blended_store(compressed, store_mask, ptr);
            count
        }
    }

    #[inline(always)]
    fn odd_even_blocks<T: Lane>(
        self,
        _odd: V128<T>,
        even: V128<T>,
    ) -> V128<T> {
        // SSE2: only 1 block (128-bit), block 0 = even.
        even
    }

    #[inline(always)]
    fn reverse_blocks<T: Lane>(self, v: V128<T>) -> V128<T> {
        // SSE2: only 1 block, nothing to reverse.
        v
    }

    #[inline(always)]
    fn compress_not<T: Lane>(self, v: V128<T>, mask: M128<T>) -> V128<T> {
        self.compress(v, self.not_mask(mask))
    }

    #[inline(always)]
    fn compress_blocks_not(self, v: V128<u64>, _mask: M128<u64>) -> V128<u64> {
        // Single 128-bit block, no-op
        v
    }

    #[inline(always)]
    fn broadcast_block<T: Lane, const IDX: usize>(self, v: V128<T>) -> V128<T> {
        // Single block, return as-is
        v
    }

    #[inline(always)]
    unsafe fn compress_bits<T: Lane>(self, v: V128<T>, bits: *const u8) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let byte_val = bits.read();
            // Build mask from bits
            let mut mask_arr = [0u8; 16];
            for i in 0..lanes {
                let bit_idx = i;
                let byte_idx = bit_idx / 8;
                let bit_in_byte = bit_idx % 8;
                let b = if byte_idx == 0 { byte_val } else { bits.add(byte_idx).read() };
                if (b >> bit_in_byte) & 1 != 0 {
                    for k in 0..T::BYTES {
                        mask_arr[i * T::BYTES + k] = 0xFF;
                    }
                }
            }
            let mask = M128::from_raw(_mm_loadu_si128(mask_arr.as_ptr().cast()));
            self.compress(v, mask)
        }
    }

    #[inline(always)]
    unsafe fn compress_bits_store<T: Lane>(self, v: V128<T>, bits: *const u8, ptr: *mut T) -> usize {
        unsafe {
            let lanes = 16 / T::BYTES;
            let byte_val = bits.read();
            let mut mask_arr = [0u8; 16];
            for i in 0..lanes {
                let bit_idx = i;
                let byte_idx = bit_idx / 8;
                let bit_in_byte = bit_idx % 8;
                let b = if byte_idx == 0 { byte_val } else { bits.add(byte_idx).read() };
                if (b >> bit_in_byte) & 1 != 0 {
                    for k in 0..T::BYTES {
                        mask_arr[i * T::BYTES + k] = 0xFF;
                    }
                }
            }
            let mask = M128::from_raw(_mm_loadu_si128(mask_arr.as_ptr().cast()));
            // compress_store recomputes and returns the written count.
            self.compress_store(v, mask, ptr)
        }
    }

    #[inline(always)]
    fn lower_half<T: Lane>(self, v: V128<T>) -> V128<T> {
        // SSE2: half = full (identity)
        v
    }

    #[inline(always)]
    fn upper_half<T: Lane>(self, _v: V128<T>) -> V128<T> {
        // SSE2: no upper half, return zero
        unsafe { V128::from_raw(_mm_setzero_si128()) }
    }

    #[inline(always)]
    fn combine<T: Lane>(self, lo: V128<T>, _hi: V128<T>) -> V128<T> {
        // SSE2: half = full, return lo
        lo
    }

    #[inline(always)]
    fn insert_block<T: Lane, const IDX: usize>(self, _v: V128<T>, blk: V128<T>) -> V128<T> {
        // SSE2: single block, IDX must be 0
        blk
    }

    #[inline(always)]
    fn extract_block<T: Lane, const IDX: usize>(self, v: V128<T>) -> V128<T> {
        // SSE2: single block
        v
    }

    #[inline(always)]
    fn interleave_whole_lower<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // SSE2: single block, same as interleave_lower
        self.interleave_lower(a, b)
    }

    #[inline(always)]
    fn interleave_whole_upper<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // SSE2: single block, same as interleave_upper
        self.interleave_upper(a, b)
    }

    #[inline(always)]
    fn interleave_even<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // Take even lanes and interleave: [a0, b0, a2, b2, ...]
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr_a = [0u8; 16];
            let mut arr_b = [0u8; 16];
            let mut result = [0u8; 16];
            _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
            _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
            let mut dst = 0;
            let mut src_idx = 0;
            while src_idx < lanes {
                // copy a[src_idx]
                let off = src_idx * T::BYTES;
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_a[off..off + T::BYTES]);
                dst += 1;
                // copy b[src_idx]
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_b[off..off + T::BYTES]);
                dst += 1;
                src_idx += 2;
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn interleave_odd<T: Lane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // Take odd lanes and interleave: [a1, b1, a3, b3, ...]
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr_a = [0u8; 16];
            let mut arr_b = [0u8; 16];
            let mut result = [0u8; 16];
            _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
            _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
            let mut dst = 0;
            let mut src_idx = 1;
            while src_idx < lanes {
                // copy a[src_idx]
                let off = src_idx * T::BYTES;
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_a[off..off + T::BYTES]);
                dst += 1;
                // copy b[src_idx]
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_b[off..off + T::BYTES]);
                dst += 1;
                src_idx += 2;
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn two_tables_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        a: V128<T>,
        b: V128<T>,
        idx: V128<I>,
    ) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr_a = [0u8; 16];
            let mut arr_b = [0u8; 16];
            let mut arr_idx = [0u8; 16];
            let mut result = [0u8; 16];
            _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
            _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
            _mm_storeu_si128(arr_idx.as_mut_ptr().cast(), idx.raw);
            for i in 0..lanes {
                let idx_off = i * I::BYTES;
                let lane_idx: usize = match I::BYTES {
                    1 => arr_idx[idx_off] as usize,
                    2 => u16::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1]]) as usize,
                    4 => u32::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3]]) as usize,
                    _ => u64::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3], arr_idx[idx_off+4], arr_idx[idx_off+5], arr_idx[idx_off+6], arr_idx[idx_off+7]]) as usize,
                };
                let src_off = if lane_idx < lanes {
                    lane_idx * T::BYTES
                } else {
                    (lane_idx - lanes) * T::BYTES
                };
                let src = if lane_idx < lanes { &arr_a } else { &arr_b };
                let dst_off = i * T::BYTES;
                if lane_idx < 2 * lanes {
                    result[dst_off..dst_off+T::BYTES].copy_from_slice(&src[src_off..src_off+T::BYTES]);
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn table_lookup_lanes_or0<T: Lane, I: IntegerLane>(
        self,
        v: V128<T>,
        idx: V128<I>,
    ) -> V128<T> {
        unsafe {
            let lanes = 16 / T::BYTES;
            let mut arr_v = [0u8; 16];
            let mut arr_idx = [0u8; 16];
            let mut result = [0u8; 16];
            _mm_storeu_si128(arr_v.as_mut_ptr().cast(), v.raw);
            _mm_storeu_si128(arr_idx.as_mut_ptr().cast(), idx.raw);
            for i in 0..lanes {
                let idx_off = i * I::BYTES;
                // Read index as signed to check high bit
                let lane_idx_signed: i64 = match I::BYTES {
                    1 => arr_idx[idx_off] as i8 as i64,
                    2 => i16::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1]]) as i64,
                    4 => i32::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3]]) as i64,
                    _ => i64::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3], arr_idx[idx_off+4], arr_idx[idx_off+5], arr_idx[idx_off+6], arr_idx[idx_off+7]]),
                };
                let dst_off = i * T::BYTES;
                if lane_idx_signed < 0 {
                    // Zero
                    for k in 0..T::BYTES {
                        result[dst_off + k] = 0;
                    }
                } else {
                    let lane_idx = lane_idx_signed as usize;
                    if lane_idx < lanes {
                        let src_off = lane_idx * T::BYTES;
                        result[dst_off..dst_off+T::BYTES].copy_from_slice(&arr_v[src_off..src_off+T::BYTES]);
                    } else {
                        for k in 0..T::BYTES {
                            result[dst_off + k] = 0;
                        }
                    }
                }
            }
            V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
        }
    }
}

// ---------------------------------------------------------------------------
// SimdReduce
// ---------------------------------------------------------------------------

// SAFETY: All operations use SSE2 intrinsics.
unsafe impl SimdReduce for Sse2 {
    #[inline(always)]
    fn sum_of_lanes<T: Lane>(self, v: V128<T>) -> T {
        unsafe {
            // SIMD horizontal reduction via shuffle + add.
            // Each step adds the upper half to the lower half, halving the
            // number of live lanes until one remains.
            let mut r = v.raw;
            match T::BYTES {
                1 => {
                    // 16 lanes -> 8 -> 4 -> 2 -> 1
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 8));
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 4));
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 2));
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 1));
                }
                2 => {
                    // 8 lanes -> 4 -> 2 -> 1
                    r = _mm_add_epi16(r, _mm_srli_si128(r, 8));
                    r = _mm_add_epi16(r, _mm_srli_si128(r, 4));
                    r = _mm_add_epi16(r, _mm_srli_si128(r, 2));
                }
                4 => {
                    if is_type::<T, f32>() {
                        let mut f = _mm_castsi128_ps(r);
                        f = _mm_add_ps(f, _mm_movehl_ps(f, f));
                        f = _mm_add_ss(f, _mm_shuffle_ps(f, f, 1));
                        r = _mm_castps_si128(f);
                    } else {
                        // 4 lanes -> 2 -> 1
                        r = _mm_add_epi32(r, _mm_srli_si128(r, 8));
                        r = _mm_add_epi32(r, _mm_srli_si128(r, 4));
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        let mut f = _mm_castsi128_pd(r);
                        f = _mm_add_pd(f, _mm_shuffle_pd(f, f, 1));
                        r = _mm_castpd_si128(f);
                    } else {
                        // 2 lanes -> 1
                        r = _mm_add_epi64(r, _mm_srli_si128(r, 8));
                    }
                }
                _ => unreachable!(),
            }
            // Extract lane 0
            core::mem::transmute_copy(&_mm_cvtsi128_si64(r))
        }
    }

    #[inline(always)]
    fn min_of_lanes<T: Lane>(self, v: V128<T>) -> T {
        unsafe {
            // Tree reduction: shuffle upper half down, min with lower, repeat.
            let mut r = v;
            match T::BYTES {
                1 => {
                    r = self.min(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                    r = self.min(r, V128::from_raw(_mm_srli_si128::<4>(r.raw)));
                    r = self.min(r, V128::from_raw(_mm_srli_si128::<2>(r.raw)));
                    r = self.min(r, V128::from_raw(_mm_srli_si128::<1>(r.raw)));
                }
                2 => {
                    r = self.min(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                    r = self.min(r, V128::from_raw(_mm_srli_si128::<4>(r.raw)));
                    r = self.min(r, V128::from_raw(_mm_srli_si128::<2>(r.raw)));
                }
                4 => {
                    if is_type::<T, f32>() {
                        let mut f = _mm_castsi128_ps(r.raw);
                        f = _mm_min_ps(f, _mm_movehl_ps(f, f));
                        f = _mm_min_ss(f, _mm_shuffle_ps(f, f, 1));
                        r = V128::from_raw(_mm_castps_si128(f));
                    } else {
                        r = self.min(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                        r = self.min(r, V128::from_raw(_mm_srli_si128::<4>(r.raw)));
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        let f = _mm_castsi128_pd(r.raw);
                        let hi = _mm_shuffle_pd(f, f, 1);
                        r = V128::from_raw(_mm_castpd_si128(_mm_min_pd(f, hi)));
                    } else {
                        r = self.min(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                    }
                }
                _ => unreachable!(),
            }
            core::mem::transmute_copy(&_mm_cvtsi128_si64(r.raw))
        }
    }

    #[inline(always)]
    fn max_of_lanes<T: Lane>(self, v: V128<T>) -> T {
        unsafe {
            let mut r = v;
            match T::BYTES {
                1 => {
                    r = self.max(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                    r = self.max(r, V128::from_raw(_mm_srli_si128::<4>(r.raw)));
                    r = self.max(r, V128::from_raw(_mm_srli_si128::<2>(r.raw)));
                    r = self.max(r, V128::from_raw(_mm_srli_si128::<1>(r.raw)));
                }
                2 => {
                    r = self.max(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                    r = self.max(r, V128::from_raw(_mm_srli_si128::<4>(r.raw)));
                    r = self.max(r, V128::from_raw(_mm_srli_si128::<2>(r.raw)));
                }
                4 => {
                    if is_type::<T, f32>() {
                        let mut f = _mm_castsi128_ps(r.raw);
                        f = _mm_max_ps(f, _mm_movehl_ps(f, f));
                        f = _mm_max_ss(f, _mm_shuffle_ps(f, f, 1));
                        r = V128::from_raw(_mm_castps_si128(f));
                    } else {
                        r = self.max(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                        r = self.max(r, V128::from_raw(_mm_srli_si128::<4>(r.raw)));
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        let f = _mm_castsi128_pd(r.raw);
                        let hi = _mm_shuffle_pd(f, f, 1);
                        r = V128::from_raw(_mm_castpd_si128(_mm_max_pd(f, hi)));
                    } else {
                        r = self.max(r, V128::from_raw(_mm_srli_si128::<8>(r.raw)));
                    }
                }
                _ => unreachable!(),
            }
            core::mem::transmute_copy(&_mm_cvtsi128_si64(r.raw))
        }
    }

    #[inline(always)]
    fn sums_of_8_abs_diff(
        self,
        a: V128<u8>,
        b: V128<u8>,
    ) -> V128<u64> {
        V128::from_raw(unsafe { _mm_sad_epu8(a.raw, b.raw) })
    }

    #[inline(always)]
    fn sums_of_2<T: NarrowLane>(self, v: V128<T>) -> V128<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    if is_signed::<T>() {
                        // i8 -> i16: sign-extend pairs and add
                        let even = _mm_srai_epi16(_mm_slli_epi16(v.raw, 8), 8);
                        let odd = _mm_srai_epi16(v.raw, 8);
                        _mm_add_epi16(even, odd)
                    } else {
                        // u8 -> u16: mask even, shift odd right, add as u16
                        let even = _mm_and_si128(v.raw, _mm_set1_epi16(0x00FF));
                        let odd = _mm_srli_epi16(v.raw, 8);
                        _mm_add_epi16(even, odd)
                    }
                }
                2 => {
                    if is_signed::<T>() {
                        // i16 -> i32: use PMADDWD with all-ones (pairwise add)
                        _mm_madd_epi16(v.raw, _mm_set1_epi16(1))
                    } else {
                        // u16 -> u32: mask even 16-bit, shift odd right, add as u32
                        let mask = _mm_set1_epi32(0x0000FFFFu32 as i32);
                        let even = _mm_and_si128(v.raw, mask);
                        let odd = _mm_srli_epi32(v.raw, 16);
                        _mm_add_epi32(even, odd)
                    }
                }
                4 => {
                    if is_type::<T, f32>() {
                        // f32 -> f64: convert pairs to f64 and add
                        let ps = _mm_castsi128_ps(v.raw);
                        let lo_pd = _mm_cvtps_pd(ps); // [v0 as f64, v1 as f64]
                        let hi_ps = _mm_castsi128_ps(_mm_srli_si128::<8>(v.raw));
                        let hi_pd = _mm_cvtps_pd(hi_ps); // [v2 as f64, v3 as f64]
                        // Add adjacent: result[0] = lo[0]+lo[1], result[1] = hi[0]+hi[1]
                        let lo_swap = _mm_shuffle_pd(lo_pd, lo_pd, 1); // [v1, v0]
                        let lo_sum = _mm_add_pd(lo_pd, lo_swap); // [v0+v1, v1+v0]
                        let hi_swap = _mm_shuffle_pd(hi_pd, hi_pd, 1);
                        let hi_sum = _mm_add_pd(hi_pd, hi_swap);
                        _mm_castpd_si128(_mm_unpacklo_pd(lo_sum, hi_sum))
                    } else if is_signed::<T>() {
                        // i32 -> i64: scalar fallback for sign extension
                        let mut arr = [0i32; 4];
                        _mm_storeu_si128(arr.as_mut_ptr().cast(), v.raw);
                        let r0 = (arr[0] as i64) + (arr[1] as i64);
                        let r1 = (arr[2] as i64) + (arr[3] as i64);
                        _mm_set_epi64x(r1, r0)
                    } else {
                        // u32 -> u64: mask even u32, shift odd right, add as u64
                        let mask = _mm_set1_epi64x(0x00000000FFFFFFFFi64);
                        let even = _mm_and_si128(v.raw, mask);
                        let odd = _mm_srli_epi64(v.raw, 32);
                        _mm_add_epi64(even, odd)
                    }
                }
                _ => v.raw,
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn sums_of_4<T: NarrowLane>(
        self,
        v: V128<T>,
    ) -> V128<<T::Wide as NarrowLane>::Wide>
    where
        T::Wide: NarrowLane + Lane,
        <T::Wide as NarrowLane>::Wide: Lane,
    {
        {
            let mid = self.sums_of_2(v);
            self.sums_of_2(mid)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdFloat
// ---------------------------------------------------------------------------

// SAFETY: All float ops use SSE2 intrinsics.
unsafe impl SimdFloat for Sse2 {
    #[inline(always)]
    fn sqrt<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm_castps_si128(_mm_sqrt_ps(_mm_castsi128_ps(v.raw)))
            } else {
                _mm_castpd_si128(_mm_sqrt_pd(_mm_castsi128_pd(v.raw)))
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn approx_reciprocal<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            if T::BYTES == 4 {
                V128::from_raw(_mm_castps_si128(_mm_rcp_ps(_mm_castsi128_ps(v.raw))))
            } else {
                // No rcp for f64; use 1.0/x
                let ones = _mm_set1_pd(1.0);
                V128::from_raw(_mm_castpd_si128(_mm_div_pd(ones, _mm_castsi128_pd(v.raw))))
            }
        }
    }

    #[inline(always)]
    fn approx_reciprocal_sqrt<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            if T::BYTES == 4 {
                V128::from_raw(_mm_castps_si128(_mm_rsqrt_ps(_mm_castsi128_ps(v.raw))))
            } else {
                let ones = _mm_set1_pd(1.0);
                let sq = _mm_sqrt_pd(_mm_castsi128_pd(v.raw));
                V128::from_raw(_mm_castpd_si128(_mm_div_pd(ones, sq)))
            }
        }
    }

    #[inline(always)]
    fn round<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            if T::BYTES == 4 {
                // SSE2 doesn't have _mm_round_ps; emulate using cvt
                let i = _mm_cvtps_epi32(_mm_castsi128_ps(v.raw));
                V128::from_raw(_mm_castps_si128(_mm_cvtepi32_ps(i)))
            } else {
                // Round to nearest even: add copysign(2^52, v), then subtract it back.
                // Values with |v| >= 2^52 are already integers.
                let v_pd = _mm_castsi128_pd(v.raw);
                let neg_zero = _mm_set1_pd(-0.0);
                let sign = _mm_and_pd(v_pd, neg_zero);
                let magic = _mm_or_pd(_mm_set1_pd(4503599627370496.0), sign); // copysign(2^52, v)
                let rounded = _mm_sub_pd(_mm_add_pd(v_pd, magic), magic);
                // Guard: |v| >= 2^52 -> return v unchanged
                let abs_v = _mm_andnot_pd(neg_zero, v_pd);
                let in_range = _mm_cmplt_pd(abs_v, _mm_set1_pd(4503599627370496.0));
                V128::from_raw(_mm_castpd_si128(_mm_or_pd(
                    _mm_and_pd(in_range, rounded),
                    _mm_andnot_pd(in_range, v_pd),
                )))
            }
        }
    }

    #[inline(always)]
    fn trunc<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            if T::BYTES == 4 {
                // Truncate: convert to int and back
                let i = _mm_cvttps_epi32(_mm_castsi128_ps(v.raw));
                V128::from_raw(_mm_castps_si128(_mm_cvtepi32_ps(i)))
            } else {
                // Build trunc from round: if |rounded| > |v|, subtract copysign(1, v).
                let v_pd = _mm_castsi128_pd(v.raw);
                let neg_zero = _mm_set1_pd(-0.0);
                let sign = _mm_and_pd(v_pd, neg_zero);
                let magic = _mm_or_pd(_mm_set1_pd(4503599627370496.0), sign);
                let rounded = _mm_sub_pd(_mm_add_pd(v_pd, magic), magic);
                // Fix: if |rounded| > |v|, we rounded away from zero
                let abs_rounded = _mm_andnot_pd(neg_zero, rounded);
                let abs_v = _mm_andnot_pd(neg_zero, v_pd);
                let too_big = _mm_cmpgt_pd(abs_rounded, abs_v);
                let fix = _mm_and_pd(too_big, _mm_or_pd(_mm_set1_pd(1.0), sign));
                let trunc = _mm_sub_pd(rounded, fix);
                // Guard: |v| >= 2^52 -> return v unchanged
                let in_range = _mm_cmplt_pd(abs_v, _mm_set1_pd(4503599627370496.0));
                V128::from_raw(_mm_castpd_si128(_mm_or_pd(
                    _mm_and_pd(in_range, trunc),
                    _mm_andnot_pd(in_range, v_pd),
                )))
            }
        }
    }

    #[inline(always)]
    fn ceil<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            if T::BYTES == 4 {
                // ceil via truncate + conditional +1
                let v_ps = _mm_castsi128_ps(v.raw);
                let neg_zero = _mm_set1_ps(-0.0f32);
                let trunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_ps));
                let needs_up = _mm_cmplt_ps(trunc, v_ps);
                let one = _mm_set1_ps(1.0f32);
                let ceil = _mm_add_ps(trunc, _mm_and_ps(needs_up, one));
                // Guard: |v| >= 2^23 -> already integer
                let abs_v = _mm_andnot_ps(neg_zero, v_ps);
                let in_range = _mm_cmplt_ps(abs_v, _mm_set1_ps(8388608.0));
                V128::from_raw(_mm_castps_si128(_mm_or_ps(
                    _mm_and_ps(in_range, ceil),
                    _mm_andnot_ps(in_range, v_ps),
                )))
            } else {
                // ceil from round: if rounded < v, add 1
                let v_pd = _mm_castsi128_pd(v.raw);
                let neg_zero = _mm_set1_pd(-0.0);
                let sign = _mm_and_pd(v_pd, neg_zero);
                let magic = _mm_or_pd(_mm_set1_pd(4503599627370496.0), sign);
                let rounded = _mm_sub_pd(_mm_add_pd(v_pd, magic), magic);
                let needs_up = _mm_cmplt_pd(rounded, v_pd);
                let ceil = _mm_add_pd(rounded, _mm_and_pd(needs_up, _mm_set1_pd(1.0)));
                // Guard: |v| >= 2^52 -> return v unchanged
                let abs_v = _mm_andnot_pd(neg_zero, v_pd);
                let in_range = _mm_cmplt_pd(abs_v, _mm_set1_pd(4503599627370496.0));
                V128::from_raw(_mm_castpd_si128(_mm_or_pd(
                    _mm_and_pd(in_range, ceil),
                    _mm_andnot_pd(in_range, v_pd),
                )))
            }
        }
    }

    #[inline(always)]
    fn floor<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        unsafe {
            if T::BYTES == 4 {
                // floor via truncate + conditional -1
                let v_ps = _mm_castsi128_ps(v.raw);
                let neg_zero = _mm_set1_ps(-0.0f32);
                let trunc = _mm_cvtepi32_ps(_mm_cvttps_epi32(v_ps));
                let needs_down = _mm_cmpgt_ps(trunc, v_ps);
                let one = _mm_set1_ps(1.0f32);
                let floor = _mm_sub_ps(trunc, _mm_and_ps(needs_down, one));
                // Guard: |v| >= 2^23 -> already integer
                let abs_v = _mm_andnot_ps(neg_zero, v_ps);
                let in_range = _mm_cmplt_ps(abs_v, _mm_set1_ps(8388608.0));
                V128::from_raw(_mm_castps_si128(_mm_or_ps(
                    _mm_and_ps(in_range, floor),
                    _mm_andnot_ps(in_range, v_ps),
                )))
            } else {
                // floor from round: if rounded > v, subtract 1
                let v_pd = _mm_castsi128_pd(v.raw);
                let neg_zero = _mm_set1_pd(-0.0);
                let sign = _mm_and_pd(v_pd, neg_zero);
                let magic = _mm_or_pd(_mm_set1_pd(4503599627370496.0), sign);
                let rounded = _mm_sub_pd(_mm_add_pd(v_pd, magic), magic);
                let needs_down = _mm_cmpgt_pd(rounded, v_pd);
                let floor = _mm_sub_pd(rounded, _mm_and_pd(needs_down, _mm_set1_pd(1.0)));
                // Guard: |v| >= 2^52 -> return v unchanged
                let abs_v = _mm_andnot_pd(neg_zero, v_pd);
                let in_range = _mm_cmplt_pd(abs_v, _mm_set1_pd(4503599627370496.0));
                V128::from_raw(_mm_castpd_si128(_mm_or_pd(
                    _mm_and_pd(in_range, floor),
                    _mm_andnot_pd(in_range, v_pd),
                )))
            }
        }
    }

    #[inline(always)]
    fn mul_add<T: FloatLane>(self, a: V128<T>, b: V128<T>, c: V128<T>) -> V128<T> {
        unsafe {
            // SSE2 doesn't have FMA; emulate as a*b + c
            if T::BYTES == 4 {
                let prod = _mm_mul_ps(_mm_castsi128_ps(a.raw), _mm_castsi128_ps(b.raw));
                V128::from_raw(_mm_castps_si128(_mm_add_ps(prod, _mm_castsi128_ps(c.raw))))
            } else {
                let prod = _mm_mul_pd(_mm_castsi128_pd(a.raw), _mm_castsi128_pd(b.raw));
                V128::from_raw(_mm_castpd_si128(_mm_add_pd(prod, _mm_castsi128_pd(c.raw))))
            }
        }
    }

    #[inline(always)]
    fn neg_mul_add<T: FloatLane>(self, a: V128<T>, b: V128<T>, c: V128<T>) -> V128<T> {
        unsafe {
            // -(a*b) + c = c - a*b
            if T::BYTES == 4 {
                let prod = _mm_mul_ps(_mm_castsi128_ps(a.raw), _mm_castsi128_ps(b.raw));
                V128::from_raw(_mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(c.raw), prod)))
            } else {
                let prod = _mm_mul_pd(_mm_castsi128_pd(a.raw), _mm_castsi128_pd(b.raw));
                V128::from_raw(_mm_castpd_si128(_mm_sub_pd(_mm_castsi128_pd(c.raw), prod)))
            }
        }
    }

    #[inline(always)]
    fn mul_sub<T: FloatLane>(self, a: V128<T>, b: V128<T>, c: V128<T>) -> V128<T> {
        unsafe {
            // a*b - c
            if T::BYTES == 4 {
                let prod = _mm_mul_ps(_mm_castsi128_ps(a.raw), _mm_castsi128_ps(b.raw));
                V128::from_raw(_mm_castps_si128(_mm_sub_ps(prod, _mm_castsi128_ps(c.raw))))
            } else {
                let prod = _mm_mul_pd(_mm_castsi128_pd(a.raw), _mm_castsi128_pd(b.raw));
                V128::from_raw(_mm_castpd_si128(_mm_sub_pd(prod, _mm_castsi128_pd(c.raw))))
            }
        }
    }

    #[inline(always)]
    fn neg_mul_sub<T: FloatLane>(self, a: V128<T>, b: V128<T>, c: V128<T>) -> V128<T> {
        unsafe {
            // -(a*b) - c
            if T::BYTES == 4 {
                let prod = _mm_mul_ps(_mm_castsi128_ps(a.raw), _mm_castsi128_ps(b.raw));
                let neg_prod = _mm_sub_ps(_mm_setzero_ps(), prod);
                V128::from_raw(_mm_castps_si128(_mm_sub_ps(
                    neg_prod,
                    _mm_castsi128_ps(c.raw),
                )))
            } else {
                let prod = _mm_mul_pd(_mm_castsi128_pd(a.raw), _mm_castsi128_pd(b.raw));
                let neg_prod = _mm_sub_pd(_mm_setzero_pd(), prod);
                V128::from_raw(_mm_castpd_si128(_mm_sub_pd(
                    neg_prod,
                    _mm_castsi128_pd(c.raw),
                )))
            }
        }
    }

    #[inline(always)]
    fn copy_sign<T: FloatLane>(self, mag: V128<T>, sign: V128<T>) -> V128<T> {
        unsafe {
            if T::BYTES == 4 {
                let sign_mask = _mm_set1_epi32(0x8000_0000u32 as i32);
                let abs_mag = _mm_andnot_si128(sign_mask, mag.raw);
                let sign_bit = _mm_and_si128(sign_mask, sign.raw);
                V128::from_raw(_mm_or_si128(abs_mag, sign_bit))
            } else {
                let sign_mask = _mm_set_epi64x(
                    0x8000_0000_0000_0000u64 as i64,
                    0x8000_0000_0000_0000u64 as i64,
                );
                let abs_mag = _mm_andnot_si128(sign_mask, mag.raw);
                let sign_bit = _mm_and_si128(sign_mask, sign.raw);
                V128::from_raw(_mm_or_si128(abs_mag, sign_bit))
            }
        }
    }

    #[inline(always)]
    fn is_nan<T: FloatLane>(self, v: V128<T>) -> M128<T> {
        unsafe {
            // NaN: unordered compare with itself
            let raw = if T::BYTES == 4 {
                let ps = _mm_castsi128_ps(v.raw);
                _mm_castps_si128(_mm_cmpunord_ps(ps, ps))
            } else {
                let pd = _mm_castsi128_pd(v.raw);
                _mm_castpd_si128(_mm_cmpunord_pd(pd, pd))
            };
            M128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn is_inf<T: FloatLane>(self, v: V128<T>) -> M128<T> {
        unsafe {
            // Inf: abs(v) == max_finite + 1 bit... or compare to inf constant
            if T::BYTES == 4 {
                let abs_mask = _mm_set1_epi32(0x7FFF_FFFFu32 as i32);
                let inf_bits = _mm_set1_epi32(0x7F80_0000u32 as i32);
                let abs_v = _mm_and_si128(v.raw, abs_mask);
                M128::from_raw(_mm_cmpeq_epi32(abs_v, inf_bits))
            } else {
                let abs_mask = _mm_set_epi64x(
                    0x7FFF_FFFF_FFFF_FFFFu64 as i64,
                    0x7FFF_FFFF_FFFF_FFFFu64 as i64,
                );
                let inf_bits = _mm_set_epi64x(
                    0x7FF0_0000_0000_0000u64 as i64,
                    0x7FF0_0000_0000_0000u64 as i64,
                );
                let abs_v = _mm_and_si128(v.raw, abs_mask);
                // 64-bit eq: check both 32-bit halves
                let eq32 = _mm_cmpeq_epi32(abs_v, inf_bits);
                let eq32_hi = _mm_shuffle_epi32(eq32, 0xB1);
                M128::from_raw(_mm_and_si128(eq32, eq32_hi))
            }
        }
    }

    #[inline(always)]
    fn zero_if_negative<T: FloatLane>(self, v: V128<T>) -> V128<T> {
        // AndNot(BroadcastSignBit(v), v): zero lanes where sign bit is set.
        unsafe {
            let raw = if T::BYTES == 4 {
                let sign = _mm_srai_epi32(v.raw, 31);
                _mm_andnot_si128(sign, v.raw)
            } else {
                // f64: srai_epi32 by 31, then broadcast the sign to both 32-bit halves
                let sign32 = _mm_srai_epi32(v.raw, 31);
                let sign64 = _mm_shuffle_epi32(sign32, 0xF5); // replicate high dword
                _mm_andnot_si128(sign64, v.raw)
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn is_finite<T: FloatLane>(self, v: V128<T>) -> M128<T> {
        unsafe {
            if T::BYTES == 4 {
                // Exponent field < 0xFF: shift away sign+mantissa, compare < 0xFF
                let shifted = _mm_srli_epi32(_mm_slli_epi32(v.raw, 1), 24);
                let max_exp = _mm_set1_epi32(0xFF);
                // SSE2 has no unsigned less-than. Use: exp < 0xFF <==> !(exp == 0xFF) AND ...
                // Actually: cmpgt(0xFF, exp) works since both are small positive values.
                M128::from_raw(_mm_cmplt_epi32(shifted, max_exp))
            } else {
                // f64: exponent is bits 52..62 (11 bits). Check < 0x7FF.
                // Shift away sign (slli 1), then srli 53 to isolate exponent.
                // SSE2 doesn't have slli_epi64 for 64-bit arithmetic shift,
                // so use mask approach: abs, then compare exponent field.
                let abs_mask = _mm_set_epi64x(
                    0x7FFF_FFFF_FFFF_FFFFu64 as i64,
                    0x7FFF_FFFF_FFFF_FFFFu64 as i64,
                );
                let abs_v = _mm_and_si128(v.raw, abs_mask);
                // Compare abs_v < inf_bits (0x7FF0_0000_0000_0000).
                // For 64-bit: finite iff abs_v < inf. Use scalar fallback.
                let inf_bits = _mm_set_epi64x(
                    0x7FF0_0000_0000_0000u64 as i64,
                    0x7FF0_0000_0000_0000u64 as i64,
                );
                // 64-bit less-than via: compare high 32 bits, then low 32 bits
                // abs_v < inf iff high32(abs_v) < high32(inf) OR
                //   (high32(abs_v) == high32(inf) AND low32(abs_v) < low32(inf))
                // Since inf low32 = 0, the second condition is never true.
                // So: finite iff high32(abs_v) < high32(inf).
                // high32(inf) = 0x7FF00000.
                let hi32_abs = _mm_srli_epi64(abs_v, 32);
                let hi32_inf = _mm_srli_epi64(inf_bits, 32);
                // Both values fit in i32 range (positive), so signed cmplt works.
                let lt32 = _mm_cmplt_epi32(hi32_abs, hi32_inf);
                // Broadcast the result to both 32-bit halves of each 64-bit lane.
                M128::from_raw(_mm_shuffle_epi32(lt32, 0xF5))
            }
        }
    }

    #[inline(always)]
    fn add_sub<T: FloatLane>(
        self,
        a: V128<T>,
        b: V128<T>,
    ) -> V128<T> {
        // Even lanes subtract, odd lanes add. No SSE3 addsub, so emulate.
        unsafe {
            let raw = if T::BYTES == 4 {
                // Negate even lanes of b by XOR with sign mask, then add.
                let sign_mask = _mm_castps_si128(_mm_set_ps(0.0, -0.0, 0.0, -0.0));
                let neg_even_b = _mm_xor_si128(b.raw, sign_mask);
                _mm_castps_si128(_mm_add_ps(
                    _mm_castsi128_ps(a.raw),
                    _mm_castsi128_ps(neg_even_b),
                ))
            } else {
                // f64: 2 lanes. Lane 0 = even (sub), lane 1 = odd (add).
                let sign_mask = _mm_castpd_si128(_mm_set_pd(0.0, -0.0));
                let neg_even_b = _mm_xor_si128(b.raw, sign_mask);
                _mm_castpd_si128(_mm_add_pd(
                    _mm_castsi128_pd(a.raw),
                    _mm_castsi128_pd(neg_even_b),
                ))
            };
            V128::from_raw(raw)
        }
    }

    #[inline(always)]
    fn min_number<T: FloatLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // IfThenElse(IsNaN(a), b, Min(a, b))
        {
            let nan_a = self.is_nan(a);
            let min_ab = self.min(a, b);
            self.if_then_else(nan_a, b, min_ab)
        }
    }

    #[inline(always)]
    fn max_number<T: FloatLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        // IfThenElse(IsNaN(a), b, Max(a, b))
        {
            let nan_a = self.is_nan(a);
            let max_ab = self.max(a, b);
            self.if_then_else(nan_a, b, max_ab)
        }
    }

    #[inline(always)]
    fn min_magnitude<T: FloatLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        {
            let abs_a = self.abs(a);
            let abs_b = self.abs(b);
            let abs_eq = self.eq(abs_a, abs_b);
            let abs_lt = self.lt(abs_a, abs_b);
            // If |a| == |b|, use min(a, b) for tie-breaking
            let eq_case = self.min(a, b);
            let sel = self.if_then_else(abs_eq, eq_case, b);
            self.if_then_else(abs_lt, a, sel)
        }
    }

    #[inline(always)]
    fn max_magnitude<T: FloatLane>(self, a: V128<T>, b: V128<T>) -> V128<T> {
        {
            let abs_a = self.abs(a);
            let abs_b = self.abs(b);
            let abs_eq = self.eq(abs_a, abs_b);
            let abs_gt = self.gt(abs_a, abs_b);
            // If |a| == |b|, use max(a, b) for tie-breaking
            let eq_case = self.max(a, b);
            let sel = self.if_then_else(abs_eq, eq_case, b);
            self.if_then_else(abs_gt, a, sel)
        }
    }

    #[inline(always)]
    fn is_either_nan<T: FloatLane>(self, a: V128<T>, b: V128<T>) -> M128<T> {
        self.or_mask(self.is_nan(a), self.is_nan(b))
    }
}

// ---------------------------------------------------------------------------
// Some utils
// ---------------------------------------------------------------------------

use crate::lane::is_type;

/// Check if a lane type is a signed type.
#[inline(always)]
const fn is_signed<T: Lane>() -> bool {
    is_type::<T, i8>() || is_type::<T, i16>() || is_type::<T, i32>() || is_type::<T, i64>()
}

/// Leading zero count for u32 lanes using the float-conversion trick.
///
/// Converts each u32 to f32 and extracts the biased exponent to determine
/// the position of the highest set bit. This is the standard SSE2 approach
#[inline]
unsafe fn lzc_u32(v: __m128i) -> __m128i {
    // SAFETY: all intrinsics below require SSE2, guaranteed by the caller.
    unsafe {
        // Normalize: clear bit at (MSB_pos - 24) to ensure float truncation
        // rounds down for values >= 2^24.
        let normalized = _mm_andnot_si128(_mm_srli_epi32(v, 24), v);
        // Convert to f32 via signed i32->f32 (single instruction on SSE2).
        // The signed conversion is correct because normalization clears enough
        // high bits; for negative interpretations the exponent is still valid.
        let as_f32 = _mm_cvtepi32_ps(normalized);
        let f32_bits = _mm_castps_si128(as_f32);
        // Extract biased exponent (bits 30..23).
        let biased_exp = _mm_srli_epi32(f32_bits, 23);
        // Clamp to 158 via i16 min (handles sign-bit leakage from negative floats).
        let clamped = _mm_min_epi16(biased_exp, _mm_set1_epi32(158));
        // lzc = 158 - biased_exp (where 158 = 127 + 31).
        let lzc = _mm_sub_epi32(_mm_set1_epi32(158), clamped);
        // Clamp to 32 for v=0 case (biased_exp=0 -> lzc=158, need 32).
        _mm_min_epu8(lzc, _mm_set1_epi32(32))
    }
}

/// Read lane bits as u64.
#[inline(always)]
fn read_lane_bits(arr: &[u8; 16], offset: usize, bytes: usize) -> u64 {
    let mut val = 0u64;
    unsafe {
        core::ptr::copy_nonoverlapping(
            arr.as_ptr().add(offset),
            core::ptr::from_mut(&mut val).cast::<u8>(),
            bytes,
        );
    }
    val
}

// ---------------------------------------------------------------------------
// SimdCrypto
// ---------------------------------------------------------------------------

// SAFETY: All AES/CLMul intrinsics are guarded by runtime feature detection.
// When hardware support is unavailable, the software fallback is used.
unsafe impl crate::ops::SimdCrypto for Sse2 {
    #[inline(always)]
    fn aes_round(self, state: V128<u8>, round_key: V128<u8>) -> V128<u8> {
        unsafe {
            if is_x86_feature_detected!("aes") {
                V128::from_raw(_mm_aesenc_si128(state.raw, round_key.raw))
            } else {
                let mut block = [0u8; 16];
                let mut key = [0u8; 16];
                _mm_storeu_si128(block.as_mut_ptr().cast(), state.raw);
                _mm_storeu_si128(key.as_mut_ptr().cast(), round_key.raw);
                super::crypto_soft::aes_round(&mut block, &key);
                V128::from_raw(_mm_loadu_si128(block.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn aes_last_round(self, state: V128<u8>, round_key: V128<u8>) -> V128<u8> {
        unsafe {
            if is_x86_feature_detected!("aes") {
                V128::from_raw(_mm_aesenclast_si128(state.raw, round_key.raw))
            } else {
                let mut block = [0u8; 16];
                let mut key = [0u8; 16];
                _mm_storeu_si128(block.as_mut_ptr().cast(), state.raw);
                _mm_storeu_si128(key.as_mut_ptr().cast(), round_key.raw);
                super::crypto_soft::aes_last_round(&mut block, &key);
                V128::from_raw(_mm_loadu_si128(block.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn aes_round_inv(self, state: V128<u8>, round_key: V128<u8>) -> V128<u8> {
        unsafe {
            if is_x86_feature_detected!("aes") {
                V128::from_raw(_mm_aesdec_si128(state.raw, round_key.raw))
            } else {
                let mut block = [0u8; 16];
                let mut key = [0u8; 16];
                _mm_storeu_si128(block.as_mut_ptr().cast(), state.raw);
                _mm_storeu_si128(key.as_mut_ptr().cast(), round_key.raw);
                super::crypto_soft::aes_round_inv(&mut block, &key);
                V128::from_raw(_mm_loadu_si128(block.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn aes_last_round_inv(self, state: V128<u8>, round_key: V128<u8>) -> V128<u8> {
        unsafe {
            if is_x86_feature_detected!("aes") {
                V128::from_raw(_mm_aesdeclast_si128(state.raw, round_key.raw))
            } else {
                let mut block = [0u8; 16];
                let mut key = [0u8; 16];
                _mm_storeu_si128(block.as_mut_ptr().cast(), state.raw);
                _mm_storeu_si128(key.as_mut_ptr().cast(), round_key.raw);
                super::crypto_soft::aes_last_round_inv(&mut block, &key);
                V128::from_raw(_mm_loadu_si128(block.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn cl_mul_lower(self, a: V128<u64>, b: V128<u64>) -> V128<u64> {
        unsafe {
            if is_x86_feature_detected!("pclmulqdq") {
                V128::from_raw(_mm_clmulepi64_si128(a.raw, b.raw, 0x00))
            } else {
                let mut arr_a = [0u64; 2];
                let mut arr_b = [0u64; 2];
                _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
                _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
                let (lo, hi) = super::crypto_soft::clmul_64(arr_a[0], arr_b[0]);
                let result = [lo, hi];
                V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn cl_mul_upper(self, a: V128<u64>, b: V128<u64>) -> V128<u64> {
        unsafe {
            if is_x86_feature_detected!("pclmulqdq") {
                V128::from_raw(_mm_clmulepi64_si128(a.raw, b.raw, 0x11))
            } else {
                let mut arr_a = [0u64; 2];
                let mut arr_b = [0u64; 2];
                _mm_storeu_si128(arr_a.as_mut_ptr().cast(), a.raw);
                _mm_storeu_si128(arr_b.as_mut_ptr().cast(), b.raw);
                let (lo, hi) = super::crypto_soft::clmul_64(arr_a[1], arr_b[1]);
                let result = [lo, hi];
                V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn aes_key_gen_assist<const RCON: i32>(self, v: V128<u8>) -> V128<u8> {
        unsafe {
            if is_x86_feature_detected!("aes") {
                V128::from_raw(_mm_aeskeygenassist_si128(v.raw, RCON))
            } else {
                let mut block = [0u8; 16];
                _mm_storeu_si128(block.as_mut_ptr().cast(), v.raw);
                let result = super::crypto_soft::aes_key_gen_assist(&block, RCON as u8);
                V128::from_raw(_mm_loadu_si128(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn aes_inv_mix_columns(self, v: V128<u8>) -> V128<u8> {
        unsafe {
            if is_x86_feature_detected!("aes") {
                V128::from_raw(_mm_aesimc_si128(v.raw))
            } else {
                let mut block = [0u8; 16];
                _mm_storeu_si128(block.as_mut_ptr().cast(), v.raw);
                super::crypto_soft::aes_inv_mix_columns_block(&mut block);
                V128::from_raw(_mm_loadu_si128(block.as_ptr().cast()))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_arch = "x86_64")]
mod tests {
    #![allow(unused_unsafe)]
    use super::*;

    fn has_sse2() -> bool {
        is_x86_feature_detected!("sse2")
    }

    #[test]
    fn test_sse2_add_i32() {
        if !has_sse2() {
            return;
        }
        let s = Sse2::new();
        unsafe {
            let a = s.splat::<i32>(10);
            let b = s.splat::<i32>(32);
            let c: V128<i32> = s.add(a, b);
            assert_eq!(s.extract_lane(c, 0), 42);
            assert_eq!(s.extract_lane(c, 1), 42);
            assert_eq!(s.extract_lane(c, 2), 42);
            assert_eq!(s.extract_lane(c, 3), 42);
        }
    }

    #[test]
    fn test_sse2_add_f32() {
        if !has_sse2() {
            return;
        }
        let s = Sse2::new();
        unsafe {
            let a = s.splat::<f32>(1.5);
            let b = s.splat::<f32>(2.5);
            let c: V128<f32> = s.add(a, b);
            let r: f32 = s.extract_lane(c, 0);
            assert!((r - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sse2_load_store() {
        if !has_sse2() {
            return;
        }
        let s = Sse2::new();
        unsafe {
            let data: [i32; 4] = [1, 2, 3, 4];
            let v: V128<i32> = s.load_u(data.as_ptr());
            assert_eq!(s.extract_lane(v, 0), 1);
            assert_eq!(s.extract_lane(v, 1), 2);
            assert_eq!(s.extract_lane(v, 2), 3);
            assert_eq!(s.extract_lane(v, 3), 4);

            let mut out = [0i32; 4];
            s.store_u(v, out.as_mut_ptr());
            assert_eq!(out, [1, 2, 3, 4]);
        }
    }

    #[test]
    fn test_sse2_compare() {
        if !has_sse2() {
            return;
        }
        let s = Sse2::new();
        unsafe {
            let a = s.splat::<i32>(5);
            let b = s.splat::<i32>(10);
            let lt_mask = s.lt::<i32>(a, b);
            assert!(s.all_true::<i32>(lt_mask));
            assert!(!s.all_false::<i32>(lt_mask));
        }
    }

    #[test]
    fn test_sse2_bitwise() {
        if !has_sse2() {
            return;
        }
        let s = Sse2::new();
        unsafe {
            let a = s.splat::<u32>(0xFF00FF00);
            let b = s.splat::<u32>(0x00FF00FF);
            let result: V128<u32> = s.and(a, b);
            assert_eq!(s.extract_lane(result, 0), 0u32);

            let result: V128<u32> = s.or(a, b);
            assert_eq!(s.extract_lane(result, 0), 0xFFFFFFFF);
        }
    }

    #[test]
    fn test_sse2_sum_of_lanes() {
        if !has_sse2() {
            return;
        }
        let s = Sse2::new();
        unsafe {
            let data: [i32; 4] = [1, 2, 3, 4];
            let v: V128<i32> = s.load_u(data.as_ptr());
            let sum: i32 = s.sum_of_lanes(v);
            assert_eq!(sum, 10);
        }
    }

    #[test]
    fn test_sse2_sqrt() {
        if !has_sse2() {
            return;
        }
        let s = Sse2::new();
        unsafe {
            let v = s.splat::<f32>(4.0);
            let sq: V128<f32> = s.sqrt(v);
            let r: f32 = s.extract_lane(sq, 0);
            assert!((r - 2.0).abs() < 1e-6);
        }
    }
}
