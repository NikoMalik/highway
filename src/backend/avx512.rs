// All unsafe blocks in this module wrap AVX-512 intrinsics or transmute_copy
// for type-punning. Safety invariants are documented on the outer `unsafe impl`
// blocks; individual intrinsic calls are safe when inputs are valid __m512i.
#![allow(clippy::undocumented_unsafe_blocks)]

/// AVX-512 backend.
///
/// Provides 512-bit SIMD operations via `core::arch::x86_64::_mm512_*` intrinsics.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::marker::PhantomData;

use crate::lane::{
    FloatLane, IntegerLane, Lane, NarrowLane, UnsignedLane, WideLane,
};
use crate::ops::{
    A64, Aligned, SimdArith, SimdBitwise, SimdCompare, SimdConvert, SimdCore,
    SimdFloat, SimdMask, SimdMemory, SimdReduce, SimdShuffle,
};
use crate::simd::{self, Simd};

// ---------------------------------------------------------------------------
// Target type
// ---------------------------------------------------------------------------

/// The AVX-512 SIMD target (512-bit vectors).
///
/// This token is a *proof* that the required AVX-512 features are available on
/// the running CPU: it cannot be constructed from safe code (the inner field is
/// private). It is handed to kernels only by the dispatch machinery after a
/// runtime feature check, which is why all AVX-512 vector operations can have
/// safe signatures.
#[derive(Clone, Copy, Debug)]
pub struct Avx512(());

impl Avx512 {
    /// Construct an AVX-512 token without checking CPU support.
    ///
    /// # Safety
    /// The caller must ensure AVX-512F/BW/CD/DQ/VL are available on the running
    /// CPU. Prefer obtaining a token through `dispatch`/`dispatch_to`, which
    /// checks at runtime.
    #[inline(always)]
    pub unsafe fn new_unchecked() -> Self {
        Avx512(())
    }

    #[inline(always)]
    pub(crate) fn new() -> Self {
        Avx512(())
    }
}

// ---------------------------------------------------------------------------
// Vector and Mask types
// ---------------------------------------------------------------------------

/// A 512-bit SIMD vector holding lanes of type `T`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct V512<T: Lane> {
    raw: __m512i,
    _marker: PhantomData<T>,
}

impl<T: Lane> core::fmt::Debug for V512<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("V512").finish_non_exhaustive()
    }
}

/// A 512-bit mask using AVX-512 bitmask representation.
/// Stores a `__mmask64` internally; only the lower `64 / T::BYTES` bits are used.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct M512<T: Lane> {
    raw: __mmask64,
    _marker: PhantomData<T>,
}

impl<T: Lane> V512<T> {
    #[inline(always)]
    pub(crate) fn from_raw(raw: __m512i) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }
}

impl<T: Lane> M512<T> {
    #[inline(always)]
    fn from_bits(raw: u64) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    fn from_raw(raw: __mmask64) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    #[inline(always)]
    const fn lane_count() -> usize {
        64 / T::BYTES
    }

    #[inline(always)]
    fn all_lanes_mask() -> u64 {
        if Self::lane_count() == 64 {
            u64::MAX
        } else {
            (1u64 << Self::lane_count()) - 1
        }
    }
}

// SAFETY: AVX-512 vectors are 512 bits = 64 bytes.
unsafe impl Simd for Avx512 {
    type Vec<T: Lane> = V512<T>;
    type Mask<T: Lane> = M512<T>;
    // Half-width of 512-bit is 256-bit (AVX2 types).
    type VecHalf<T: Lane> = crate::backend::avx2::V256<T>;
    type MaskHalf<T: Lane> = crate::backend::avx2::M256<T>;
    const VECTOR_BYTES: usize = 64;
}

// ---------------------------------------------------------------------------
// Utils
// ---------------------------------------------------------------------------

use crate::lane::is_type;

#[inline(always)]
const fn is_signed<T: Lane>() -> bool {
    is_type::<T, i8>()
        || is_type::<T, i16>()
        || is_type::<T, i32>()
        || is_type::<T, i64>()
}

// Native VPOPCNTDQ (32/64-bit) — available on Ice Lake+.
#[target_feature(enable = "avx512vpopcntdq")]
unsafe fn native_popcnt_epi32(v: __m512i) -> __m512i {
    _mm512_popcnt_epi32(v)
}
#[target_feature(enable = "avx512vpopcntdq")]
unsafe fn native_popcnt_epi64(v: __m512i) -> __m512i {
    _mm512_popcnt_epi64(v)
}

// Native BITALG (8/16-bit) — available on Ice Lake+.
#[target_feature(enable = "avx512bitalg")]
unsafe fn native_popcnt_epi8(v: __m512i) -> __m512i {
    _mm512_popcnt_epi8(v)
}
#[target_feature(enable = "avx512bitalg")]
unsafe fn native_popcnt_epi16(v: __m512i) -> __m512i {
    _mm512_popcnt_epi16(v)
}

// Native VBMI2 compress (8/16-bit) — available on Ice Lake+.
#[target_feature(enable = "avx512vbmi2")]
unsafe fn native_compress_epi8(k: __mmask64, v: __m512i) -> __m512i {
    _mm512_maskz_compress_epi8(k, v)
}
#[target_feature(enable = "avx512vbmi2")]
unsafe fn native_compress_epi16(k: __mmask32, v: __m512i) -> __m512i {
    _mm512_maskz_compress_epi16(k, v)
}
#[target_feature(enable = "avx512vbmi2")]
unsafe fn native_compressstoreu_epi8(ptr: *mut u8, k: __mmask64, v: __m512i) {
    unsafe {
        _mm512_mask_compressstoreu_epi8(ptr.cast(), k, v);
    }
}
#[target_feature(enable = "avx512vbmi2")]
unsafe fn native_compressstoreu_epi16(ptr: *mut u8, k: __mmask32, v: __m512i) {
    unsafe {
        _mm512_mask_compressstoreu_epi16(ptr.cast(), k, v);
    }
}

// Native VBMI permute (8-bit) — available on Ice Lake+.
#[target_feature(enable = "avx512vbmi")]
unsafe fn native_permutexvar_epi8(idx: __m512i, v: __m512i) -> __m512i {
    _mm512_permutexvar_epi8(idx, v)
}

#[inline(always)]
fn read_lane<T: Lane>(arr: &[u8; 64], offset: usize) -> T {
    let mut val = T::default();
    unsafe {
        core::ptr::copy_nonoverlapping(
            arr.as_ptr().add(offset),
            core::ptr::from_mut(&mut val).cast::<u8>(),
            T::BYTES,
        );
    }
    val
}

#[inline(always)]
fn write_lane<T: Lane>(arr: &mut [u8; 64], offset: usize, val: T) {
    unsafe {
        core::ptr::copy_nonoverlapping(
            core::ptr::from_ref(&val).cast::<u8>(),
            arr.as_mut_ptr().add(offset),
            T::BYTES,
        );
    }
}

// ---------------------------------------------------------------------------
// SimdCore
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX-512F+BW+DQ+VL, guaranteed by dispatch.
unsafe impl SimdCore for Avx512 {
    #[inline(always)]
    fn zero<T: Lane>(self) -> V512<T> {
        V512::from_raw(unsafe { _mm512_setzero_si512() })
    }

    #[inline(always)]
    fn splat<T: Lane>(self, value: T) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    let b: u8 = core::mem::transmute_copy(&value);
                    _mm512_set1_epi8(b as i8)
                }
                2 => {
                    let h: u16 = core::mem::transmute_copy(&value);
                    _mm512_set1_epi16(h as i16)
                }
                4 => {
                    let w: u32 = core::mem::transmute_copy(&value);
                    _mm512_set1_epi32(w as i32)
                }
                8 => {
                    let d: u64 = core::mem::transmute_copy(&value);
                    _mm512_set1_epi64(d as i64)
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn undefined<T: Lane>(self) -> V512<T> {
        V512::from_raw(unsafe { _mm512_setzero_si512() })
    }

    #[inline(always)]
    fn bitcast<T: Lane, U: Lane>(self, v: V512<T>) -> V512<U> {
        V512::from_raw(v.raw)
    }

    #[inline(always)]
    unsafe fn extract_lane<T: Lane>(self, v: V512<T>, index: usize) -> T {
        unsafe {
            let mut a: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            _mm512_store_si512(a.as_mut_ptr().cast(), v.raw);
            read_lane(a.as_ref(), index * T::BYTES)
        }
    }

    #[inline(always)]
    unsafe fn insert_lane<T: Lane>(
        self,
        v: V512<T>,
        index: usize,
        value: T,
    ) -> V512<T> {
        unsafe {
            let mut a: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            _mm512_store_si512(a.as_mut_ptr().cast(), v.raw);
            write_lane(a.as_mut(), index * T::BYTES, value);
            V512::from_raw(_mm512_load_si512(a.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn iota<T: Lane>(self, base: T) -> V512<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    let b: u8 = core::mem::transmute_copy(&base);
                    let iota = _mm512_set_epi8(
                        63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50,
                        49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36,
                        35, 34, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22,
                        21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8,
                        7, 6, 5, 4, 3, 2, 1, 0,
                    );
                    V512::from_raw(_mm512_add_epi8(
                        iota,
                        _mm512_set1_epi8(b as i8),
                    ))
                }
                2 => {
                    let h: u16 = core::mem::transmute_copy(&base);
                    let iota = _mm512_set_epi16(
                        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                        17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                        1, 0,
                    );
                    V512::from_raw(_mm512_add_epi16(
                        iota,
                        _mm512_set1_epi16(h as i16),
                    ))
                }
                4 => {
                    if is_type::<T, f32>() {
                        let b: f32 = core::mem::transmute_copy(&base);
                        let iota = _mm512_set_epi32(
                            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                            0,
                        );
                        V512::from_raw(_mm512_castps_si512(_mm512_add_ps(
                            _mm512_set1_ps(b),
                            _mm512_cvtepi32_ps(iota),
                        )))
                    } else {
                        let w: u32 = core::mem::transmute_copy(&base);
                        let iota = _mm512_set_epi32(
                            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                            0,
                        );
                        V512::from_raw(_mm512_add_epi32(
                            iota,
                            _mm512_set1_epi32(w as i32),
                        ))
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        let b: f64 = core::mem::transmute_copy(&base);
                        let iota_f64 = _mm512_set_pd(
                            7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0,
                        );
                        V512::from_raw(_mm512_castpd_si512(_mm512_add_pd(
                            _mm512_set1_pd(b),
                            iota_f64,
                        )))
                    } else {
                        let d: u64 = core::mem::transmute_copy(&base);
                        let iota = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
                        V512::from_raw(_mm512_add_epi64(
                            iota,
                            _mm512_set1_epi64(d as i64),
                        ))
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

// SAFETY: All intrinsics require AVX-512F.
unsafe impl SimdMemory for Avx512 {
    #[inline(always)]
    unsafe fn load<T: Lane>(self, ptr: *const T) -> V512<T> {
        V512::from_raw(unsafe { _mm512_load_si512(ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn load_u<T: Lane>(self, ptr: *const T) -> V512<T> {
        V512::from_raw(unsafe { _mm512_loadu_si512(ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn store<T: Lane>(self, v: V512<T>, ptr: *mut T) {
        unsafe { _mm512_store_si512(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn store_u<T: Lane>(self, v: V512<T>, ptr: *mut T) {
        unsafe { _mm512_storeu_si512(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn stream<T: Lane>(self, v: V512<T>, ptr: *mut T) {
        unsafe { _mm512_stream_si512(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn load_dup128<T: Lane>(self, ptr: *const T) -> V512<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm512_castps_si512(_mm512_broadcast_f32x4(_mm_loadu_ps(
                    ptr.cast(),
                )))
            } else if is_type::<T, f64>() {
                // 128 bits = 2 f64, broadcast to 4 copies
                _mm512_castpd_si512(_mm512_broadcast_f64x2(_mm_loadu_pd(
                    ptr.cast(),
                )))
            } else {
                _mm512_broadcast_i32x4(_mm_loadu_si128(ptr.cast()))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn masked_load<T: Lane>(
        self,
        mask: M512<T>,
        ptr: *const T,
    ) -> V512<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm512_castps_si512(_mm512_maskz_loadu_ps(
                    mask.raw as __mmask16,
                    ptr.cast(),
                ))
            } else if is_type::<T, f64>() {
                _mm512_castpd_si512(_mm512_maskz_loadu_pd(
                    mask.raw as __mmask8,
                    ptr.cast(),
                ))
            } else {
                match T::BYTES {
                    1 => _mm512_maskz_loadu_epi8(
                        mask.raw as __mmask64,
                        ptr.cast(),
                    ),
                    2 => _mm512_maskz_loadu_epi16(
                        mask.raw as __mmask32,
                        ptr.cast(),
                    ),
                    4 => _mm512_maskz_loadu_epi32(
                        mask.raw as __mmask16,
                        ptr.cast(),
                    ),
                    8 => _mm512_maskz_loadu_epi64(
                        mask.raw as __mmask8,
                        ptr.cast(),
                    ),
                    _ => unreachable!(),
                }
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn blended_store<T: Lane>(
        self,
        v: V512<T>,
        mask: M512<T>,
        ptr: *mut T,
    ) {
        unsafe {
            if is_type::<T, f32>() {
                _mm512_mask_storeu_ps(
                    ptr.cast(),
                    mask.raw as __mmask16,
                    _mm512_castsi512_ps(v.raw),
                );
            } else if is_type::<T, f64>() {
                _mm512_mask_storeu_pd(
                    ptr.cast(),
                    mask.raw as __mmask8,
                    _mm512_castsi512_pd(v.raw),
                );
            } else {
                match T::BYTES {
                    1 => _mm512_mask_storeu_epi8(
                        ptr.cast(),
                        mask.raw as __mmask64,
                        v.raw,
                    ),
                    2 => _mm512_mask_storeu_epi16(
                        ptr.cast(),
                        mask.raw as __mmask32,
                        v.raw,
                    ),
                    4 => _mm512_mask_storeu_epi32(
                        ptr.cast(),
                        mask.raw as __mmask16,
                        v.raw,
                    ),
                    8 => _mm512_mask_storeu_epi64(
                        ptr.cast(),
                        mask.raw as __mmask8,
                        v.raw,
                    ),
                    _ => unreachable!(),
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn gather_index<T: Lane>(
        self,
        base: *const T,
        idx: V512<i32>,
    ) -> V512<T> {
        unsafe {
            if is_type::<T, u32>() || is_type::<T, i32>() {
                V512::from_raw(_mm512_i32gather_epi32::<4>(
                    idx.raw,
                    base.cast::<i32>(),
                ))
            } else if is_type::<T, f32>() {
                V512::from_raw(_mm512_castps_si512(_mm512_i32gather_ps::<4>(
                    idx.raw,
                    base.cast::<f32>(),
                )))
            } else if is_type::<T, u64>() || is_type::<T, i64>() {
                let idx256 = _mm512_castsi512_si256(idx.raw);
                V512::from_raw(_mm512_i32gather_epi64::<8>(
                    idx256,
                    base.cast::<i64>(),
                ))
            } else if is_type::<T, f64>() {
                let idx256 = _mm512_castsi512_si256(idx.raw);
                V512::from_raw(_mm512_castpd_si512(_mm512_i32gather_pd::<8>(
                    idx256,
                    base.cast::<f64>(),
                )))
            } else {
                // u8/i8/u16/i16: scalar fallback
                // Only use as many indices as available (16 i32 slots in 512-bit vector)
                let lanes = (64 / T::BYTES).min(16);
                let mut idx_arr: Aligned<A64, [u8; 64]> =
                    Aligned::new([0u8; 64]);
                _mm512_store_si512(idx_arr.as_mut_ptr().cast(), idx.raw);
                let mut result: Aligned<A64, [u8; 64]> =
                    Aligned::new([0u8; 64]);
                for i in 0..lanes {
                    let index = read_lane::<i32>(&idx_arr, i * 4) as usize;
                    let val: T = *base.add(index);
                    write_lane(result.as_mut(), i * T::BYTES, val);
                }
                V512::from_raw(_mm512_load_si512(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn scatter_index<T: Lane>(
        self,
        v: V512<T>,
        base: *mut T,
        idx: V512<i32>,
    ) {
        unsafe {
            if is_type::<T, u32>() || is_type::<T, i32>() {
                _mm512_i32scatter_epi32::<4>(
                    base.cast::<i32>(),
                    idx.raw,
                    v.raw,
                );
            } else if is_type::<T, f32>() {
                _mm512_i32scatter_ps::<4>(
                    base.cast::<f32>(),
                    idx.raw,
                    _mm512_castsi512_ps(v.raw),
                );
            } else if is_type::<T, u64>() || is_type::<T, i64>() {
                let idx256 = _mm512_castsi512_si256(idx.raw);
                _mm512_i32scatter_epi64::<8>(base.cast::<i64>(), idx256, v.raw);
            } else if is_type::<T, f64>() {
                let idx256 = _mm512_castsi512_si256(idx.raw);
                _mm512_i32scatter_pd::<8>(
                    base.cast::<f64>(),
                    idx256,
                    _mm512_castsi512_pd(v.raw),
                );
            } else {
                // u8/i8/u16/i16: scalar fallback
                // Only use as many indices as available (16 i32 slots in 512-bit vector)
                let lanes = (64 / T::BYTES).min(16);
                let mut v_arr: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
                let mut idx_arr: Aligned<A64, [u8; 64]> =
                    Aligned::new([0u8; 64]);
                _mm512_store_si512(v_arr.as_mut_ptr().cast(), v.raw);
                _mm512_store_si512(idx_arr.as_mut_ptr().cast(), idx.raw);
                for i in 0..lanes {
                    let index = read_lane::<i32>(&idx_arr, i * 4) as usize;
                    let val: T = read_lane(&v_arr, i * T::BYTES);
                    *base.add(index) = val;
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_2<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V512<T>, V512<T>) {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut buf0: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut buf1: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let src = ptr.cast::<u8>();
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    src.add((i * 2) * T::BYTES),
                    buf0.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    src.add((i * 2 + 1) * T::BYTES),
                    buf1.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V512::from_raw(_mm512_load_si512(buf0.as_ptr().cast())),
                V512::from_raw(_mm512_load_si512(buf1.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_3<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V512<T>, V512<T>, V512<T>) {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut buf0: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut buf1: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut buf2: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let src = ptr.cast::<u8>();
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    src.add((i * 3) * T::BYTES),
                    buf0.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    src.add((i * 3 + 1) * T::BYTES),
                    buf1.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    src.add((i * 3 + 2) * T::BYTES),
                    buf2.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V512::from_raw(_mm512_load_si512(buf0.as_ptr().cast())),
                V512::from_raw(_mm512_load_si512(buf1.as_ptr().cast())),
                V512::from_raw(_mm512_load_si512(buf2.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_4<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V512<T>, V512<T>, V512<T>, V512<T>) {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut buf0: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut buf1: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut buf2: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut buf3: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let src = ptr.cast::<u8>();
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    src.add((i * 4) * T::BYTES),
                    buf0.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    src.add((i * 4 + 1) * T::BYTES),
                    buf1.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    src.add((i * 4 + 2) * T::BYTES),
                    buf2.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    src.add((i * 4 + 3) * T::BYTES),
                    buf3.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V512::from_raw(_mm512_load_si512(buf0.as_ptr().cast())),
                V512::from_raw(_mm512_load_si512(buf1.as_ptr().cast())),
                V512::from_raw(_mm512_load_si512(buf2.as_ptr().cast())),
                V512::from_raw(_mm512_load_si512(buf3.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_2<T: Lane>(
        self,
        v0: V512<T>,
        v1: V512<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut a0: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut a1: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            _mm512_store_si512(a0.as_mut_ptr().cast(), v0.raw);
            _mm512_store_si512(a1.as_mut_ptr().cast(), v1.raw);
            let dst = ptr.cast::<u8>();
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    a0.as_ptr().add(i * T::BYTES),
                    dst.add((i * 2) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    a1.as_ptr().add(i * T::BYTES),
                    dst.add((i * 2 + 1) * T::BYTES),
                    T::BYTES,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_3<T: Lane>(
        self,
        v0: V512<T>,
        v1: V512<T>,
        v2: V512<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut a0: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut a1: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut a2: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            _mm512_store_si512(a0.as_mut_ptr().cast(), v0.raw);
            _mm512_store_si512(a1.as_mut_ptr().cast(), v1.raw);
            _mm512_store_si512(a2.as_mut_ptr().cast(), v2.raw);
            let dst = ptr.cast::<u8>();
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    a0.as_ptr().add(i * T::BYTES),
                    dst.add((i * 3) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    a1.as_ptr().add(i * T::BYTES),
                    dst.add((i * 3 + 1) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    a2.as_ptr().add(i * T::BYTES),
                    dst.add((i * 3 + 2) * T::BYTES),
                    T::BYTES,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_4<T: Lane>(
        self,
        v0: V512<T>,
        v1: V512<T>,
        v2: V512<T>,
        v3: V512<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut a0: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut a1: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut a2: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            let mut a3: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
            _mm512_store_si512(a0.as_mut_ptr().cast(), v0.raw);
            _mm512_store_si512(a1.as_mut_ptr().cast(), v1.raw);
            _mm512_store_si512(a2.as_mut_ptr().cast(), v2.raw);
            _mm512_store_si512(a3.as_mut_ptr().cast(), v3.raw);
            let dst = ptr.cast::<u8>();
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    a0.as_ptr().add(i * T::BYTES),
                    dst.add((i * 4) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    a1.as_ptr().add(i * T::BYTES),
                    dst.add((i * 4 + 1) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    a2.as_ptr().add(i * T::BYTES),
                    dst.add((i * 4 + 2) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    a3.as_ptr().add(i * T::BYTES),
                    dst.add((i * 4 + 3) * T::BYTES),
                    T::BYTES,
                );
            }
        }
    }

    #[inline(always)]
    unsafe fn load_expand<T: Lane>(
        self,
        mask: M512<T>,
        ptr: *const T,
    ) -> V512<T> {
        // AVX-512 has native expand-load instructions
        unsafe {
            if is_type::<T, f32>() {
                V512::from_raw(_mm512_castps_si512(
                    _mm512_maskz_expandloadu_ps(
                        mask.raw as __mmask16,
                        ptr.cast(),
                    ),
                ))
            } else if is_type::<T, f64>() {
                V512::from_raw(_mm512_castpd_si512(
                    _mm512_maskz_expandloadu_pd(
                        mask.raw as __mmask8,
                        ptr.cast(),
                    ),
                ))
            } else {
                match T::BYTES {
                    4 => V512::from_raw(_mm512_maskz_expandloadu_epi32(
                        mask.raw as __mmask16,
                        ptr.cast(),
                    )),
                    8 => V512::from_raw(_mm512_maskz_expandloadu_epi64(
                        mask.raw as __mmask8,
                        ptr.cast(),
                    )),
                    _ => {
                        // For 1/2-byte types, no native expand-load. Use expand after load.
                        let loaded = self.load_u(ptr);
                        self.expand(loaded, mask)
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimdArith
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX-512F/BW.
unsafe impl SimdArith for Avx512 {
    #[inline(always)]
    fn add<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm512_add_epi8(a.raw, b.raw),
                2 => _mm512_add_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm512_castps_si512(_mm512_add_ps(
                            _mm512_castsi512_ps(a.raw),
                            _mm512_castsi512_ps(b.raw),
                        ))
                    } else {
                        _mm512_add_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm512_castpd_si512(_mm512_add_pd(
                            _mm512_castsi512_pd(a.raw),
                            _mm512_castsi512_pd(b.raw),
                        ))
                    } else {
                        _mm512_add_epi64(a.raw, b.raw)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn sub<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm512_sub_epi8(a.raw, b.raw),
                2 => _mm512_sub_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm512_castps_si512(_mm512_sub_ps(
                            _mm512_castsi512_ps(a.raw),
                            _mm512_castsi512_ps(b.raw),
                        ))
                    } else {
                        _mm512_sub_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm512_castpd_si512(_mm512_sub_pd(
                            _mm512_castsi512_pd(a.raw),
                            _mm512_castsi512_pd(b.raw),
                        ))
                    } else {
                        _mm512_sub_epi64(a.raw, b.raw)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn mul<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Emulate 8-bit mul via 16-bit
                    let mask = _mm512_set1_epi16(0x00FF);
                    let a_lo = _mm512_and_si512(a.raw, mask);
                    let b_lo = _mm512_and_si512(b.raw, mask);
                    let mul_lo =
                        _mm512_and_si512(_mm512_mullo_epi16(a_lo, b_lo), mask);
                    let a_hi = _mm512_srli_epi16(a.raw, 8);
                    let b_hi = _mm512_srli_epi16(b.raw, 8);
                    let mul_hi =
                        _mm512_slli_epi16(_mm512_mullo_epi16(a_hi, b_hi), 8);
                    _mm512_or_si512(mul_lo, mul_hi)
                }
                2 => _mm512_mullo_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm512_castps_si512(_mm512_mul_ps(
                            _mm512_castsi512_ps(a.raw),
                            _mm512_castsi512_ps(b.raw),
                        ))
                    } else {
                        _mm512_mullo_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm512_castpd_si512(_mm512_mul_pd(
                            _mm512_castsi512_pd(a.raw),
                            _mm512_castsi512_pd(b.raw),
                        ))
                    } else {
                        _mm512_mullo_epi64(a.raw, b.raw)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn div<T: FloatLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_div_ps(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                ))
            } else {
                _mm512_castpd_si512(_mm512_div_pd(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn saturated_add<T: IntegerLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    if is_type::<T, u8>() {
                        _mm512_adds_epu8(a.raw, b.raw)
                    } else {
                        _mm512_adds_epi8(a.raw, b.raw)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm512_adds_epu16(a.raw, b.raw)
                    } else {
                        _mm512_adds_epi16(a.raw, b.raw)
                    }
                }
                4 => {
                    if is_type::<T, u32>() {
                        // Unsigned saturating add: sum = a + b; if sum < a then overflow -> MAX
                        let sum = _mm512_add_epi32(a.raw, b.raw);
                        let overflow = _mm512_cmplt_epu32_mask(sum, a.raw);
                        _mm512_mask_set1_epi32(sum, overflow, -1i32)
                    } else {
                        // Signed saturating add: detect overflow via sign bits
                        let sum = _mm512_add_epi32(a.raw, b.raw);
                        // Overflow: a and b same sign, sum different sign
                        // pos + pos -> neg: saturate to MAX (0x7FFFFFFF)
                        // neg + neg -> pos: saturate to MIN (0x80000000)
                        let sign_a =
                            _mm512_sra_epi32(a.raw, _mm_cvtsi64_si128(31));
                        // Overflow when a^sum has sign bit set AND a^b does NOT have sign bit set
                        let overflow_mask = _mm512_andnot_si512(
                            _mm512_xor_si512(a.raw, b.raw),
                            _mm512_xor_si512(a.raw, sum),
                        );
                        let overflow = _mm512_movepi32_mask(overflow_mask);
                        // Saturated value: if a was positive -> MAX, else MIN
                        let sat_val = _mm512_xor_si512(
                            sign_a,
                            _mm512_set1_epi32(0x7FFF_FFFFu32 as i32),
                        );
                        _mm512_mask_mov_epi32(sum, overflow, sat_val)
                    }
                }
                8 => {
                    if is_type::<T, u64>() {
                        let sum = _mm512_add_epi64(a.raw, b.raw);
                        let overflow = _mm512_cmplt_epu64_mask(sum, a.raw);
                        _mm512_mask_set1_epi64(sum, overflow, -1i64)
                    } else {
                        let sum = _mm512_add_epi64(a.raw, b.raw);
                        let sign_a =
                            _mm512_sra_epi64(a.raw, _mm_cvtsi64_si128(63));
                        let overflow_mask = _mm512_andnot_si512(
                            _mm512_xor_si512(a.raw, b.raw),
                            _mm512_xor_si512(a.raw, sum),
                        );
                        let overflow = _mm512_movepi64_mask(overflow_mask);
                        let sat_val = _mm512_xor_si512(
                            sign_a,
                            _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64),
                        );
                        _mm512_mask_mov_epi64(sum, overflow, sat_val)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn saturated_sub<T: IntegerLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    if is_type::<T, u8>() {
                        _mm512_subs_epu8(a.raw, b.raw)
                    } else {
                        _mm512_subs_epi8(a.raw, b.raw)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm512_subs_epu16(a.raw, b.raw)
                    } else {
                        _mm512_subs_epi16(a.raw, b.raw)
                    }
                }
                4 => {
                    if is_type::<T, u32>() {
                        // Unsigned saturating sub: diff = a - b; if a < b then underflow -> 0
                        let diff = _mm512_sub_epi32(a.raw, b.raw);
                        let underflow = _mm512_cmplt_epu32_mask(a.raw, b.raw);
                        _mm512_mask_set1_epi32(diff, underflow, 0)
                    } else {
                        // Signed saturating sub: a - b, detect overflow
                        let diff = _mm512_sub_epi32(a.raw, b.raw);
                        // Overflow when a and b have different signs AND diff sign differs from a
                        let sign_a =
                            _mm512_sra_epi32(a.raw, _mm_cvtsi64_si128(31));
                        let overflow_mask = _mm512_and_si512(
                            _mm512_xor_si512(a.raw, b.raw),
                            _mm512_xor_si512(a.raw, diff),
                        );
                        let overflow = _mm512_movepi32_mask(overflow_mask);
                        let sat_val = _mm512_xor_si512(
                            sign_a,
                            _mm512_set1_epi32(0x7FFF_FFFFu32 as i32),
                        );
                        _mm512_mask_mov_epi32(diff, overflow, sat_val)
                    }
                }
                8 => {
                    if is_type::<T, u64>() {
                        let diff = _mm512_sub_epi64(a.raw, b.raw);
                        let underflow = _mm512_cmplt_epu64_mask(a.raw, b.raw);
                        _mm512_mask_set1_epi64(diff, underflow, 0)
                    } else {
                        let diff = _mm512_sub_epi64(a.raw, b.raw);
                        let sign_a =
                            _mm512_sra_epi64(a.raw, _mm_cvtsi64_si128(63));
                        let overflow_mask = _mm512_and_si512(
                            _mm512_xor_si512(a.raw, b.raw),
                            _mm512_xor_si512(a.raw, diff),
                        );
                        let overflow = _mm512_movepi64_mask(overflow_mask);
                        let sat_val = _mm512_xor_si512(
                            sign_a,
                            _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64),
                        );
                        _mm512_mask_mov_epi64(diff, overflow, sat_val)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn abs<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            if is_type::<T, f32>() {
                let mask = _mm512_set1_epi32(0x7FFF_FFFFu32 as i32);
                V512::from_raw(_mm512_and_si512(v.raw, mask))
            } else if is_type::<T, f64>() {
                let mask = _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64);
                V512::from_raw(_mm512_and_si512(v.raw, mask))
            } else if is_signed::<T>() {
                let raw = match T::BYTES {
                    1 => _mm512_abs_epi8(v.raw),
                    2 => _mm512_abs_epi16(v.raw),
                    4 => _mm512_abs_epi32(v.raw),
                    8 => _mm512_abs_epi64(v.raw),
                    _ => unreachable!(),
                };
                V512::from_raw(raw)
            } else {
                v
            }
        }
    }

    #[inline(always)]
    fn neg<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            if is_type::<T, f32>() {
                let sign = _mm512_set1_epi32(0x8000_0000u32 as i32);
                V512::from_raw(_mm512_xor_si512(v.raw, sign))
            } else if is_type::<T, f64>() {
                let sign = _mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64);
                V512::from_raw(_mm512_xor_si512(v.raw, sign))
            } else {
                let z = _mm512_setzero_si512();
                let raw = match T::BYTES {
                    1 => _mm512_sub_epi8(z, v.raw),
                    2 => _mm512_sub_epi16(z, v.raw),
                    4 => _mm512_sub_epi32(z, v.raw),
                    8 => _mm512_sub_epi64(z, v.raw),
                    _ => unreachable!(),
                };
                V512::from_raw(raw)
            }
        }
    }

    #[inline(always)]
    fn min<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm512_castps_si512(_mm512_min_ps(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                ))
            } else if is_type::<T, f64>() {
                _mm512_castpd_si512(_mm512_min_pd(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                ))
            } else {
                match T::BYTES {
                    1 => {
                        if is_signed::<T>() {
                            _mm512_min_epi8(a.raw, b.raw)
                        } else {
                            _mm512_min_epu8(a.raw, b.raw)
                        }
                    }
                    2 => {
                        if is_signed::<T>() {
                            _mm512_min_epi16(a.raw, b.raw)
                        } else {
                            _mm512_min_epu16(a.raw, b.raw)
                        }
                    }
                    4 => {
                        if is_signed::<T>() {
                            _mm512_min_epi32(a.raw, b.raw)
                        } else {
                            _mm512_min_epu32(a.raw, b.raw)
                        }
                    }
                    8 => {
                        if is_signed::<T>() {
                            _mm512_min_epi64(a.raw, b.raw)
                        } else {
                            _mm512_min_epu64(a.raw, b.raw)
                        }
                    }
                    _ => unreachable!(),
                }
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn max<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm512_castps_si512(_mm512_max_ps(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                ))
            } else if is_type::<T, f64>() {
                _mm512_castpd_si512(_mm512_max_pd(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                ))
            } else {
                match T::BYTES {
                    1 => {
                        if is_signed::<T>() {
                            _mm512_max_epi8(a.raw, b.raw)
                        } else {
                            _mm512_max_epu8(a.raw, b.raw)
                        }
                    }
                    2 => {
                        if is_signed::<T>() {
                            _mm512_max_epi16(a.raw, b.raw)
                        } else {
                            _mm512_max_epu16(a.raw, b.raw)
                        }
                    }
                    4 => {
                        if is_signed::<T>() {
                            _mm512_max_epi32(a.raw, b.raw)
                        } else {
                            _mm512_max_epu32(a.raw, b.raw)
                        }
                    }
                    8 => {
                        if is_signed::<T>() {
                            _mm512_max_epi64(a.raw, b.raw)
                        } else {
                            _mm512_max_epu64(a.raw, b.raw)
                        }
                    }
                    _ => unreachable!(),
                }
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn mul_high<T: IntegerLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    let mask_lo = _mm512_set1_epi16(0x00FF);
                    let mask_hi = _mm512_set1_epi16(0xFF00u16 as i16);
                    if is_type::<T, u8>() {
                        let a_even = _mm512_and_si512(a.raw, mask_lo);
                        let b_even = _mm512_and_si512(b.raw, mask_lo);
                        let prod_even = _mm512_mullo_epi16(a_even, b_even);
                        let hi_even = _mm512_srli_epi16(prod_even, 8);
                        let a_odd = _mm512_srli_epi16(a.raw, 8);
                        let b_odd = _mm512_srli_epi16(b.raw, 8);
                        let prod_odd = _mm512_mullo_epi16(a_odd, b_odd);
                        let hi_odd = _mm512_and_si512(prod_odd, mask_hi);
                        _mm512_or_si512(hi_even, hi_odd)
                    } else {
                        let a_even =
                            _mm512_srai_epi16(_mm512_slli_epi16(a.raw, 8), 8);
                        let b_even =
                            _mm512_srai_epi16(_mm512_slli_epi16(b.raw, 8), 8);
                        let prod_even = _mm512_mullo_epi16(a_even, b_even);
                        let hi_even = _mm512_srli_epi16(prod_even, 8);
                        let a_odd = _mm512_srai_epi16(a.raw, 8);
                        let b_odd = _mm512_srai_epi16(b.raw, 8);
                        let prod_odd = _mm512_mullo_epi16(a_odd, b_odd);
                        let hi_odd = _mm512_and_si512(prod_odd, mask_hi);
                        _mm512_or_si512(hi_even, hi_odd)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm512_mulhi_epu16(a.raw, b.raw)
                    } else {
                        _mm512_mulhi_epi16(a.raw, b.raw)
                    }
                }
                _ => {
                    if is_type::<T, u32>() {
                        let p_even = _mm512_mul_epu32(a.raw, b.raw);
                        let a_odd = _mm512_srli_epi64(a.raw, 32);
                        let b_odd = _mm512_srli_epi64(b.raw, 32);
                        let p_odd = _mm512_mul_epu32(a_odd, b_odd);
                        let hi_even = _mm512_srli_epi64(p_even, 32);
                        let mask_hi32 = _mm512_set_epi32(
                            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                            -1, 0,
                        );
                        let hi_odd = _mm512_and_si512(p_odd, mask_hi32);
                        _mm512_or_si512(hi_even, hi_odd)
                    } else {
                        // i32: use native _mm512_mul_epi32 with shift+mask+or
                        let p_even = _mm512_mul_epi32(a.raw, b.raw);
                        let a_odd = _mm512_srli_epi64(a.raw, 32);
                        let b_odd = _mm512_srli_epi64(b.raw, 32);
                        let p_odd = _mm512_mul_epi32(a_odd, b_odd);
                        let hi_even = _mm512_srli_epi64(p_even, 32);
                        let mask_hi32 = _mm512_set_epi32(
                            -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0,
                            -1, 0,
                        );
                        let hi_odd = _mm512_and_si512(p_odd, mask_hi32);
                        _mm512_or_si512(hi_even, hi_odd)
                    }
                }
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn average_round<T: UnsignedLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm512_avg_epu8(a.raw, b.raw),
                2 => _mm512_avg_epu16(a.raw, b.raw),
                4 => {
                    // avg(a,b) = (a>>1) + (b>>1) + ((a|b) & 1)
                    let one = _mm512_set1_epi32(1);
                    let a_half = _mm512_srli_epi32(a.raw, 1);
                    let b_half = _mm512_srli_epi32(b.raw, 1);
                    let carry =
                        _mm512_and_si512(_mm512_or_si512(a.raw, b.raw), one);
                    _mm512_add_epi32(_mm512_add_epi32(a_half, b_half), carry)
                }
                8 => {
                    // avg(a,b) = (a>>1) + (b>>1) + ((a|b) & 1)
                    let one = _mm512_set1_epi64(1);
                    let a_half = _mm512_srli_epi64(a.raw, 1);
                    let b_half = _mm512_srli_epi64(b.raw, 1);
                    let carry =
                        _mm512_and_si512(_mm512_or_si512(a.raw, b.raw), one);
                    _mm512_add_epi64(_mm512_add_epi64(a_half, b_half), carry)
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn abs_diff<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        {
            let mx = self.max(a, b);
            let mn = self.min(a, b);
            self.sub(mx, mn)
        }
    }

    #[inline(always)]
    fn clamp<T: Lane>(self, v: V512<T>, lo: V512<T>, hi: V512<T>) -> V512<T> {
        self.min(self.max(v, lo), hi)
    }

    #[inline(always)]
    fn mul_even<T: NarrowLane>(self, a: V512<T>, b: V512<T>) -> V512<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            let raw = match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        // f32 -> f64: gather even f32 lanes, promote, multiply
                        let idx = _mm512_set_epi32(
                            0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0,
                        );
                        let a_even = _mm512_permutexvar_epi32(idx, a.raw);
                        let b_even = _mm512_permutexvar_epi32(idx, b.raw);
                        let a_f64 = _mm512_cvtps_pd(_mm256_castsi256_ps(
                            _mm512_castsi512_si256(a_even),
                        ));
                        let b_f64 = _mm512_cvtps_pd(_mm256_castsi256_ps(
                            _mm512_castsi512_si256(b_even),
                        ));
                        _mm512_castpd_si512(_mm512_mul_pd(a_f64, b_f64))
                    } else if is_signed::<T>() {
                        _mm512_mul_epi32(a.raw, b.raw)
                    } else {
                        _mm512_mul_epu32(a.raw, b.raw)
                    }
                }
                1 => {
                    // u8/i8 -> u16/i16: extract even bytes, widen, multiply
                    if is_signed::<T>() {
                        let a16 =
                            _mm512_srai_epi16(_mm512_slli_epi16(a.raw, 8), 8);
                        let b16 =
                            _mm512_srai_epi16(_mm512_slli_epi16(b.raw, 8), 8);
                        _mm512_mullo_epi16(a16, b16)
                    } else {
                        let mask = _mm512_set1_epi16(0x00FFu16 as i16);
                        let a16 = _mm512_and_si512(a.raw, mask);
                        let b16 = _mm512_and_si512(b.raw, mask);
                        _mm512_mullo_epi16(a16, b16)
                    }
                }
                2 => {
                    // u16/i16 -> u32/i32: extract even 16-bit lanes, widen, multiply
                    if is_signed::<T>() {
                        let a32 =
                            _mm512_srai_epi32(_mm512_slli_epi32(a.raw, 16), 16);
                        let b32 =
                            _mm512_srai_epi32(_mm512_slli_epi32(b.raw, 16), 16);
                        _mm512_mullo_epi32(a32, b32)
                    } else {
                        let mask = _mm512_set1_epi32(0x0000FFFFu32 as i32);
                        let a32 = _mm512_and_si512(a.raw, mask);
                        let b32 = _mm512_and_si512(b.raw, mask);
                        _mm512_mullo_epi32(a32, b32)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn mul_odd<T: NarrowLane>(self, a: V512<T>, b: V512<T>) -> V512<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            let raw = match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        // f32 -> f64: gather odd f32 lanes, promote, multiply
                        let idx = _mm512_set_epi32(
                            0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1,
                        );
                        let a_odd = _mm512_permutexvar_epi32(idx, a.raw);
                        let b_odd = _mm512_permutexvar_epi32(idx, b.raw);
                        let a_f64 = _mm512_cvtps_pd(_mm256_castsi256_ps(
                            _mm512_castsi512_si256(a_odd),
                        ));
                        let b_f64 = _mm512_cvtps_pd(_mm256_castsi256_ps(
                            _mm512_castsi512_si256(b_odd),
                        ));
                        _mm512_castpd_si512(_mm512_mul_pd(a_f64, b_f64))
                    } else {
                        // Shift right by 32 bits to move odd lanes into even positions
                        let a_shifted = _mm512_srli_epi64(a.raw, 32);
                        let b_shifted = _mm512_srli_epi64(b.raw, 32);
                        if is_signed::<T>() {
                            _mm512_mul_epi32(a_shifted, b_shifted)
                        } else {
                            _mm512_mul_epu32(a_shifted, b_shifted)
                        }
                    }
                }
                1 => {
                    // u8/i8 -> u16/i16: extract odd bytes, widen, multiply
                    if is_signed::<T>() {
                        let a16 = _mm512_srai_epi16(a.raw, 8);
                        let b16 = _mm512_srai_epi16(b.raw, 8);
                        _mm512_mullo_epi16(a16, b16)
                    } else {
                        let a16 = _mm512_srli_epi16(a.raw, 8);
                        let b16 = _mm512_srli_epi16(b.raw, 8);
                        _mm512_mullo_epi16(a16, b16)
                    }
                }
                2 => {
                    // u16/i16 -> u32/i32: extract odd 16-bit lanes, widen, multiply
                    if is_signed::<T>() {
                        let a32 = _mm512_srai_epi32(a.raw, 16);
                        let b32 = _mm512_srai_epi32(b.raw, 16);
                        _mm512_mullo_epi32(a32, b32)
                    } else {
                        let a32 = _mm512_srli_epi32(a.raw, 16);
                        let b32 = _mm512_srli_epi32(b.raw, 16);
                        _mm512_mullo_epi32(a32, b32)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn widen_mul_pairwise_add_i16(
        self,
        a: V512<i16>,
        b: V512<i16>,
    ) -> V512<i32> {
        V512::from_raw(unsafe { _mm512_madd_epi16(a.raw, b.raw) })
    }

    #[inline(always)]
    fn sat_widen_mul_pairwise_add(self, a: V512<u8>, b: V512<i8>) -> V512<i16> {
        V512::from_raw(unsafe { _mm512_maddubs_epi16(a.raw, b.raw) })
    }

    #[inline(always)]
    fn mul_fixed_point_15(self, a: V512<i16>, b: V512<i16>) -> V512<i16> {
        V512::from_raw(unsafe { _mm512_mulhrs_epi16(a.raw, b.raw) })
    }

    #[inline(always)]
    fn reorder_widen_mul_accumulate(
        self,
        a: V512<i16>,
        b: V512<i16>,
        sum: V512<i32>,
    ) -> V512<i32> {
        unsafe {
            V512::from_raw(_mm512_add_epi32(
                sum.raw,
                _mm512_madd_epi16(a.raw, b.raw),
            ))
        }
    }

    #[inline(always)]
    fn saturated_neg<T: IntegerLane>(self, v: V512<T>) -> V512<T> {
        self.saturated_sub(self.zero::<T>(), v)
    }

    #[inline(always)]
    fn saturated_abs<T: IntegerLane>(self, v: V512<T>) -> V512<T> {
        self.max(v, self.saturated_neg(v))
    }

    #[inline(always)]
    fn masked_min_or<T: Lane>(
        self,
        no: V512<T>,
        mask: M512<T>,
        a: V512<T>,
        b: V512<T>,
    ) -> V512<T> {
        self.if_then_else(mask, self.min(a, b), no)
    }

    #[inline(always)]
    fn masked_max_or<T: Lane>(
        self,
        no: V512<T>,
        mask: M512<T>,
        a: V512<T>,
        b: V512<T>,
    ) -> V512<T> {
        self.if_then_else(mask, self.max(a, b), no)
    }

    #[inline(always)]
    fn masked_add_or<T: Lane>(
        self,
        no: V512<T>,
        mask: M512<T>,
        a: V512<T>,
        b: V512<T>,
    ) -> V512<T> {
        self.if_then_else(mask, self.add(a, b), no)
    }

    #[inline(always)]
    fn masked_sub_or<T: Lane>(
        self,
        no: V512<T>,
        mask: M512<T>,
        a: V512<T>,
        b: V512<T>,
    ) -> V512<T> {
        self.if_then_else(mask, self.sub(a, b), no)
    }

    #[inline(always)]
    fn masked_mul_or<T: Lane>(
        self,
        no: V512<T>,
        mask: M512<T>,
        a: V512<T>,
        b: V512<T>,
    ) -> V512<T> {
        self.if_then_else(mask, self.mul(a, b), no)
    }
}

// ---------------------------------------------------------------------------
// SimdBitwise
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX-512F.
unsafe impl SimdBitwise for Avx512 {
    #[inline(always)]
    fn and<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        V512::from_raw(unsafe { _mm512_and_si512(a.raw, b.raw) })
    }

    #[inline(always)]
    fn or<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        V512::from_raw(unsafe { _mm512_or_si512(a.raw, b.raw) })
    }

    #[inline(always)]
    fn xor<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        V512::from_raw(unsafe { _mm512_xor_si512(a.raw, b.raw) })
    }

    #[inline(always)]
    fn not<T: Lane>(self, v: V512<T>) -> V512<T> {
        let all_ones = unsafe { _mm512_set1_epi8(!0) };
        V512::from_raw(unsafe { _mm512_xor_si512(v.raw, all_ones) })
    }

    #[inline(always)]
    fn and_not<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        V512::from_raw(unsafe { _mm512_andnot_si512(a.raw, b.raw) })
    }

    #[inline(always)]
    fn shift_left<T: IntegerLane, const BITS: u32>(
        self,
        v: V512<T>,
    ) -> V512<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(BITS as i64);
            let raw = match T::BYTES {
                1 => {
                    let shifted = _mm512_sll_epi16(v.raw, count);
                    let mask =
                        _mm512_set1_epi8((0xFFu8.wrapping_shl(BITS)) as i8);
                    _mm512_and_si512(shifted, mask)
                }
                2 => _mm512_sll_epi16(v.raw, count),
                4 => _mm512_sll_epi32(v.raw, count),
                8 => _mm512_sll_epi64(v.raw, count),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_right<T: IntegerLane, const BITS: u32>(
        self,
        v: V512<T>,
    ) -> V512<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(BITS as i64);
            let raw = match T::BYTES {
                1 => {
                    if is_signed::<T>() {
                        let count8 = _mm_cvtsi64_si128(8i64);
                        let count_plus_8 = _mm_cvtsi64_si128((BITS + 8) as i64);
                        let shifted = _mm512_sra_epi16(
                            _mm512_sll_epi16(v.raw, count8),
                            count_plus_8,
                        );
                        let mask = _mm512_set1_epi16(0x00FF);
                        let lo = _mm512_and_si512(shifted, mask);
                        let hi = _mm512_andnot_si512(
                            mask,
                            _mm512_sra_epi16(v.raw, count),
                        );
                        _mm512_or_si512(lo, hi)
                    } else {
                        let shifted = _mm512_srl_epi16(v.raw, count);
                        let mask =
                            _mm512_set1_epi8((0xFFu8.wrapping_shr(BITS)) as i8);
                        _mm512_and_si512(shifted, mask)
                    }
                }
                2 => {
                    if is_signed::<T>() {
                        _mm512_sra_epi16(v.raw, count)
                    } else {
                        _mm512_srl_epi16(v.raw, count)
                    }
                }
                4 => {
                    if is_signed::<T>() {
                        _mm512_sra_epi32(v.raw, count)
                    } else {
                        _mm512_srl_epi32(v.raw, count)
                    }
                }
                8 => {
                    if is_signed::<T>() {
                        _mm512_sra_epi64(v.raw, count)
                    } else {
                        _mm512_srl_epi64(v.raw, count)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn rotate_right<T: IntegerLane, const BITS: u32>(
        self,
        v: V512<T>,
    ) -> V512<T> {
        unsafe {
            let type_bits = (T::BYTES * 8) as u32;
            let raw = match T::BYTES {
                4 | 8 => {
                    let right_count = _mm_cvtsi64_si128(BITS as i64);
                    let left_count =
                        _mm_cvtsi64_si128((type_bits - BITS) as i64);
                    let right = if T::BYTES == 4 {
                        _mm512_srl_epi32(v.raw, right_count)
                    } else {
                        _mm512_srl_epi64(v.raw, right_count)
                    };
                    let left = if T::BYTES == 4 {
                        _mm512_sll_epi32(v.raw, left_count)
                    } else {
                        _mm512_sll_epi64(v.raw, left_count)
                    };
                    _mm512_or_si512(right, left)
                }
                2 => {
                    // rotate_right for u16/i16: shift right + shift left + OR, masked to 16-bit lanes
                    let right_count = _mm_cvtsi64_si128(BITS as i64);
                    let left_count =
                        _mm_cvtsi64_si128((type_bits - BITS) as i64);
                    let right = _mm512_srl_epi16(v.raw, right_count);
                    let left = _mm512_sll_epi16(v.raw, left_count);
                    _mm512_or_si512(right, left)
                }
                1 => {
                    // rotate_right for u8/i8: emulate via 16-bit shifts with per-byte masking
                    let right_count = _mm_cvtsi64_si128(BITS as i64);
                    let left_count =
                        _mm_cvtsi64_si128((type_bits - BITS) as i64);
                    let right = _mm512_and_si512(
                        _mm512_srl_epi16(v.raw, right_count),
                        _mm512_set1_epi8((0xFFu8.wrapping_shr(BITS)) as i8),
                    );
                    let left = _mm512_and_si512(
                        _mm512_sll_epi16(v.raw, left_count),
                        _mm512_set1_epi8(
                            (0xFFu8.wrapping_shl(type_bits - BITS)) as i8,
                        ),
                    );
                    _mm512_or_si512(right, left)
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_left_same<T: IntegerLane>(self, v: V512<T>, bits: u32) -> V512<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(bits as i64);
            let raw = match T::BYTES {
                1 => {
                    let shifted = _mm512_sll_epi16(v.raw, count);
                    let mask =
                        _mm512_set1_epi8((0xFFu8.wrapping_shl(bits)) as i8);
                    _mm512_and_si512(shifted, mask)
                }
                2 => _mm512_sll_epi16(v.raw, count),
                4 => _mm512_sll_epi32(v.raw, count),
                8 => _mm512_sll_epi64(v.raw, count),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_right_same<T: IntegerLane>(
        self,
        v: V512<T>,
        bits: u32,
    ) -> V512<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(bits as i64);
            let raw = match T::BYTES {
                1 => {
                    if is_signed::<T>() {
                        // Emulate arithmetic shift right for i8
                        let count8 = _mm_cvtsi64_si128(8i64);
                        let count_plus_8 = _mm_cvtsi64_si128((bits + 8) as i64);
                        let shifted = _mm512_sra_epi16(
                            _mm512_sll_epi16(v.raw, count8),
                            count_plus_8,
                        );
                        let mask = _mm512_set1_epi16(0x00FF);
                        let lo = _mm512_and_si512(shifted, mask);
                        let hi_shifted = _mm512_sra_epi16(v.raw, count);
                        let hi = _mm512_andnot_si512(mask, hi_shifted);
                        _mm512_or_si512(lo, hi)
                    } else {
                        let shifted = _mm512_srl_epi16(v.raw, count);
                        let mask =
                            _mm512_set1_epi8((0xFFu8.wrapping_shr(bits)) as i8);
                        _mm512_and_si512(shifted, mask)
                    }
                }
                2 => {
                    if is_signed::<T>() {
                        _mm512_sra_epi16(v.raw, count)
                    } else {
                        _mm512_srl_epi16(v.raw, count)
                    }
                }
                4 => {
                    if is_signed::<T>() {
                        _mm512_sra_epi32(v.raw, count)
                    } else {
                        _mm512_srl_epi32(v.raw, count)
                    }
                }
                8 => {
                    if is_signed::<T>() {
                        _mm512_sra_epi64(v.raw, count)
                    } else {
                        _mm512_srl_epi64(v.raw, count)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_left_bytes<T: Lane, const BYTES: usize>(
        self,
        v: V512<T>,
    ) -> V512<T> {
        unsafe {
            let raw = match BYTES {
                0 => v.raw,
                1 => _mm512_bslli_epi128::<1>(v.raw),
                2 => _mm512_bslli_epi128::<2>(v.raw),
                3 => _mm512_bslli_epi128::<3>(v.raw),
                4 => _mm512_bslli_epi128::<4>(v.raw),
                5 => _mm512_bslli_epi128::<5>(v.raw),
                6 => _mm512_bslli_epi128::<6>(v.raw),
                7 => _mm512_bslli_epi128::<7>(v.raw),
                8 => _mm512_bslli_epi128::<8>(v.raw),
                9 => _mm512_bslli_epi128::<9>(v.raw),
                10 => _mm512_bslli_epi128::<10>(v.raw),
                11 => _mm512_bslli_epi128::<11>(v.raw),
                12 => _mm512_bslli_epi128::<12>(v.raw),
                13 => _mm512_bslli_epi128::<13>(v.raw),
                14 => _mm512_bslli_epi128::<14>(v.raw),
                15 => _mm512_bslli_epi128::<15>(v.raw),
                _ => _mm512_setzero_si512(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn shift_right_bytes<T: Lane, const BYTES: usize>(
        self,
        v: V512<T>,
    ) -> V512<T> {
        unsafe {
            let raw = match BYTES {
                0 => v.raw,
                1 => _mm512_bsrli_epi128::<1>(v.raw),
                2 => _mm512_bsrli_epi128::<2>(v.raw),
                3 => _mm512_bsrli_epi128::<3>(v.raw),
                4 => _mm512_bsrli_epi128::<4>(v.raw),
                5 => _mm512_bsrli_epi128::<5>(v.raw),
                6 => _mm512_bsrli_epi128::<6>(v.raw),
                7 => _mm512_bsrli_epi128::<7>(v.raw),
                8 => _mm512_bsrli_epi128::<8>(v.raw),
                9 => _mm512_bsrli_epi128::<9>(v.raw),
                10 => _mm512_bsrli_epi128::<10>(v.raw),
                11 => _mm512_bsrli_epi128::<11>(v.raw),
                12 => _mm512_bsrli_epi128::<12>(v.raw),
                13 => _mm512_bsrli_epi128::<13>(v.raw),
                14 => _mm512_bsrli_epi128::<14>(v.raw),
                15 => _mm512_bsrli_epi128::<15>(v.raw),
                _ => _mm512_setzero_si512(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn population_count<T: IntegerLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            // Use native VPOPCNTDQ/BITALG when available (Ice Lake+).
            // is_x86_feature_detected! caches after the first call.
            if T::BYTES == 4 && is_x86_feature_detected!("avx512vpopcntdq") {
                return V512::from_raw(native_popcnt_epi32(v.raw));
            }
            if T::BYTES == 8 && is_x86_feature_detected!("avx512vpopcntdq") {
                return V512::from_raw(native_popcnt_epi64(v.raw));
            }
            if T::BYTES == 1 && is_x86_feature_detected!("avx512bitalg") {
                return V512::from_raw(native_popcnt_epi8(v.raw));
            }
            if T::BYTES == 2 && is_x86_feature_detected!("avx512bitalg") {
                return V512::from_raw(native_popcnt_epi16(v.raw));
            }

            // Fallback: nibble-lookup popcount via vpshufb LUT,
            // then reduce within each lane width.
            let nibble_lut = _mm512_broadcast_i32x4(_mm_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            ));
            let lo_mask = _mm512_set1_epi8(0x0Fu8 as i8);
            let lo = _mm512_and_si512(v.raw, lo_mask);
            let hi = _mm512_and_si512(_mm512_srli_epi16(v.raw, 4), lo_mask);
            let cnt = _mm512_add_epi8(
                _mm512_shuffle_epi8(nibble_lut, lo),
                _mm512_shuffle_epi8(nibble_lut, hi),
            );
            // cnt holds per-byte popcount; reduce for wider lanes
            let raw = match T::BYTES {
                1 => cnt,
                2 => _mm512_maddubs_epi16(cnt, _mm512_set1_epi8(1)),
                4 => {
                    let sum16 = _mm512_maddubs_epi16(cnt, _mm512_set1_epi8(1));
                    _mm512_madd_epi16(sum16, _mm512_set1_epi16(1))
                }
                8 => _mm512_sad_epu8(cnt, _mm512_setzero_si512()),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn leading_zero_count<T: IntegerLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            // AVX-512CD native lzcnt for 32/64-bit lanes.
            if T::BYTES == 4 {
                V512::from_raw(_mm512_lzcnt_epi32(v.raw))
            } else if T::BYTES == 8 {
                V512::from_raw(_mm512_lzcnt_epi64(v.raw))
            } else if T::BYTES == 2 {
                // 16-bit: promote each half to 32-bit, lzcnt, subtract 16, pack back
                let lo_256 = _mm512_castsi512_si256(v.raw);
                let hi_256 = _mm512_extracti64x4_epi64(v.raw, 1);
                let lo_32 = _mm512_cvtepu16_epi32(lo_256);
                let hi_32 = _mm512_cvtepu16_epi32(hi_256);
                let bias = _mm512_set1_epi32(16);
                let lzc_lo = _mm512_sub_epi32(_mm512_lzcnt_epi32(lo_32), bias);
                let lzc_hi = _mm512_sub_epi32(_mm512_lzcnt_epi32(hi_32), bias);
                let lo_16 = _mm512_cvtepi32_epi16(lzc_lo);
                let hi_16 = _mm512_cvtepi32_epi16(lzc_hi);
                V512::from_raw(_mm512_inserti64x4(
                    _mm512_castsi256_si512(lo_16),
                    hi_16,
                    1,
                ))
            } else {
                // 8-bit: process in 4 batches of 16 bytes -> 16 x i32
                let bias = _mm512_set1_epi32(24);
                let src = v.raw;

                // Extract 4 x 128-bit chunks, zero-extend u8->u32 (16 lanes each)
                let chunk0 = _mm512_castsi512_si128(src);
                let chunk1 = _mm512_extracti32x4_epi32(src, 1);
                let chunk2 = _mm512_extracti32x4_epi32(src, 2);
                let chunk3 = _mm512_extracti32x4_epi32(src, 3);

                let w0 = _mm512_cvtepu8_epi32(chunk0);
                let w1 = _mm512_cvtepu8_epi32(chunk1);
                let w2 = _mm512_cvtepu8_epi32(chunk2);
                let w3 = _mm512_cvtepu8_epi32(chunk3);

                let lz0 = _mm512_sub_epi32(_mm512_lzcnt_epi32(w0), bias);
                let lz1 = _mm512_sub_epi32(_mm512_lzcnt_epi32(w1), bias);
                let lz2 = _mm512_sub_epi32(_mm512_lzcnt_epi32(w2), bias);
                let lz3 = _mm512_sub_epi32(_mm512_lzcnt_epi32(w3), bias);

                // Pack 32->16: two batches of 16 i32 -> 16 i16 each
                let p01_lo = _mm512_cvtepi32_epi16(lz0); // __m256i, 16 x i16
                let p01_hi = _mm512_cvtepi32_epi16(lz1);
                let p23_lo = _mm512_cvtepi32_epi16(lz2);
                let p23_hi = _mm512_cvtepi32_epi16(lz3);

                // Combine into two 512-bit vectors of i16
                let half0 = _mm512_inserti64x4(
                    _mm512_castsi256_si512(p01_lo),
                    p01_hi,
                    1,
                );
                let half1 = _mm512_inserti64x4(
                    _mm512_castsi256_si512(p23_lo),
                    p23_hi,
                    1,
                );

                // Pack 16->8: use _mm512_cvtepi16_epi8 to get __m256i from each
                let bytes0 = _mm512_cvtepi16_epi8(half0); // __m256i, 32 bytes
                let bytes1 = _mm512_cvtepi16_epi8(half1);

                V512::from_raw(_mm512_inserti64x4(
                    _mm512_castsi256_si512(bytes0),
                    bytes1,
                    1,
                ))
            }
        }
    }

    #[inline(always)]
    fn trailing_zero_count<T: IntegerLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            // tzcnt(x) = popcount((x - 1) & ~x)
            // XOR with all-ones is type-agnostic NOT
            let all_ones = _mm512_set1_epi64(-1);
            let not_x = _mm512_xor_si512(v.raw, all_ones);
            let xm1 = match T::BYTES {
                1 => _mm512_sub_epi8(v.raw, _mm512_set1_epi8(1)),
                2 => _mm512_sub_epi16(v.raw, _mm512_set1_epi16(1)),
                4 => _mm512_sub_epi32(v.raw, _mm512_set1_epi32(1)),
                8 => _mm512_sub_epi64(v.raw, _mm512_set1_epi64(1)),
                _ => unreachable!(),
            };
            let isolated = _mm512_and_si512(xm1, not_x);
            self.population_count(V512::from_raw(isolated))
        }
    }

    #[inline(always)]
    fn reverse_lane_bytes<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                1 => v, // No-op for single-byte lanes
                2 => {
                    // Swap bytes within each u16: [1,0, 3,2, ...]
                    let idx = _mm512_set_epi8(
                        62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51,
                        48, 49, 46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37,
                        34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23,
                        20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9,
                        6, 7, 4, 5, 2, 3, 0, 1,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                4 => {
                    // Reverse bytes within each u32: [3,2,1,0, 7,6,5,4, ...]
                    let idx = _mm512_set_epi8(
                        60, 61, 62, 63, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49,
                        50, 51, 44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39,
                        32, 33, 34, 35, 28, 29, 30, 31, 24, 25, 26, 27, 20, 21,
                        22, 23, 16, 17, 18, 19, 12, 13, 14, 15, 8, 9, 10, 11,
                        4, 5, 6, 7, 0, 1, 2, 3,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                8 => {
                    // Reverse bytes within each u64: [7,6,5,4,3,2,1,0, ...]
                    let idx = _mm512_set_epi8(
                        56, 57, 58, 59, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53,
                        54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35,
                        36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17,
                        18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15,
                        0, 1, 2, 3, 4, 5, 6, 7,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn reverse_bits<T: IntegerLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            // Nibble-lookup to reverse bits within each byte, then reverse bytes for wider lanes
            let nibble_rev = _mm512_broadcast_i32x4(_mm_setr_epi8(
                0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE, 0x1, 0x9, 0x5, 0xD,
                0x3, 0xB, 0x7, 0xF,
            ));
            let lo_mask = _mm512_set1_epi8(0x0Fu8 as i8);
            let hi_mask = _mm512_set1_epi8(0xF0u8 as i8);
            let lo = _mm512_and_si512(v.raw, lo_mask);
            let hi = _mm512_and_si512(_mm512_srli_epi16(v.raw, 4), lo_mask);
            // rev_lo: reversed low nibble -> goes to high nibble position
            let rev_lo = _mm512_shuffle_epi8(nibble_rev, lo);
            // rev_hi: reversed high nibble -> goes to low nibble position
            let rev_hi = _mm512_shuffle_epi8(nibble_rev, hi);
            // Combine: shift rev_lo to high nibble, keep rev_hi in low nibble
            // Use AND mask to prevent cross-byte leakage from slli_epi16
            let reversed_bytes = _mm512_or_si512(
                _mm512_and_si512(_mm512_slli_epi16(rev_lo, 4), hi_mask),
                rev_hi,
            );
            if T::BYTES == 1 {
                V512::from_raw(reversed_bytes)
            } else {
                // For wider lanes, reverse the byte order within each lane
                self.reverse_lane_bytes(V512::from_raw(reversed_bytes))
            }
        }
    }

    #[inline(always)]
    fn shl<T: IntegerLane>(self, v: V512<T>, bits: V512<T>) -> V512<T> {
        unsafe {
            if is_type::<T, u32>() || is_type::<T, i32>() {
                V512::from_raw(_mm512_sllv_epi32(v.raw, bits.raw))
            } else if is_type::<T, u64>() || is_type::<T, i64>() {
                V512::from_raw(_mm512_sllv_epi64(v.raw, bits.raw))
            } else if is_type::<T, u16>() || is_type::<T, i16>() {
                V512::from_raw(_mm512_sllv_epi16(v.raw, bits.raw))
            } else {
                // u8/i8: promote to 16-bit, shift, mask, combine
                let mask_lo = _mm512_set1_epi16(0x00FF);
                let v_even = _mm512_and_si512(v.raw, mask_lo);
                let b_even = _mm512_and_si512(bits.raw, mask_lo);
                let v_odd = _mm512_srli_epi16(v.raw, 8);
                let b_odd = _mm512_srli_epi16(bits.raw, 8);
                let r_even = _mm512_and_si512(
                    _mm512_sllv_epi16(v_even, b_even),
                    mask_lo,
                );
                let r_odd = _mm512_slli_epi16(
                    _mm512_and_si512(_mm512_sllv_epi16(v_odd, b_odd), mask_lo),
                    8,
                );
                V512::from_raw(_mm512_or_si512(r_even, r_odd))
            }
        }
    }

    #[inline(always)]
    fn shr<T: IntegerLane>(self, v: V512<T>, bits: V512<T>) -> V512<T> {
        unsafe {
            if is_type::<T, u32>() {
                V512::from_raw(_mm512_srlv_epi32(v.raw, bits.raw))
            } else if is_type::<T, i32>() {
                V512::from_raw(_mm512_srav_epi32(v.raw, bits.raw))
            } else if is_type::<T, u64>() {
                V512::from_raw(_mm512_srlv_epi64(v.raw, bits.raw))
            } else if is_type::<T, i64>() {
                V512::from_raw(_mm512_srav_epi64(v.raw, bits.raw))
            } else if is_type::<T, u16>() {
                V512::from_raw(_mm512_srlv_epi16(v.raw, bits.raw))
            } else if is_type::<T, i16>() {
                V512::from_raw(_mm512_srav_epi16(v.raw, bits.raw))
            } else if is_type::<T, u8>() {
                // u8: even/odd split with logical right shift
                let mask_lo = _mm512_set1_epi16(0x00FF);
                let v_even = _mm512_and_si512(v.raw, mask_lo);
                let b_even = _mm512_and_si512(bits.raw, mask_lo);
                let v_odd = _mm512_srli_epi16(v.raw, 8);
                let b_odd = _mm512_srli_epi16(bits.raw, 8);
                let r_even = _mm512_and_si512(
                    _mm512_srlv_epi16(v_even, b_even),
                    mask_lo,
                );
                let r_odd =
                    _mm512_slli_epi16(_mm512_srlv_epi16(v_odd, b_odd), 8);
                V512::from_raw(_mm512_or_si512(r_even, r_odd))
            } else {
                // i8: even/odd split with arithmetic right shift
                let mask_lo = _mm512_set1_epi16(0x00FF);
                // Sign-extend even bytes to 16-bit
                let v_even = _mm512_srai_epi16(_mm512_slli_epi16(v.raw, 8), 8);
                let b_even = _mm512_and_si512(bits.raw, mask_lo);
                // Odd bytes are already in the high byte position
                let v_odd = _mm512_srai_epi16(v.raw, 8);
                let b_odd = _mm512_srli_epi16(bits.raw, 8);
                let r_even = _mm512_and_si512(
                    _mm512_srav_epi16(v_even, b_even),
                    mask_lo,
                );
                let r_odd =
                    _mm512_slli_epi16(_mm512_srav_epi16(v_odd, b_odd), 8);
                V512::from_raw(_mm512_or_si512(r_even, r_odd))
            }
        }
    }

    #[inline(always)]
    fn ror<T: IntegerLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => V512::from_raw(_mm512_rorv_epi32(a.raw, b.raw)),
                8 => V512::from_raw(_mm512_rorv_epi64(a.raw, b.raw)),
                _ => {
                    let lanes = 64 / T::BYTES;
                    let mut result = [0u8; 64];
                    let mut arr_a = [0u8; 64];
                    let mut arr_b = [0u8; 64];
                    _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
                    _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
                    for i in 0..lanes {
                        let offset = i * T::BYTES;
                        match T::BYTES {
                            1 => {
                                let va = arr_a[offset];
                                let vb = arr_b[offset] & 7;
                                result[offset] = va.rotate_right(vb as u32);
                            }
                            2 => {
                                let va = u16::from_le_bytes([
                                    arr_a[offset],
                                    arr_a[offset + 1],
                                ]);
                                let vb = u16::from_le_bytes([
                                    arr_b[offset],
                                    arr_b[offset + 1],
                                ]) & 15;
                                let rb =
                                    va.rotate_right(vb as u32).to_le_bytes();
                                result[offset] = rb[0];
                                result[offset + 1] = rb[1];
                            }
                            _ => unreachable!(),
                        }
                    }
                    V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
                }
            }
        }
    }

    #[inline(always)]
    fn rol<T: IntegerLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => V512::from_raw(_mm512_rolv_epi32(a.raw, b.raw)),
                8 => V512::from_raw(_mm512_rolv_epi64(a.raw, b.raw)),
                _ => {
                    let lanes = 64 / T::BYTES;
                    let mut result = [0u8; 64];
                    let mut arr_a = [0u8; 64];
                    let mut arr_b = [0u8; 64];
                    _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
                    _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
                    for i in 0..lanes {
                        let offset = i * T::BYTES;
                        match T::BYTES {
                            1 => {
                                let va = arr_a[offset];
                                let vb = arr_b[offset] & 7;
                                result[offset] = va.rotate_left(vb as u32);
                            }
                            2 => {
                                let va = u16::from_le_bytes([
                                    arr_a[offset],
                                    arr_a[offset + 1],
                                ]);
                                let vb = u16::from_le_bytes([
                                    arr_b[offset],
                                    arr_b[offset + 1],
                                ]) & 15;
                                let rb =
                                    va.rotate_left(vb as u32).to_le_bytes();
                                result[offset] = rb[0];
                                result[offset + 1] = rb[1];
                            }
                            _ => unreachable!(),
                        }
                    }
                    V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
                }
            }
        }
    }

    #[inline(always)]
    fn rotate_left<T: IntegerLane, const BITS: u32>(
        self,
        v: V512<T>,
    ) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => V512::from_raw(_mm512_rolv_epi32(
                    v.raw,
                    _mm512_set1_epi32(BITS as i32),
                )),
                8 => V512::from_raw(_mm512_rolv_epi64(
                    v.raw,
                    _mm512_set1_epi64(BITS as i64),
                )),
                _ => {
                    let lanes = 64 / T::BYTES;
                    let mut result = [0u8; 64];
                    let mut arr = [0u8; 64];
                    _mm512_storeu_si512(arr.as_mut_ptr().cast(), v.raw);
                    for i in 0..lanes {
                        let offset = i * T::BYTES;
                        match T::BYTES {
                            1 => result[offset] = arr[offset].rotate_left(BITS),
                            2 => {
                                let va = u16::from_le_bytes([
                                    arr[offset],
                                    arr[offset + 1],
                                ]);
                                let rb = va.rotate_left(BITS).to_le_bytes();
                                result[offset] = rb[0];
                                result[offset + 1] = rb[1];
                            }
                            _ => unreachable!(),
                        }
                    }
                    V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
                }
            }
        }
    }

    #[inline(always)]
    fn broadcast_sign_bit<T: IntegerLane>(self, v: V512<T>) -> V512<T> {
        // All-ones if the MSB (sign bit) is set, else all-zeros. Matches C++ Highway.
        unsafe {
            let raw = match T::BYTES {
                1 => _mm512_movm_epi8(_mm512_movepi8_mask(v.raw)),
                2 => _mm512_srai_epi16(v.raw, 15),
                4 => _mm512_srai_epi32(v.raw, 31),
                _ => _mm512_srai_epi64(v.raw, 63),
            };
            V512::from_raw(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdCompare
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX-512F/BW.
unsafe impl SimdCompare for Avx512 {
    #[inline(always)]
    fn eq<T: Lane>(self, a: V512<T>, b: V512<T>) -> M512<T> {
        unsafe {
            let bits = if is_type::<T, f32>() {
                _mm512_cmp_ps_mask(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _CMP_EQ_OQ,
                ) as u64
            } else if is_type::<T, f64>() {
                _mm512_cmp_pd_mask(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _CMP_EQ_OQ,
                ) as u64
            } else {
                match T::BYTES {
                    1 => _mm512_cmpeq_epi8_mask(a.raw, b.raw) as u64,
                    2 => _mm512_cmpeq_epi16_mask(a.raw, b.raw) as u64,
                    4 => _mm512_cmpeq_epi32_mask(a.raw, b.raw) as u64,
                    8 => _mm512_cmpeq_epi64_mask(a.raw, b.raw) as u64,
                    _ => unreachable!(),
                }
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn ne<T: Lane>(self, a: V512<T>, b: V512<T>) -> M512<T> {
        // For floats, use _CMP_NEQ_OQ directly (ordered, quiet) to match
        // C++ Highway semantics: NaN != x returns false.
        // Using !eq would give unordered semantics (NaN != x returns true).
        unsafe {
            let bits = if is_type::<T, f32>() {
                _mm512_cmp_ps_mask(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _CMP_NEQ_OQ,
                ) as u64
            } else if is_type::<T, f64>() {
                _mm512_cmp_pd_mask(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _CMP_NEQ_OQ,
                ) as u64
            } else {
                let eq = self.eq::<T>(a, b);
                !eq.raw & M512::<T>::all_lanes_mask()
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn lt<T: Lane>(self, a: V512<T>, b: V512<T>) -> M512<T> {
        unsafe {
            let bits = if is_type::<T, f32>() {
                _mm512_cmp_ps_mask(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _CMP_LT_OQ,
                ) as u64
            } else if is_type::<T, f64>() {
                _mm512_cmp_pd_mask(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _CMP_LT_OQ,
                ) as u64
            } else if is_signed::<T>() {
                match T::BYTES {
                    1 => _mm512_cmplt_epi8_mask(a.raw, b.raw) as u64,
                    2 => _mm512_cmplt_epi16_mask(a.raw, b.raw) as u64,
                    4 => _mm512_cmplt_epi32_mask(a.raw, b.raw) as u64,
                    8 => _mm512_cmplt_epi64_mask(a.raw, b.raw) as u64,
                    _ => unreachable!(),
                }
            } else {
                match T::BYTES {
                    1 => _mm512_cmplt_epu8_mask(a.raw, b.raw) as u64,
                    2 => _mm512_cmplt_epu16_mask(a.raw, b.raw) as u64,
                    4 => _mm512_cmplt_epu32_mask(a.raw, b.raw) as u64,
                    8 => _mm512_cmplt_epu64_mask(a.raw, b.raw) as u64,
                    _ => unreachable!(),
                }
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn le<T: Lane>(self, a: V512<T>, b: V512<T>) -> M512<T> {
        unsafe {
            let bits = if is_type::<T, f32>() {
                _mm512_cmp_ps_mask(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _CMP_LE_OQ,
                ) as u64
            } else if is_type::<T, f64>() {
                _mm512_cmp_pd_mask(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _CMP_LE_OQ,
                ) as u64
            } else if is_signed::<T>() {
                match T::BYTES {
                    1 => _mm512_cmple_epi8_mask(a.raw, b.raw) as u64,
                    2 => _mm512_cmple_epi16_mask(a.raw, b.raw) as u64,
                    4 => _mm512_cmple_epi32_mask(a.raw, b.raw) as u64,
                    8 => _mm512_cmple_epi64_mask(a.raw, b.raw) as u64,
                    _ => unreachable!(),
                }
            } else {
                match T::BYTES {
                    1 => _mm512_cmple_epu8_mask(a.raw, b.raw) as u64,
                    2 => _mm512_cmple_epu16_mask(a.raw, b.raw) as u64,
                    4 => _mm512_cmple_epu32_mask(a.raw, b.raw) as u64,
                    8 => _mm512_cmple_epu64_mask(a.raw, b.raw) as u64,
                    _ => unreachable!(),
                }
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn gt<T: Lane>(self, a: V512<T>, b: V512<T>) -> M512<T> {
        self.lt(b, a)
    }

    #[inline(always)]
    fn ge<T: Lane>(self, a: V512<T>, b: V512<T>) -> M512<T> {
        self.le(b, a)
    }

    #[inline(always)]
    fn test_bit<T: IntegerLane>(self, v: V512<T>, bit: V512<T>) -> M512<T> {
        unsafe {
            let bits = match T::BYTES {
                1 => _mm512_test_epi8_mask(v.raw, bit.raw) as u64,
                2 => _mm512_test_epi16_mask(v.raw, bit.raw) as u64,
                4 => _mm512_test_epi32_mask(v.raw, bit.raw) as u64,
                8 => _mm512_test_epi64_mask(v.raw, bit.raw) as u64,
                _ => unreachable!(),
            };
            M512::from_bits(bits)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdMask
// ---------------------------------------------------------------------------

// SAFETY: AVX-512 mask operations.
unsafe impl SimdMask for Avx512 {
    #[inline(always)]
    fn mask_from_vec<T: Lane>(self, v: V512<T>) -> M512<T> {
        unsafe {
            let zero = _mm512_setzero_si512();
            let bits = match T::BYTES {
                1 => !_mm512_cmpeq_epi8_mask(v.raw, zero) as u64,
                2 => {
                    !_mm512_cmpeq_epi16_mask(v.raw, zero) as u64
                        & M512::<T>::all_lanes_mask()
                }
                4 => {
                    !_mm512_cmpeq_epi32_mask(v.raw, zero) as u64
                        & M512::<T>::all_lanes_mask()
                }
                8 => {
                    !_mm512_cmpeq_epi64_mask(v.raw, zero) as u64
                        & M512::<T>::all_lanes_mask()
                }
                _ => unreachable!(),
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn vec_from_mask<T: Lane>(self, m: M512<T>) -> V512<T> {
        unsafe {
            let zero = _mm512_setzero_si512();
            let ones = _mm512_set1_epi8(!0);
            let raw = match T::BYTES {
                1 => _mm512_mask_mov_epi8(zero, m.raw as __mmask64, ones),
                2 => _mm512_mask_mov_epi16(zero, m.raw as __mmask32, ones),
                4 => _mm512_mask_mov_epi32(zero, m.raw as __mmask16, ones),
                8 => _mm512_mask_mov_epi64(zero, m.raw as __mmask8, ones),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn first_n<T: Lane>(self, n: usize) -> M512<T> {
        unsafe {
            let lanes = M512::<T>::lane_count();
            let clamped = n.min(lanes);
            // _bzhi_u64: single BMI2 instruction, zeros bits from position `clamped` upward.
            // All AVX-512 CPUs support BMI2.
            let bits = _bzhi_u64(u64::MAX, clamped as u32);
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn count_true<T: Lane>(self, m: M512<T>) -> usize {
        (m.raw & M512::<T>::all_lanes_mask()).count_ones() as usize
    }

    #[inline(always)]
    fn all_true<T: Lane>(self, m: M512<T>) -> bool {
        (m.raw & M512::<T>::all_lanes_mask()) == M512::<T>::all_lanes_mask()
    }

    #[inline(always)]
    fn all_false<T: Lane>(self, m: M512<T>) -> bool {
        (m.raw & M512::<T>::all_lanes_mask()) == 0
    }

    #[inline(always)]
    fn find_first_true<T: Lane>(self, m: M512<T>) -> Option<usize> {
        let bits = m.raw & M512::<T>::all_lanes_mask();
        if bits == 0 {
            None
        } else {
            Some(bits.trailing_zeros() as usize)
        }
    }

    #[inline(always)]
    fn if_then_else<T: Lane>(
        self,
        mask: M512<T>,
        yes: V512<T>,
        no: V512<T>,
    ) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    _mm512_mask_mov_epi8(no.raw, mask.raw as __mmask64, yes.raw)
                }
                2 => _mm512_mask_mov_epi16(
                    no.raw,
                    mask.raw as __mmask32,
                    yes.raw,
                ),
                4 => _mm512_mask_mov_epi32(
                    no.raw,
                    mask.raw as __mmask16,
                    yes.raw,
                ),
                8 => {
                    _mm512_mask_mov_epi64(no.raw, mask.raw as __mmask8, yes.raw)
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn if_then_else_zero<T: Lane>(
        self,
        mask: M512<T>,
        yes: V512<T>,
    ) -> V512<T> {
        unsafe {
            let zero = _mm512_setzero_si512();
            let raw = match T::BYTES {
                1 => _mm512_mask_mov_epi8(zero, mask.raw as __mmask64, yes.raw),
                2 => {
                    _mm512_mask_mov_epi16(zero, mask.raw as __mmask32, yes.raw)
                }
                4 => {
                    _mm512_mask_mov_epi32(zero, mask.raw as __mmask16, yes.raw)
                }
                8 => _mm512_mask_mov_epi64(zero, mask.raw as __mmask8, yes.raw),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn if_then_zero_else<T: Lane>(self, mask: M512<T>, no: V512<T>) -> V512<T> {
        unsafe {
            let zero = _mm512_setzero_si512();
            // Invert mask
            let inv = !mask.raw & M512::<T>::all_lanes_mask();
            let raw = match T::BYTES {
                1 => _mm512_mask_mov_epi8(zero, inv as __mmask64, no.raw),
                2 => _mm512_mask_mov_epi16(zero, inv as __mmask32, no.raw),
                4 => _mm512_mask_mov_epi32(zero, inv as __mmask16, no.raw),
                8 => _mm512_mask_mov_epi64(zero, inv as __mmask8, no.raw),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn and_mask<T: Lane>(self, a: M512<T>, b: M512<T>) -> M512<T> {
        M512::from_raw(unsafe { _kand_mask64(a.raw, b.raw) })
    }

    #[inline(always)]
    fn or_mask<T: Lane>(self, a: M512<T>, b: M512<T>) -> M512<T> {
        M512::from_raw(unsafe { _kor_mask64(a.raw, b.raw) })
    }

    #[inline(always)]
    fn not_mask<T: Lane>(self, m: M512<T>) -> M512<T> {
        M512::from_bits(!m.raw & M512::<T>::all_lanes_mask())
    }

    #[inline(always)]
    fn xor_mask<T: Lane>(self, a: M512<T>, b: M512<T>) -> M512<T> {
        M512::from_raw(a.raw ^ b.raw)
    }

    #[inline(always)]
    fn find_last_true<T: Lane>(self, m: M512<T>) -> Option<usize> {
        let bits = m.raw & M512::<T>::all_lanes_mask();
        if bits == 0 {
            None
        } else {
            Some(63 - (bits.leading_zeros() as usize))
        }
    }

    #[inline(always)]
    fn bits_from_mask<T: Lane>(self, m: M512<T>) -> u64 {
        m.raw & M512::<T>::all_lanes_mask()
    }

    #[inline(always)]
    fn exclusive_neither<T: Lane>(self, a: M512<T>, b: M512<T>) -> M512<T> {
        // NOR: true only where neither a nor b is set (C++ ExclusiveNeither
        // documented "neither" semantic). NOR over valid lanes only.
        let nor = !(a.raw | b.raw) & M512::<T>::all_lanes_mask();
        M512::from_bits(nor)
    }

    #[inline(always)]
    fn slide_mask_1_up<T: Lane>(self, mask: M512<T>) -> M512<T> {
        let shifted = (mask.raw << 1) & M512::<T>::all_lanes_mask();
        M512::from_bits(shifted)
    }

    #[inline(always)]
    fn slide_mask_1_down<T: Lane>(self, mask: M512<T>) -> M512<T> {
        M512::from_bits(mask.raw >> 1)
    }

    #[inline(always)]
    fn if_negative_then_else<T: Lane>(
        self,
        v: V512<T>,
        yes: V512<T>,
        no: V512<T>,
    ) -> V512<T> {
        unsafe { self.if_then_else(avx512_sign_mask::<T>(v.raw), yes, no) }
    }

    #[inline(always)]
    fn if_negative_then_else_zero<T: Lane>(
        self,
        v: V512<T>,
        yes: V512<T>,
    ) -> V512<T> {
        unsafe { self.if_then_else_zero(avx512_sign_mask::<T>(v.raw), yes) }
    }

    #[inline(always)]
    fn if_negative_then_zero_else<T: Lane>(
        self,
        v: V512<T>,
        no: V512<T>,
    ) -> V512<T> {
        unsafe { self.if_then_zero_else(avx512_sign_mask::<T>(v.raw), no) }
    }
}

/// Build a per-lane sign-bit mask: bit set where the lane's MSB is set.
/// Works for signed integers and floats (via the bit-pattern's top bit).
#[inline(always)]
unsafe fn avx512_sign_mask<T: Lane>(raw: __m512i) -> M512<T> {
    unsafe {
        let bits = match T::BYTES {
            1 => _mm512_movepi8_mask(raw) as u64,
            2 => _mm512_movepi16_mask(raw) as u64,
            4 => _mm512_movepi32_mask(raw) as u64,
            _ => _mm512_movepi64_mask(raw) as u64,
        };
        M512::from_bits(bits)
    }
}

// ---------------------------------------------------------------------------
// SimdConvert
// ---------------------------------------------------------------------------

// SAFETY: AVX-512 conversion intrinsics.
unsafe impl SimdConvert for Avx512 {
    #[inline(always)]
    fn promote_to<N: NarrowLane>(self, v: V512<N>) -> V512<N::Wide>
    where
        N::Wide: Lane,
    {
        unsafe {
            let raw = match N::BYTES {
                1 => {
                    if is_signed::<N>() {
                        _mm512_cvtepi8_epi16(_mm512_castsi512_si256(v.raw))
                    } else {
                        _mm512_cvtepu8_epi16(_mm512_castsi512_si256(v.raw))
                    }
                }
                2 => {
                    if is_signed::<N>() {
                        _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v.raw))
                    } else {
                        _mm512_cvtepu16_epi32(_mm512_castsi512_si256(v.raw))
                    }
                }
                4 => {
                    if is_type::<N, f32>() {
                        _mm512_castpd_si512(_mm512_cvtps_pd(
                            _mm256_castsi256_ps(_mm512_castsi512_si256(v.raw)),
                        ))
                    } else if is_signed::<N>() {
                        _mm512_cvtepi32_epi64(_mm512_castsi512_si256(v.raw))
                    } else {
                        _mm512_cvtepu32_epi64(_mm512_castsi512_si256(v.raw))
                    }
                }
                _ => v.raw,
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn demote_to<W: WideLane>(self, v: V512<W>) -> V512<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = match W::BYTES {
                2 => {
                    if is_signed::<W>() {
                        _mm512_castsi256_si512(_mm512_cvtsepi16_epi8(v.raw))
                    } else {
                        _mm512_castsi256_si512(_mm512_cvtusepi16_epi8(v.raw))
                    }
                }
                4 => {
                    if is_signed::<W>() {
                        _mm512_castsi256_si512(_mm512_cvtsepi32_epi16(v.raw))
                    } else {
                        _mm512_castsi256_si512(_mm512_cvtusepi32_epi16(v.raw))
                    }
                }
                8 => {
                    if is_type::<W, f64>() {
                        let ps = _mm512_cvtpd_ps(_mm512_castsi512_pd(v.raw));
                        _mm512_castsi256_si512(_mm256_castps_si256(ps))
                    } else if is_signed::<W>() {
                        _mm512_castsi256_si512(_mm512_cvtsepi64_epi32(v.raw))
                    } else {
                        _mm512_castsi256_si512(_mm512_cvtusepi64_epi32(v.raw))
                    }
                }
                _ => v.raw,
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn convert_to_int<F: FloatLane>(self, v: V512<F>) -> V512<F::Int> {
        unsafe {
            let raw = if F::BYTES == 4 {
                _mm512_cvttps_epi32(_mm512_castsi512_ps(v.raw))
            } else {
                // f64 -> i64: cvttpd_epi64 returns i64::MIN for positive overflow.
                // Saturate positive overflow to i64::MAX like C++ Highway.
                let v_pd = _mm512_castsi512_pd(v.raw);
                let overflow = _mm512_cmp_pd_mask(
                    v_pd,
                    _mm512_set1_pd(9.223372036854776e18),
                    _CMP_GE_OQ,
                );
                let converted = _mm512_cvttpd_epi64(v_pd);
                _mm512_mask_mov_epi64(
                    converted,
                    overflow,
                    _mm512_set1_epi64(i64::MAX),
                )
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn convert_to_float<F: FloatLane>(self, v: V512<F::Int>) -> V512<F> {
        unsafe {
            let raw = if F::BYTES == 4 {
                _mm512_castps_si512(_mm512_cvtepi32_ps(v.raw))
            } else {
                _mm512_castpd_si512(_mm512_cvtepi64_pd(v.raw))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn truncate_to<W: WideLane>(self, v: V512<W>) -> V512<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = match W::BYTES {
                2 => {
                    // u16/i16 -> u8/i8: _mm512_cvtepi16_epi8 returns __m256i (32 bytes)
                    _mm512_castsi256_si512(_mm512_cvtepi16_epi8(v.raw))
                }
                4 => {
                    // u32/i32 -> u16/i16: _mm512_cvtepi32_epi16 returns __m256i (16 values)
                    _mm512_castsi256_si512(_mm512_cvtepi32_epi16(v.raw))
                }
                8 => {
                    // u64/i64 -> u32/i32: _mm512_cvtepi64_epi32 returns __m256i (8 values)
                    _mm512_castsi256_si512(_mm512_cvtepi64_epi32(v.raw))
                }
                _ => v.raw,
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn ordered_demote_2_to<W: WideLane>(
        self,
        lo: V512<W>,
        hi: V512<W>,
    ) -> V512<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let perm = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
            let raw = match W::BYTES {
                4 => {
                    if is_type::<W, u32>() {
                        // u32 -> u16: clamp to 0x7FFFFFFF so packus treats as positive
                        let max_i32 = _mm512_set1_epi32(0x7FFFFFFFu32 as i32);
                        let lo_clamped = _mm512_min_epu32(lo.raw, max_i32);
                        let hi_clamped = _mm512_min_epu32(hi.raw, max_i32);
                        let packed =
                            _mm512_packus_epi32(lo_clamped, hi_clamped);
                        _mm512_permutexvar_epi64(perm, packed)
                    } else {
                        // i32 -> i16
                        let packed = _mm512_packs_epi32(lo.raw, hi.raw);
                        _mm512_permutexvar_epi64(perm, packed)
                    }
                }
                2 => {
                    if is_type::<W, u16>() {
                        // u16 -> u8: clamp to 0x7FFF so packus treats as positive
                        let max_i16 = _mm512_set1_epi16(0x7FFFu16 as i16);
                        let lo_clamped = _mm512_min_epu16(lo.raw, max_i16);
                        let hi_clamped = _mm512_min_epu16(hi.raw, max_i16);
                        let packed =
                            _mm512_packus_epi16(lo_clamped, hi_clamped);
                        _mm512_permutexvar_epi64(perm, packed)
                    } else {
                        // i16 -> i8
                        let packed = _mm512_packs_epi16(lo.raw, hi.raw);
                        _mm512_permutexvar_epi64(perm, packed)
                    }
                }
                8 => {
                    // u64/i64 -> u32/i32: saturating demote
                    if is_type::<W, u64>() {
                        let lo_nar = _mm512_cvtusepi64_epi32(lo.raw);
                        let hi_nar = _mm512_cvtusepi64_epi32(hi.raw);
                        _mm512_inserti64x4(
                            _mm512_castsi256_si512(lo_nar),
                            hi_nar,
                            1,
                        )
                    } else {
                        // i64 -> i32: signed saturating truncate
                        let lo_nar = _mm512_cvtsepi64_epi32(lo.raw);
                        let hi_nar = _mm512_cvtsepi64_epi32(hi.raw);
                        _mm512_inserti64x4(
                            _mm512_castsi256_si512(lo_nar),
                            hi_nar,
                            1,
                        )
                    }
                }
                _ => lo.raw,
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn nearest_int<F: FloatLane>(self, v: V512<F>) -> V512<F::Int> {
        unsafe {
            let raw = if F::BYTES == 4 {
                // _mm512_cvtps_epi32: round-to-nearest using current mode.
                // Clamp >= 2^31 to i32::MAX.
                let ps = _mm512_castsi512_ps(v.raw);
                let overflow = _mm512_cmp_ps_mask(
                    ps,
                    _mm512_set1_ps(2147483648.0f32),
                    _CMP_GE_OQ,
                );
                let max_f = _mm512_set1_ps(2147483520.0f32);
                let clamped = _mm512_min_ps(ps, max_f);
                let converted = _mm512_cvtps_epi32(clamped);
                _mm512_mask_mov_epi32(
                    converted,
                    overflow,
                    _mm512_set1_epi32(i32::MAX),
                )
            } else {
                // f64 -> i64: _mm512_cvtpd_epi64 rounds to nearest.
                let pd = _mm512_castsi512_pd(v.raw);
                let overflow = _mm512_cmp_pd_mask(
                    pd,
                    _mm512_set1_pd(9.223372036854776e18),
                    _CMP_GE_OQ,
                );
                let converted = _mm512_cvtpd_epi64(pd);
                _mm512_mask_mov_epi64(
                    converted,
                    overflow,
                    _mm512_set1_epi64(i64::MAX),
                )
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn reorder_demote_2_to<W: WideLane>(
        self,
        a: V512<W>,
        b: V512<W>,
    ) -> V512<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            if W::BYTES == 4 && W::Narrow::BYTES == 2 {
                if is_type::<W, i32>() {
                    return V512::from_raw(_mm512_packs_epi32(a.raw, b.raw));
                } else if is_type::<W, u32>() {
                    let max_val = _mm512_set1_epi32(0xFFFF);
                    let a_c = _mm512_min_epu32(a.raw, max_val);
                    let b_c = _mm512_min_epu32(b.raw, max_val);
                    return V512::from_raw(_mm512_packus_epi32(a_c, b_c));
                }
            }
            if W::BYTES == 2 && W::Narrow::BYTES == 1 {
                if is_type::<W, i16>() {
                    return V512::from_raw(_mm512_packs_epi16(a.raw, b.raw));
                } else if is_type::<W, u16>() {
                    let max_val = _mm512_set1_epi16(0xFF_i16);
                    let a_c = _mm512_min_epu16(a.raw, max_val);
                    let b_c = _mm512_min_epu16(b.raw, max_val);
                    return V512::from_raw(_mm512_packus_epi16(a_c, b_c));
                }
            }
            // Fallback: truncation
            self.ordered_demote_2_to(a, b)
        }
    }

    #[inline(always)]
    fn demote_in_range_to<W: WideLane>(self, v: V512<W>) -> V512<W::Narrow>
    where
        W::Narrow: Lane,
    {
        self.demote_to(v)
    }

    #[inline(always)]
    fn convert_in_range_to_int<F: FloatLane>(self, v: V512<F>) -> V512<F::Int> {
        self.convert_to_int(v)
    }

    #[inline(always)]
    fn promote_lower_to<N: NarrowLane>(self, v: V512<N>) -> V512<N::Wide>
    where
        N::Wide: Lane,
    {
        unsafe {
            let lo = _mm512_castsi512_si256(v.raw);
            let raw = match N::BYTES {
                1 => {
                    if is_type::<N, u8>() {
                        _mm512_cvtepu8_epi16(lo)
                    } else {
                        _mm512_cvtepi8_epi16(lo)
                    }
                }
                2 => {
                    if is_type::<N, u16>() {
                        _mm512_cvtepu16_epi32(lo)
                    } else {
                        _mm512_cvtepi16_epi32(lo)
                    }
                }
                4 => {
                    if is_type::<N, u32>() {
                        _mm512_cvtepu32_epi64(lo)
                    } else if is_type::<N, i32>() {
                        _mm512_cvtepi32_epi64(lo)
                    } else {
                        _mm512_castpd_si512(_mm512_cvtps_pd(
                            _mm256_castsi256_ps(lo),
                        ))
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn promote_upper_to<N: NarrowLane>(self, v: V512<N>) -> V512<N::Wide>
    where
        N::Wide: Lane,
    {
        unsafe {
            let hi = _mm512_extracti64x4_epi64(v.raw, 1);
            let raw = match N::BYTES {
                1 => {
                    if is_type::<N, u8>() {
                        _mm512_cvtepu8_epi16(hi)
                    } else {
                        _mm512_cvtepi8_epi16(hi)
                    }
                }
                2 => {
                    if is_type::<N, u16>() {
                        _mm512_cvtepu16_epi32(hi)
                    } else {
                        _mm512_cvtepi16_epi32(hi)
                    }
                }
                4 => {
                    if is_type::<N, u32>() {
                        _mm512_cvtepu32_epi64(hi)
                    } else if is_type::<N, i32>() {
                        _mm512_cvtepi32_epi64(hi)
                    } else {
                        _mm512_castpd_si512(_mm512_cvtps_pd(
                            _mm256_castsi256_ps(hi),
                        ))
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn ordered_truncate_2_to<W: WideLane>(
        self,
        lo: V512<W>,
        hi: V512<W>,
    ) -> V512<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // OrderedTruncate2To = ConcatEven of the narrow-reinterpreted vectors.
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

// SAFETY: AVX-512 shuffle intrinsics.
unsafe impl SimdShuffle for Avx512 {
    #[inline(always)]
    fn reverse<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    // Reverse bytes within each 128-bit lane, then reverse 128-bit lane order
                    let byte_rev = _mm512_broadcast_i32x4(_mm_setr_epi8(
                        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                    ));
                    let within_lanes = _mm512_shuffle_epi8(v.raw, byte_rev);
                    // Reverse the four 128-bit lanes: 0b_00_01_10_11 = 0x1B
                    V512::from_raw(_mm512_shuffle_i64x2(
                        within_lanes,
                        within_lanes,
                        0x1B,
                    ))
                }
                2 => {
                    // 32 u16 lanes reversed via permutexvar
                    let idx = _mm512_set_epi16(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                        30, 31,
                    );
                    V512::from_raw(_mm512_permutexvar_epi16(idx, v.raw))
                }
                4 => {
                    let idx = _mm512_set_epi32(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                    );
                    V512::from_raw(_mm512_permutexvar_epi32(idx, v.raw))
                }
                8 => {
                    let idx = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
                    V512::from_raw(_mm512_permutexvar_epi64(idx, v.raw))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn broadcast_lane<T: Lane, const IDX: usize>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => V512::from_raw(_mm512_permutexvar_epi32(
                    _mm512_set1_epi32(IDX as i32),
                    v.raw,
                )),
                8 => V512::from_raw(_mm512_permutexvar_epi64(
                    _mm512_set1_epi64(IDX as i64),
                    v.raw,
                )),
                2 => V512::from_raw(_mm512_permutexvar_epi16(
                    _mm512_set1_epi16(IDX as i16),
                    v.raw,
                )),
                _ => {
                    // 1-byte: use VBMI permutexvar_epi8 when available (Ice Lake+).
                    if is_x86_feature_detected!("avx512vbmi") {
                        V512::from_raw(native_permutexvar_epi8(
                            _mm512_set1_epi8(IDX as i8),
                            v.raw,
                        ))
                    } else {
                        let val: T = self.extract_lane(v, IDX);
                        self.splat(val)
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn interleave_lower<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm512_unpacklo_epi8(a.raw, b.raw),
                2 => _mm512_unpacklo_epi16(a.raw, b.raw),
                4 => _mm512_unpacklo_epi32(a.raw, b.raw),
                8 => _mm512_unpacklo_epi64(a.raw, b.raw),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn interleave_upper<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm512_unpackhi_epi8(a.raw, b.raw),
                2 => _mm512_unpackhi_epi16(a.raw, b.raw),
                4 => _mm512_unpackhi_epi32(a.raw, b.raw),
                8 => _mm512_unpackhi_epi64(a.raw, b.raw),
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn zip_lower<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        self.interleave_lower(a, b)
    }

    #[inline(always)]
    fn zip_upper<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        self.interleave_upper(a, b)
    }

    #[inline(always)]
    fn table_lookup_bytes<T: Lane>(
        self,
        table: V512<T>,
        idx: V512<T>,
    ) -> V512<T> {
        // AVX-512BW has vpshufb within 128-bit lanes
        V512::from_raw(unsafe { _mm512_shuffle_epi8(table.raw, idx.raw) })
    }

    #[inline(always)]
    fn table_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        v: V512<T>,
        idx: V512<I>,
    ) -> V512<T> {
        unsafe {
            if T::BYTES == 4 {
                V512::from_raw(_mm512_permutexvar_epi32(idx.raw, v.raw))
            } else if T::BYTES == 8 {
                V512::from_raw(_mm512_permutexvar_epi64(idx.raw, v.raw))
            } else if T::BYTES == 2 {
                V512::from_raw(_mm512_permutexvar_epi16(idx.raw, v.raw))
            } else if is_x86_feature_detected!("avx512vbmi") {
                // 8-bit: use permutexvar_epi8 (AVX512VBMI, Ice Lake+)
                V512::from_raw(native_permutexvar_epi8(idx.raw, v.raw))
            } else {
                // 8-bit scalar fallback (no VBMI)
                let lanes = simd::lanes::<T, Avx512>();
                let mut data: Aligned<A64, [u8; 64]> = Aligned::new([0u8; 64]);
                let mut indices: Aligned<A64, [u8; 64]> =
                    Aligned::new([0u8; 64]);
                _mm512_store_si512(data.as_mut_ptr().cast(), v.raw);
                _mm512_store_si512(indices.as_mut_ptr().cast(), idx.raw);
                let mut result: Aligned<A64, [u8; 64]> =
                    Aligned::new([0u8; 64]);
                for (i, slot) in result.iter_mut().enumerate().take(lanes) {
                    let idx_off = i * I::BYTES;
                    let lane_idx = (indices[idx_off] as usize) % lanes;
                    *slot = data[lane_idx];
                }
                V512 {
                    raw: _mm512_load_si512(result.as_ptr().cast()),
                    _marker: PhantomData,
                }
            }
        }
    }

    #[inline(always)]
    fn reverse2<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    // Swap adjacent bytes using vpshufb
                    let idx = _mm512_set_epi8(
                        62, 63, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51,
                        48, 49, 46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37,
                        34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23,
                        20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9,
                        6, 7, 4, 5, 2, 3, 0, 1,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                2 => {
                    // Swap adjacent u16 pairs using vpshufb
                    let idx = _mm512_set_epi8(
                        61, 60, 63, 62, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48,
                        51, 50, 45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38,
                        33, 32, 35, 34, 29, 28, 31, 30, 25, 24, 27, 26, 21, 20,
                        23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10,
                        5, 4, 7, 6, 1, 0, 3, 2,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                4 => {
                    // Swap adjacent u32 pairs: 0b10_11_00_01
                    V512::from_raw(_mm512_shuffle_epi32(v.raw, _MM_PERM_CDAB))
                }
                8 => {
                    // Swap adjacent u64 pairs within each 128-bit block
                    V512::from_raw(_mm512_shuffle_epi32(v.raw, _MM_PERM_BADC))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn reverse4<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    // Reverse groups of 4 bytes using vpshufb
                    let idx = _mm512_set_epi8(
                        60, 61, 62, 63, 56, 57, 58, 59, 52, 53, 54, 55, 48, 49,
                        50, 51, 44, 45, 46, 47, 40, 41, 42, 43, 36, 37, 38, 39,
                        32, 33, 34, 35, 28, 29, 30, 31, 24, 25, 26, 27, 20, 21,
                        22, 23, 16, 17, 18, 19, 12, 13, 14, 15, 8, 9, 10, 11,
                        4, 5, 6, 7, 0, 1, 2, 3,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                2 => {
                    // Reverse groups of 4 u16 using vpshufb
                    let idx = _mm512_set_epi8(
                        57, 56, 59, 58, 61, 60, 63, 62, 49, 48, 51, 50, 53, 52,
                        55, 54, 41, 40, 43, 42, 45, 44, 47, 46, 33, 32, 35, 34,
                        37, 36, 39, 38, 25, 24, 27, 26, 29, 28, 31, 30, 17, 16,
                        19, 18, 21, 20, 23, 22, 9, 8, 11, 10, 13, 12, 15, 14,
                        1, 0, 3, 2, 5, 4, 7, 6,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                4 => {
                    // Reverse groups of 4 u32: 0b00_01_10_11
                    V512::from_raw(_mm512_shuffle_epi32(v.raw, _MM_PERM_ABCD))
                }
                8 => {
                    // Reverse groups of 4 u64 within each 256-bit half
                    V512::from_raw(_mm512_permutex_epi64(v.raw, _MM_PERM_ABCD))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn reverse8<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    // Reverse groups of 8 bytes using vpshufb
                    let idx = _mm512_set_epi8(
                        56, 57, 58, 59, 60, 61, 62, 63, 48, 49, 50, 51, 52, 53,
                        54, 55, 40, 41, 42, 43, 44, 45, 46, 47, 32, 33, 34, 35,
                        36, 37, 38, 39, 24, 25, 26, 27, 28, 29, 30, 31, 16, 17,
                        18, 19, 20, 21, 22, 23, 8, 9, 10, 11, 12, 13, 14, 15,
                        0, 1, 2, 3, 4, 5, 6, 7,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                2 => {
                    // Reverse 8 u16 within each 128-bit block using vpshufb
                    let idx = _mm512_set_epi8(
                        49, 48, 51, 50, 53, 52, 55, 54, 57, 56, 59, 58, 61, 60,
                        63, 62, 33, 32, 35, 34, 37, 36, 39, 38, 41, 40, 43, 42,
                        45, 44, 47, 46, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24,
                        27, 26, 29, 28, 31, 30, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8,
                        11, 10, 13, 12, 15, 14,
                    );
                    V512::from_raw(_mm512_shuffle_epi8(v.raw, idx))
                }
                4 => {
                    // Reverse groups of 8 u32: need full-vector permute
                    let idx = _mm512_set_epi32(
                        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
                    );
                    V512::from_raw(_mm512_permutexvar_epi32(idx, v.raw))
                }
                8 => {
                    // Reverse all 8 u64 lanes
                    let idx = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
                    V512::from_raw(_mm512_permutexvar_epi64(idx, v.raw))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn concat_upper_lower<T: Lane>(self, hi: V512<T>, lo: V512<T>) -> V512<T> {
        // Upper 256 bits from hi, lower 256 bits from lo.
        // Mask 0xF0 means upper 4 u64 lanes (bits 4..7) come from hi.
        unsafe { V512::from_raw(_mm512_mask_blend_epi64(0xF0, lo.raw, hi.raw)) }
    }

    #[inline(always)]
    fn concat_lower_upper<T: Lane>(self, hi: V512<T>, lo: V512<T>) -> V512<T> {
        // Lower 256 bits from hi, upper 256 bits from lo.
        // Mask 0x0F means lower 4 u64 lanes (bits 0..3) come from hi.
        unsafe { V512::from_raw(_mm512_mask_blend_epi64(0x0F, lo.raw, hi.raw)) }
    }

    #[inline(always)]
    fn concat_even<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    // Even lanes: a[0],a[2],a[4],...,a[14],b[0],b[2],...,b[14]
                    let idx = _mm512_set_epi32(
                        30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2,
                        0,
                    );
                    V512::from_raw(_mm512_permutex2var_epi32(a.raw, idx, b.raw))
                }
                8 => {
                    // Even lanes: a[0],a[2],a[4],a[6],b[0],b[2],b[4],b[6]
                    let idx = _mm512_set_epi64(14, 12, 10, 8, 6, 4, 2, 0);
                    V512::from_raw(_mm512_permutex2var_epi64(a.raw, idx, b.raw))
                }
                2 => {
                    // Even u16 lanes
                    let idx = _mm512_set_epi16(
                        62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36,
                        34, 32, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8,
                        6, 4, 2, 0,
                    );
                    V512::from_raw(_mm512_permutex2var_epi16(a.raw, idx, b.raw))
                }
                1 => {
                    // SIMD via packus: isolate even bytes (AND 0x00FF), pack, fix block order.
                    let mask16 = _mm512_set1_epi16(0x00FF);
                    let a16 = _mm512_and_si512(a.raw, mask16);
                    let b16 = _mm512_and_si512(b.raw, mask16);
                    let packed = _mm512_packus_epi16(a16, b16);
                    // packus interleaves 128-bit blocks; fix with u64 permute.
                    let fix_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
                    V512::from_raw(_mm512_permutexvar_epi64(fix_idx, packed))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn concat_odd<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    // Odd lanes: a[1],a[3],...,a[15],b[1],b[3],...,b[15]
                    let idx = _mm512_set_epi32(
                        31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3,
                        1,
                    );
                    V512::from_raw(_mm512_permutex2var_epi32(a.raw, idx, b.raw))
                }
                8 => {
                    // Odd lanes: a[1],a[3],a[5],a[7],b[1],b[3],b[5],b[7]
                    let idx = _mm512_set_epi64(15, 13, 11, 9, 7, 5, 3, 1);
                    V512::from_raw(_mm512_permutex2var_epi64(a.raw, idx, b.raw))
                }
                2 => {
                    // Odd u16 lanes
                    let idx = _mm512_set_epi16(
                        63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37,
                        35, 33, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9,
                        7, 5, 3, 1,
                    );
                    V512::from_raw(_mm512_permutex2var_epi16(a.raw, idx, b.raw))
                }
                1 => {
                    // SIMD via packus: shift odd bytes into even position, pack, fix block order.
                    let a16 = _mm512_srli_epi16(a.raw, 8);
                    let b16 = _mm512_srli_epi16(b.raw, 8);
                    let packed = _mm512_packus_epi16(a16, b16);
                    let fix_idx = _mm512_set_epi64(7, 5, 3, 1, 6, 4, 2, 0);
                    V512::from_raw(_mm512_permutexvar_epi64(fix_idx, packed))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn odd_even<T: Lane>(self, odd: V512<T>, even: V512<T>) -> V512<T> {
        unsafe {
            // Select even lanes from `even`, odd lanes from `odd`
            // Mask bit i: 0 = even (from `even`), 1 = odd (from `odd`)
            match T::BYTES {
                1 => {
                    let mask: __mmask64 = 0xAAAA_AAAA_AAAA_AAAA;
                    V512::from_raw(_mm512_mask_mov_epi8(
                        even.raw, mask, odd.raw,
                    ))
                }
                2 => {
                    let mask: __mmask32 = 0xAAAA_AAAA;
                    V512::from_raw(_mm512_mask_mov_epi16(
                        even.raw, mask, odd.raw,
                    ))
                }
                4 => {
                    let mask: __mmask16 = 0xAAAA;
                    V512::from_raw(_mm512_mask_mov_epi32(
                        even.raw, mask, odd.raw,
                    ))
                }
                8 => {
                    let mask: __mmask8 = 0xAA;
                    V512::from_raw(_mm512_mask_mov_epi64(
                        even.raw, mask, odd.raw,
                    ))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn slide_up_lanes<T: Lane>(self, v: V512<T>, n: usize) -> V512<T> {
        unsafe {
            let lanes = 64 / T::BYTES;
            if n >= lanes {
                return self.zero();
            }
            // Build index vector: lane i gets value from lane (i - n), or zero if i < n
            // Use maskz permute: mask off lanes where i < n
            let mask_bits = if n >= 64 {
                0u64
            } else {
                M512::<T>::all_lanes_mask() << n
            };
            match T::BYTES {
                4 => {
                    let iota = _mm512_set_epi32(
                        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                    );
                    let idx =
                        _mm512_sub_epi32(iota, _mm512_set1_epi32(n as i32));
                    V512::from_raw(_mm512_maskz_permutexvar_epi32(
                        mask_bits as __mmask16,
                        idx,
                        v.raw,
                    ))
                }
                8 => {
                    let iota = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
                    let idx =
                        _mm512_sub_epi64(iota, _mm512_set1_epi64(n as i64));
                    V512::from_raw(_mm512_maskz_permutexvar_epi64(
                        mask_bits as __mmask8,
                        idx,
                        v.raw,
                    ))
                }
                2 => {
                    let iota = _mm512_set_epi16(
                        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                        17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                        1, 0,
                    );
                    let idx =
                        _mm512_sub_epi16(iota, _mm512_set1_epi16(n as i16));
                    V512::from_raw(_mm512_maskz_permutexvar_epi16(
                        mask_bits as __mmask32,
                        idx,
                        v.raw,
                    ))
                }
                _ => {
                    // 1-byte: aligned store + unaligned load (no VBMI)
                    let byte_shift = n * T::BYTES;
                    let mut buf: Aligned<A64, [u8; 128]> =
                        Aligned::new([0u8; 128]);
                    _mm512_store_si512(buf.as_mut_ptr().add(64).cast(), v.raw);
                    V512::from_raw(_mm512_loadu_si512(
                        buf.as_ptr().add(64 - byte_shift).cast(),
                    ))
                }
            }
        }
    }

    #[inline(always)]
    fn slide_down_lanes<T: Lane>(self, v: V512<T>, n: usize) -> V512<T> {
        unsafe {
            let lanes = 64 / T::BYTES;
            if n >= lanes {
                return self.zero();
            }
            // Build index vector: lane i gets value from lane (i + n), or zero if i + n >= lanes
            let remaining = lanes - n;
            let mask_bits = if remaining >= 64 {
                u64::MAX
            } else {
                (1u64 << remaining) - 1
            };
            match T::BYTES {
                4 => {
                    let iota = _mm512_set_epi32(
                        15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
                    );
                    let idx =
                        _mm512_add_epi32(iota, _mm512_set1_epi32(n as i32));
                    V512::from_raw(_mm512_maskz_permutexvar_epi32(
                        mask_bits as __mmask16,
                        idx,
                        v.raw,
                    ))
                }
                8 => {
                    let iota = _mm512_set_epi64(7, 6, 5, 4, 3, 2, 1, 0);
                    let idx =
                        _mm512_add_epi64(iota, _mm512_set1_epi64(n as i64));
                    V512::from_raw(_mm512_maskz_permutexvar_epi64(
                        mask_bits as __mmask8,
                        idx,
                        v.raw,
                    ))
                }
                2 => {
                    let iota = _mm512_set_epi16(
                        31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,
                        17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2,
                        1, 0,
                    );
                    let idx =
                        _mm512_add_epi16(iota, _mm512_set1_epi16(n as i16));
                    V512::from_raw(_mm512_maskz_permutexvar_epi16(
                        mask_bits as __mmask32,
                        idx,
                        v.raw,
                    ))
                }
                _ => {
                    // 1-byte: aligned store + unaligned load (no VBMI)
                    let byte_shift = n * T::BYTES;
                    let mut buf: Aligned<A64, [u8; 128]> =
                        Aligned::new([0u8; 128]);
                    _mm512_store_si512(buf.as_mut_ptr().cast(), v.raw);
                    V512::from_raw(_mm512_loadu_si512(
                        buf.as_ptr().add(byte_shift).cast(),
                    ))
                }
            }
        }
    }

    #[inline(always)]
    fn compress<T: Lane>(self, v: V512<T>, mask: M512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        V512::from_raw(_mm512_castps_si512(
                            _mm512_maskz_compress_ps(
                                mask.raw as __mmask16,
                                _mm512_castsi512_ps(v.raw),
                            ),
                        ))
                    } else {
                        V512::from_raw(_mm512_maskz_compress_epi32(
                            mask.raw as __mmask16,
                            v.raw,
                        ))
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        V512::from_raw(_mm512_castpd_si512(
                            _mm512_maskz_compress_pd(
                                mask.raw as __mmask8,
                                _mm512_castsi512_pd(v.raw),
                            ),
                        ))
                    } else {
                        V512::from_raw(_mm512_maskz_compress_epi64(
                            mask.raw as __mmask8,
                            v.raw,
                        ))
                    }
                }
                _ => {
                    // Use native VBMI2 compress when available (Ice Lake+).
                    if T::BYTES == 1 && is_x86_feature_detected!("avx512vbmi2")
                    {
                        V512::from_raw(native_compress_epi8(mask.raw, v.raw))
                    } else if T::BYTES == 2
                        && is_x86_feature_detected!("avx512vbmi2")
                    {
                        V512::from_raw(native_compress_epi16(
                            mask.raw as __mmask32,
                            v.raw,
                        ))
                    } else if T::BYTES == 2 {
                        // Promote u16->u32, native compress, demote u32->u16
                        let mbits = mask.raw & M512::<T>::all_lanes_mask();
                        let lo_256 = _mm512_castsi512_si256(v.raw);
                        let hi_256 = _mm512_extracti64x4_epi64(v.raw, 1);
                        let lo_32 = _mm512_cvtepu16_epi32(lo_256);
                        let hi_32 = _mm512_cvtepu16_epi32(hi_256);
                        let lo_mask = (mbits & 0xFFFF) as __mmask16;
                        let hi_mask = ((mbits >> 16) & 0xFFFF) as __mmask16;
                        let lo_comp =
                            _mm512_maskz_compress_epi32(lo_mask, lo_32);
                        let hi_comp =
                            _mm512_maskz_compress_epi32(hi_mask, hi_32);
                        let lo_cnt = (lo_mask as u32).count_ones() as usize;
                        let lo_nar = _mm512_cvtepi32_epi16(lo_comp); // __m256i
                        let hi_nar = _mm512_cvtepi32_epi16(hi_comp); // __m256i
                        let mut buf = [0u8; 64];
                        _mm256_storeu_si256(buf.as_mut_ptr().cast(), lo_nar);
                        _mm256_storeu_si256(
                            buf.as_mut_ptr().add(lo_cnt * 2).cast(),
                            hi_nar,
                        );
                        V512::from_raw(_mm512_loadu_si512(buf.as_ptr().cast()))
                    } else {
                        // T::BYTES == 1
                        // Promote u8->u32 in 4 chunks, native compress, demote u32->u8
                        let mbits = mask.raw & M512::<T>::all_lanes_mask();
                        let chunk0 = _mm512_castsi512_si128(v.raw);
                        let chunk1 = _mm512_extracti32x4_epi32(v.raw, 1);
                        let chunk2 = _mm512_extracti32x4_epi32(v.raw, 2);
                        let chunk3 = _mm512_extracti32x4_epi32(v.raw, 3);
                        let w0 = _mm512_cvtepu8_epi32(chunk0);
                        let w1 = _mm512_cvtepu8_epi32(chunk1);
                        let w2 = _mm512_cvtepu8_epi32(chunk2);
                        let w3 = _mm512_cvtepu8_epi32(chunk3);
                        let m0 = (mbits & 0xFFFF) as __mmask16;
                        let m1 = ((mbits >> 16) & 0xFFFF) as __mmask16;
                        let m2 = ((mbits >> 32) & 0xFFFF) as __mmask16;
                        let m3 = ((mbits >> 48) & 0xFFFF) as __mmask16;
                        let c0 = _mm512_maskz_compress_epi32(m0, w0);
                        let c1 = _mm512_maskz_compress_epi32(m1, w1);
                        let c2 = _mm512_maskz_compress_epi32(m2, w2);
                        let c3 = _mm512_maskz_compress_epi32(m3, w3);
                        let cnt0 = (m0 as u32).count_ones() as usize;
                        let cnt1 = (m1 as u32).count_ones() as usize;
                        let cnt2 = (m2 as u32).count_ones() as usize;
                        let n0 = _mm512_cvtepi32_epi8(c0); // __m128i
                        let n1 = _mm512_cvtepi32_epi8(c1);
                        let n2 = _mm512_cvtepi32_epi8(c2);
                        let n3 = _mm512_cvtepi32_epi8(c3);
                        let mut buf = [0u8; 64];
                        _mm_storeu_si128(buf.as_mut_ptr().cast(), n0);
                        _mm_storeu_si128(buf.as_mut_ptr().add(cnt0).cast(), n1);
                        _mm_storeu_si128(
                            buf.as_mut_ptr().add(cnt0 + cnt1).cast(),
                            n2,
                        );
                        _mm_storeu_si128(
                            buf.as_mut_ptr().add(cnt0 + cnt1 + cnt2).cast(),
                            n3,
                        );
                        V512::from_raw(_mm512_loadu_si512(buf.as_ptr().cast()))
                    }
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn compress_store<T: Lane>(
        self,
        v: V512<T>,
        mask: M512<T>,
        ptr: *mut T,
    ) -> usize {
        unsafe {
            let mbits = mask.raw & M512::<T>::all_lanes_mask();
            let count = mbits.count_ones() as usize;
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        _mm512_mask_compressstoreu_ps(
                            ptr.cast(),
                            mask.raw as __mmask16,
                            _mm512_castsi512_ps(v.raw),
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi32(
                            ptr.cast(),
                            mask.raw as __mmask16,
                            v.raw,
                        );
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm512_mask_compressstoreu_pd(
                            ptr.cast(),
                            mask.raw as __mmask8,
                            _mm512_castsi512_pd(v.raw),
                        );
                    } else {
                        _mm512_mask_compressstoreu_epi64(
                            ptr.cast(),
                            mask.raw as __mmask8,
                            v.raw,
                        );
                    }
                }
                _ => {
                    // Use native VBMI2 compress-store when available (Ice Lake+).
                    if T::BYTES == 1 && is_x86_feature_detected!("avx512vbmi2")
                    {
                        native_compressstoreu_epi8(ptr.cast(), mask.raw, v.raw);
                    } else if T::BYTES == 2
                        && is_x86_feature_detected!("avx512vbmi2")
                    {
                        native_compressstoreu_epi16(
                            ptr.cast(),
                            mask.raw as __mmask32,
                            v.raw,
                        );
                    } else {
                        // Promote->compress->demote via self.compress(), then copy
                        let compressed = self.compress(v, mask);
                        let mut buf = [0u8; 64];
                        _mm512_storeu_si512(
                            buf.as_mut_ptr().cast(),
                            compressed.raw,
                        );
                        core::ptr::copy_nonoverlapping(
                            buf.as_ptr(),
                            ptr.cast::<u8>(),
                            count * T::BYTES,
                        );
                    }
                }
            }
            count
        }
    }

    #[inline(always)]
    fn dup_even<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        V512::from_raw(_mm512_castps_si512(_mm512_moveldup_ps(
                            _mm512_castsi512_ps(v.raw),
                        )))
                    } else {
                        // 0xA0 = 0b10_10_00_00 => [0,0,2,2] within each 128-bit lane
                        V512::from_raw(_mm512_shuffle_epi32(v.raw, 0xA0))
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        V512::from_raw(_mm512_castpd_si512(_mm512_movedup_pd(
                            _mm512_castsi512_pd(v.raw),
                        )))
                    } else {
                        V512::from_raw(_mm512_unpacklo_epi64(v.raw, v.raw))
                    }
                }
                2 => {
                    // Duplicate even u16: mask even lanes, shift left 16, OR
                    let mask = _mm512_set1_epi32(0x0000FFFF_u32 as i32);
                    let even = _mm512_and_si512(v.raw, mask);
                    let dup =
                        _mm512_or_si512(even, _mm512_slli_epi32(even, 16));
                    V512::from_raw(dup)
                }
                1 => {
                    // Duplicate even u8: mask even bytes, shift left 8, OR
                    let mask = _mm512_set1_epi16(0x00FF);
                    let even = _mm512_and_si512(v.raw, mask);
                    let dup = _mm512_or_si512(even, _mm512_slli_epi16(even, 8));
                    V512::from_raw(dup)
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn dup_odd<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        V512::from_raw(_mm512_castps_si512(_mm512_movehdup_ps(
                            _mm512_castsi512_ps(v.raw),
                        )))
                    } else {
                        // 0xF5 = 0b11_11_01_01 => [1,1,3,3] within each 128-bit lane
                        V512::from_raw(_mm512_shuffle_epi32(v.raw, 0xF5))
                    }
                }
                8 => V512::from_raw(_mm512_unpackhi_epi64(v.raw, v.raw)),
                2 => {
                    // Duplicate odd u16: shift right 16 to get odd into even, OR with original masked
                    let odd = _mm512_srli_epi32(v.raw, 16);
                    let mask = _mm512_set1_epi32(0xFFFF0000_u32 as i32);
                    let dup =
                        _mm512_or_si512(odd, _mm512_and_si512(v.raw, mask));
                    V512::from_raw(dup)
                }
                1 => {
                    // Duplicate odd u8: shift right 8 to get odd into even, OR with original masked
                    let odd = _mm512_srli_epi16(v.raw, 8);
                    let mask = _mm512_set1_epi16(0xFF00_u16 as i16);
                    let dup =
                        _mm512_or_si512(odd, _mm512_and_si512(v.raw, mask));
                    V512::from_raw(dup)
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    fn concat_lower_lower<T: Lane>(self, hi: V512<T>, lo: V512<T>) -> V512<T> {
        // Lower 256 bits of lo in low, lower 256 bits of hi in high
        // 0x44 = 0b01_00_01_00: selects q0,q1 from lo (first src) and q0,q1 from hi (second src)
        unsafe { V512::from_raw(_mm512_shuffle_i64x2(lo.raw, hi.raw, 0x44)) }
    }

    #[inline(always)]
    fn concat_upper_upper<T: Lane>(self, hi: V512<T>, lo: V512<T>) -> V512<T> {
        // Upper 256 bits of lo in low, upper 256 bits of hi in high
        // 0xEE = 0b11_10_11_10: selects q2,q3 from lo (first src) and q2,q3 from hi (second src)
        unsafe { V512::from_raw(_mm512_shuffle_i64x2(lo.raw, hi.raw, 0xEE)) }
    }

    #[inline(always)]
    fn slide_1_up<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            if T::BYTES == 4 {
                V512::from_raw(_mm512_alignr_epi32(
                    v.raw,
                    _mm512_setzero_si512(),
                    15,
                ))
            } else if T::BYTES == 8 {
                V512::from_raw(_mm512_alignr_epi64(
                    v.raw,
                    _mm512_setzero_si512(),
                    7,
                ))
            } else {
                // Byte-level fallback for 1/2-byte lanes
                let byte_shift = T::BYTES;
                let mut buf: Aligned<A64, [u8; 128]> = Aligned::new([0u8; 128]);
                _mm512_store_si512(buf.as_mut_ptr().add(64).cast(), v.raw);
                V512::from_raw(_mm512_loadu_si512(
                    buf.as_ptr().add(64 - byte_shift).cast(),
                ))
            }
        }
    }

    #[inline(always)]
    fn slide_1_down<T: Lane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            if T::BYTES == 4 {
                V512::from_raw(_mm512_alignr_epi32(
                    _mm512_setzero_si512(),
                    v.raw,
                    1,
                ))
            } else if T::BYTES == 8 {
                V512::from_raw(_mm512_alignr_epi64(
                    _mm512_setzero_si512(),
                    v.raw,
                    1,
                ))
            } else {
                // Byte-level fallback for 1/2-byte lanes
                let byte_shift = T::BYTES;
                let mut buf: Aligned<A64, [u8; 128]> = Aligned::new([0u8; 128]);
                _mm512_store_si512(buf.as_mut_ptr().cast(), v.raw);
                V512::from_raw(_mm512_loadu_si512(
                    buf.as_ptr().add(byte_shift).cast(),
                ))
            }
        }
    }

    #[inline(always)]
    fn expand<T: Lane>(self, v: V512<T>, mask: M512<T>) -> V512<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        V512::from_raw(_mm512_castps_si512(
                            _mm512_mask_expand_ps(
                                _mm512_setzero_ps(),
                                mask.raw as __mmask16,
                                _mm512_castsi512_ps(v.raw),
                            ),
                        ))
                    } else {
                        V512::from_raw(_mm512_mask_expand_epi32(
                            _mm512_setzero_si512(),
                            mask.raw as __mmask16,
                            v.raw,
                        ))
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        V512::from_raw(_mm512_castpd_si512(
                            _mm512_mask_expand_pd(
                                _mm512_setzero_pd(),
                                mask.raw as __mmask8,
                                _mm512_castsi512_pd(v.raw),
                            ),
                        ))
                    } else {
                        V512::from_raw(_mm512_mask_expand_epi64(
                            _mm512_setzero_si512(),
                            mask.raw as __mmask8,
                            v.raw,
                        ))
                    }
                }
                _ => {
                    // u8/i8/u16/i16: scalar fallback
                    let lanes = 64 / T::BYTES;
                    let mut src: Aligned<A64, [u8; 64]> =
                        Aligned::new([0u8; 64]);
                    _mm512_store_si512(src.as_mut_ptr().cast(), v.raw);
                    let mut result: Aligned<A64, [u8; 64]> =
                        Aligned::new([0u8; 64]);
                    let mbits = mask.raw & M512::<T>::all_lanes_mask();
                    let mut src_idx = 0usize;
                    for i in 0..lanes {
                        if (mbits >> i) & 1 != 0 {
                            core::ptr::copy_nonoverlapping(
                                src.as_ptr().add(src_idx * T::BYTES),
                                result.as_mut_ptr().add(i * T::BYTES),
                                T::BYTES,
                            );
                            src_idx += 1;
                        }
                        // else: result already zeroed
                    }
                    V512::from_raw(_mm512_load_si512(result.as_ptr().cast()))
                }
            }
        }
    }

    #[inline(always)]
    fn combine_shift_right_bytes<T: Lane, const BYTES: usize>(
        self,
        hi: V512<T>,
        lo: V512<T>,
    ) -> V512<T> {
        // _mm512_alignr_epi8 is per-128-bit-lane PALIGNR.
        unsafe {
            let raw = match BYTES {
                0 => lo.raw,
                1 => _mm512_alignr_epi8::<1>(hi.raw, lo.raw),
                2 => _mm512_alignr_epi8::<2>(hi.raw, lo.raw),
                3 => _mm512_alignr_epi8::<3>(hi.raw, lo.raw),
                4 => _mm512_alignr_epi8::<4>(hi.raw, lo.raw),
                5 => _mm512_alignr_epi8::<5>(hi.raw, lo.raw),
                6 => _mm512_alignr_epi8::<6>(hi.raw, lo.raw),
                7 => _mm512_alignr_epi8::<7>(hi.raw, lo.raw),
                8 => _mm512_alignr_epi8::<8>(hi.raw, lo.raw),
                9 => _mm512_alignr_epi8::<9>(hi.raw, lo.raw),
                10 => _mm512_alignr_epi8::<10>(hi.raw, lo.raw),
                11 => _mm512_alignr_epi8::<11>(hi.raw, lo.raw),
                12 => _mm512_alignr_epi8::<12>(hi.raw, lo.raw),
                13 => _mm512_alignr_epi8::<13>(hi.raw, lo.raw),
                14 => _mm512_alignr_epi8::<14>(hi.raw, lo.raw),
                15 => _mm512_alignr_epi8::<15>(hi.raw, lo.raw),
                _ => hi.raw,
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn compress_blended_store<T: Lane>(
        self,
        v: V512<T>,
        mask: M512<T>,
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
    fn odd_even_blocks<T: Lane>(self, odd: V512<T>, even: V512<T>) -> V512<T> {
        // AVX-512: 4 blocks. Blocks 0,2 from even; blocks 1,3 from odd.
        // Use _mm512_mask_blend_epi64 with mask 0x33:
        // bits: 0b00110011 -> lanes 0,1,4,5 (blocks 0,2) from even; 2,3,6,7 (blocks 1,3) from odd.
        unsafe {
            V512::from_raw(_mm512_mask_blend_epi64(0x33, odd.raw, even.raw))
        }
    }

    #[inline(always)]
    fn reverse_blocks<T: Lane>(self, v: V512<T>) -> V512<T> {
        // Reverse 4 * 128-bit blocks: 0x1B = 0b00_01_10_11 -> block order 3,2,1,0.
        unsafe { V512::from_raw(_mm512_shuffle_i32x4(v.raw, v.raw, 0x1B)) }
    }

    #[inline(always)]
    fn compress_not<T: Lane>(self, v: V512<T>, mask: M512<T>) -> V512<T> {
        self.compress(v, self.not_mask(mask))
    }

    #[inline(always)]
    fn compress_blocks_not(self, v: V512<u64>, mask: M512<u64>) -> V512<u64> {
        self.compress(v, self.not_mask(mask))
    }

    #[inline(always)]
    fn broadcast_block<T: Lane, const IDX: usize>(self, v: V512<T>) -> V512<T> {
        unsafe {
            // Extract 128-bit block IDX, broadcast to all 4 positions
            let block = match IDX {
                0 => _mm512_castsi512_si128(v.raw),
                1 => _mm512_extracti32x4_epi32(v.raw, 1),
                2 => _mm512_extracti32x4_epi32(v.raw, 2),
                3 => _mm512_extracti32x4_epi32(v.raw, 3),
                _ => unreachable!(),
            };
            V512::from_raw(_mm512_broadcast_i32x4(block))
        }
    }

    #[inline(always)]
    unsafe fn compress_bits<T: Lane>(
        self,
        v: V512<T>,
        bits: *const u8,
    ) -> V512<T> {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut mask_bits: u64 = 0;
            for i in 0..lanes {
                let byte_idx = i / 8;
                let bit_in_byte = i % 8;
                let b = bits.add(byte_idx).read();
                if (b >> bit_in_byte) & 1 != 0 {
                    mask_bits |= 1u64 << i;
                }
            }
            let mask = M512::<T>::from_raw(mask_bits as __mmask64);
            self.compress(v, mask)
        }
    }

    #[inline(always)]
    unsafe fn compress_bits_store<T: Lane>(
        self,
        v: V512<T>,
        bits: *const u8,
        ptr: *mut T,
    ) -> usize {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut mask_bits: u64 = 0;
            for i in 0..lanes {
                let byte_idx = i / 8;
                let bit_in_byte = i % 8;
                let b = bits.add(byte_idx).read();
                if (b >> bit_in_byte) & 1 != 0 {
                    mask_bits |= 1u64 << i;
                }
            }
            let mask = M512::<T>::from_raw(mask_bits as __mmask64);
            self.compress_store(v, mask, ptr)
        }
    }

    #[inline(always)]
    fn lower_half<T: Lane>(self, v: V512<T>) -> crate::backend::avx2::V256<T> {
        unsafe {
            crate::backend::avx2::V256::from_raw(_mm512_castsi512_si256(v.raw))
        }
    }

    #[inline(always)]
    fn upper_half<T: Lane>(self, v: V512<T>) -> crate::backend::avx2::V256<T> {
        unsafe {
            crate::backend::avx2::V256::from_raw(_mm512_extracti64x4_epi64(
                v.raw, 1,
            ))
        }
    }

    #[inline(always)]
    fn combine<T: Lane>(
        self,
        lo: crate::backend::avx2::V256<T>,
        hi: crate::backend::avx2::V256<T>,
    ) -> V512<T> {
        unsafe {
            let lo512 = _mm512_castsi256_si512(lo.raw());
            V512::from_raw(_mm512_inserti64x4(lo512, hi.raw(), 1))
        }
    }

    #[inline(always)]
    fn insert_block<T: Lane, const IDX: usize>(
        self,
        v: V512<T>,
        blk: crate::backend::avx2::V256<T>,
    ) -> V512<T> {
        unsafe {
            if IDX == 0 {
                V512::from_raw(_mm512_inserti64x4(v.raw, blk.raw(), 0))
            } else {
                V512::from_raw(_mm512_inserti64x4(v.raw, blk.raw(), 1))
            }
        }
    }

    #[inline(always)]
    fn extract_block<T: Lane, const IDX: usize>(
        self,
        v: V512<T>,
    ) -> crate::backend::avx2::V256<T> {
        unsafe {
            if IDX == 0 {
                crate::backend::avx2::V256::from_raw(_mm512_castsi512_si256(
                    v.raw,
                ))
            } else {
                crate::backend::avx2::V256::from_raw(_mm512_extracti64x4_epi64(
                    v.raw, 1,
                ))
            }
        }
    }

    #[inline(always)]
    fn interleave_whole_lower<T: Lane>(
        self,
        a: V512<T>,
        b: V512<T>,
    ) -> V512<T> {
        // Whole-vector (cross-block) interleave of the lower halves:
        // result[2i] = a[i], result[2i+1] = b[i] for i in 0..N/2.
        // The 2-block ConcatLowerLower fallback is WRONG for 512-bit (4 blocks),
        // so use an explicit scalar interleave (matches C++ permutex2var indices).
        unsafe {
            let lanes = 64 / T::BYTES;
            let half = lanes / 2;
            let mut arr_a = [0u8; 64];
            let mut arr_b = [0u8; 64];
            let mut result = [0u8; 64];
            _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
            _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
            for i in 0..half {
                let sa = i * T::BYTES;
                let da = (2 * i) * T::BYTES;
                let db = (2 * i + 1) * T::BYTES;
                result[da..da + T::BYTES]
                    .copy_from_slice(&arr_a[sa..sa + T::BYTES]);
                result[db..db + T::BYTES]
                    .copy_from_slice(&arr_b[sa..sa + T::BYTES]);
            }
            V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn interleave_whole_upper<T: Lane>(
        self,
        a: V512<T>,
        b: V512<T>,
    ) -> V512<T> {
        // Whole-vector interleave of the upper halves:
        // result[2i] = a[N/2+i], result[2i+1] = b[N/2+i] for i in 0..N/2.
        unsafe {
            let lanes = 64 / T::BYTES;
            let half = lanes / 2;
            let mut arr_a = [0u8; 64];
            let mut arr_b = [0u8; 64];
            let mut result = [0u8; 64];
            _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
            _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
            for i in 0..half {
                let sa = (half + i) * T::BYTES;
                let da = (2 * i) * T::BYTES;
                let db = (2 * i + 1) * T::BYTES;
                result[da..da + T::BYTES]
                    .copy_from_slice(&arr_a[sa..sa + T::BYTES]);
                result[db..db + T::BYTES]
                    .copy_from_slice(&arr_b[sa..sa + T::BYTES]);
            }
            V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn interleave_even<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut arr_a = [0u8; 64];
            let mut arr_b = [0u8; 64];
            let mut result = [0u8; 64];
            _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
            _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
            let mut dst = 0;
            let mut src_idx = 0;
            while src_idx < lanes {
                let off = src_idx * T::BYTES;
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_a[off..off + T::BYTES]);
                dst += 1;
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_b[off..off + T::BYTES]);
                dst += 1;
                src_idx += 2;
            }
            V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn interleave_odd<T: Lane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut arr_a = [0u8; 64];
            let mut arr_b = [0u8; 64];
            let mut result = [0u8; 64];
            _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
            _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
            let mut dst = 0;
            let mut src_idx = 1;
            while src_idx < lanes {
                let off = src_idx * T::BYTES;
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_a[off..off + T::BYTES]);
                dst += 1;
                result[dst * T::BYTES..dst * T::BYTES + T::BYTES]
                    .copy_from_slice(&arr_b[off..off + T::BYTES]);
                dst += 1;
                src_idx += 2;
            }
            V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn two_tables_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        a: V512<T>,
        b: V512<T>,
        idx: V512<I>,
    ) -> V512<T> {
        unsafe {
            // AVX-512 has native _mm512_permutex2var for 32-bit and 64-bit lanes
            if T::BYTES == 4 {
                V512::from_raw(_mm512_permutex2var_epi32(a.raw, idx.raw, b.raw))
            } else if T::BYTES == 8 {
                V512::from_raw(_mm512_permutex2var_epi64(a.raw, idx.raw, b.raw))
            } else {
                // For 8-bit and 16-bit: scalar fallback
                let lanes = 64 / T::BYTES;
                let mut arr_a = [0u8; 64];
                let mut arr_b = [0u8; 64];
                let mut arr_idx = [0u8; 64];
                let mut result = [0u8; 64];
                _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
                _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
                _mm512_storeu_si512(arr_idx.as_mut_ptr().cast(), idx.raw);
                for i in 0..lanes {
                    let idx_off = i * I::BYTES;
                    let lane_idx: usize = match I::BYTES {
                        1 => arr_idx[idx_off] as usize,
                        2 => u16::from_le_bytes([
                            arr_idx[idx_off],
                            arr_idx[idx_off + 1],
                        ]) as usize,
                        4 => u32::from_le_bytes([
                            arr_idx[idx_off],
                            arr_idx[idx_off + 1],
                            arr_idx[idx_off + 2],
                            arr_idx[idx_off + 3],
                        ]) as usize,
                        _ => u64::from_le_bytes([
                            arr_idx[idx_off],
                            arr_idx[idx_off + 1],
                            arr_idx[idx_off + 2],
                            arr_idx[idx_off + 3],
                            arr_idx[idx_off + 4],
                            arr_idx[idx_off + 5],
                            arr_idx[idx_off + 6],
                            arr_idx[idx_off + 7],
                        ]) as usize,
                    };
                    let dst_off = i * T::BYTES;
                    if lane_idx < lanes {
                        let src_off = lane_idx * T::BYTES;
                        result[dst_off..dst_off + T::BYTES].copy_from_slice(
                            &arr_a[src_off..src_off + T::BYTES],
                        );
                    } else if lane_idx < 2 * lanes {
                        let src_off = (lane_idx - lanes) * T::BYTES;
                        result[dst_off..dst_off + T::BYTES].copy_from_slice(
                            &arr_b[src_off..src_off + T::BYTES],
                        );
                    }
                }
                V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    fn table_lookup_lanes_or0<T: Lane, I: IntegerLane>(
        self,
        v: V512<T>,
        idx: V512<I>,
    ) -> V512<T> {
        unsafe {
            let lanes = 64 / T::BYTES;
            let mut arr_v = [0u8; 64];
            let mut arr_idx = [0u8; 64];
            let mut result = [0u8; 64];
            _mm512_storeu_si512(arr_v.as_mut_ptr().cast(), v.raw);
            _mm512_storeu_si512(arr_idx.as_mut_ptr().cast(), idx.raw);
            for i in 0..lanes {
                let idx_off = i * I::BYTES;
                let lane_idx_signed: i64 = match I::BYTES {
                    1 => arr_idx[idx_off] as i8 as i64,
                    2 => i16::from_le_bytes([
                        arr_idx[idx_off],
                        arr_idx[idx_off + 1],
                    ]) as i64,
                    4 => i32::from_le_bytes([
                        arr_idx[idx_off],
                        arr_idx[idx_off + 1],
                        arr_idx[idx_off + 2],
                        arr_idx[idx_off + 3],
                    ]) as i64,
                    _ => i64::from_le_bytes([
                        arr_idx[idx_off],
                        arr_idx[idx_off + 1],
                        arr_idx[idx_off + 2],
                        arr_idx[idx_off + 3],
                        arr_idx[idx_off + 4],
                        arr_idx[idx_off + 5],
                        arr_idx[idx_off + 6],
                        arr_idx[idx_off + 7],
                    ]),
                };
                let dst_off = i * T::BYTES;
                if lane_idx_signed < 0 || lane_idx_signed as usize >= lanes {
                    for k in 0..T::BYTES {
                        result[dst_off + k] = 0;
                    }
                } else {
                    let src_off = (lane_idx_signed as usize) * T::BYTES;
                    result[dst_off..dst_off + T::BYTES]
                        .copy_from_slice(&arr_v[src_off..src_off + T::BYTES]);
                }
            }
            V512::from_raw(_mm512_loadu_si512(result.as_ptr().cast()))
        }
    }
}

// ---------------------------------------------------------------------------
// SimdReduce
// ---------------------------------------------------------------------------

// SAFETY: AVX-512 reduction intrinsics.
unsafe impl SimdReduce for Avx512 {
    #[inline(always)]
    fn sum_of_lanes<T: Lane>(self, v: V512<T>) -> T {
        unsafe {
            if is_type::<T, i32>() || is_type::<T, u32>() {
                let sum = _mm512_reduce_add_epi32(v.raw);
                core::mem::transmute_copy(&sum)
            } else if is_type::<T, i64>() || is_type::<T, u64>() {
                let sum = _mm512_reduce_add_epi64(v.raw);
                core::mem::transmute_copy(&sum)
            } else if is_type::<T, f32>() {
                let sum = _mm512_reduce_add_ps(_mm512_castsi512_ps(v.raw));
                core::mem::transmute_copy(&sum)
            } else if is_type::<T, f64>() {
                let sum = _mm512_reduce_add_pd(_mm512_castsi512_pd(v.raw));
                core::mem::transmute_copy(&sum)
            } else if is_type::<T, u8>() {
                // u8: use _mm512_sad_epu8 to sum groups of 8 bytes into u64 lanes
                let sums = _mm512_sad_epu8(v.raw, _mm512_setzero_si512());
                // sums has 8 u64 lanes, each holding sum of 8 bytes
                let total = _mm512_reduce_add_epi64(sums) as u8;
                core::mem::transmute_copy(&total)
            } else if is_type::<T, i8>() {
                // i8: bias to unsigned, use sad, then subtract bias
                // bias each byte by 128 to make unsigned, sum, then subtract 64*128
                let bias = _mm512_set1_epi8(-128i8);
                let biased = _mm512_xor_si512(v.raw, bias); // flip sign bit: i8 -> pseudo-u8
                let sums = _mm512_sad_epu8(biased, _mm512_setzero_si512());
                let total = (_mm512_reduce_add_epi64(sums) - 64 * 128) as i8;
                core::mem::transmute_copy(&total)
            } else if is_type::<T, u16>() || is_type::<T, i16>() {
                // u16/i16: use _mm512_madd_epi16 with 1s to sum pairs into i32, then reduce
                let ones = _mm512_set1_epi16(1);
                let pair_sums = _mm512_madd_epi16(v.raw, ones); // 16 x i32
                let total = _mm512_reduce_add_epi32(pair_sums);
                if is_type::<T, u16>() {
                    let r = total as u16;
                    core::mem::transmute_copy(&r)
                } else {
                    let r = total as i16;
                    core::mem::transmute_copy(&r)
                }
            } else {
                unreachable!()
            }
        }
    }

    #[inline(always)]
    fn min_of_lanes<T: Lane>(self, v: V512<T>) -> T {
        unsafe {
            if is_type::<T, i32>() {
                let r = _mm512_reduce_min_epi32(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, u32>() {
                let r = _mm512_reduce_min_epu32(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, i64>() {
                let r = _mm512_reduce_min_epi64(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, u64>() {
                let r = _mm512_reduce_min_epu64(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, f32>() {
                let r = _mm512_reduce_min_ps(_mm512_castsi512_ps(v.raw));
                core::mem::transmute_copy(&r)
            } else if is_type::<T, f64>() {
                let r = _mm512_reduce_min_pd(_mm512_castsi512_pd(v.raw));
                core::mem::transmute_copy(&r)
            } else {
                // 8/16-bit: tree reduction using SIMD min + cross-lane shuffles
                // Fold upper half into lower half, repeat until 1 lane
                let cur = v.raw;
                if T::BYTES == 1 {
                    // 64 bytes -> fold via 256-bit halves, 128-bit, 64-bit, 32-bit, 16-bit chunks
                    // Step 1: fold upper 256 bits into lower 256 bits
                    let hi256 = _mm512_extracti64x4_epi64(cur, 1);
                    let lo256 = _mm512_castsi512_si256(cur);
                    let folded256 = if is_signed::<T>() {
                        _mm256_min_epi8(lo256, hi256)
                    } else {
                        _mm256_min_epu8(lo256, hi256)
                    };
                    // Now we have 32 bytes in a __m256i; fold to __m128i
                    let hi128 = _mm256_extracti128_si256(folded256, 1);
                    let lo128 = _mm256_castsi256_si128(folded256);
                    let folded128 = if is_signed::<T>() {
                        _mm_min_epi8(lo128, hi128)
                    } else {
                        _mm_min_epu8(lo128, hi128)
                    };
                    // 16 bytes in __m128i; fold pairs
                    let shifted64 = _mm_srli_si128::<8>(folded128);
                    let folded64 = if is_signed::<T>() {
                        _mm_min_epi8(folded128, shifted64)
                    } else {
                        _mm_min_epu8(folded128, shifted64)
                    };
                    let shifted32 = _mm_srli_si128::<4>(folded64);
                    let folded32 = if is_signed::<T>() {
                        _mm_min_epi8(folded64, shifted32)
                    } else {
                        _mm_min_epu8(folded64, shifted32)
                    };
                    let shifted16 = _mm_srli_si128::<2>(folded32);
                    let folded16 = if is_signed::<T>() {
                        _mm_min_epi8(folded32, shifted16)
                    } else {
                        _mm_min_epu8(folded32, shifted16)
                    };
                    let shifted8 = _mm_srli_si128::<1>(folded16);
                    let final_val = if is_signed::<T>() {
                        _mm_min_epi8(folded16, shifted8)
                    } else {
                        _mm_min_epu8(folded16, shifted8)
                    };
                    let r = _mm_extract_epi8(final_val, 0) as u8;
                    core::mem::transmute_copy(&r)
                } else {
                    // 16-bit: 32 lanes
                    let hi256 = _mm512_extracti64x4_epi64(cur, 1);
                    let lo256 = _mm512_castsi512_si256(cur);
                    let folded256 = if is_signed::<T>() {
                        _mm256_min_epi16(lo256, hi256)
                    } else {
                        _mm256_min_epu16(lo256, hi256)
                    };
                    let hi128 = _mm256_extracti128_si256(folded256, 1);
                    let lo128 = _mm256_castsi256_si128(folded256);
                    let folded128 = if is_signed::<T>() {
                        _mm_min_epi16(lo128, hi128)
                    } else {
                        _mm_min_epu16(lo128, hi128)
                    };
                    let shifted64 = _mm_srli_si128::<8>(folded128);
                    let folded64 = if is_signed::<T>() {
                        _mm_min_epi16(folded128, shifted64)
                    } else {
                        _mm_min_epu16(folded128, shifted64)
                    };
                    let shifted32 = _mm_srli_si128::<4>(folded64);
                    let folded32 = if is_signed::<T>() {
                        _mm_min_epi16(folded64, shifted32)
                    } else {
                        _mm_min_epu16(folded64, shifted32)
                    };
                    let shifted16 = _mm_srli_si128::<2>(folded32);
                    let final_val = if is_signed::<T>() {
                        _mm_min_epi16(folded32, shifted16)
                    } else {
                        _mm_min_epu16(folded32, shifted16)
                    };
                    let r = _mm_extract_epi16(final_val, 0) as u16;
                    core::mem::transmute_copy(&r)
                }
            }
        }
    }

    #[inline(always)]
    fn max_of_lanes<T: Lane>(self, v: V512<T>) -> T {
        unsafe {
            if is_type::<T, i32>() {
                let r = _mm512_reduce_max_epi32(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, u32>() {
                let r = _mm512_reduce_max_epu32(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, i64>() {
                let r = _mm512_reduce_max_epi64(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, u64>() {
                let r = _mm512_reduce_max_epu64(v.raw);
                core::mem::transmute_copy(&r)
            } else if is_type::<T, f32>() {
                let r = _mm512_reduce_max_ps(_mm512_castsi512_ps(v.raw));
                core::mem::transmute_copy(&r)
            } else if is_type::<T, f64>() {
                let r = _mm512_reduce_max_pd(_mm512_castsi512_pd(v.raw));
                core::mem::transmute_copy(&r)
            } else {
                // 8/16-bit: tree reduction using SIMD max + cross-lane shuffles
                let cur = v.raw;
                if T::BYTES == 1 {
                    let hi256 = _mm512_extracti64x4_epi64(cur, 1);
                    let lo256 = _mm512_castsi512_si256(cur);
                    let folded256 = if is_signed::<T>() {
                        _mm256_max_epi8(lo256, hi256)
                    } else {
                        _mm256_max_epu8(lo256, hi256)
                    };
                    let hi128 = _mm256_extracti128_si256(folded256, 1);
                    let lo128 = _mm256_castsi256_si128(folded256);
                    let folded128 = if is_signed::<T>() {
                        _mm_max_epi8(lo128, hi128)
                    } else {
                        _mm_max_epu8(lo128, hi128)
                    };
                    let shifted64 = _mm_srli_si128::<8>(folded128);
                    let folded64 = if is_signed::<T>() {
                        _mm_max_epi8(folded128, shifted64)
                    } else {
                        _mm_max_epu8(folded128, shifted64)
                    };
                    let shifted32 = _mm_srli_si128::<4>(folded64);
                    let folded32 = if is_signed::<T>() {
                        _mm_max_epi8(folded64, shifted32)
                    } else {
                        _mm_max_epu8(folded64, shifted32)
                    };
                    let shifted16 = _mm_srli_si128::<2>(folded32);
                    let folded16 = if is_signed::<T>() {
                        _mm_max_epi8(folded32, shifted16)
                    } else {
                        _mm_max_epu8(folded32, shifted16)
                    };
                    let shifted8 = _mm_srli_si128::<1>(folded16);
                    let final_val = if is_signed::<T>() {
                        _mm_max_epi8(folded16, shifted8)
                    } else {
                        _mm_max_epu8(folded16, shifted8)
                    };
                    let r = _mm_extract_epi8(final_val, 0) as u8;
                    core::mem::transmute_copy(&r)
                } else {
                    // 16-bit: 32 lanes
                    let hi256 = _mm512_extracti64x4_epi64(cur, 1);
                    let lo256 = _mm512_castsi512_si256(cur);
                    let folded256 = if is_signed::<T>() {
                        _mm256_max_epi16(lo256, hi256)
                    } else {
                        _mm256_max_epu16(lo256, hi256)
                    };
                    let hi128 = _mm256_extracti128_si256(folded256, 1);
                    let lo128 = _mm256_castsi256_si128(folded256);
                    let folded128 = if is_signed::<T>() {
                        _mm_max_epi16(lo128, hi128)
                    } else {
                        _mm_max_epu16(lo128, hi128)
                    };
                    let shifted64 = _mm_srli_si128::<8>(folded128);
                    let folded64 = if is_signed::<T>() {
                        _mm_max_epi16(folded128, shifted64)
                    } else {
                        _mm_max_epu16(folded128, shifted64)
                    };
                    let shifted32 = _mm_srli_si128::<4>(folded64);
                    let folded32 = if is_signed::<T>() {
                        _mm_max_epi16(folded64, shifted32)
                    } else {
                        _mm_max_epu16(folded64, shifted32)
                    };
                    let shifted16 = _mm_srli_si128::<2>(folded32);
                    let final_val = if is_signed::<T>() {
                        _mm_max_epi16(folded32, shifted16)
                    } else {
                        _mm_max_epu16(folded32, shifted16)
                    };
                    let r = _mm_extract_epi16(final_val, 0) as u16;
                    core::mem::transmute_copy(&r)
                }
            }
        }
    }

    #[inline(always)]
    fn sums_of_8_abs_diff(self, a: V512<u8>, b: V512<u8>) -> V512<u64> {
        V512::from_raw(unsafe { _mm512_sad_epu8(a.raw, b.raw) })
    }

    #[inline(always)]
    fn sums_of_2<T: NarrowLane>(self, v: V512<T>) -> V512<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // u8/i8 -> u16/i16: pairwise add adjacent bytes
                    if is_signed::<T>() {
                        // Sign-extend even bytes, sign-extend odd bytes, add as i16
                        let even =
                            _mm512_srai_epi16(_mm512_slli_epi16(v.raw, 8), 8);
                        let odd = _mm512_srai_epi16(v.raw, 8);
                        _mm512_add_epi16(even, odd)
                    } else {
                        // Use _mm512_maddubs_epi16(v, 1s) = sum of pairs as u8*i8
                        // Since values are u8 and multiplier is 1, this is a pairwise sum
                        _mm512_maddubs_epi16(v.raw, _mm512_set1_epi8(1))
                    }
                }
                2 => {
                    if is_signed::<T>() {
                        // i16 -> i32: use _mm512_madd_epi16 with 1s for pairwise add
                        _mm512_madd_epi16(v.raw, _mm512_set1_epi16(1))
                    } else {
                        // u16 -> u32: mask even u16, shift odd right, add as u32
                        // (madd_epi16 treats inputs as signed, wrong for u16 >= 0x8000)
                        let mask = _mm512_set1_epi32(0x0000FFFFu32 as i32);
                        let even = _mm512_and_si512(v.raw, mask);
                        let odd = _mm512_srli_epi32(v.raw, 16);
                        _mm512_add_epi32(even, odd)
                    }
                }
                4 => {
                    if is_type::<T, f32>() {
                        // f32 -> f64: add even and odd pairs
                        let ps = _mm512_castsi512_ps(v.raw);
                        // Even indices: 0,2,4,6,8,10,12,14 -> lower 8 f32
                        let idx_even = _mm512_set_epi32(
                            0, 0, 0, 0, 0, 0, 0, 0, 14, 12, 10, 8, 6, 4, 2, 0,
                        );
                        let idx_odd = _mm512_set_epi32(
                            0, 0, 0, 0, 0, 0, 0, 0, 15, 13, 11, 9, 7, 5, 3, 1,
                        );
                        let even = _mm512_permutexvar_ps(idx_even, ps);
                        let odd = _mm512_permutexvar_ps(idx_odd, ps);
                        // Convert lower 8 f32 to f64 and add
                        let even_pd =
                            _mm512_cvtps_pd(_mm512_castps512_ps256(even));
                        let odd_pd =
                            _mm512_cvtps_pd(_mm512_castps512_ps256(odd));
                        _mm512_castpd_si512(_mm512_add_pd(even_pd, odd_pd))
                    } else if is_signed::<T>() {
                        // i32 -> i64: sign-extend even and odd, add
                        let even = _mm512_and_si512(
                            v.raw,
                            _mm512_set1_epi64(0x00000000FFFFFFFF_u64 as i64),
                        );
                        let even_ext =
                            _mm512_srai_epi64(_mm512_slli_epi64(even, 32), 32);
                        let odd_ext = _mm512_srai_epi64(v.raw, 32);
                        _mm512_add_epi64(even_ext, odd_ext)
                    } else {
                        // u32 -> u64: zero-extend even and odd, add
                        let mask =
                            _mm512_set1_epi64(0x00000000FFFFFFFF_u64 as i64);
                        let even = _mm512_and_si512(v.raw, mask);
                        let odd = _mm512_srli_epi64(v.raw, 32);
                        _mm512_add_epi64(even, odd)
                    }
                }
                _ => unreachable!(),
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn sums_of_4<T: NarrowLane>(
        self,
        v: V512<T>,
    ) -> V512<<T::Wide as NarrowLane>::Wide>
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

// SAFETY: AVX-512F float intrinsics.
unsafe impl SimdFloat for Avx512 {
    #[inline(always)]
    fn sqrt<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_sqrt_ps(_mm512_castsi512_ps(v.raw)))
            } else {
                _mm512_castpd_si512(_mm512_sqrt_pd(_mm512_castsi512_pd(v.raw)))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn approx_reciprocal<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            if T::BYTES == 4 {
                V512::from_raw(_mm512_castps_si512(_mm512_rcp14_ps(
                    _mm512_castsi512_ps(v.raw),
                )))
            } else {
                V512::from_raw(_mm512_castpd_si512(_mm512_rcp14_pd(
                    _mm512_castsi512_pd(v.raw),
                )))
            }
        }
    }

    #[inline(always)]
    fn approx_reciprocal_sqrt<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            if T::BYTES == 4 {
                V512::from_raw(_mm512_castps_si512(_mm512_rsqrt14_ps(
                    _mm512_castsi512_ps(v.raw),
                )))
            } else {
                V512::from_raw(_mm512_castpd_si512(_mm512_rsqrt14_pd(
                    _mm512_castsi512_pd(v.raw),
                )))
            }
        }
    }

    #[inline(always)]
    fn round<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_roundscale_ps(
                    _mm512_castsi512_ps(v.raw),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                ))
            } else {
                _mm512_castpd_si512(_mm512_roundscale_pd(
                    _mm512_castsi512_pd(v.raw),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn trunc<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_roundscale_ps(
                    _mm512_castsi512_ps(v.raw),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC,
                ))
            } else {
                _mm512_castpd_si512(_mm512_roundscale_pd(
                    _mm512_castsi512_pd(v.raw),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC,
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn ceil<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_roundscale_ps(
                    _mm512_castsi512_ps(v.raw),
                    _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
                ))
            } else {
                _mm512_castpd_si512(_mm512_roundscale_pd(
                    _mm512_castsi512_pd(v.raw),
                    _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC,
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn floor<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_roundscale_ps(
                    _mm512_castsi512_ps(v.raw),
                    _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
                ))
            } else {
                _mm512_castpd_si512(_mm512_roundscale_pd(
                    _mm512_castsi512_pd(v.raw),
                    _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC,
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn mul_add<T: FloatLane>(
        self,
        a: V512<T>,
        b: V512<T>,
        c: V512<T>,
    ) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_fmadd_ps(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _mm512_castsi512_ps(c.raw),
                ))
            } else {
                _mm512_castpd_si512(_mm512_fmadd_pd(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _mm512_castsi512_pd(c.raw),
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn neg_mul_add<T: FloatLane>(
        self,
        a: V512<T>,
        b: V512<T>,
        c: V512<T>,
    ) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_fnmadd_ps(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _mm512_castsi512_ps(c.raw),
                ))
            } else {
                _mm512_castpd_si512(_mm512_fnmadd_pd(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _mm512_castsi512_pd(c.raw),
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn mul_sub<T: FloatLane>(
        self,
        a: V512<T>,
        b: V512<T>,
        c: V512<T>,
    ) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_fmsub_ps(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _mm512_castsi512_ps(c.raw),
                ))
            } else {
                _mm512_castpd_si512(_mm512_fmsub_pd(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _mm512_castsi512_pd(c.raw),
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn neg_mul_sub<T: FloatLane>(
        self,
        a: V512<T>,
        b: V512<T>,
        c: V512<T>,
    ) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm512_castps_si512(_mm512_fnmsub_ps(
                    _mm512_castsi512_ps(a.raw),
                    _mm512_castsi512_ps(b.raw),
                    _mm512_castsi512_ps(c.raw),
                ))
            } else {
                _mm512_castpd_si512(_mm512_fnmsub_pd(
                    _mm512_castsi512_pd(a.raw),
                    _mm512_castsi512_pd(b.raw),
                    _mm512_castsi512_pd(c.raw),
                ))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn copy_sign<T: FloatLane>(self, mag: V512<T>, sign: V512<T>) -> V512<T> {
        unsafe {
            if T::BYTES == 4 {
                let sign_mask = _mm512_set1_epi32(0x8000_0000u32 as i32);
                let abs_mag = _mm512_andnot_si512(sign_mask, mag.raw);
                let sign_bit = _mm512_and_si512(sign_mask, sign.raw);
                V512::from_raw(_mm512_or_si512(abs_mag, sign_bit))
            } else {
                let sign_mask =
                    _mm512_set1_epi64(0x8000_0000_0000_0000u64 as i64);
                let abs_mag = _mm512_andnot_si512(sign_mask, mag.raw);
                let sign_bit = _mm512_and_si512(sign_mask, sign.raw);
                V512::from_raw(_mm512_or_si512(abs_mag, sign_bit))
            }
        }
    }

    #[inline(always)]
    fn is_nan<T: FloatLane>(self, v: V512<T>) -> M512<T> {
        unsafe {
            let bits = if T::BYTES == 4 {
                let ps = _mm512_castsi512_ps(v.raw);
                _mm512_cmp_ps_mask(ps, ps, _CMP_UNORD_Q) as u64
            } else {
                let pd = _mm512_castsi512_pd(v.raw);
                _mm512_cmp_pd_mask(pd, pd, _CMP_UNORD_Q) as u64
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn is_inf<T: FloatLane>(self, v: V512<T>) -> M512<T> {
        unsafe {
            let bits = if T::BYTES == 4 {
                let abs_mask = _mm512_set1_epi32(0x7FFF_FFFFu32 as i32);
                let inf_bits = _mm512_set1_epi32(0x7F80_0000u32 as i32);
                let abs_v = _mm512_and_si512(v.raw, abs_mask);
                _mm512_cmpeq_epi32_mask(abs_v, inf_bits) as u64
            } else {
                let abs_mask =
                    _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64);
                let inf_bits =
                    _mm512_set1_epi64(0x7FF0_0000_0000_0000u64 as i64);
                let abs_v = _mm512_and_si512(v.raw, abs_mask);
                _mm512_cmpeq_epi64_mask(abs_v, inf_bits) as u64
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn zero_if_negative<T: FloatLane>(self, v: V512<T>) -> V512<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                let sign = _mm512_srai_epi32(v.raw, 31);
                _mm512_andnot_si512(sign, v.raw)
            } else {
                let sign = _mm512_srai_epi64(v.raw, 63);
                _mm512_andnot_si512(sign, v.raw)
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn is_finite<T: FloatLane>(self, v: V512<T>) -> M512<T> {
        unsafe {
            let bits = if T::BYTES == 4 {
                // Exponent < 0xFF means finite.
                let shifted =
                    _mm512_srli_epi32(_mm512_slli_epi32(v.raw, 1), 24);
                let max_exp = _mm512_set1_epi32(0xFF);
                // cmplt unsigned: cmplt_epu32 not available, but both values are small,
                // so signed cmpgt works.
                _mm512_cmpgt_epi32_mask(max_exp, shifted) as u64
            } else {
                // f64: exponent < 0x7FF means finite.
                let abs_mask =
                    _mm512_set1_epi64(0x7FFF_FFFF_FFFF_FFFFu64 as i64);
                let abs_v = _mm512_and_si512(v.raw, abs_mask);
                let inf = _mm512_set1_epi64(0x7FF0_0000_0000_0000u64 as i64);
                // abs_v < inf iff finite. Use cmpgt_epi64(inf, abs_v).
                // For unsigned comparison: since abs_v has sign bit cleared and inf < i64::MAX,
                // signed comparison works here.
                _mm512_cmpgt_epi64_mask(inf, abs_v) as u64
            };
            M512::from_bits(bits)
        }
    }

    #[inline(always)]
    fn add_sub<T: FloatLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        // AVX-512 has no _mm512_addsub. Compose: OddEven(Add(a,b), Sub(a,b)).
        unsafe {
            let raw = if T::BYTES == 4 {
                let ps_a = _mm512_castsi512_ps(a.raw);
                let ps_b = _mm512_castsi512_ps(b.raw);
                let sum = _mm512_add_ps(ps_a, ps_b);
                let diff = _mm512_sub_ps(ps_a, ps_b);
                // Even lanes from diff, odd lanes from sum.
                // Use mask blend: mask bit = 1 selects from sum (odd), 0 from diff (even).
                // 16 lanes: odd positions = 0xAAAA
                _mm512_castps_si512(_mm512_mask_blend_ps(0xAAAA, diff, sum))
            } else {
                let pd_a = _mm512_castsi512_pd(a.raw);
                let pd_b = _mm512_castsi512_pd(b.raw);
                let sum = _mm512_add_pd(pd_a, pd_b);
                let diff = _mm512_sub_pd(pd_a, pd_b);
                // 8 lanes: odd positions = 0xAA
                _mm512_castpd_si512(_mm512_mask_blend_pd(0xAA, diff, sum))
            };
            V512::from_raw(raw)
        }
    }

    #[inline(always)]
    fn min_number<T: FloatLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        {
            let nan_a = self.is_nan(a);
            let min_ab = self.min(a, b);
            self.if_then_else(nan_a, b, min_ab)
        }
    }

    #[inline(always)]
    fn max_number<T: FloatLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        {
            let nan_a = self.is_nan(a);
            let max_ab = self.max(a, b);
            self.if_then_else(nan_a, b, max_ab)
        }
    }

    #[inline(always)]
    fn min_magnitude<T: FloatLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        {
            let abs_a = self.abs(a);
            let abs_b = self.abs(b);
            let abs_eq = self.eq(abs_a, abs_b);
            let abs_lt = self.lt(abs_a, abs_b);
            let eq_case = self.min(a, b);
            let sel = self.if_then_else(abs_eq, eq_case, b);
            self.if_then_else(abs_lt, a, sel)
        }
    }

    #[inline(always)]
    fn max_magnitude<T: FloatLane>(self, a: V512<T>, b: V512<T>) -> V512<T> {
        {
            let abs_a = self.abs(a);
            let abs_b = self.abs(b);
            let abs_eq = self.eq(abs_a, abs_b);
            let abs_gt = self.gt(abs_a, abs_b);
            let eq_case = self.max(a, b);
            let sel = self.if_then_else(abs_eq, eq_case, b);
            self.if_then_else(abs_gt, a, sel)
        }
    }

    #[inline(always)]
    fn is_either_nan<T: FloatLane>(self, a: V512<T>, b: V512<T>) -> M512<T> {
        self.or_mask(self.is_nan(a), self.is_nan(b))
    }
}

// ---------------------------------------------------------------------------
// SimdCrypto
// ---------------------------------------------------------------------------

/// Apply a 128-bit AES-NI intrinsic to each 128-bit block of a 512-bit vector.
macro_rules! avx512_aes_per_block {
    ($state:expr, $key:expr, $aes_intrinsic:ident, $soft_fn:ident) => {{
        let s0 = _mm512_castsi512_si128($state.raw);
        let s1 = _mm512_extracti32x4_epi32($state.raw, 1);
        let s2 = _mm512_extracti32x4_epi32($state.raw, 2);
        let s3 = _mm512_extracti32x4_epi32($state.raw, 3);
        let k0 = _mm512_castsi512_si128($key.raw);
        let k1 = _mm512_extracti32x4_epi32($key.raw, 1);
        let k2 = _mm512_extracti32x4_epi32($key.raw, 2);
        let k3 = _mm512_extracti32x4_epi32($key.raw, 3);
        if is_x86_feature_detected!("aes") {
            let r0 = $aes_intrinsic(s0, k0);
            let r1 = $aes_intrinsic(s1, k1);
            let r2 = $aes_intrinsic(s2, k2);
            let r3 = $aes_intrinsic(s3, k3);
            let lo = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
            let hi = _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
            V512::from_raw(_mm512_inserti64x4(
                _mm512_castsi256_si512(lo),
                hi,
                1,
            ))
        } else {
            let mut b0 = [0u8; 16];
            let mut b1 = [0u8; 16];
            let mut b2 = [0u8; 16];
            let mut b3 = [0u8; 16];
            let mut kk0 = [0u8; 16];
            let mut kk1 = [0u8; 16];
            let mut kk2 = [0u8; 16];
            let mut kk3 = [0u8; 16];
            _mm_storeu_si128(b0.as_mut_ptr().cast(), s0);
            _mm_storeu_si128(b1.as_mut_ptr().cast(), s1);
            _mm_storeu_si128(b2.as_mut_ptr().cast(), s2);
            _mm_storeu_si128(b3.as_mut_ptr().cast(), s3);
            _mm_storeu_si128(kk0.as_mut_ptr().cast(), k0);
            _mm_storeu_si128(kk1.as_mut_ptr().cast(), k1);
            _mm_storeu_si128(kk2.as_mut_ptr().cast(), k2);
            _mm_storeu_si128(kk3.as_mut_ptr().cast(), k3);
            super::crypto_soft::$soft_fn(&mut b0, &kk0);
            super::crypto_soft::$soft_fn(&mut b1, &kk1);
            super::crypto_soft::$soft_fn(&mut b2, &kk2);
            super::crypto_soft::$soft_fn(&mut b3, &kk3);
            let r0 = _mm_loadu_si128(b0.as_ptr().cast());
            let r1 = _mm_loadu_si128(b1.as_ptr().cast());
            let r2 = _mm_loadu_si128(b2.as_ptr().cast());
            let r3 = _mm_loadu_si128(b3.as_ptr().cast());
            let lo = _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
            let hi = _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
            V512::from_raw(_mm512_inserti64x4(
                _mm512_castsi256_si512(lo),
                hi,
                1,
            ))
        }
    }};
}

// SAFETY: AES/CLMul intrinsics are guarded by runtime feature detection.
// Falls back to per-128-bit-block AES-NI or software implementation.
unsafe impl crate::ops::SimdCrypto for Avx512 {
    #[inline(always)]
    fn aes_round(self, state: V512<u8>, round_key: V512<u8>) -> V512<u8> {
        unsafe {
            avx512_aes_per_block!(state, round_key, _mm_aesenc_si128, aes_round)
        }
    }

    #[inline(always)]
    fn aes_last_round(self, state: V512<u8>, round_key: V512<u8>) -> V512<u8> {
        unsafe {
            avx512_aes_per_block!(
                state,
                round_key,
                _mm_aesenclast_si128,
                aes_last_round
            )
        }
    }

    #[inline(always)]
    fn aes_round_inv(self, state: V512<u8>, round_key: V512<u8>) -> V512<u8> {
        unsafe {
            avx512_aes_per_block!(
                state,
                round_key,
                _mm_aesdec_si128,
                aes_round_inv
            )
        }
    }

    #[inline(always)]
    fn aes_last_round_inv(
        self,
        state: V512<u8>,
        round_key: V512<u8>,
    ) -> V512<u8> {
        unsafe {
            avx512_aes_per_block!(
                state,
                round_key,
                _mm_aesdeclast_si128,
                aes_last_round_inv
            )
        }
    }

    #[inline(always)]
    fn cl_mul_lower(self, a: V512<u64>, b: V512<u64>) -> V512<u64> {
        unsafe {
            // Split into 4 * 128-bit blocks, CLMul each pair's lower u64
            let mut arr_a = [0u64; 8];
            let mut arr_b = [0u64; 8];
            _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
            _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
            if is_x86_feature_detected!("pclmulqdq") {
                for i in (0..8).step_by(2) {
                    let va = _mm_loadu_si128(arr_a[i..].as_ptr().cast());
                    let vb = _mm_loadu_si128(arr_b[i..].as_ptr().cast());
                    let r = _mm_clmulepi64_si128(va, vb, 0x00);
                    _mm_storeu_si128(arr_a[i..].as_mut_ptr().cast(), r);
                }
            } else {
                for i in (0..8).step_by(2) {
                    let (lo, hi) =
                        super::crypto_soft::clmul_64(arr_a[i], arr_b[i]);
                    arr_a[i] = lo;
                    arr_a[i + 1] = hi;
                }
            }
            V512::from_raw(_mm512_loadu_si512(arr_a.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn cl_mul_upper(self, a: V512<u64>, b: V512<u64>) -> V512<u64> {
        unsafe {
            let mut arr_a = [0u64; 8];
            let mut arr_b = [0u64; 8];
            _mm512_storeu_si512(arr_a.as_mut_ptr().cast(), a.raw);
            _mm512_storeu_si512(arr_b.as_mut_ptr().cast(), b.raw);
            if is_x86_feature_detected!("pclmulqdq") {
                for i in (0..8).step_by(2) {
                    let va = _mm_loadu_si128(arr_a[i..].as_ptr().cast());
                    let vb = _mm_loadu_si128(arr_b[i..].as_ptr().cast());
                    let r = _mm_clmulepi64_si128(va, vb, 0x11);
                    _mm_storeu_si128(arr_a[i..].as_mut_ptr().cast(), r);
                }
            } else {
                for i in (0..8).step_by(2) {
                    let (lo, hi) = super::crypto_soft::clmul_64(
                        arr_a[i + 1],
                        arr_b[i + 1],
                    );
                    arr_a[i] = lo;
                    arr_a[i + 1] = hi;
                }
            }
            V512::from_raw(_mm512_loadu_si512(arr_a.as_ptr().cast()))
        }
    }

    #[inline(always)]
    fn aes_key_gen_assist<const RCON: i32>(self, v: V512<u8>) -> V512<u8> {
        unsafe {
            let s0 = _mm512_castsi512_si128(v.raw);
            let s1 = _mm512_extracti32x4_epi32(v.raw, 1);
            let s2 = _mm512_extracti32x4_epi32(v.raw, 2);
            let s3 = _mm512_extracti32x4_epi32(v.raw, 3);
            if is_x86_feature_detected!("aes") {
                let r0 = _mm_aeskeygenassist_si128(s0, RCON);
                let r1 = _mm_aeskeygenassist_si128(s1, RCON);
                let r2 = _mm_aeskeygenassist_si128(s2, RCON);
                let r3 = _mm_aeskeygenassist_si128(s3, RCON);
                let lo =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
                let hi =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
                V512::from_raw(_mm512_inserti64x4(
                    _mm512_castsi256_si512(lo),
                    hi,
                    1,
                ))
            } else {
                let mut b0 = [0u8; 16];
                let mut b1 = [0u8; 16];
                let mut b2 = [0u8; 16];
                let mut b3 = [0u8; 16];
                _mm_storeu_si128(b0.as_mut_ptr().cast(), s0);
                _mm_storeu_si128(b1.as_mut_ptr().cast(), s1);
                _mm_storeu_si128(b2.as_mut_ptr().cast(), s2);
                _mm_storeu_si128(b3.as_mut_ptr().cast(), s3);
                let r0 =
                    super::crypto_soft::aes_key_gen_assist(&b0, RCON as u8);
                let r1 =
                    super::crypto_soft::aes_key_gen_assist(&b1, RCON as u8);
                let r2 =
                    super::crypto_soft::aes_key_gen_assist(&b2, RCON as u8);
                let r3 =
                    super::crypto_soft::aes_key_gen_assist(&b3, RCON as u8);
                let v0 = _mm_loadu_si128(r0.as_ptr().cast());
                let v1 = _mm_loadu_si128(r1.as_ptr().cast());
                let v2 = _mm_loadu_si128(r2.as_ptr().cast());
                let v3 = _mm_loadu_si128(r3.as_ptr().cast());
                let lo =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(v0), v1, 1);
                let hi =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(v2), v3, 1);
                V512::from_raw(_mm512_inserti64x4(
                    _mm512_castsi256_si512(lo),
                    hi,
                    1,
                ))
            }
        }
    }

    #[inline(always)]
    fn aes_inv_mix_columns(self, v: V512<u8>) -> V512<u8> {
        unsafe {
            let s0 = _mm512_castsi512_si128(v.raw);
            let s1 = _mm512_extracti32x4_epi32(v.raw, 1);
            let s2 = _mm512_extracti32x4_epi32(v.raw, 2);
            let s3 = _mm512_extracti32x4_epi32(v.raw, 3);
            if is_x86_feature_detected!("aes") {
                let r0 = _mm_aesimc_si128(s0);
                let r1 = _mm_aesimc_si128(s1);
                let r2 = _mm_aesimc_si128(s2);
                let r3 = _mm_aesimc_si128(s3);
                let lo =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(r0), r1, 1);
                let hi =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(r2), r3, 1);
                V512::from_raw(_mm512_inserti64x4(
                    _mm512_castsi256_si512(lo),
                    hi,
                    1,
                ))
            } else {
                let mut b0 = [0u8; 16];
                let mut b1 = [0u8; 16];
                let mut b2 = [0u8; 16];
                let mut b3 = [0u8; 16];
                _mm_storeu_si128(b0.as_mut_ptr().cast(), s0);
                _mm_storeu_si128(b1.as_mut_ptr().cast(), s1);
                _mm_storeu_si128(b2.as_mut_ptr().cast(), s2);
                _mm_storeu_si128(b3.as_mut_ptr().cast(), s3);
                super::crypto_soft::aes_inv_mix_columns_block(&mut b0);
                super::crypto_soft::aes_inv_mix_columns_block(&mut b1);
                super::crypto_soft::aes_inv_mix_columns_block(&mut b2);
                super::crypto_soft::aes_inv_mix_columns_block(&mut b3);
                let v0 = _mm_loadu_si128(b0.as_ptr().cast());
                let v1 = _mm_loadu_si128(b1.as_ptr().cast());
                let v2 = _mm_loadu_si128(b2.as_ptr().cast());
                let v3 = _mm_loadu_si128(b3.as_ptr().cast());
                let lo =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(v0), v1, 1);
                let hi =
                    _mm256_inserti128_si256(_mm256_castsi128_si256(v2), v3, 1);
                V512::from_raw(_mm512_inserti64x4(
                    _mm512_castsi256_si512(lo),
                    hi,
                    1,
                ))
            }
        }
    }
}
