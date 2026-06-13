// All unsafe blocks in this module wrap AVX2/FMA intrinsics or transmute_copy
// for type-punning. Safety invariants are documented on the outer `unsafe impl`
// blocks; individual intrinsic calls are safe when inputs are valid __m256i.
#![allow(clippy::undocumented_unsafe_blocks)]

/// AVX2 backend.
///
/// Provides 256-bit SIMD operations via `core::arch::x86_64::_mm256_*` intrinsics.
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::marker::PhantomData;

use crate::lane::{FloatLane, IntegerLane, Lane, NarrowLane, UnsignedLane, WideLane};
use crate::ops::{
    SimdArith, SimdBitwise, SimdCompare, SimdConvert, SimdCore, SimdFloat, SimdMask, SimdMemory,
    SimdReduce, SimdShuffle,
};
use crate::simd::{self, Simd};
use crate::{A16, A32, Aligned};

// ---------------------------------------------------------------------------
// Target type
// ---------------------------------------------------------------------------

/// The AVX2 SIMD target (256-bit vectors).
#[derive(Clone, Copy, Debug)]
pub struct Avx2;

// ---------------------------------------------------------------------------
// Vector and Mask types
// ---------------------------------------------------------------------------

/// A 256-bit SIMD vector holding lanes of type `T`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct V256<T: Lane> {
    raw: __m256i,
    _marker: PhantomData<T>,
}

impl<T: Lane> core::fmt::Debug for V256<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("V256").finish_non_exhaustive()
    }
}

/// A 256-bit mask corresponding to vectors of type `T`.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct M256<T: Lane> {
    raw: __m256i,
    _marker: PhantomData<T>,
}

impl<T: Lane> core::fmt::Debug for M256<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("M256").finish_non_exhaustive()
    }
}

impl<T: Lane> V256<T> {
    #[inline(always)]
    pub(crate) fn from_raw(raw: __m256i) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }

    /// Access the raw intrinsic type.
    #[inline(always)]
    pub(crate) fn raw(self) -> __m256i {
        self.raw
    }
}

impl<T: Lane> M256<T> {
    #[inline(always)]
    pub(crate) fn from_raw(raw: __m256i) -> Self {
        Self {
            raw,
            _marker: PhantomData,
        }
    }
}

// SAFETY: AVX2 vectors are 256 bits = 32 bytes.
unsafe impl Simd for Avx2 {
    type Vec<T: Lane> = V256<T>;
    type Mask<T: Lane> = M256<T>;
    // Half-width of 256-bit is 128-bit (SSE2 types).
    type VecHalf<T: Lane> = crate::backend::sse2::V128<T>;
    type MaskHalf<T: Lane> = crate::backend::sse2::M128<T>;
    const VECTOR_BYTES: usize = 32;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use crate::lane::is_type;

#[inline(always)]
const fn is_signed<T: Lane>() -> bool {
    is_type::<T, i8>() || is_type::<T, i16>() || is_type::<T, i32>() || is_type::<T, i64>()
}

/// LUT for AVX2 Compress of 8 * u32 lanes.
/// Each entry packs 8 * 4-bit lane indices for the given 8-bit mask pattern.
/// Entry `i` contains the permutation indices for mask bits `i`.
#[rustfmt::skip]
static COMPRESS_32X8_LUT: Aligned<A16, [u32; 256]> = Aligned::new([
    0x76543210, 0x76543218, 0x76543209, 0x76543298, 0x7654310a, 0x765431a8,
    0x765430a9, 0x76543a98, 0x7654210b, 0x765421b8, 0x765420b9, 0x76542b98,
    0x765410ba, 0x76541ba8, 0x76540ba9, 0x7654ba98, 0x7653210c, 0x765321c8,
    0x765320c9, 0x76532c98, 0x765310ca, 0x76531ca8, 0x76530ca9, 0x7653ca98,
    0x765210cb, 0x76521cb8, 0x76520cb9, 0x7652cb98, 0x76510cba, 0x7651cba8,
    0x7650cba9, 0x765cba98, 0x7643210d, 0x764321d8, 0x764320d9, 0x76432d98,
    0x764310da, 0x76431da8, 0x76430da9, 0x7643da98, 0x764210db, 0x76421db8,
    0x76420db9, 0x7642db98, 0x76410dba, 0x7641dba8, 0x7640dba9, 0x764dba98,
    0x763210dc, 0x76321dc8, 0x76320dc9, 0x7632dc98, 0x76310dca, 0x7631dca8,
    0x7630dca9, 0x763dca98, 0x76210dcb, 0x7621dcb8, 0x7620dcb9, 0x762dcb98,
    0x7610dcba, 0x761dcba8, 0x760dcba9, 0x76dcba98, 0x7543210e, 0x754321e8,
    0x754320e9, 0x75432e98, 0x754310ea, 0x75431ea8, 0x75430ea9, 0x7543ea98,
    0x754210eb, 0x75421eb8, 0x75420eb9, 0x7542eb98, 0x75410eba, 0x7541eba8,
    0x7540eba9, 0x754eba98, 0x753210ec, 0x75321ec8, 0x75320ec9, 0x7532ec98,
    0x75310eca, 0x7531eca8, 0x7530eca9, 0x753eca98, 0x75210ecb, 0x7521ecb8,
    0x7520ecb9, 0x752ecb98, 0x7510ecba, 0x751ecba8, 0x750ecba9, 0x75ecba98,
    0x743210ed, 0x74321ed8, 0x74320ed9, 0x7432ed98, 0x74310eda, 0x7431eda8,
    0x7430eda9, 0x743eda98, 0x74210edb, 0x7421edb8, 0x7420edb9, 0x742edb98,
    0x7410edba, 0x741edba8, 0x740edba9, 0x74edba98, 0x73210edc, 0x7321edc8,
    0x7320edc9, 0x732edc98, 0x7310edca, 0x731edca8, 0x730edca9, 0x73edca98,
    0x7210edcb, 0x721edcb8, 0x720edcb9, 0x72edcb98, 0x710edcba, 0x71edcba8,
    0x70edcba9, 0x7edcba98, 0x6543210f, 0x654321f8, 0x654320f9, 0x65432f98,
    0x654310fa, 0x65431fa8, 0x65430fa9, 0x6543fa98, 0x654210fb, 0x65421fb8,
    0x65420fb9, 0x6542fb98, 0x65410fba, 0x6541fba8, 0x6540fba9, 0x654fba98,
    0x653210fc, 0x65321fc8, 0x65320fc9, 0x6532fc98, 0x65310fca, 0x6531fca8,
    0x6530fca9, 0x653fca98, 0x65210fcb, 0x6521fcb8, 0x6520fcb9, 0x652fcb98,
    0x6510fcba, 0x651fcba8, 0x650fcba9, 0x65fcba98, 0x643210fd, 0x64321fd8,
    0x64320fd9, 0x6432fd98, 0x64310fda, 0x6431fda8, 0x6430fda9, 0x643fda98,
    0x64210fdb, 0x6421fdb8, 0x6420fdb9, 0x642fdb98, 0x6410fdba, 0x641fdba8,
    0x640fdba9, 0x64fdba98, 0x63210fdc, 0x6321fdc8, 0x6320fdc9, 0x632fdc98,
    0x6310fdca, 0x631fdca8, 0x630fdca9, 0x63fdca98, 0x6210fdcb, 0x621fdcb8,
    0x620fdcb9, 0x62fdcb98, 0x610fdcba, 0x61fdcba8, 0x60fdcba9, 0x6fdcba98,
    0x543210fe, 0x54321fe8, 0x54320fe9, 0x5432fe98, 0x54310fea, 0x5431fea8,
    0x5430fea9, 0x543fea98, 0x54210feb, 0x5421feb8, 0x5420feb9, 0x542feb98,
    0x5410feba, 0x541feba8, 0x540feba9, 0x54feba98, 0x53210fec, 0x5321fec8,
    0x5320fec9, 0x532fec98, 0x5310feca, 0x531feca8, 0x530feca9, 0x53feca98,
    0x5210fecb, 0x521fecb8, 0x520fecb9, 0x52fecb98, 0x510fecba, 0x51fecba8,
    0x50fecba9, 0x5fecba98, 0x43210fed, 0x4321fed8, 0x4320fed9, 0x432fed98,
    0x4310feda, 0x431feda8, 0x430feda9, 0x43feda98, 0x4210fedb, 0x421fedb8,
    0x420fedb9, 0x42fedb98, 0x410fedba, 0x41fedba8, 0x40fedba9, 0x4fedba98,
    0x3210fedc, 0x321fedc8, 0x320fedc9, 0x32fedc98, 0x310fedca, 0x31fedca8,
    0x30fedca9, 0x3fedca98, 0x210fedcb, 0x21fedcb8, 0x20fedcb9, 0x2fedcb98,
    0x10fedcba, 0x1fedcba8, 0x0fedcba9, 0xfedcba98,
]);

/// LUT for AVX2 Compress of 4 * u64 lanes.
/// 16 entries * 8 u32 indices = 128 u32 values. Each group of 8 u32s is a
/// permutation for `_mm256_permutevar8x32_epi32` (pairs of u32 per u64 lane).
/// Aligned to 32 bytes so each 8*u32 entry can be loaded with `_mm256_load_si256`.
#[rustfmt::skip]
static COMPRESS_64X4_LUT: Aligned<A32, [u32; 128]> = Aligned::new([
    0,  1,  2,  3,  4,  5,  6, 7, 8, 9, 2,  3,  4,  5,  6,  7,
    10, 11, 0,  1,  4,  5,  6, 7, 8, 9, 10, 11, 4,  5,  6,  7,
    12, 13, 0,  1,  2,  3,  6, 7, 8, 9, 12, 13, 2,  3,  6,  7,
    10, 11, 12, 13, 0,  1,  6, 7, 8, 9, 10, 11, 12, 13, 6,  7,
    14, 15, 0,  1,  2,  3,  4, 5, 8, 9, 14, 15, 2,  3,  4,  5,
    10, 11, 14, 15, 0,  1,  4, 5, 8, 9, 10, 11, 14, 15, 4,  5,
    12, 13, 14, 15, 0,  1,  2, 3, 8, 9, 12, 13, 14, 15, 2,  3,
    10, 11, 12, 13, 14, 15, 0, 1, 8, 9, 10, 11, 12, 13, 14, 15,
]);

#[inline(always)]
fn read_lane<T: Lane>(arr: &[u8; 32], offset: usize) -> T {
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
fn write_lane<T: Lane>(arr: &mut [u8; 32], offset: usize, val: T) {
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

// SAFETY: All intrinsics require AVX2, guaranteed by dispatch trampoline.
unsafe impl SimdCore for Avx2 {
    #[inline(always)]
    unsafe fn zero<T: Lane>(self) -> V256<T> {
        V256::from_raw(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    unsafe fn splat<T: Lane>(self, value: T) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    let b: u8 = core::mem::transmute_copy(&value);
                    _mm256_set1_epi8(b as i8)
                }
                2 => {
                    let h: u16 = core::mem::transmute_copy(&value);
                    _mm256_set1_epi16(h as i16)
                }
                4 => {
                    let w: u32 = core::mem::transmute_copy(&value);
                    _mm256_set1_epi32(w as i32)
                }
                8 => {
                    let d: u64 = core::mem::transmute_copy(&value);
                    _mm256_set1_epi64x(d as i64)
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn undefined<T: Lane>(self) -> V256<T> {
        V256::from_raw(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    unsafe fn bitcast<T: Lane, U: Lane>(self, v: V256<T>) -> V256<U> {
        V256::from_raw(v.raw)
    }

    #[inline(always)]
    unsafe fn extract_lane<T: Lane>(self, v: V256<T>, index: usize) -> T {
        unsafe {
            let mut arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            _mm256_store_si256(arr.as_mut_ptr().cast(), v.raw);
            read_lane(arr.as_ref(), index * T::BYTES)
        }
    }

    #[inline(always)]
    unsafe fn insert_lane<T: Lane>(self, v: V256<T>, index: usize, value: T) -> V256<T> {
        unsafe {
            let mut arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            _mm256_store_si256(arr.as_mut_ptr().cast(), v.raw);
            write_lane(arr.as_mut(), index * T::BYTES, value);
            V256::from_raw(_mm256_load_si256(arr.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn iota<T: Lane>(self, base: T) -> V256<T> {
        unsafe {
            let indices = match T::BYTES {
                1 => _mm256_setr_epi8(
                    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                    22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                ),
                2 => _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
                4 => _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
                8 => _mm256_setr_epi64x(0, 1, 2, 3),
                _ => unreachable!(),
            };
            let base_splat = self.splat(base);
            let raw = match T::BYTES {
                1 => _mm256_add_epi8(base_splat.raw, indices),
                2 => _mm256_add_epi16(base_splat.raw, indices),
                4 => {
                    if is_type::<T, f32>() {
                        // For f32, convert indices to float, then add
                        _mm256_castps_si256(_mm256_add_ps(
                            _mm256_castsi256_ps(base_splat.raw),
                            _mm256_cvtepi32_ps(indices),
                        ))
                    } else {
                        _mm256_add_epi32(base_splat.raw, indices)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        // For f64, need to convert i64 indices to f64
                        let idx_f64 = _mm256_setr_pd(0.0, 1.0, 2.0, 3.0);
                        _mm256_castpd_si256(_mm256_add_pd(
                            _mm256_castsi256_pd(base_splat.raw),
                            idx_f64,
                        ))
                    } else {
                        _mm256_add_epi64(base_splat.raw, indices)
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdMemory
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdMemory for Avx2 {
    #[inline(always)]
    unsafe fn load<T: Lane>(self, ptr: *const T) -> V256<T> {
        V256::from_raw(unsafe { _mm256_load_si256(ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn load_u<T: Lane>(self, ptr: *const T) -> V256<T> {
        V256::from_raw(unsafe { _mm256_loadu_si256(ptr.cast()) })
    }

    #[inline(always)]
    unsafe fn store<T: Lane>(self, v: V256<T>, ptr: *mut T) {
        unsafe { _mm256_store_si256(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn store_u<T: Lane>(self, v: V256<T>, ptr: *mut T) {
        unsafe { _mm256_storeu_si256(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn stream<T: Lane>(self, v: V256<T>, ptr: *mut T) {
        unsafe { _mm256_stream_si256(ptr.cast(), v.raw) }
    }

    #[inline(always)]
    unsafe fn load_dup128<T: Lane>(self, ptr: *const T) -> V256<T> {
        // Use unaligned load + broadcast; ptr may not be 16-byte aligned.
        unsafe {
            let lo = _mm_loadu_si128(ptr.cast());
            let raw = _mm256_broadcastsi128_si256(lo);
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn masked_load<T: Lane>(self, mask: M256<T>, ptr: *const T) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        _mm256_castps_si256(_mm256_maskload_ps(ptr.cast(), mask.raw))
                    } else {
                        _mm256_maskload_epi32(ptr.cast(), mask.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm256_castpd_si256(_mm256_maskload_pd(ptr.cast(), mask.raw))
                    } else {
                        _mm256_maskload_epi64(ptr.cast(), mask.raw)
                    }
                }
                _ => {
                    // For 8-bit and 16-bit types, use if_then_else_zero with load_u
                    let loaded = self.load_u(ptr);
                    self.if_then_else_zero(mask, loaded).raw
                }
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn blended_store<T: Lane>(self, v: V256<T>, mask: M256<T>, ptr: *mut T) {
        unsafe {
            match T::BYTES {
                4 => {
                    if is_type::<T, f32>() {
                        _mm256_maskstore_ps(ptr.cast(), mask.raw, _mm256_castsi256_ps(v.raw));
                    } else {
                        _mm256_maskstore_epi32(ptr.cast(), mask.raw, v.raw);
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm256_maskstore_pd(ptr.cast(), mask.raw, _mm256_castsi256_pd(v.raw));
                    } else {
                        _mm256_maskstore_epi64(ptr.cast(), mask.raw, v.raw);
                    }
                }
                _ => {
                    // For 8-bit and 16-bit: load existing, blend, store
                    let existing = self.load_u(ptr);
                    let blended = self.if_then_else(mask, v, existing);
                    self.store_u(blended, ptr);
                }
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn gather_index<T: Lane>(
        self,
        base: *const T,
        idx: V256<i32>,
    ) -> V256<T> {
        unsafe {
            if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    V256::from_raw(_mm256_castps_si256(
                        _mm256_i32gather_ps::<4>(base.cast::<f32>(), idx.raw),
                    ))
                } else {
                    V256::from_raw(_mm256_i32gather_epi32::<4>(base.cast::<i32>(), idx.raw))
                }
            } else if T::BYTES == 8 {
                let idx128 = _mm256_castsi256_si128(idx.raw); // only need 4 indices
                if is_type::<T, f64>() {
                    V256::from_raw(_mm256_castpd_si256(
                        _mm256_i32gather_pd::<8>(base.cast::<f64>(), idx128),
                    ))
                } else {
                    V256::from_raw(_mm256_i32gather_epi64::<8>(base.cast::<i64>(), idx128))
                }
            } else {
                // scalar fallback for u8/u16 etc
                // Only use as many indices as available (8 i32 slots)
                let lanes = (32 / T::BYTES).min(8);
                let mut idx_arr = [0i32; 8];
                _mm256_storeu_si256(idx_arr.as_mut_ptr().cast(), idx.raw);
                let mut result: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                for i in 0..lanes {
                    let src = base.offset(idx_arr[i] as isize);
                    core::ptr::copy_nonoverlapping(
                        src.cast::<u8>(),
                        result.as_mut_ptr().add(i * T::BYTES),
                        T::BYTES,
                    );
                }
                V256::from_raw(_mm256_load_si256(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn scatter_index<T: Lane>(
        self,
        v: V256<T>,
        base: *mut T,
        idx: V256<i32>,
    ) {
        unsafe {
            // AVX2 has no native scatter; scalar loop for all types
            // Only use as many indices as available (8 i32 slots)
            let lanes = (32 / T::BYTES).min(8);
            let mut data: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut idx_arr = [0i32; 8];
            _mm256_store_si256(data.as_mut_ptr().cast(), v.raw);
            _mm256_storeu_si256(idx_arr.as_mut_ptr().cast(), idx.raw);
            for i in 0..lanes {
                let dst = base.offset(idx_arr[i] as isize);
                core::ptr::copy_nonoverlapping(
                    data.as_ptr().add(i * T::BYTES),
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
    ) -> (V256<T>, V256<T>) {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut raw: Aligned<A32, [u8; 64]> = Aligned::new([0u8; 64]);
            core::ptr::copy_nonoverlapping(
                ptr.cast::<u8>(),
                raw.as_mut_ptr(),
                lanes * 2 * T::BYTES,
            );
            let mut a: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut b: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 2) * T::BYTES),
                    a.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 2 + 1) * T::BYTES),
                    b.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V256::from_raw(_mm256_load_si256(a.as_ptr().cast())),
                V256::from_raw(_mm256_load_si256(b.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_3<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V256<T>, V256<T>, V256<T>) {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut raw: Aligned<A32, [u8; 96]> = Aligned::new([0u8; 96]);
            core::ptr::copy_nonoverlapping(
                ptr.cast::<u8>(),
                raw.as_mut_ptr(),
                lanes * 3 * T::BYTES,
            );
            let mut a: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut b: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut c: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 3) * T::BYTES),
                    a.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 3 + 1) * T::BYTES),
                    b.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 3 + 2) * T::BYTES),
                    c.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V256::from_raw(_mm256_load_si256(a.as_ptr().cast())),
                V256::from_raw(_mm256_load_si256(b.as_ptr().cast())),
                V256::from_raw(_mm256_load_si256(c.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn load_interleaved_4<T: Lane>(
        self,
        ptr: *const T,
    ) -> (V256<T>, V256<T>, V256<T>, V256<T>) {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut raw: Aligned<A32, [u8; 128]> = Aligned::new([0u8; 128]);
            core::ptr::copy_nonoverlapping(
                ptr.cast::<u8>(),
                raw.as_mut_ptr(),
                lanes * 4 * T::BYTES,
            );
            let mut a: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut b: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut c: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut d: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 4) * T::BYTES),
                    a.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 4 + 1) * T::BYTES),
                    b.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 4 + 2) * T::BYTES),
                    c.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    raw.as_ptr().add((i * 4 + 3) * T::BYTES),
                    d.as_mut_ptr().add(i * T::BYTES),
                    T::BYTES,
                );
            }
            (
                V256::from_raw(_mm256_load_si256(a.as_ptr().cast())),
                V256::from_raw(_mm256_load_si256(b.as_ptr().cast())),
                V256::from_raw(_mm256_load_si256(c.as_ptr().cast())),
                V256::from_raw(_mm256_load_si256(d.as_ptr().cast())),
            )
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_2<T: Lane>(
        self,
        v0: V256<T>,
        v1: V256<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut a: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut b: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            _mm256_store_si256(a.as_mut_ptr().cast(), v0.raw);
            _mm256_store_si256(b.as_mut_ptr().cast(), v1.raw);
            let mut raw: Aligned<A32, [u8; 64]> = Aligned::new([0u8; 64]);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    a.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 2) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    b.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 2 + 1) * T::BYTES),
                    T::BYTES,
                );
            }
            core::ptr::copy_nonoverlapping(
                raw.as_ptr(),
                ptr.cast::<u8>(),
                lanes * 2 * T::BYTES,
            );
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_3<T: Lane>(
        self,
        v0: V256<T>,
        v1: V256<T>,
        v2: V256<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut a: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut b: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut c: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            _mm256_store_si256(a.as_mut_ptr().cast(), v0.raw);
            _mm256_store_si256(b.as_mut_ptr().cast(), v1.raw);
            _mm256_store_si256(c.as_mut_ptr().cast(), v2.raw);
            let mut raw: Aligned<A32, [u8; 96]> = Aligned::new([0u8; 96]);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    a.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 3) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    b.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 3 + 1) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    c.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 3 + 2) * T::BYTES),
                    T::BYTES,
                );
            }
            core::ptr::copy_nonoverlapping(
                raw.as_ptr(),
                ptr.cast::<u8>(),
                lanes * 3 * T::BYTES,
            );
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_4<T: Lane>(
        self,
        v0: V256<T>,
        v1: V256<T>,
        v2: V256<T>,
        v3: V256<T>,
        ptr: *mut T,
    ) {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut a: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut b: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut c: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut d: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            _mm256_store_si256(a.as_mut_ptr().cast(), v0.raw);
            _mm256_store_si256(b.as_mut_ptr().cast(), v1.raw);
            _mm256_store_si256(c.as_mut_ptr().cast(), v2.raw);
            _mm256_store_si256(d.as_mut_ptr().cast(), v3.raw);
            let mut raw: Aligned<A32, [u8; 128]> = Aligned::new([0u8; 128]);
            for i in 0..lanes {
                core::ptr::copy_nonoverlapping(
                    a.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 4) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    b.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 4 + 1) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    c.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 4 + 2) * T::BYTES),
                    T::BYTES,
                );
                core::ptr::copy_nonoverlapping(
                    d.as_ptr().add(i * T::BYTES),
                    raw.as_mut_ptr().add((i * 4 + 3) * T::BYTES),
                    T::BYTES,
                );
            }
            core::ptr::copy_nonoverlapping(
                raw.as_ptr(),
                ptr.cast::<u8>(),
                lanes * 4 * T::BYTES,
            );
        }
    }

    #[inline(always)]
    unsafe fn load_expand<T: Lane>(self, mask: M256<T>, ptr: *const T) -> V256<T> {
        unsafe {
            let loaded = self.load_u(ptr);
            self.expand(loaded, mask)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdArith
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdArith for Avx2 {
    #[inline(always)]
    unsafe fn add<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm256_add_epi8(a.raw, b.raw),
                2 => _mm256_add_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm256_castps_si256(_mm256_add_ps(
                            _mm256_castsi256_ps(a.raw),
                            _mm256_castsi256_ps(b.raw),
                        ))
                    } else {
                        _mm256_add_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm256_castpd_si256(_mm256_add_pd(
                            _mm256_castsi256_pd(a.raw),
                            _mm256_castsi256_pd(b.raw),
                        ))
                    } else {
                        _mm256_add_epi64(a.raw, b.raw)
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn sub<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm256_sub_epi8(a.raw, b.raw),
                2 => _mm256_sub_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm256_castps_si256(_mm256_sub_ps(
                            _mm256_castsi256_ps(a.raw),
                            _mm256_castsi256_ps(b.raw),
                        ))
                    } else {
                        _mm256_sub_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm256_castpd_si256(_mm256_sub_pd(
                            _mm256_castsi256_pd(a.raw),
                            _mm256_castsi256_pd(b.raw),
                        ))
                    } else {
                        _mm256_sub_epi64(a.raw, b.raw)
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn mul<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // No _mm256_mullo_epi8; emulate with 16-bit
                    let mask = _mm256_set1_epi16(0x00FF);
                    let a_lo = _mm256_and_si256(a.raw, mask);
                    let b_lo = _mm256_and_si256(b.raw, mask);
                    let mul_lo = _mm256_and_si256(_mm256_mullo_epi16(a_lo, b_lo), mask);
                    let a_hi = _mm256_srli_epi16(a.raw, 8);
                    let b_hi = _mm256_srli_epi16(b.raw, 8);
                    let mul_hi = _mm256_slli_epi16(_mm256_mullo_epi16(a_hi, b_hi), 8);
                    _mm256_or_si256(mul_lo, mul_hi)
                }
                2 => _mm256_mullo_epi16(a.raw, b.raw),
                4 => {
                    if is_type::<T, f32>() {
                        _mm256_castps_si256(_mm256_mul_ps(
                            _mm256_castsi256_ps(a.raw),
                            _mm256_castsi256_ps(b.raw),
                        ))
                    } else {
                        _mm256_mullo_epi32(a.raw, b.raw)
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        _mm256_castpd_si256(_mm256_mul_pd(
                            _mm256_castsi256_pd(a.raw),
                            _mm256_castsi256_pd(b.raw),
                        ))
                    } else {
                        // 64-bit integer mul emulation
                        let a_hi = _mm256_srli_epi64(a.raw, 32);
                        let b_hi = _mm256_srli_epi64(b.raw, 32);
                        let mul_ll = _mm256_mul_epu32(a.raw, b.raw);
                        let mul_lh = _mm256_mul_epu32(a.raw, b_hi);
                        let mul_hl = _mm256_mul_epu32(a_hi, b.raw);
                        let cross = _mm256_add_epi64(mul_lh, mul_hl);
                        _mm256_add_epi64(mul_ll, _mm256_slli_epi64(cross, 32))
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn div<T: FloatLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_div_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                ))
            } else {
                _mm256_castpd_si256(_mm256_div_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn saturated_add<T: IntegerLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    if is_type::<T, u8>() {
                        _mm256_adds_epu8(a.raw, b.raw)
                    } else {
                        _mm256_adds_epi8(a.raw, b.raw)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm256_adds_epu16(a.raw, b.raw)
                    } else {
                        _mm256_adds_epi16(a.raw, b.raw)
                    }
                }
                4 => {
                    if is_type::<T, u32>() {
                        // Unsigned: saturated_add(a, b) = a + min(b, ~a)
                        // ~a is the max value we can add without overflow
                        let not_a = _mm256_xor_si256(a.raw, _mm256_set1_epi8(!0));
                        let clamped_b = _mm256_min_epu32(b.raw, not_a);
                        _mm256_add_epi32(a.raw, clamped_b)
                    } else {
                        // Signed i32: detect overflow with sign checks
                        // Overflow if a and b have same sign but result has different sign
                        let sum = _mm256_add_epi32(a.raw, b.raw);
                        // Overflow when: (a ^ sum) & (b ^ sum) has sign bit set
                        let overflow = _mm256_and_si256(
                            _mm256_xor_si256(a.raw, sum),
                            _mm256_xor_si256(b.raw, sum),
                        );
                        // overflow has sign bit set in lanes that overflowed
                        let overflow_mask = _mm256_srai_epi32(overflow, 31); // all-ones if overflowed
                        // Saturated value: if a was positive (overflow to +), use MAX; if negative, use MIN
                        let a_sign = _mm256_srai_epi32(a.raw, 31); // all-ones if a < 0
                        // sat_val = a < 0 ? i32::MIN : i32::MAX
                        let sat_val = _mm256_xor_si256(
                            _mm256_set1_epi32(i32::MAX),
                            a_sign, // XOR with all-ones flips to MIN (0x80000000)
                        );
                        // Select: if overflowed, use sat_val; else use sum
                        _mm256_or_si256(
                            _mm256_and_si256(overflow_mask, sat_val),
                            _mm256_andnot_si256(overflow_mask, sum),
                        )
                    }
                }
                8 => {
                    if is_type::<T, u64>() {
                        // Unsigned u64: saturated_add(a, b) = a + min(b, ~a)
                        // No _mm256_min_epu64 in AVX2, use comparison
                        let not_a = _mm256_xor_si256(a.raw, _mm256_set1_epi8(!0));
                        // Unsigned 64-bit compare: flip sign bits and use signed compare
                        let sign_flip = _mm256_set1_epi64x(i64::MIN);
                        let not_a_s = _mm256_xor_si256(not_a, sign_flip);
                        let b_s = _mm256_xor_si256(b.raw, sign_flip);
                        // b > not_a means overflow
                        let b_gt = _mm256_cmpgt_epi64(b_s, not_a_s); // all-ones if b > ~a
                        // If overflow, clamp b to ~a
                        let clamped_b = _mm256_or_si256(
                            _mm256_andnot_si256(b_gt, b.raw),
                            _mm256_and_si256(b_gt, not_a),
                        );
                        _mm256_add_epi64(a.raw, clamped_b)
                    } else {
                        // Signed i64
                        let sum = _mm256_add_epi64(a.raw, b.raw);
                        let overflow = _mm256_and_si256(
                            _mm256_xor_si256(a.raw, sum),
                            _mm256_xor_si256(b.raw, sum),
                        );
                        // Need sign bit extended to full 64-bit mask
                        // No _mm256_srai_epi64 in AVX2; use shuffle+srai_epi32
                        // Sign of 64-bit value is in bit 63 = high bit of high dword
                        let overflow_sign = _mm256_srai_epi32(overflow, 31);
                        // Broadcast the sign of the high dword to both dwords: shuffle [1,1,3,3,5,5,7,7]
                        let overflow_mask = _mm256_shuffle_epi32(overflow_sign, 0xF5);

                        let a_sign = _mm256_srai_epi32(a.raw, 31);
                        let a_sign_mask = _mm256_shuffle_epi32(a_sign, 0xF5);
                        let sat_val = _mm256_xor_si256(_mm256_set1_epi64x(i64::MAX), a_sign_mask);
                        _mm256_or_si256(
                            _mm256_and_si256(overflow_mask, sat_val),
                            _mm256_andnot_si256(overflow_mask, sum),
                        )
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn saturated_sub<T: IntegerLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    if is_type::<T, u8>() {
                        _mm256_subs_epu8(a.raw, b.raw)
                    } else {
                        _mm256_subs_epi8(a.raw, b.raw)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm256_subs_epu16(a.raw, b.raw)
                    } else {
                        _mm256_subs_epi16(a.raw, b.raw)
                    }
                }
                4 => {
                    if is_type::<T, u32>() {
                        // Unsigned: saturated_sub(a, b) = a - min(a, b)
                        let min_ab = _mm256_min_epu32(a.raw, b.raw);
                        _mm256_sub_epi32(a.raw, min_ab)
                    } else {
                        // Signed i32: detect underflow
                        let diff = _mm256_sub_epi32(a.raw, b.raw);
                        // Underflow when a and b have different signs and result sign != a's sign
                        // Equivalently: (a ^ b) & (a ^ diff) has sign bit set
                        let underflow = _mm256_and_si256(
                            _mm256_xor_si256(a.raw, b.raw),
                            _mm256_xor_si256(a.raw, diff),
                        );
                        let underflow_mask = _mm256_srai_epi32(underflow, 31);
                        let a_sign = _mm256_srai_epi32(a.raw, 31);
                        // sat_val = a >= 0 ? i32::MAX : i32::MIN
                        let sat_val = _mm256_xor_si256(_mm256_set1_epi32(i32::MAX), a_sign);
                        _mm256_or_si256(
                            _mm256_and_si256(underflow_mask, sat_val),
                            _mm256_andnot_si256(underflow_mask, diff),
                        )
                    }
                }
                8 => {
                    if is_type::<T, u64>() {
                        // Unsigned u64: saturated_sub(a, b) = a - min(a, b)
                        // No _mm256_min_epu64 in AVX2, emulate with compare
                        let sign_flip = _mm256_set1_epi64x(i64::MIN);
                        let a_s = _mm256_xor_si256(a.raw, sign_flip);
                        let b_s = _mm256_xor_si256(b.raw, sign_flip);
                        // a < b (unsigned) iff a_s < b_s (signed)
                        let b_gt_a = _mm256_cmpgt_epi64(b_s, a_s); // all-ones if b > a
                        // min(a, b): if b > a then a, else b
                        let min_ab = _mm256_or_si256(
                            _mm256_and_si256(b_gt_a, a.raw),
                            _mm256_andnot_si256(b_gt_a, b.raw),
                        );
                        _mm256_sub_epi64(a.raw, min_ab)
                    } else {
                        // Signed i64
                        let diff = _mm256_sub_epi64(a.raw, b.raw);
                        let underflow = _mm256_and_si256(
                            _mm256_xor_si256(a.raw, b.raw),
                            _mm256_xor_si256(a.raw, diff),
                        );
                        let underflow_sign = _mm256_srai_epi32(underflow, 31);
                        let underflow_mask = _mm256_shuffle_epi32(underflow_sign, 0xF5);

                        let a_sign = _mm256_srai_epi32(a.raw, 31);
                        let a_sign_mask = _mm256_shuffle_epi32(a_sign, 0xF5);
                        let sat_val = _mm256_xor_si256(_mm256_set1_epi64x(i64::MAX), a_sign_mask);
                        _mm256_or_si256(
                            _mm256_and_si256(underflow_mask, sat_val),
                            _mm256_andnot_si256(underflow_mask, diff),
                        )
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn abs<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            if is_type::<T, f32>() {
                let mask = _mm256_set1_epi32(0x7FFF_FFFFu32 as i32);
                V256::from_raw(_mm256_and_si256(v.raw, mask))
            } else if is_type::<T, f64>() {
                let mask = _mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64);
                V256::from_raw(_mm256_and_si256(v.raw, mask))
            } else if is_signed::<T>() {
                let raw = match T::BYTES {
                    1 => _mm256_abs_epi8(v.raw),
                    2 => _mm256_abs_epi16(v.raw),
                    4 => _mm256_abs_epi32(v.raw),
                    8 => {
                        // No _mm256_abs_epi64 in AVX2; emulate
                        let sign = _mm256_cmpgt_epi64(_mm256_setzero_si256(), v.raw);
                        _mm256_sub_epi64(_mm256_xor_si256(v.raw, sign), sign)
                    }
                    _ => unreachable!(),
                };
                V256::from_raw(raw)
            } else {
                v // unsigned: already positive
            }
        }
    }

    #[inline(always)]
    unsafe fn neg<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            if is_type::<T, f32>() {
                let sign = _mm256_set1_epi32(0x8000_0000u32 as i32);
                V256::from_raw(_mm256_xor_si256(v.raw, sign))
            } else if is_type::<T, f64>() {
                let sign = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
                V256::from_raw(_mm256_xor_si256(v.raw, sign))
            } else {
                let z = _mm256_setzero_si256();
                let raw = match T::BYTES {
                    1 => _mm256_sub_epi8(z, v.raw),
                    2 => _mm256_sub_epi16(z, v.raw),
                    4 => _mm256_sub_epi32(z, v.raw),
                    8 => _mm256_sub_epi64(z, v.raw),
                    _ => unreachable!(),
                };
                V256::from_raw(raw)
            }
        }
    }

    #[inline(always)]
    unsafe fn min<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm256_castps_si256(_mm256_min_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                ))
            } else if is_type::<T, f64>() {
                _mm256_castpd_si256(_mm256_min_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                ))
            } else {
                match T::BYTES {
                    1 => {
                        if is_signed::<T>() {
                            _mm256_min_epi8(a.raw, b.raw)
                        } else {
                            _mm256_min_epu8(a.raw, b.raw)
                        }
                    }
                    2 => {
                        if is_signed::<T>() {
                            _mm256_min_epi16(a.raw, b.raw)
                        } else {
                            _mm256_min_epu16(a.raw, b.raw)
                        }
                    }
                    4 => {
                        if is_signed::<T>() {
                            _mm256_min_epi32(a.raw, b.raw)
                        } else {
                            _mm256_min_epu32(a.raw, b.raw)
                        }
                    }
                    8 => {
                        // Emulate 64-bit min
                        let lt = self.lt::<T>(a, b).raw;
                        _mm256_or_si256(_mm256_and_si256(lt, a.raw), _mm256_andnot_si256(lt, b.raw))
                    }
                    _ => unreachable!(),
                }
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn max<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm256_castps_si256(_mm256_max_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                ))
            } else if is_type::<T, f64>() {
                _mm256_castpd_si256(_mm256_max_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                ))
            } else {
                match T::BYTES {
                    1 => {
                        if is_signed::<T>() {
                            _mm256_max_epi8(a.raw, b.raw)
                        } else {
                            _mm256_max_epu8(a.raw, b.raw)
                        }
                    }
                    2 => {
                        if is_signed::<T>() {
                            _mm256_max_epi16(a.raw, b.raw)
                        } else {
                            _mm256_max_epu16(a.raw, b.raw)
                        }
                    }
                    4 => {
                        if is_signed::<T>() {
                            _mm256_max_epi32(a.raw, b.raw)
                        } else {
                            _mm256_max_epu32(a.raw, b.raw)
                        }
                    }
                    8 => {
                        let gt = self.gt::<T>(a, b).raw;
                        _mm256_or_si256(_mm256_and_si256(gt, a.raw), _mm256_andnot_si256(gt, b.raw))
                    }
                    _ => unreachable!(),
                }
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn mul_high<T: IntegerLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    let mask_lo = _mm256_set1_epi16(0x00FF);
                    let mask_hi = _mm256_set1_epi16(0xFF00u16 as i16);
                    if is_type::<T, u8>() {
                        let a_even = _mm256_and_si256(a.raw, mask_lo);
                        let b_even = _mm256_and_si256(b.raw, mask_lo);
                        let prod_even = _mm256_mullo_epi16(a_even, b_even);
                        let hi_even = _mm256_srli_epi16(prod_even, 8);
                        let a_odd = _mm256_srli_epi16(a.raw, 8);
                        let b_odd = _mm256_srli_epi16(b.raw, 8);
                        let prod_odd = _mm256_mullo_epi16(a_odd, b_odd);
                        let hi_odd = _mm256_and_si256(prod_odd, mask_hi);
                        _mm256_or_si256(hi_even, hi_odd)
                    } else {
                        let a_even = _mm256_srai_epi16(_mm256_slli_epi16(a.raw, 8), 8);
                        let b_even = _mm256_srai_epi16(_mm256_slli_epi16(b.raw, 8), 8);
                        let prod_even = _mm256_mullo_epi16(a_even, b_even);
                        let hi_even = _mm256_srli_epi16(prod_even, 8);
                        let a_odd = _mm256_srai_epi16(a.raw, 8);
                        let b_odd = _mm256_srai_epi16(b.raw, 8);
                        let prod_odd = _mm256_mullo_epi16(a_odd, b_odd);
                        let hi_odd = _mm256_and_si256(prod_odd, mask_hi);
                        _mm256_or_si256(hi_even, hi_odd)
                    }
                }
                2 => {
                    if is_type::<T, u16>() {
                        _mm256_mulhi_epu16(a.raw, b.raw)
                    } else {
                        _mm256_mulhi_epi16(a.raw, b.raw)
                    }
                }
                _ => {
                    if is_type::<T, u32>() {
                        let p_even = _mm256_mul_epu32(a.raw, b.raw);
                        let a_odd = _mm256_srli_epi64(a.raw, 32);
                        let b_odd = _mm256_srli_epi64(b.raw, 32);
                        let p_odd = _mm256_mul_epu32(a_odd, b_odd);
                        let hi_even = _mm256_srli_epi64(p_even, 32);
                        let mask_hi32 = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
                        let hi_odd = _mm256_and_si256(p_odd, mask_hi32);
                        _mm256_or_si256(hi_even, hi_odd)
                    } else {
                        // i32: use native _mm256_mul_epi32 with shift+mask+or
                        let p_even = _mm256_mul_epi32(a.raw, b.raw);
                        let a_odd = _mm256_srli_epi64(a.raw, 32);
                        let b_odd = _mm256_srli_epi64(b.raw, 32);
                        let p_odd = _mm256_mul_epi32(a_odd, b_odd);
                        let hi_even = _mm256_srli_epi64(p_even, 32);
                        let mask_hi32 = _mm256_set_epi32(-1, 0, -1, 0, -1, 0, -1, 0);
                        let hi_odd = _mm256_and_si256(p_odd, mask_hi32);
                        _mm256_or_si256(hi_even, hi_odd)
                    }
                }
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn average_round<T: UnsignedLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm256_avg_epu8(a.raw, b.raw),
                2 => _mm256_avg_epu16(a.raw, b.raw),
                4 => {
                    // avg(a,b) = (a >> 1) + (b >> 1) + ((a | b) & 1)
                    // Uses logical right shift since T: UnsignedLane
                    let a_half = _mm256_srli_epi32(a.raw, 1);
                    let b_half = _mm256_srli_epi32(b.raw, 1);
                    let carry =
                        _mm256_and_si256(_mm256_or_si256(a.raw, b.raw), _mm256_set1_epi32(1));
                    _mm256_add_epi32(_mm256_add_epi32(a_half, b_half), carry)
                }
                8 => {
                    let a_half = _mm256_srli_epi64(a.raw, 1);
                    let b_half = _mm256_srli_epi64(b.raw, 1);
                    let carry =
                        _mm256_and_si256(_mm256_or_si256(a.raw, b.raw), _mm256_set1_epi64x(1));
                    _mm256_add_epi64(_mm256_add_epi64(a_half, b_half), carry)
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn abs_diff<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.sub(self.max(a, b), self.min(a, b)) }
    }

    #[inline(always)]
    unsafe fn clamp<T: Lane>(self, v: V256<T>, lo: V256<T>, hi: V256<T>) -> V256<T> {
        unsafe { self.min(self.max(v, lo), hi) }
    }

    #[inline(always)]
    unsafe fn mul_even<T: NarrowLane>(self, a: V256<T>, b: V256<T>) -> V256<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            if is_type::<T, u32>() {
                V256::from_raw(_mm256_mul_epu32(a.raw, b.raw))
            } else if is_type::<T, i32>() {
                V256::from_raw(_mm256_mul_epi32(a.raw, b.raw))
            } else if T::BYTES == 1 {
                // u8/i8 -> u16/i16: extract even bytes, widen, multiply
                if is_signed::<T>() {
                    let a16 = _mm256_srai_epi16(_mm256_slli_epi16(a.raw, 8), 8);
                    let b16 = _mm256_srai_epi16(_mm256_slli_epi16(b.raw, 8), 8);
                    V256::from_raw(_mm256_mullo_epi16(a16, b16))
                } else {
                    let mask = _mm256_set1_epi16(0x00FFu16 as i16);
                    let a16 = _mm256_and_si256(a.raw, mask);
                    let b16 = _mm256_and_si256(b.raw, mask);
                    V256::from_raw(_mm256_mullo_epi16(a16, b16))
                }
            } else if T::BYTES == 2 {
                // u16/i16 -> u32/i32: extract even 16-bit lanes, widen, multiply
                if is_signed::<T>() {
                    let a32 = _mm256_srai_epi32(_mm256_slli_epi32(a.raw, 16), 16);
                    let b32 = _mm256_srai_epi32(_mm256_slli_epi32(b.raw, 16), 16);
                    V256::from_raw(_mm256_mullo_epi32(a32, b32))
                } else {
                    let mask = _mm256_set1_epi32(0x0000FFFFu32 as i32);
                    let a32 = _mm256_and_si256(a.raw, mask);
                    let b32 = _mm256_and_si256(b.raw, mask);
                    V256::from_raw(_mm256_mullo_epi32(a32, b32))
                }
            } else if is_type::<T, f32>() {
                // f32 -> f64: extract even f32 lanes, convert to f64, multiply
                let a_ps = _mm256_castsi256_ps(a.raw);
                let b_ps = _mm256_castsi256_ps(b.raw);
                let a_lo = _mm256_castps256_ps128(a_ps);
                let a_hi = _mm256_extractf128_ps(a_ps, 1);
                let a_even = _mm_shuffle_ps(a_lo, a_hi, 0x88); // [a0,a2,a4,a6]
                let b_lo = _mm256_castps256_ps128(b_ps);
                let b_hi = _mm256_extractf128_ps(b_ps, 1);
                let b_even = _mm_shuffle_ps(b_lo, b_hi, 0x88);
                let a_pd = _mm256_cvtps_pd(a_even);
                let b_pd = _mm256_cvtps_pd(b_even);
                V256::from_raw(_mm256_castpd_si256(_mm256_mul_pd(a_pd, b_pd)))
            } else {
                unreachable!()
            }
        }
    }

    #[inline(always)]
    unsafe fn mul_odd<T: NarrowLane>(self, a: V256<T>, b: V256<T>) -> V256<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            if is_type::<T, u32>() {
                // Shift right by 32 to move odd lanes to even positions, then mul_epu32
                let a_odd = _mm256_srli_epi64(a.raw, 32);
                let b_odd = _mm256_srli_epi64(b.raw, 32);
                V256::from_raw(_mm256_mul_epu32(a_odd, b_odd))
            } else if is_type::<T, i32>() {
                let a_odd = _mm256_srli_epi64(a.raw, 32);
                let b_odd = _mm256_srli_epi64(b.raw, 32);
                V256::from_raw(_mm256_mul_epi32(a_odd, b_odd))
            } else if T::BYTES == 1 {
                // u8/i8 -> u16/i16: extract odd bytes, widen, multiply
                if is_signed::<T>() {
                    let a16 = _mm256_srai_epi16(a.raw, 8);
                    let b16 = _mm256_srai_epi16(b.raw, 8);
                    V256::from_raw(_mm256_mullo_epi16(a16, b16))
                } else {
                    let a16 = _mm256_srli_epi16(a.raw, 8);
                    let b16 = _mm256_srli_epi16(b.raw, 8);
                    V256::from_raw(_mm256_mullo_epi16(a16, b16))
                }
            } else if T::BYTES == 2 {
                // u16/i16 -> u32/i32: extract odd 16-bit lanes, widen, multiply
                if is_signed::<T>() {
                    let a32 = _mm256_srai_epi32(a.raw, 16);
                    let b32 = _mm256_srai_epi32(b.raw, 16);
                    V256::from_raw(_mm256_mullo_epi32(a32, b32))
                } else {
                    let a32 = _mm256_srli_epi32(a.raw, 16);
                    let b32 = _mm256_srli_epi32(b.raw, 16);
                    V256::from_raw(_mm256_mullo_epi32(a32, b32))
                }
            } else if is_type::<T, f32>() {
                // f32 -> f64: extract odd f32 lanes, convert to f64, multiply
                let a_ps = _mm256_castsi256_ps(a.raw);
                let b_ps = _mm256_castsi256_ps(b.raw);
                let a_lo = _mm256_castps256_ps128(a_ps);
                let a_hi = _mm256_extractf128_ps(a_ps, 1);
                let a_odd = _mm_shuffle_ps(a_lo, a_hi, 0xDD); // [a1,a3,a5,a7]
                let b_lo = _mm256_castps256_ps128(b_ps);
                let b_hi = _mm256_extractf128_ps(b_ps, 1);
                let b_odd = _mm_shuffle_ps(b_lo, b_hi, 0xDD);
                let a_pd = _mm256_cvtps_pd(a_odd);
                let b_pd = _mm256_cvtps_pd(b_odd);
                V256::from_raw(_mm256_castpd_si256(_mm256_mul_pd(a_pd, b_pd)))
            } else {
                unreachable!()
            }
        }
    }

    #[inline(always)]
    unsafe fn widen_mul_pairwise_add_i16(
        self,
        a: V256<i16>,
        b: V256<i16>,
    ) -> V256<i32> {
        V256::from_raw(unsafe { _mm256_madd_epi16(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn sat_widen_mul_pairwise_add(
        self,
        a: V256<u8>,
        b: V256<i8>,
    ) -> V256<i16> {
        V256::from_raw(unsafe { _mm256_maddubs_epi16(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn mul_fixed_point_15(
        self,
        a: V256<i16>,
        b: V256<i16>,
    ) -> V256<i16> {
        V256::from_raw(unsafe { _mm256_mulhrs_epi16(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn reorder_widen_mul_accumulate(
        self,
        a: V256<i16>,
        b: V256<i16>,
        sum: V256<i32>,
    ) -> V256<i32> {
        V256::from_raw(unsafe { _mm256_add_epi32(sum.raw, _mm256_madd_epi16(a.raw, b.raw)) })
    }

    #[inline(always)]
    unsafe fn saturated_neg<T: IntegerLane>(self, v: V256<T>) -> V256<T> {
        unsafe { self.saturated_sub(self.zero::<T>(), v) }
    }

    #[inline(always)]
    unsafe fn saturated_abs<T: IntegerLane>(self, v: V256<T>) -> V256<T> {
        unsafe { self.max(v, self.saturated_neg(v)) }
    }

    #[inline(always)]
    unsafe fn masked_min_or<T: Lane>(self, no: V256<T>, mask: M256<T>, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.if_then_else(mask, self.min(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_max_or<T: Lane>(self, no: V256<T>, mask: M256<T>, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.if_then_else(mask, self.max(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_add_or<T: Lane>(self, no: V256<T>, mask: M256<T>, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.if_then_else(mask, self.add(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_sub_or<T: Lane>(self, no: V256<T>, mask: M256<T>, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.if_then_else(mask, self.sub(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_mul_or<T: Lane>(self, no: V256<T>, mask: M256<T>, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.if_then_else(mask, self.mul(a, b), no) }
    }
}

// ---------------------------------------------------------------------------
// SimdBitwise
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdBitwise for Avx2 {
    #[inline(always)]
    unsafe fn and<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        V256::from_raw(unsafe { _mm256_and_si256(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn or<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        V256::from_raw(unsafe { _mm256_or_si256(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn xor<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        V256::from_raw(unsafe { _mm256_xor_si256(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn not<T: Lane>(self, v: V256<T>) -> V256<T> {
        let all_ones = unsafe { _mm256_set1_epi8(!0) };
        V256::from_raw(unsafe { _mm256_xor_si256(v.raw, all_ones) })
    }

    #[inline(always)]
    unsafe fn and_not<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        V256::from_raw(unsafe { _mm256_andnot_si256(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn shift_left<T: IntegerLane, const BITS: u32>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(BITS as i64);
            let raw = match T::BYTES {
                1 => {
                    let shifted = _mm256_sll_epi16(v.raw, count);
                    let mask = _mm256_set1_epi8((0xFFu8.wrapping_shl(BITS)) as i8);
                    _mm256_and_si256(shifted, mask)
                }
                2 => _mm256_sll_epi16(v.raw, count),
                4 => _mm256_sll_epi32(v.raw, count),
                8 => _mm256_sll_epi64(v.raw, count),
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn shift_right<T: IntegerLane, const BITS: u32>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(BITS as i64);
            let raw = match T::BYTES {
                1 => {
                    if is_signed::<T>() {
                        let count8 = _mm_cvtsi64_si128(8i64);
                        let count_plus_8 = _mm_cvtsi64_si128((BITS + 8) as i64);
                        let shifted =
                            _mm256_sra_epi16(_mm256_sll_epi16(v.raw, count8), count_plus_8);
                        let mask = _mm256_set1_epi16(0x00FF);
                        let lo = _mm256_and_si256(shifted, mask);
                        let hi = _mm256_andnot_si256(mask, _mm256_sra_epi16(v.raw, count));
                        _mm256_or_si256(lo, hi)
                    } else {
                        let shifted = _mm256_srl_epi16(v.raw, count);
                        let mask = _mm256_set1_epi8((0xFFu8.wrapping_shr(BITS)) as i8);
                        _mm256_and_si256(shifted, mask)
                    }
                }
                2 => {
                    if is_signed::<T>() {
                        _mm256_sra_epi16(v.raw, count)
                    } else {
                        _mm256_srl_epi16(v.raw, count)
                    }
                }
                4 => {
                    if is_signed::<T>() {
                        _mm256_sra_epi32(v.raw, count)
                    } else {
                        _mm256_srl_epi32(v.raw, count)
                    }
                }
                8 => {
                    if is_signed::<T>() {
                        // AVX2 has no _mm256_sra_epi64. Emulate arithmetic
                        // right shift: sign-extend the high bit, then blend.
                        let sign = _mm256_srai_epi32(v.raw, 31);
                        // Broadcast each 64-bit lane's sign to all 64 bits:
                        let sign64 = _mm256_shuffle_epi32(sign, 0xF5); // [1,1,3,3,5,5,7,7]
                        // Logical shift the value
                        let shifted = _mm256_srl_epi64(v.raw, count);
                        // Build a mask of the sign bits that were shifted in
                        let ones = _mm256_set1_epi64x(-1i64);
                        let sign_mask =
                            _mm256_sll_epi64(ones, _mm_cvtsi64_si128(64i64 - BITS as i64));
                        // Where sign is negative, fill shifted-in positions with 1s
                        let fill = _mm256_and_si256(sign64, sign_mask);
                        _mm256_or_si256(shifted, fill)
                    } else {
                        _mm256_srl_epi64(v.raw, count)
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn rotate_right<T: IntegerLane, const BITS: u32>(self, v: V256<T>) -> V256<T> {
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
                    // No native 8-bit shifts; use 16-bit shifts with masking
                    let shr = _mm256_and_si256(
                        _mm256_srl_epi16(v.raw, count_r),
                        _mm256_set1_epi8((0xFFu8.wrapping_shr(right)) as i8),
                    );
                    let shl = _mm256_and_si256(
                        _mm256_sll_epi16(v.raw, count_l),
                        _mm256_set1_epi8((0xFFu8.wrapping_shl(left)) as i8),
                    );
                    _mm256_or_si256(shr, shl)
                }
                2 => {
                    let shr = _mm256_srl_epi16(v.raw, count_r);
                    let shl = _mm256_sll_epi16(v.raw, count_l);
                    _mm256_or_si256(shr, shl)
                }
                4 => {
                    let shr = _mm256_srl_epi32(v.raw, count_r);
                    let shl = _mm256_sll_epi32(v.raw, count_l);
                    _mm256_or_si256(shr, shl)
                }
                8 => {
                    let shr = _mm256_srl_epi64(v.raw, count_r);
                    let shl = _mm256_sll_epi64(v.raw, count_l);
                    _mm256_or_si256(shr, shl)
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn shift_left_same<T: IntegerLane>(self, v: V256<T>, bits: u32) -> V256<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(bits as i64);
            let raw = match T::BYTES {
                1 => {
                    let shifted = _mm256_sll_epi16(v.raw, count);
                    let mask = _mm256_set1_epi8((0xFFu8.wrapping_shl(bits)) as i8);
                    _mm256_and_si256(shifted, mask)
                }
                2 => _mm256_sll_epi16(v.raw, count),
                4 => _mm256_sll_epi32(v.raw, count),
                8 => _mm256_sll_epi64(v.raw, count),
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn shift_right_same<T: IntegerLane>(self, v: V256<T>, bits: u32) -> V256<T> {
        unsafe {
            let count = _mm_cvtsi64_si128(bits as i64);
            let raw = match T::BYTES {
                1 => {
                    if is_signed::<T>() {
                        // Emulate arithmetic shift right for i8
                        let count8 = _mm_cvtsi64_si128(8i64);
                        let count_plus_8 = _mm_cvtsi64_si128((bits + 8) as i64);
                        let shifted =
                            _mm256_sra_epi16(_mm256_sll_epi16(v.raw, count8), count_plus_8);
                        let mask = _mm256_set1_epi16(0x00FF);
                        let lo = _mm256_and_si256(shifted, mask);
                        let hi_shifted = _mm256_sra_epi16(v.raw, count);
                        let hi = _mm256_andnot_si256(mask, hi_shifted);
                        _mm256_or_si256(lo, hi)
                    } else {
                        let shifted = _mm256_srl_epi16(v.raw, count);
                        let mask = _mm256_set1_epi8((0xFFu8.wrapping_shr(bits)) as i8);
                        _mm256_and_si256(shifted, mask)
                    }
                }
                2 => {
                    if is_signed::<T>() {
                        _mm256_sra_epi16(v.raw, count)
                    } else {
                        _mm256_srl_epi16(v.raw, count)
                    }
                }
                4 => {
                    if is_signed::<T>() {
                        _mm256_sra_epi32(v.raw, count)
                    } else {
                        _mm256_srl_epi32(v.raw, count)
                    }
                }
                8 => {
                    if is_signed::<T>() {
                        let sign = _mm256_srai_epi32(v.raw, 31);
                        let sign64 = _mm256_shuffle_epi32(sign, 0xF5);
                        let shifted = _mm256_srl_epi64(v.raw, count);
                        let ones = _mm256_set1_epi64x(-1i64);
                        let sign_mask =
                            _mm256_sll_epi64(ones, _mm_cvtsi64_si128(64i64 - bits as i64));
                        let fill = _mm256_and_si256(sign64, sign_mask);
                        _mm256_or_si256(shifted, fill)
                    } else {
                        _mm256_srl_epi64(v.raw, count)
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn shift_left_bytes<T: Lane, const BYTES: usize>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = match BYTES {
                0 => v.raw,
                1 => _mm256_slli_si256::<1>(v.raw),
                2 => _mm256_slli_si256::<2>(v.raw),
                3 => _mm256_slli_si256::<3>(v.raw),
                4 => _mm256_slli_si256::<4>(v.raw),
                5 => _mm256_slli_si256::<5>(v.raw),
                6 => _mm256_slli_si256::<6>(v.raw),
                7 => _mm256_slli_si256::<7>(v.raw),
                8 => _mm256_slli_si256::<8>(v.raw),
                9 => _mm256_slli_si256::<9>(v.raw),
                10 => _mm256_slli_si256::<10>(v.raw),
                11 => _mm256_slli_si256::<11>(v.raw),
                12 => _mm256_slli_si256::<12>(v.raw),
                13 => _mm256_slli_si256::<13>(v.raw),
                14 => _mm256_slli_si256::<14>(v.raw),
                15 => _mm256_slli_si256::<15>(v.raw),
                _ => _mm256_setzero_si256(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn shift_right_bytes<T: Lane, const BYTES: usize>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = match BYTES {
                0 => v.raw,
                1 => _mm256_srli_si256::<1>(v.raw),
                2 => _mm256_srli_si256::<2>(v.raw),
                3 => _mm256_srli_si256::<3>(v.raw),
                4 => _mm256_srli_si256::<4>(v.raw),
                5 => _mm256_srli_si256::<5>(v.raw),
                6 => _mm256_srli_si256::<6>(v.raw),
                7 => _mm256_srli_si256::<7>(v.raw),
                8 => _mm256_srli_si256::<8>(v.raw),
                9 => _mm256_srli_si256::<9>(v.raw),
                10 => _mm256_srli_si256::<10>(v.raw),
                11 => _mm256_srli_si256::<11>(v.raw),
                12 => _mm256_srli_si256::<12>(v.raw),
                13 => _mm256_srli_si256::<13>(v.raw),
                14 => _mm256_srli_si256::<14>(v.raw),
                15 => _mm256_srli_si256::<15>(v.raw),
                _ => _mm256_setzero_si256(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn population_count<T: IntegerLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            // Nibble-lookup popcount: count set bits per byte, then accumulate
            let nibble_lut = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                2, 3, 3, 4,
            );
            let lo_mask = _mm256_set1_epi8(0x0F);
            let lo = _mm256_and_si256(v.raw, lo_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(v.raw, 4), lo_mask);
            let cnt = _mm256_add_epi8(
                _mm256_shuffle_epi8(nibble_lut, lo),
                _mm256_shuffle_epi8(nibble_lut, hi),
            );
            // cnt has per-byte popcount. For wider lanes, accumulate bytes.
            let raw = match T::BYTES {
                1 => cnt,
                2 => {
                    // Sum pairs of bytes: _mm256_maddubs_epi16 treats first arg
                    // as unsigned bytes and multiplies pairwise with second arg
                    // (signed bytes), then adds adjacent pairs into i16.
                    // With multiplier=1 this sums each pair of byte popcounts.
                    _mm256_maddubs_epi16(cnt, _mm256_set1_epi8(1))
                }
                4 => {
                    // Sum bytes within each 32-bit lane: maddubs for pairs,
                    // then madd to sum the two 16-bit halves per 32-bit lane.
                    let sum16 = _mm256_maddubs_epi16(cnt, _mm256_set1_epi8(1));
                    _mm256_madd_epi16(sum16, _mm256_set1_epi16(1))
                }
                8 => {
                    // sad_epu8 sums absolute differences vs zero across groups
                    // of 8 bytes into 64-bit accumulators — perfect for 64-bit popcount.
                    _mm256_sad_epu8(cnt, _mm256_setzero_si256())
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn leading_zero_count<T: IntegerLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            match T::BYTES {
                1 => {
                    // Nibble-lookup for CLZ of high nibble, then adjust for low nibble.
                    // clz4[nibble] gives the leading zero count of a 4-bit value.
                    let clz4 = _mm256_setr_epi8(
                        4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 2, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0,
                    );
                    let lo_mask = _mm256_set1_epi8(0x0F);
                    let lo = _mm256_and_si256(v.raw, lo_mask);
                    let hi = _mm256_and_si256(_mm256_srli_epi16(v.raw, 4), lo_mask);
                    let clz_lo = _mm256_shuffle_epi8(clz4, lo);
                    let clz_hi = _mm256_shuffle_epi8(clz4, hi);
                    // If high nibble is nonzero, clz = clz_hi; else clz = 4 + clz_lo
                    // clz_hi < 4 means high nibble was nonzero
                    let hi_nonzero = _mm256_cmpgt_epi8(_mm256_set1_epi8(4), clz_hi);
                    // Where hi_nonzero: use clz_hi; else: 4 + clz_lo
                    let clz_full = _mm256_add_epi8(clz_lo, _mm256_set1_epi8(4));
                    V256::from_raw(_mm256_or_si256(
                        _mm256_and_si256(hi_nonzero, clz_hi),
                        _mm256_andnot_si256(hi_nonzero, clz_full),
                    ))
                }
                2 => {
                    // CLZ for 16-bit lanes: compute CLZ for each byte, then combine.
                    // clz(u16) = if hi_byte != 0 { clz(hi_byte) } else { 8 + clz(lo_byte) }
                    let clz4 = _mm256_setr_epi8(
                        4, 3, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 4, 3, 2, 2, 1, 1, 1, 1, 0,
                        0, 0, 0, 0, 0, 0, 0,
                    );
                    let lo_mask = _mm256_set1_epi8(0x0F);
                    let lo_nib = _mm256_and_si256(v.raw, lo_mask);
                    let hi_nib = _mm256_and_si256(_mm256_srli_epi16(v.raw, 4), lo_mask);
                    let clz_lo_nib = _mm256_shuffle_epi8(clz4, lo_nib);
                    let clz_hi_nib = _mm256_shuffle_epi8(clz4, hi_nib);
                    // Per-byte CLZ
                    let hi_nz = _mm256_cmpgt_epi8(_mm256_set1_epi8(4), clz_hi_nib);
                    let byte_clz = _mm256_or_si256(
                        _mm256_and_si256(hi_nz, clz_hi_nib),
                        _mm256_andnot_si256(
                            hi_nz,
                            _mm256_add_epi8(clz_lo_nib, _mm256_set1_epi8(4)),
                        ),
                    );
                    // Now combine bytes within 16-bit lanes.
                    // High byte of each 16-bit lane is at odd byte positions.
                    // If high byte != 0 (i.e., byte_clz of high byte < 8), use that;
                    // else 8 + byte_clz of low byte.
                    // Shift byte_clz right by 8 bits to get high byte's clz in even positions
                    let hi_byte_clz = _mm256_srli_epi16(byte_clz, 8);
                    let lo_byte_clz = _mm256_and_si256(byte_clz, _mm256_set1_epi16(0x00FF));
                    // high byte is nonzero if hi_byte_clz < 8
                    let hi_byte_nz = _mm256_cmpgt_epi16(_mm256_set1_epi16(8), hi_byte_clz);
                    let full_clz = _mm256_add_epi16(lo_byte_clz, _mm256_set1_epi16(8));
                    V256::from_raw(_mm256_or_si256(
                        _mm256_and_si256(hi_byte_nz, hi_byte_clz),
                        _mm256_andnot_si256(hi_byte_nz, full_clz),
                    ))
                }
                4 => {
                    // Float-conversion trick for 32-bit CLZ.
                    // _mm256_cvtepi32_ps treats input as signed i32, so we clear
                    // bit 31 and handle it separately.
                    let msb_mask = _mm256_set1_epi32(i32::MIN); // 0x80000000
                    let has_msb = _mm256_and_si256(v.raw, msb_mask);
                    // has_msb has bit 31 set where v had bit 31 set; else zero
                    let has_msb_mask = _mm256_cmpeq_epi32(has_msb, msb_mask); // all-ones if MSB set

                    // Clear MSB, then apply float trick on remaining 31 bits.
                    // For these values, clz(v & 0x7FFFFFFF) = clz(v) - 1 when MSB is set
                    // but we handle MSB case separately (clz = 0).
                    let v_no_msb = _mm256_andnot_si256(msb_mask, v.raw);
                    // Normalize: clear bit at (MSB_pos - 24) to ensure float conversion
                    // rounds down for values >= 2^24 (matching C++ NormalizeForUIntTruncConvToF32).
                    let v_normalized = _mm256_andnot_si256(_mm256_srli_epi32(v_no_msb, 24), v_no_msb);
                    let v_or_1 = _mm256_or_si256(v_normalized, _mm256_set1_epi32(1));
                    let as_float = _mm256_cvtepi32_ps(v_or_1);
                    let float_bits = _mm256_castps_si256(as_float);
                    let exponent = _mm256_and_si256(
                        _mm256_srli_epi32(float_bits, 23),
                        _mm256_set1_epi32(0xFF),
                    );
                    // For non-zero v_no_msb: clz(v_no_msb) = 158 - exponent
                    // But v_no_msb has bit 31 clear, so the leading 0 is bit 31,
                    // meaning clz = 1 + clz(remaining 31 bits).
                    // Actually: v_no_msb is a 32-bit value with bit 31 = 0.
                    // The float conversion is correct for values 0..0x7FFFFFFF.
                    // clz(v) when MSB is not set = 158 - exponent (of float(v|1))
                    //   v = 0x40000000: float = 2^30, exp = 127+30 = 157, clz = 158-157 = 1. Correct.
                    //   v = 1: float = 1.0, exp = 127, clz = 158-127 = 31. Correct.
                    //   v = 0: v|1 = 1, float = 1.0, exp = 127, clz = 31. Wrong (should be 32).
                    let clz_no_msb = _mm256_sub_epi32(_mm256_set1_epi32(158), exponent);
                    // Fix v == 0 case
                    let is_zero = _mm256_cmpeq_epi32(v.raw, _mm256_setzero_si256());
                    // MSB set -> clz = 0; v == 0 -> clz = 32; else -> clz_no_msb
                    let result = _mm256_or_si256(
                        _mm256_and_si256(is_zero, _mm256_set1_epi32(32)),
                        _mm256_andnot_si256(is_zero, clz_no_msb),
                    );
                    // Override with 0 for lanes where MSB is set
                    V256::from_raw(_mm256_andnot_si256(has_msb_mask, result))
                }
                8 => {
                    // For 64-bit lanes, split into high and low 32-bit halves,
                    // compute CLZ-32 for each, then combine.
                    // clz(u64) = if hi32 != 0 { clz32(hi32) } else { 32 + clz32(lo32) }
                    let hi32 = _mm256_srli_epi64(v.raw, 32);
                    let lo32 = _mm256_and_si256(v.raw, _mm256_set1_epi64x(0xFFFFFFFF));

                    // Helper: CLZ-32 with MSB-safe float trick.
                    // Clear bit 31, compute CLZ on remaining 31 bits, handle MSB.
                    let msb32 = _mm256_set1_epi32(i32::MIN);
                    let one32 = _mm256_set1_epi32(1);
                    let c158 = _mm256_set1_epi32(158);
                    let zero = _mm256_setzero_si256();

                    // CLZ of hi32
                    let hi_has_msb = _mm256_cmpeq_epi32(_mm256_and_si256(hi32, msb32), msb32);
                    let hi_no_msb = _mm256_andnot_si256(msb32, hi32);
                    let hi_norm = _mm256_andnot_si256(_mm256_srli_epi32(hi_no_msb, 24), hi_no_msb);
                    let hi_f = _mm256_cvtepi32_ps(_mm256_or_si256(hi_norm, one32));
                    let hi_exp = _mm256_and_si256(
                        _mm256_srli_epi32(_mm256_castps_si256(hi_f), 23),
                        _mm256_set1_epi32(0xFF),
                    );
                    let hi_clz_raw = _mm256_sub_epi32(c158, hi_exp);
                    let hi_is_zero = _mm256_cmpeq_epi32(hi32, zero);
                    let hi_clz = _mm256_andnot_si256(
                        hi_has_msb,
                        _mm256_or_si256(
                            _mm256_and_si256(hi_is_zero, _mm256_set1_epi32(32)),
                            _mm256_andnot_si256(hi_is_zero, hi_clz_raw),
                        ),
                    );

                    // CLZ of lo32
                    let lo_has_msb = _mm256_cmpeq_epi32(_mm256_and_si256(lo32, msb32), msb32);
                    let lo_no_msb = _mm256_andnot_si256(msb32, lo32);
                    let lo_norm = _mm256_andnot_si256(_mm256_srli_epi32(lo_no_msb, 24), lo_no_msb);
                    let lo_f = _mm256_cvtepi32_ps(_mm256_or_si256(lo_norm, one32));
                    let lo_exp = _mm256_and_si256(
                        _mm256_srli_epi32(_mm256_castps_si256(lo_f), 23),
                        _mm256_set1_epi32(0xFF),
                    );
                    let lo_clz_raw = _mm256_sub_epi32(c158, lo_exp);
                    let lo_is_zero = _mm256_cmpeq_epi32(lo32, zero);
                    let lo_clz = _mm256_andnot_si256(
                        lo_has_msb,
                        _mm256_or_si256(
                            _mm256_and_si256(lo_is_zero, _mm256_set1_epi32(32)),
                            _mm256_andnot_si256(lo_is_zero, lo_clz_raw),
                        ),
                    );

                    // Combine: if hi32 != 0: clz64 = hi_clz; else: clz64 = 32 + lo_clz
                    let hi_zero_64 = _mm256_cmpeq_epi64(hi32, zero);
                    let lo_clz_64 = _mm256_and_si256(lo_clz, _mm256_set1_epi64x(0xFFFFFFFF));
                    let hi_clz_64 = _mm256_and_si256(hi_clz, _mm256_set1_epi64x(0xFFFFFFFF));
                    let lo_plus_32 = _mm256_add_epi64(lo_clz_64, _mm256_set1_epi64x(32));
                    V256::from_raw(_mm256_or_si256(
                        _mm256_and_si256(hi_zero_64, lo_plus_32),
                        _mm256_andnot_si256(hi_zero_64, hi_clz_64),
                    ))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    unsafe fn trailing_zero_count<T: IntegerLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            // tzc(x) = popcount((x - 1) & ~x)
            // For x == 0: x-1 wraps to all-ones, ~x = all-ones, so result = type_bits. Correct.
            let one = match T::BYTES {
                1 => _mm256_set1_epi8(1),
                2 => _mm256_set1_epi16(1),
                4 => _mm256_set1_epi32(1),
                8 => _mm256_set1_epi64x(1),
                _ => unreachable!(),
            };
            let x_minus_1 = match T::BYTES {
                1 => _mm256_sub_epi8(v.raw, one),
                2 => _mm256_sub_epi16(v.raw, one),
                4 => _mm256_sub_epi32(v.raw, one),
                8 => _mm256_sub_epi64(v.raw, one),
                _ => unreachable!(),
            };
            let not_x = _mm256_xor_si256(v.raw, _mm256_set1_epi8(!0));
            let isolated = _mm256_and_si256(x_minus_1, not_x);
            // Use the optimized population_count
            self.population_count(V256::<T>::from_raw(isolated))
        }
    }

    #[inline(always)]
    unsafe fn reverse_lane_bytes<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => v.raw, // single byte, nothing to reverse
                2 => {
                    // Swap bytes within each 16-bit lane: use shuffle_epi8
                    let idx = _mm256_set_epi8(
                        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10,
                        11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                4 => {
                    // Reverse 4 bytes within each 32-bit lane
                    let idx = _mm256_set_epi8(
                        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9,
                        10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                8 => {
                    // Reverse 8 bytes within each 64-bit lane
                    let idx = _mm256_set_epi8(
                        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                        14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn reverse_bits<T: IntegerLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            // Nibble-lookup to reverse bits within each nibble
            let nibble_rev = _mm256_setr_epi8(
                0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE, 0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
                0x0, 0x8, 0x4, 0xC, 0x2, 0xA, 0x6, 0xE, 0x1, 0x9, 0x5, 0xD, 0x3, 0xB, 0x7, 0xF,
            );
            let lo_mask = _mm256_set1_epi8(0x0F);
            let hi_mask = _mm256_set1_epi8(0xF0u8 as i8);
            let lo = _mm256_and_si256(v.raw, lo_mask);
            let hi = _mm256_and_si256(_mm256_srli_epi16(v.raw, 4), lo_mask);
            // Reverse nibbles: lo becomes high nibble, hi becomes low nibble
            let reversed_bytes = _mm256_or_si256(
                _mm256_and_si256(
                    _mm256_slli_epi16(_mm256_shuffle_epi8(nibble_rev, lo), 4),
                    hi_mask,
                ),
                _mm256_shuffle_epi8(nibble_rev, hi),
            );
            // For multi-byte lanes, also reverse byte order within each lane
            if T::BYTES == 1 {
                V256::from_raw(reversed_bytes)
            } else {
                // Reuse the existing reverse_lane_bytes which uses _mm256_shuffle_epi8
                let byte_reversed = V256::<T>::from_raw(reversed_bytes);
                self.reverse_lane_bytes(byte_reversed)
            }
        }
    }

    #[inline(always)]
    unsafe fn shl<T: IntegerLane>(
        self,
        v: V256<T>,
        bits: V256<T>,
    ) -> V256<T> {
        unsafe {
            if is_type::<T, u32>() || is_type::<T, i32>() {
                V256::from_raw(_mm256_sllv_epi32(v.raw, bits.raw))
            } else if is_type::<T, u64>() || is_type::<T, i64>() {
                V256::from_raw(_mm256_sllv_epi64(v.raw, bits.raw))
            } else if T::BYTES == 2 {
                // u16/i16: promote to 32-bit, shift, demote
                let lo_v = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v.raw));
                let hi_v = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v.raw, 1));
                let lo_b = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(bits.raw));
                let hi_b = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(bits.raw, 1));
                let lo_r = _mm256_sllv_epi32(lo_v, lo_b);
                let hi_r = _mm256_sllv_epi32(hi_v, hi_b);
                // Pack back: mask to 16 bits, then pack
                let mask16 = _mm256_set1_epi32(0xFFFF);
                let lo_masked = _mm256_and_si256(lo_r, mask16);
                let hi_masked = _mm256_and_si256(hi_r, mask16);
                let packed = _mm256_packus_epi32(lo_masked, hi_masked);
                // packus interleaves within 128-bit lanes, need permute to fix order
                V256::from_raw(_mm256_permute4x64_epi64(packed, 0xD8))
            } else {
                // u8/i8: scalar fallback
                let mut v_arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                let mut b_arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                _mm256_store_si256(v_arr.as_mut_ptr().cast(), v.raw);
                _mm256_store_si256(b_arr.as_mut_ptr().cast(), bits.raw);
                for i in 0..32 {
                    let shift = b_arr.value[i];
                    v_arr.value[i] = if shift >= 8 { 0 } else { v_arr.value[i] << shift };
                }
                V256::from_raw(_mm256_load_si256(v_arr.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn shr<T: IntegerLane>(
        self,
        v: V256<T>,
        bits: V256<T>,
    ) -> V256<T> {
        unsafe {
            if is_type::<T, u32>() {
                V256::from_raw(_mm256_srlv_epi32(v.raw, bits.raw))
            } else if is_type::<T, i32>() {
                V256::from_raw(_mm256_srav_epi32(v.raw, bits.raw))
            } else if is_type::<T, u64>() {
                V256::from_raw(_mm256_srlv_epi64(v.raw, bits.raw))
            } else if is_type::<T, i64>() {
                // AVX2 has no _mm256_srav_epi64; scalar fallback
                let mut arr: Aligned<A32, [i64; 4]> = Aligned::new([0i64; 4]);
                let mut barr: Aligned<A32, [i64; 4]> = Aligned::new([0i64; 4]);
                _mm256_store_si256(arr.as_mut_ptr().cast(), v.raw);
                _mm256_store_si256(barr.as_mut_ptr().cast(), bits.raw);
                for i in 0..4 {
                    arr.value[i] >>= barr.value[i];
                }
                V256::from_raw(_mm256_load_si256(arr.as_ptr().cast()))
            } else if T::BYTES == 2 {
                // u16/i16: promote to 32-bit, shift, pack back
                let lo_v = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v.raw));
                let hi_v = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(v.raw, 1));
                let lo_b = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(bits.raw));
                let hi_b = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(bits.raw, 1));
                if is_signed::<T>() {
                    // Sign-extend u16 to i32 for arithmetic shift
                    let lo_vs = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v.raw));
                    let hi_vs = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(v.raw, 1));
                    let lo_r = _mm256_srav_epi32(lo_vs, lo_b);
                    let hi_r = _mm256_srav_epi32(hi_vs, hi_b);
                    // Pack with signed saturation then fix order
                    let packed = _mm256_packs_epi32(lo_r, hi_r);
                    V256::from_raw(_mm256_permute4x64_epi64(packed, 0xD8))
                } else {
                    let lo_r = _mm256_srlv_epi32(lo_v, lo_b);
                    let hi_r = _mm256_srlv_epi32(hi_v, hi_b);
                    let mask16 = _mm256_set1_epi32(0xFFFF);
                    let lo_masked = _mm256_and_si256(lo_r, mask16);
                    let hi_masked = _mm256_and_si256(hi_r, mask16);
                    let packed = _mm256_packus_epi32(lo_masked, hi_masked);
                    V256::from_raw(_mm256_permute4x64_epi64(packed, 0xD8))
                }
            } else {
                // u8/i8: scalar fallback
                if is_signed::<T>() {
                    let mut v_arr: Aligned<A32, [i8; 32]> = Aligned::new([0i8; 32]);
                    let mut b_arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                    _mm256_store_si256(v_arr.as_mut_ptr().cast(), v.raw);
                    _mm256_store_si256(b_arr.as_mut_ptr().cast(), bits.raw);
                    for i in 0..32 {
                        let shift = b_arr.value[i];
                        v_arr.value[i] = if shift >= 8 {
                            v_arr.value[i] >> 7 // sign-fill
                        } else {
                            v_arr.value[i] >> shift
                        };
                    }
                    V256::from_raw(_mm256_load_si256(v_arr.as_ptr().cast()))
                } else {
                    let mut v_arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                    let mut b_arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                    _mm256_store_si256(v_arr.as_mut_ptr().cast(), v.raw);
                    _mm256_store_si256(b_arr.as_mut_ptr().cast(), bits.raw);
                    for i in 0..32 {
                        let shift = b_arr.value[i];
                        v_arr.value[i] = if shift >= 8 { 0 } else { v_arr.value[i] >> shift };
                    }
                    V256::from_raw(_mm256_load_si256(v_arr.as_ptr().cast()))
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn ror<T: IntegerLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let _bits_per_lane = (T::BYTES * 8) as u32;
            let mut result = [0u8; 32];
            let mut arr_a = [0u8; 32];
            let mut arr_b = [0u8; 32];
            _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
            _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);
            for i in 0..lanes {
                let offset = i * T::BYTES;
                match T::BYTES {
                    1 => {
                        let va = arr_a[offset];
                        let vb = arr_b[offset] & 7;
                        result[offset] = va.rotate_right(vb as u32);
                    }
                    2 => {
                        let va = u16::from_le_bytes([arr_a[offset], arr_a[offset + 1]]);
                        let vb = u16::from_le_bytes([arr_b[offset], arr_b[offset + 1]]) & 15;
                        let rb = va.rotate_right(vb as u32).to_le_bytes();
                        result[offset] = rb[0];
                        result[offset + 1] = rb[1];
                    }
                    4 => {
                        let va = u32::from_le_bytes([arr_a[offset], arr_a[offset+1], arr_a[offset+2], arr_a[offset+3]]);
                        let vb = u32::from_le_bytes([arr_b[offset], arr_b[offset+1], arr_b[offset+2], arr_b[offset+3]]) & 31;
                        let rb = va.rotate_right(vb).to_le_bytes();
                        result[offset..offset+4].copy_from_slice(&rb);
                    }
                    8 => {
                        let va = u64::from_le_bytes([arr_a[offset], arr_a[offset+1], arr_a[offset+2], arr_a[offset+3], arr_a[offset+4], arr_a[offset+5], arr_a[offset+6], arr_a[offset+7]]);
                        let vb = u64::from_le_bytes([arr_b[offset], arr_b[offset+1], arr_b[offset+2], arr_b[offset+3], arr_b[offset+4], arr_b[offset+5], arr_b[offset+6], arr_b[offset+7]]) & 63;
                        let rb = va.rotate_right(vb as u32).to_le_bytes();
                        result[offset..offset+8].copy_from_slice(&rb);
                    }
                    _ => unreachable!(),
                }
            }
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn rol<T: IntegerLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut result = [0u8; 32];
            let mut arr_a = [0u8; 32];
            let mut arr_b = [0u8; 32];
            _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
            _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);
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
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn rotate_left<T: IntegerLane, const BITS: u32>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut result = [0u8; 32];
            let mut arr = [0u8; 32];
            _mm256_storeu_si256(arr.as_mut_ptr().cast(), v.raw);
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
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn broadcast_sign_bit<T: IntegerLane>(self, v: V256<T>) -> V256<T> {
        // All-ones if the MSB (sign bit) is set, else all-zeros. Matches C++ Highway.
        unsafe {
            let raw = match T::BYTES {
                1 => _mm256_cmpgt_epi8(_mm256_setzero_si256(), v.raw),
                2 => _mm256_srai_epi16(v.raw, 15),
                4 => _mm256_srai_epi32(v.raw, 31),
                // i64: AVX2 has no _mm256_srai_epi64; broadcast the high dword's sign
                // within each 128-bit lane (shuffle operates per 128-bit lane).
                _ => {
                    let sign = _mm256_srai_epi32(v.raw, 31);
                    _mm256_shuffle_epi32(sign, 0xF5) // _MM_SHUFFLE(3, 3, 1, 1)
                }
            };
            V256::from_raw(raw)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdCompare
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdCompare for Avx2 {
    #[inline(always)]
    unsafe fn eq<T: Lane>(self, a: V256<T>, b: V256<T>) -> M256<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm256_castps_si256(_mm256_cmp_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _CMP_EQ_OQ,
                ))
            } else if is_type::<T, f64>() {
                _mm256_castpd_si256(_mm256_cmp_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _CMP_EQ_OQ,
                ))
            } else {
                match T::BYTES {
                    1 => _mm256_cmpeq_epi8(a.raw, b.raw),
                    2 => _mm256_cmpeq_epi16(a.raw, b.raw),
                    4 => _mm256_cmpeq_epi32(a.raw, b.raw),
                    8 => _mm256_cmpeq_epi64(a.raw, b.raw),
                    _ => unreachable!(),
                }
            };
            M256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn ne<T: Lane>(self, a: V256<T>, b: V256<T>) -> M256<T> {
        unsafe {
            // For floats, use _CMP_NEQ_OQ directly (ordered, quiet) to match
            // C++ Highway semantics: NaN != x returns false.
            // Using !eq would give unordered semantics (NaN != x returns true).
            let raw = if is_type::<T, f32>() {
                _mm256_castps_si256(_mm256_cmp_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _CMP_NEQ_OQ,
                ))
            } else if is_type::<T, f64>() {
                _mm256_castpd_si256(_mm256_cmp_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _CMP_NEQ_OQ,
                ))
            } else {
                let eq = self.eq::<T>(a, b);
                _mm256_xor_si256(eq.raw, _mm256_set1_epi8(!0))
            };
            M256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn lt<T: Lane>(self, a: V256<T>, b: V256<T>) -> M256<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm256_castps_si256(_mm256_cmp_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _CMP_LT_OQ,
                ))
            } else if is_type::<T, f64>() {
                _mm256_castpd_si256(_mm256_cmp_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _CMP_LT_OQ,
                ))
            } else if is_signed::<T>() {
                // gt(b, a) = lt(a, b) — AVX2 has cmpgt but not cmplt
                match T::BYTES {
                    1 => _mm256_cmpgt_epi8(b.raw, a.raw),
                    2 => _mm256_cmpgt_epi16(b.raw, a.raw),
                    4 => _mm256_cmpgt_epi32(b.raw, a.raw),
                    8 => _mm256_cmpgt_epi64(b.raw, a.raw),
                    _ => unreachable!(),
                }
            } else {
                // Unsigned: flip sign bits then use signed compare
                let sign_flip = match T::BYTES {
                    1 => _mm256_set1_epi8(i8::MIN),
                    2 => _mm256_set1_epi16(i16::MIN),
                    4 => _mm256_set1_epi32(i32::MIN),
                    8 => _mm256_set1_epi64x(i64::MIN),
                    _ => unreachable!(),
                };
                let af = _mm256_xor_si256(a.raw, sign_flip);
                let bf = _mm256_xor_si256(b.raw, sign_flip);
                match T::BYTES {
                    1 => _mm256_cmpgt_epi8(bf, af),
                    2 => _mm256_cmpgt_epi16(bf, af),
                    4 => _mm256_cmpgt_epi32(bf, af),
                    8 => _mm256_cmpgt_epi64(bf, af),
                    _ => unreachable!(),
                }
            };
            M256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn le<T: Lane>(self, a: V256<T>, b: V256<T>) -> M256<T> {
        unsafe {
            if is_type::<T, f32>() {
                M256::from_raw(_mm256_castps_si256(_mm256_cmp_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _CMP_LE_OQ,
                )))
            } else if is_type::<T, f64>() {
                M256::from_raw(_mm256_castpd_si256(_mm256_cmp_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _CMP_LE_OQ,
                )))
            } else {
                let eq = self.eq::<T>(a, b);
                let lt = self.lt::<T>(a, b);
                M256::from_raw(_mm256_or_si256(eq.raw, lt.raw))
            }
        }
    }

    #[inline(always)]
    unsafe fn gt<T: Lane>(self, a: V256<T>, b: V256<T>) -> M256<T> {
        unsafe { self.lt(b, a) }
    }

    #[inline(always)]
    unsafe fn ge<T: Lane>(self, a: V256<T>, b: V256<T>) -> M256<T> {
        unsafe { self.le(b, a) }
    }

    #[inline(always)]
    unsafe fn test_bit<T: IntegerLane>(self, v: V256<T>, bit: V256<T>) -> M256<T> {
        unsafe {
            let anded = _mm256_and_si256(v.raw, bit.raw);
            let zero = _mm256_setzero_si256();
            let eq_zero = match T::BYTES {
                1 => _mm256_cmpeq_epi8(anded, zero),
                2 => _mm256_cmpeq_epi16(anded, zero),
                4 => _mm256_cmpeq_epi32(anded, zero),
                8 => _mm256_cmpeq_epi64(anded, zero),
                _ => unreachable!(),
            };
            M256::from_raw(_mm256_xor_si256(eq_zero, _mm256_set1_epi8(!0)))
        }
    }
}

// ---------------------------------------------------------------------------
// SimdMask
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdMask for Avx2 {
    #[inline(always)]
    unsafe fn mask_from_vec<T: Lane>(self, v: V256<T>) -> M256<T> {
        unsafe {
            let zero = _mm256_setzero_si256();
            let is_zero = match T::BYTES {
                1 => _mm256_cmpeq_epi8(v.raw, zero),
                2 => _mm256_cmpeq_epi16(v.raw, zero),
                4 => _mm256_cmpeq_epi32(v.raw, zero),
                8 => _mm256_cmpeq_epi64(v.raw, zero),
                _ => unreachable!(),
            };
            M256::from_raw(_mm256_xor_si256(is_zero, _mm256_set1_epi8(!0)))
        }
    }

    #[inline(always)]
    unsafe fn vec_from_mask<T: Lane>(self, m: M256<T>) -> V256<T> {
        V256::from_raw(m.raw)
    }

    #[inline(always)]
    unsafe fn first_n<T: Lane>(self, n: usize) -> M256<T> {
        // Signed comparisons are cheaper (same as C++ Highway).
        // iota < threshold <-> cmpgt(threshold, iota)
        unsafe {
            match T::BYTES {
                1 => {
                    let iota = _mm256_setr_epi8(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                        21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                    );
                    let threshold = _mm256_set1_epi8(n as i8);
                    M256::from_raw(_mm256_cmpgt_epi8(threshold, iota))
                }
                2 => {
                    let iota =
                        _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                    let threshold = _mm256_set1_epi16(n as i16);
                    M256::from_raw(_mm256_cmpgt_epi16(threshold, iota))
                }
                4 => {
                    let iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                    let threshold = _mm256_set1_epi32(n as i32);
                    M256::from_raw(_mm256_cmpgt_epi32(threshold, iota))
                }
                _ => {
                    // 64-bit: 4 lanes. AVX2 has _mm256_cmpgt_epi64.
                    let iota = _mm256_setr_epi64x(0, 1, 2, 3);
                    let threshold = _mm256_set1_epi64x(n as i64);
                    M256::from_raw(_mm256_cmpgt_epi64(threshold, iota))
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn count_true<T: Lane>(self, m: M256<T>) -> usize {
        unsafe {
            match T::BYTES {
                1 => _mm256_movemask_epi8(m.raw).count_ones() as usize,
                2 => (_mm256_movemask_epi8(m.raw).count_ones() as usize) / 2,
                4 => _mm256_movemask_ps(_mm256_castsi256_ps(m.raw)).count_ones() as usize,
                8 => _mm256_movemask_pd(_mm256_castsi256_pd(m.raw)).count_ones() as usize,
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    unsafe fn all_true<T: Lane>(self, m: M256<T>) -> bool {
        unsafe { _mm256_movemask_epi8(m.raw) == -1i32 }
    }

    #[inline(always)]
    unsafe fn all_false<T: Lane>(self, m: M256<T>) -> bool {
        unsafe { _mm256_movemask_epi8(m.raw) == 0 }
    }

    #[inline(always)]
    unsafe fn find_first_true<T: Lane>(self, m: M256<T>) -> Option<usize> {
        unsafe {
            let bits = _mm256_movemask_epi8(m.raw) as u32;
            if bits == 0 {
                None
            } else {
                Some((bits.trailing_zeros() as usize) / T::BYTES)
            }
        }
    }

    #[inline(always)]
    unsafe fn if_then_else<T: Lane>(self, mask: M256<T>, yes: V256<T>, no: V256<T>) -> V256<T> {
        unsafe {
            V256::from_raw(_mm256_or_si256(
                _mm256_and_si256(mask.raw, yes.raw),
                _mm256_andnot_si256(mask.raw, no.raw),
            ))
        }
    }

    #[inline(always)]
    unsafe fn if_then_else_zero<T: Lane>(self, mask: M256<T>, yes: V256<T>) -> V256<T> {
        V256::from_raw(unsafe { _mm256_and_si256(mask.raw, yes.raw) })
    }

    #[inline(always)]
    unsafe fn if_then_zero_else<T: Lane>(self, mask: M256<T>, no: V256<T>) -> V256<T> {
        V256::from_raw(unsafe { _mm256_andnot_si256(mask.raw, no.raw) })
    }

    #[inline(always)]
    unsafe fn and_mask<T: Lane>(self, a: M256<T>, b: M256<T>) -> M256<T> {
        M256::from_raw(unsafe { _mm256_and_si256(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn or_mask<T: Lane>(self, a: M256<T>, b: M256<T>) -> M256<T> {
        M256::from_raw(unsafe { _mm256_or_si256(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn not_mask<T: Lane>(self, m: M256<T>) -> M256<T> {
        M256::from_raw(unsafe { _mm256_xor_si256(m.raw, _mm256_set1_epi8(!0)) })
    }

    #[inline(always)]
    unsafe fn xor_mask<T: Lane>(self, a: M256<T>, b: M256<T>) -> M256<T> {
        M256::from_raw(unsafe { _mm256_xor_si256(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn find_last_true<T: Lane>(self, m: M256<T>) -> Option<usize> {
        unsafe {
            let bits = _mm256_movemask_epi8(m.raw) as u32;
            if bits == 0 {
                None
            } else {
                // Highest set bit position in the 32-bit mask
                let highest = 31 - bits.leading_zeros() as usize;
                // Convert byte index to lane index
                Some(highest / T::BYTES)
            }
        }
    }

    #[inline(always)]
    unsafe fn bits_from_mask<T: Lane>(self, m: M256<T>) -> u64 {
        unsafe {
            match T::BYTES {
                1 => _mm256_movemask_epi8(m.raw) as u32 as u64,
                2 => {
                    // movemask_epi8 gives 32 bits. Extract odd-position bits and pack.
                    let byte_mask = _mm256_movemask_epi8(m.raw) as u64;
                    let x = (byte_mask >> 1) & 0x55555555;
                    let x = (x | (x >> 1)) & 0x33333333;
                    let x = (x | (x >> 2)) & 0x0F0F0F0F;
                    let x = (x | (x >> 4)) & 0x00FF00FF;
                    (x | (x >> 8)) & 0x0000FFFF
                }
                4 => _mm256_movemask_ps(_mm256_castsi256_ps(m.raw)) as u32 as u64,
                8 => _mm256_movemask_pd(_mm256_castsi256_pd(m.raw)) as u32 as u64,
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    unsafe fn exclusive_neither<T: Lane>(self, a: M256<T>, b: M256<T>) -> M256<T> {
        // NOR: true only where neither a nor b is set (C++ ExclusiveNeither).
        unsafe {
            let ones = _mm256_cmpeq_epi8(_mm256_setzero_si256(), _mm256_setzero_si256());
            let not_b = _mm256_xor_si256(b.raw, ones);
            M256::from_raw(_mm256_andnot_si256(a.raw, not_b))
        }
    }

    #[inline(always)]
    unsafe fn slide_mask_1_up<T: Lane>(self, mask: M256<T>) -> M256<T> {
        unsafe {
            let v = self.vec_from_mask::<T>(mask);
            let slid = self.slide_1_up(v);
            self.mask_from_vec(slid)
        }
    }

    #[inline(always)]
    unsafe fn slide_mask_1_down<T: Lane>(self, mask: M256<T>) -> M256<T> {
        unsafe {
            let v = self.vec_from_mask::<T>(mask);
            let slid = self.slide_1_down(v);
            self.mask_from_vec(slid)
        }
    }

    #[inline(always)]
    unsafe fn if_negative_then_else<T: Lane>(self, v: V256<T>, yes: V256<T>, no: V256<T>) -> V256<T> {
        unsafe {
            let sign = avx2_sign_mask::<T>(v.raw);
            let r = _mm256_or_si256(_mm256_and_si256(sign, yes.raw), _mm256_andnot_si256(sign, no.raw));
            V256::from_raw(r)
        }
    }

    #[inline(always)]
    unsafe fn if_negative_then_else_zero<T: Lane>(self, v: V256<T>, yes: V256<T>) -> V256<T> {
        unsafe {
            let sign = avx2_sign_mask::<T>(v.raw);
            V256::from_raw(_mm256_and_si256(sign, yes.raw))
        }
    }

    #[inline(always)]
    unsafe fn if_negative_then_zero_else<T: Lane>(self, v: V256<T>, no: V256<T>) -> V256<T> {
        unsafe {
            let sign = avx2_sign_mask::<T>(v.raw);
            V256::from_raw(_mm256_andnot_si256(sign, no.raw))
        }
    }
}

/// Build an all-ones/all-zeros sign-broadcast mask from raw bits, per lane width.
/// All-ones where the lane's MSB (sign bit) is set. Works for ints and floats.
#[inline(always)]
unsafe fn avx2_sign_mask<T: Lane>(raw: __m256i) -> __m256i {
    unsafe {
        match T::BYTES {
            1 => _mm256_cmpgt_epi8(_mm256_setzero_si256(), raw),
            2 => _mm256_srai_epi16(raw, 15),
            4 => _mm256_srai_epi32(raw, 31),
            _ => {
                let s = _mm256_srai_epi32(raw, 31);
                _mm256_shuffle_epi32(s, 0xF5) // _MM_SHUFFLE(3, 3, 1, 1)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimdConvert
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdConvert for Avx2 {
    #[inline(always)]
    unsafe fn promote_to<N: NarrowLane>(self, v: V256<N>) -> V256<N::Wide>
    where
        N::Wide: Lane,
    {
        unsafe {
            let raw = match N::BYTES {
                1 => {
                    if is_signed::<N>() {
                        _mm256_cvtepi8_epi16(_mm256_castsi256_si128(v.raw))
                    } else {
                        _mm256_cvtepu8_epi16(_mm256_castsi256_si128(v.raw))
                    }
                }
                2 => {
                    if is_signed::<N>() {
                        _mm256_cvtepi16_epi32(_mm256_castsi256_si128(v.raw))
                    } else {
                        _mm256_cvtepu16_epi32(_mm256_castsi256_si128(v.raw))
                    }
                }
                4 => {
                    if is_type::<N, f32>() {
                        let lo = _mm256_castsi256_si128(v.raw);
                        _mm256_castpd_si256(_mm256_cvtps_pd(_mm_castsi128_ps(lo)))
                    } else if is_signed::<N>() {
                        _mm256_cvtepi32_epi64(_mm256_castsi256_si128(v.raw))
                    } else {
                        _mm256_cvtepu32_epi64(_mm256_castsi256_si128(v.raw))
                    }
                }
                _ => v.raw,
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn demote_to<W: WideLane>(self, v: V256<W>) -> V256<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = match W::BYTES {
                2 => {
                    // i16/u16 -> i8/u8: process both 128-bit halves
                    let lo = _mm256_castsi256_si128(v.raw);
                    let hi = _mm256_extracti128_si256(v.raw, 1);
                    if is_signed::<W>() {
                        let packed = _mm_packs_epi16(lo, hi);
                        _mm256_castsi128_si256(packed)
                    } else {
                        // Clamp u16 to 255 before packus (which treats input as signed)
                        let max_val = _mm256_set1_epi16(0xFF);
                        let excess = _mm256_subs_epu16(v.raw, max_val);
                        let clamped = _mm256_sub_epi16(v.raw, excess);
                        let lo_c = _mm256_castsi256_si128(clamped);
                        let hi_c = _mm256_extracti128_si256(clamped, 1);
                        let packed = _mm_packus_epi16(lo_c, hi_c);
                        _mm256_castsi128_si256(packed)
                    }
                }
                4 => {
                    // i32/u32 -> i16/u16: process both 128-bit halves
                    let lo = _mm256_castsi256_si128(v.raw);
                    let hi = _mm256_extracti128_si256(v.raw, 1);
                    if is_signed::<W>() {
                        let packed = _mm_packs_epi32(lo, hi);
                        _mm256_castsi128_si256(packed)
                    } else {
                        // u32 -> u16: clamp to 0x7FFFFFFF so packus treats as positive
                        let max_i32 = _mm256_set1_epi32(0x7FFFFFFFu32 as i32);
                        let clamped = _mm256_min_epu32(v.raw, max_i32);
                        let lo_c = _mm256_castsi256_si128(clamped);
                        let hi_c = _mm256_extracti128_si256(clamped, 1);
                        let packed = _mm_packus_epi32(lo_c, hi_c);
                        _mm256_castsi128_si256(packed)
                    }
                }
                8 => {
                    if is_type::<W, f64>() {
                        // f64 -> f32
                        let ps = _mm256_cvtpd_ps(_mm256_castsi256_pd(v.raw));
                        _mm256_castsi128_si256(_mm_castps_si128(ps))
                    } else {
                        // 64 -> 32 bit saturating via SIMD clamp + shuffle pack
                        let clamped = if is_signed::<W>() {
                            let min_val = V256::<W>::from_raw(_mm256_set1_epi64x(i32::MIN as i64));
                            let max_val = V256::<W>::from_raw(_mm256_set1_epi64x(i32::MAX as i64));
                            self.min(self.max(v, min_val), max_val)
                        } else {
                            let max_val = V256::<W>::from_raw(_mm256_set1_epi64x(u32::MAX as i64));
                            self.min(v, max_val)
                        };
                        // Shuffle within each 128-bit lane: [lo0, lo1, ?, ?] per lane
                        let shuffled = _mm256_shuffle_epi32(clamped.raw, 0x08);
                        // Extract 128-bit halves, combine low 64 of each
                        let lo = _mm256_castsi256_si128(shuffled);
                        let hi = _mm256_extracti128_si256(shuffled, 1);
                        _mm256_castsi128_si256(_mm_unpacklo_epi64(lo, hi))
                    }
                }
                _ => v.raw,
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn convert_to_int<F: FloatLane>(self, v: V256<F>) -> V256<F::Int> {
        unsafe {
            let raw = if F::BYTES == 4 {
                _mm256_cvttps_epi32(_mm256_castsi256_ps(v.raw))
            } else {
                // f64 -> i64: extract each f64 lane and convert with saturation.
                // cvttsd_si64 returns i64::MIN for both positive and negative
                // overflow. C++ saturates positive overflow to i64::MAX.
                let v_pd = _mm256_castsi256_pd(v.raw);
                let overflow = _mm256_castpd_si256(_mm256_cmp_pd(
                    v_pd,
                    _mm256_set1_pd(9.223372036854776e18),
                    _CMP_GE_OQ,
                ));
                let lo = _mm256_castsi256_si128(v.raw);
                let hi = _mm256_extracti128_si256(v.raw, 1);
                let i0 = _mm_cvttsd_si64(_mm_castsi128_pd(lo));
                let i1 = _mm_cvttsd_si64(_mm_castsi128_pd(_mm_srli_si128(lo, 8)));
                let i2 = _mm_cvttsd_si64(_mm_castsi128_pd(hi));
                let i3 = _mm_cvttsd_si64(_mm_castsi128_pd(_mm_srli_si128(hi, 8)));
                let lo_r = _mm_set_epi64x(i1, i0);
                let hi_r = _mm_set_epi64x(i3, i2);
                let converted = _mm256_set_m128i(hi_r, lo_r);
                // Where overflow: i64::MAX; else: converted
                let max_val = _mm256_set1_epi64x(i64::MAX);
                _mm256_or_si256(
                    _mm256_and_si256(overflow, max_val),
                    _mm256_andnot_si256(overflow, converted),
                )
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn convert_to_float<F: FloatLane>(self, v: V256<F::Int>) -> V256<F> {
        unsafe {
            let raw = if F::BYTES == 4 {
                _mm256_castps_si256(_mm256_cvtepi32_ps(v.raw))
            } else {
                // i64 -> f64: Wim trick
                let k84_63 = _mm256_set1_epi64x(0x4530000080000000u64 as i64);
                let v_upper = _mm256_castpd_si256(_mm256_sub_pd(
                    _mm256_castsi256_pd(_mm256_xor_si256(_mm256_srli_epi64(v.raw, 32), k84_63)),
                    _mm256_castsi256_pd(_mm256_set1_epi64x(0x4530000080100000u64 as i64)),
                ));
                let k52 = _mm256_set1_epi64x(0x4330000000000000u64 as i64);
                // OddEven for u32 on AVX2: blend_epi32 with mask 0xAA (odd positions from k52)
                let odd_even = _mm256_blend_epi32(v.raw, k52, 0xAA);
                // v_upper contains -2^52 bias from subtraction constant;
                // odd_even as f64 = 2^52 + lower_bits, so they cancel.
                _mm256_castpd_si256(_mm256_add_pd(
                    _mm256_castsi256_pd(v_upper),
                    _mm256_castsi256_pd(odd_even),
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn truncate_to<W: WideLane>(self, v: V256<W>) -> V256<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = match W::BYTES {
                2 => {
                    // u16/i16 -> u8/i8: mask low byte, shuffle to pack
                    let mask = _mm256_set1_epi16(0x00FF);
                    let masked = _mm256_and_si256(v.raw, mask);
                    let shuf = _mm256_set_epi8(
                        -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0,
                        -1, -1, -1, -1, -1, -1, -1, -1, 14, 12, 10, 8, 6, 4, 2, 0,
                    );
                    let shuffled = _mm256_shuffle_epi8(masked, shuf);
                    // Each 128-bit lane has 8 bytes in low half; combine with permute
                    _mm256_permute4x64_epi64(shuffled, 0x08)
                }
                4 => {
                    // u32/i32 -> u16/i16: mask low 16 bits, shuffle to pack
                    let mask = _mm256_set1_epi32(0x0000FFFF);
                    let masked = _mm256_and_si256(v.raw, mask);
                    let shuf = _mm256_set_epi8(
                        -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0,
                        -1, -1, -1, -1, -1, -1, -1, -1, 13, 12, 9, 8, 5, 4, 1, 0,
                    );
                    let shuffled = _mm256_shuffle_epi8(masked, shuf);
                    _mm256_permute4x64_epi64(shuffled, 0x08)
                }
                8 => {
                    // u64/i64 -> u32/i32: extract low 32 from each 64, then permute
                    // shuffle_epi32 with 0x08 = [0,2,0,0] per 128-bit lane extracts low 32 of each 64
                    let shuffled = _mm256_shuffle_epi32(v.raw, 0x08);
                    // Now each 128-bit lane has [lo0, lo1, lo0, lo0]; the useful data is in low 64
                    let lo = _mm256_castsi256_si128(shuffled);
                    let hi = _mm256_extracti128_si256(shuffled, 1);
                    _mm256_castsi128_si256(_mm_unpacklo_epi64(lo, hi))
                }
                _ => v.raw,
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn ordered_demote_2_to<W: WideLane>(
        self,
        lo: V256<W>,
        hi: V256<W>,
    ) -> V256<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe {
            let raw = match W::BYTES {
                4 => {
                    if is_signed::<W>() {
                        // i32 -> i16
                        let packed = _mm256_packs_epi32(lo.raw, hi.raw);
                        _mm256_permute4x64_epi64(packed, 0xD8)
                    } else {
                        // u32 -> u16: clamp to 0x7FFFFFFF so packus treats as positive
                        let max_i32 = _mm256_set1_epi32(0x7FFFFFFFu32 as i32);
                        let lo_clamped = _mm256_min_epu32(lo.raw, max_i32);
                        let hi_clamped = _mm256_min_epu32(hi.raw, max_i32);
                        let packed = _mm256_packus_epi32(lo_clamped, hi_clamped);
                        _mm256_permute4x64_epi64(packed, 0xD8)
                    }
                }
                2 => {
                    if is_type::<W, u16>() {
                        // u16 -> u8: clamp to 0x7FFF so packus treats as positive
                        let max_i16 = _mm256_set1_epi16(0x7FFFu16 as i16);
                        let lo_clamped = _mm256_min_epu16(lo.raw, max_i16);
                        let hi_clamped = _mm256_min_epu16(hi.raw, max_i16);
                        let packed = _mm256_packus_epi16(lo_clamped, hi_clamped);
                        _mm256_permute4x64_epi64(packed, 0xD8)
                    } else {
                        // i16 -> i8
                        let packed = _mm256_packs_epi16(lo.raw, hi.raw);
                        _mm256_permute4x64_epi64(packed, 0xD8)
                    }
                }
                8 => {
                    // u64/i64 -> u32/i32: saturating demote
                    let lanes = 32 / W::BYTES; // 4 per vector
                    let mut lo_arr = [0i64; 4];
                    let mut hi_arr = [0i64; 4];
                    _mm256_storeu_si256(lo_arr.as_mut_ptr().cast(), lo.raw);
                    _mm256_storeu_si256(hi_arr.as_mut_ptr().cast(), hi.raw);
                    let mut result: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                    let result_u32: &mut [u32; 8] = core::mem::transmute(&mut *result);
                    if is_type::<W, u64>() {
                        for i in 0..lanes {
                            let v = lo_arr[i] as u64;
                            result_u32[i] = if v > u32::MAX as u64 { u32::MAX } else { v as u32 };
                        }
                        for i in 0..lanes {
                            let v = hi_arr[i] as u64;
                            result_u32[lanes + i] = if v > u32::MAX as u64 { u32::MAX } else { v as u32 };
                        }
                    } else {
                        // i64 -> i32
                        for i in 0..lanes {
                            let v = lo_arr[i];
                            result_u32[i] = if v > i32::MAX as i64 {
                                i32::MAX as u32
                            } else if v < i32::MIN as i64 {
                                i32::MIN as u32
                            } else {
                                v as u32
                            };
                        }
                        for i in 0..lanes {
                            let v = hi_arr[i];
                            result_u32[lanes + i] = if v > i32::MAX as i64 {
                                i32::MAX as u32
                            } else if v < i32::MIN as i64 {
                                i32::MIN as u32
                            } else {
                                v as u32
                            };
                        }
                    }
                    _mm256_load_si256(result.as_ptr().cast())
                }
                _ => lo.raw,
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn nearest_int<F: FloatLane>(self, v: V256<F>) -> V256<F::Int> {
        unsafe {
            let raw = if F::BYTES == 4 {
                // _mm256_cvtps_epi32: round-to-nearest using current mode (nearest-even).
                // Clamp >= 2^31 to i32::MAX.
                let ps = _mm256_castsi256_ps(v.raw);
                let overflow = _mm256_castps_si256(_mm256_cmp_ps(
                    ps,
                    _mm256_set1_ps(2147483648.0f32),
                    _CMP_GE_OQ,
                ));
                let max_f = _mm256_set1_ps(2147483520.0f32);
                let clamped = _mm256_min_ps(ps, max_f);
                let converted = _mm256_cvtps_epi32(clamped);
                _mm256_or_si256(
                    _mm256_and_si256(overflow, _mm256_set1_epi32(i32::MAX)),
                    _mm256_andnot_si256(overflow, converted),
                )
            } else {
                // f64 -> i64: scalar fallback with round-to-nearest-even.
                let v_pd = _mm256_castsi256_pd(v.raw);
                let overflow = _mm256_castpd_si256(_mm256_cmp_pd(
                    v_pd,
                    _mm256_set1_pd(9.223372036854776e18),
                    _CMP_GE_OQ,
                ));
                // Round to nearest-even, then truncate-convert.
                let rounded = _mm256_round_pd(v_pd, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                let lo = _mm256_castsi256_si128(_mm256_castpd_si256(rounded));
                let hi = _mm256_extracti128_si256(_mm256_castpd_si256(rounded), 1);
                let i0 = _mm_cvttsd_si64(_mm_castsi128_pd(lo));
                let i1 = _mm_cvttsd_si64(_mm_castsi128_pd(_mm_srli_si128(lo, 8)));
                let i2 = _mm_cvttsd_si64(_mm_castsi128_pd(hi));
                let i3 = _mm_cvttsd_si64(_mm_castsi128_pd(_mm_srli_si128(hi, 8)));
                let lo_r = _mm_set_epi64x(i1, i0);
                let hi_r = _mm_set_epi64x(i3, i2);
                let converted = _mm256_set_m128i(hi_r, lo_r);
                let max_val = _mm256_set1_epi64x(i64::MAX);
                _mm256_or_si256(
                    _mm256_and_si256(overflow, max_val),
                    _mm256_andnot_si256(overflow, converted),
                )
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn reorder_demote_2_to<W: WideLane>(
        self,
        a: V256<W>,
        b: V256<W>,
    ) -> V256<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // AVX2: use pack instructions WITHOUT the extra permute that ordered_demote_2_to does.
        // The pack instructions interleave blocks: [a_lo, b_lo, a_hi, b_hi] instead of
        // [a_lo, a_hi, b_lo, b_hi]. That's the "reordered" output.
        unsafe {
            // We need to call the pack intrinsics directly without the fix-up permute.
            // Let's look at what ordered_demote_2_to does and skip the final permute.
            // For simplicity and correctness, use a scalar loop approach:
            let _lanes_wide = 32 / W::BYTES;
            let mut arr_a = [0u8; 32];
            let mut arr_b = [0u8; 32];
            let mut result = [0u8; 32];
            _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
            _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);

            // Use the pack instructions which produce interleaved block order
            if W::BYTES == 4 && W::Narrow::BYTES == 2 {
                // i32->i16 or u32->u16: _mm256_packs_epi32 or _mm256_packus_epi32
                if is_type::<W, i32>() || is_type::<W, u32>() {
                    // For signed: saturating pack
                    if is_type::<W, i32>() {
                        let packed = _mm256_packs_epi32(a.raw, b.raw);
                        return V256::from_raw(packed);
                    }
                    // u32->u16: clamp then packus
                    let max_val = _mm256_set1_epi32(0xFFFF);
                    let a_clamped = _mm256_min_epu32(a.raw, max_val);
                    let b_clamped = _mm256_min_epu32(b.raw, max_val);
                    let packed = _mm256_packus_epi32(a_clamped, b_clamped);
                    return V256::from_raw(packed);
                }
            }
            if W::BYTES == 2 && W::Narrow::BYTES == 1 {
                if is_type::<W, i16>() {
                    let packed = _mm256_packs_epi16(a.raw, b.raw);
                    return V256::from_raw(packed);
                } else if is_type::<W, u16>() {
                    let max_val = _mm256_set1_epi16(0xFF_i16);
                    let a_clamped = _mm256_min_epu16(a.raw, max_val);
                    let b_clamped = _mm256_min_epu16(b.raw, max_val);
                    let packed = _mm256_packus_epi16(a_clamped, b_clamped);
                    return V256::from_raw(packed);
                }
            }
            // Fallback: scalar demote with reordered layout
            // Pack a's lower 128 bits, b's lower 128 bits, a's upper, b's upper
            let narrow_bytes = W::Narrow::BYTES;
            let mut dst = 0;
            // Lower 128 bits of a
            for i in 0..(16 / W::BYTES) {
                let off = i * W::BYTES;
                result[dst..dst+narrow_bytes].copy_from_slice(&arr_a[off..off+narrow_bytes]);
                dst += narrow_bytes;
            }
            // Lower 128 bits of b
            for i in 0..(16 / W::BYTES) {
                let off = i * W::BYTES;
                result[dst..dst+narrow_bytes].copy_from_slice(&arr_b[off..off+narrow_bytes]);
                dst += narrow_bytes;
            }
            // Upper 128 bits of a
            for i in (16 / W::BYTES)..(32 / W::BYTES) {
                let off = i * W::BYTES;
                result[dst..dst+narrow_bytes].copy_from_slice(&arr_a[off..off+narrow_bytes]);
                dst += narrow_bytes;
            }
            // Upper 128 bits of b
            for i in (16 / W::BYTES)..(32 / W::BYTES) {
                let off = i * W::BYTES;
                result[dst..dst+narrow_bytes].copy_from_slice(&arr_b[off..off+narrow_bytes]);
                dst += narrow_bytes;
            }
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn demote_in_range_to<W: WideLane>(self, v: V256<W>) -> V256<W::Narrow>
    where
        W::Narrow: Lane,
    {
        unsafe { self.demote_to(v) }
    }

    #[inline(always)]
    unsafe fn convert_in_range_to_int<F: FloatLane>(self, v: V256<F>) -> V256<F::Int> {
        unsafe { self.convert_to_int(v) }
    }

    #[inline(always)]
    unsafe fn promote_lower_to<N: NarrowLane>(self, v: V256<N>) -> V256<N::Wide>
    where
        N::Wide: Lane,
    {
        // Extract lower V128, then promote that to V256
        unsafe {
            let lo = _mm256_castsi256_si128(v.raw);
            // Promote 128-bit narrow to 256-bit wide
            let raw = match N::BYTES {
                1 => {
                    if is_type::<N, u8>() {
                        _mm256_cvtepu8_epi16(lo)
                    } else {
                        _mm256_cvtepi8_epi16(lo)
                    }
                }
                2 => {
                    if is_type::<N, u16>() {
                        _mm256_cvtepu16_epi32(lo)
                    } else {
                        _mm256_cvtepi16_epi32(lo)
                    }
                }
                4 => {
                    if is_type::<N, u32>() {
                        _mm256_cvtepu32_epi64(lo)
                    } else if is_type::<N, i32>() {
                        _mm256_cvtepi32_epi64(lo)
                    } else {
                        // f32 -> f64
                        let f = _mm_castsi128_ps(lo);
                        _mm256_castpd_si256(_mm256_cvtps_pd(f))
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn promote_upper_to<N: NarrowLane>(self, v: V256<N>) -> V256<N::Wide>
    where
        N::Wide: Lane,
    {
        // Extract upper V128, then promote that to V256
        unsafe {
            let hi = _mm256_extracti128_si256(v.raw, 1);
            let raw = match N::BYTES {
                1 => {
                    if is_type::<N, u8>() {
                        _mm256_cvtepu8_epi16(hi)
                    } else {
                        _mm256_cvtepi8_epi16(hi)
                    }
                }
                2 => {
                    if is_type::<N, u16>() {
                        _mm256_cvtepu16_epi32(hi)
                    } else {
                        _mm256_cvtepi16_epi32(hi)
                    }
                }
                4 => {
                    if is_type::<N, u32>() {
                        _mm256_cvtepu32_epi64(hi)
                    } else if is_type::<N, i32>() {
                        _mm256_cvtepi32_epi64(hi)
                    } else {
                        // f32 -> f64
                        let f = _mm_castsi128_ps(hi);
                        _mm256_castpd_si256(_mm256_cvtps_pd(f))
                    }
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn ordered_truncate_2_to<W: WideLane>(
        self,
        lo: V256<W>,
        hi: V256<W>,
    ) -> V256<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // OrderedTruncate2To = ConcatEven of the narrow-reinterpreted vectors.
        unsafe {
            let lo_n = self.bitcast::<W, W::Narrow>(lo);
            let hi_n = self.bitcast::<W, W::Narrow>(hi);
            self.concat_even(lo_n, hi_n)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdShuffle
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdShuffle for Avx2 {
    #[inline(always)]
    unsafe fn reverse<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            // First swap 128-bit halves, then reverse within each half
            let swapped = _mm256_permute2x128_si256(v.raw, v.raw, 0x01);
            let raw = match T::BYTES {
                1 => {
                    let idx = _mm256_set_epi8(
                        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6,
                        7, 8, 9, 10, 11, 12, 13, 14, 15,
                    );
                    _mm256_shuffle_epi8(swapped, idx)
                }
                2 => {
                    let idx = _mm256_set_epi8(
                        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7,
                        6, 9, 8, 11, 10, 13, 12, 15, 14,
                    );
                    _mm256_shuffle_epi8(swapped, idx)
                }
                4 => _mm256_shuffle_epi32(swapped, 0x1B),
                8 => _mm256_shuffle_epi32(swapped, 0x4E), // swap u64s within each 128-bit half
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn broadcast_lane<T: Lane, const IDX: usize>(self, v: V256<T>) -> V256<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    // _mm256_permutevar8x32_epi32 can broadcast any 32-bit lane
                    let idx_vec = _mm256_set1_epi32(IDX as i32);
                    V256::from_raw(_mm256_permutevar8x32_epi32(v.raw, idx_vec))
                }
                8 => {
                    // _mm256_permute4x64_epi64 with immediate: broadcast lane IDX
                    // The immediate selects src lane for each dst lane.
                    // We want all 4 dst lanes to come from src lane IDX.
                    // imm8 = IDX | (IDX << 2) | (IDX << 4) | (IDX << 6)
                    // Since IDX is const, the match should optimize away.
                    match IDX {
                        0 => V256::from_raw(_mm256_permute4x64_epi64(v.raw, 0x00)),
                        1 => V256::from_raw(_mm256_permute4x64_epi64(v.raw, 0x55)),
                        2 => V256::from_raw(_mm256_permute4x64_epi64(v.raw, 0xAA)),
                        3 => V256::from_raw(_mm256_permute4x64_epi64(v.raw, 0xFF)),
                        _ => {
                            // Fallback for out-of-range (shouldn't happen)
                            let val: T = self.extract_lane(v, IDX);
                            self.splat(val)
                        }
                    }
                }
                _ => {
                    // For 8-bit and 16-bit, use extract+splat (the shuffle_epi8
                    // approach would require building a byte index vector which
                    // is equally complex for arbitrary IDX)
                    let val: T = self.extract_lane(v, IDX);
                    self.splat(val)
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn interleave_lower<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm256_unpacklo_epi8(a.raw, b.raw),
                2 => _mm256_unpacklo_epi16(a.raw, b.raw),
                4 => _mm256_unpacklo_epi32(a.raw, b.raw),
                8 => _mm256_unpacklo_epi64(a.raw, b.raw),
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn interleave_upper<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => _mm256_unpackhi_epi8(a.raw, b.raw),
                2 => _mm256_unpackhi_epi16(a.raw, b.raw),
                4 => _mm256_unpackhi_epi32(a.raw, b.raw),
                8 => _mm256_unpackhi_epi64(a.raw, b.raw),
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn zip_lower<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.interleave_lower(a, b) }
    }

    #[inline(always)]
    unsafe fn zip_upper<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe { self.interleave_upper(a, b) }
    }

    #[inline(always)]
    unsafe fn table_lookup_bytes<T: Lane>(self, table: V256<T>, idx: V256<T>) -> V256<T> {
        // AVX2 pshufb operates within 128-bit lanes
        V256::from_raw(unsafe { _mm256_shuffle_epi8(table.raw, idx.raw) })
    }

    #[inline(always)]
    unsafe fn table_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        v: V256<T>,
        idx: V256<I>,
    ) -> V256<T> {
        unsafe {
            if T::BYTES == 4 {
                V256::from_raw(_mm256_permutevar8x32_epi32(v.raw, idx.raw))
            } else {
                // For other sizes, extract and reinsert
                let lanes = simd::lanes::<T, Avx2>();
                let mut data: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                let mut indices: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                _mm256_store_si256(data.as_mut_ptr().cast(), v.raw);
                _mm256_store_si256(indices.as_mut_ptr().cast(), idx.raw);
                let mut result: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                for i in 0..lanes {
                    let idx_off = i * I::BYTES;
                    let mut idx_val = 0u64;
                    core::ptr::copy_nonoverlapping(
                        indices.as_ptr().add(idx_off),
                        core::ptr::from_mut(&mut idx_val).cast::<u8>(),
                        I::BYTES,
                    );
                    let lane_idx = (idx_val as usize) % lanes;
                    let src_off = lane_idx * T::BYTES;
                    let dst_off = i * T::BYTES;
                    result[dst_off..dst_off + T::BYTES]
                        .copy_from_slice(&data[src_off..src_off + T::BYTES]);
                }
                V256::from_raw(_mm256_load_si256(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn reverse2<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Swap adjacent bytes within each 128-bit lane
                    let idx = _mm256_set_epi8(
                        14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1, 14, 15, 12, 13, 10,
                        11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                2 => {
                    // Swap adjacent 16-bit pairs using byte shuffle
                    let idx = _mm256_set_epi8(
                        13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2, 13, 12, 15, 14, 9, 8,
                        11, 10, 5, 4, 7, 6, 1, 0, 3, 2,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                4 => {
                    // Swap adjacent 32-bit pairs: [1,0,3,2] = 0b10_11_00_01
                    _mm256_shuffle_epi32(v.raw, 0b10_11_00_01)
                }
                8 => {
                    // Swap adjacent 64-bit pairs across the full 256-bit register
                    _mm256_permute4x64_epi64(v.raw, 0b10_11_00_01)
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn reverse4<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Reverse groups of 4 bytes within each 128-bit lane
                    let idx = _mm256_set_epi8(
                        12, 13, 14, 15, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9,
                        10, 11, 4, 5, 6, 7, 0, 1, 2, 3,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                2 => {
                    // Reverse groups of 4 u16 within each 128-bit lane
                    let idx = _mm256_set_epi8(
                        9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12,
                        15, 14, 1, 0, 3, 2, 5, 4, 7, 6,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                4 => {
                    // Reverse 4 lanes within each 128-bit half: [3,2,1,0] = 0b00_01_10_11
                    _mm256_shuffle_epi32(v.raw, 0b00_01_10_11)
                }
                8 => {
                    // Only 4 lanes total: full reverse
                    _mm256_permute4x64_epi64(v.raw, 0b00_01_10_11)
                }
                _ => unreachable!(),
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn reverse8<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = match T::BYTES {
                1 => {
                    // Reverse groups of 8 bytes within each 128-bit lane
                    let idx = _mm256_set_epi8(
                        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                        14, 15, 0, 1, 2, 3, 4, 5, 6, 7,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                2 => {
                    // 8 u16 per 128-bit lane: reverse within each lane
                    let idx = _mm256_set_epi8(
                        1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7,
                        6, 9, 8, 11, 10, 13, 12, 15, 14,
                    );
                    _mm256_shuffle_epi8(v.raw, idx)
                }
                4 => {
                    // 8 u32 total: reverse within each 128-bit half, then swap halves
                    let rev_halves = _mm256_shuffle_epi32(v.raw, 0b00_01_10_11);
                    _mm256_permute2x128_si256(rev_halves, rev_halves, 0x01)
                }
                _ => {
                    // 8-lane reverse not meaningful for 64-bit (only 4 lanes)
                    // Scalar fallback
                    let lanes = simd::lanes::<T, Avx2>();
                    let mut arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                    _mm256_store_si256(arr.as_mut_ptr().cast(), v.raw);
                    let mut result: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                    let group = 8.min(lanes);
                    let num_groups = lanes / group;
                    for g in 0..num_groups {
                        for i in 0..group {
                            let src = (g * group + i) * T::BYTES;
                            let dst = (g * group + (group - 1 - i)) * T::BYTES;
                            result[dst..dst + T::BYTES]
                                .copy_from_slice(arr[src..src + T::BYTES].as_ref());
                        }
                    }
                    _mm256_load_si256(result.as_ptr().cast())
                }
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn concat_upper_lower<T: Lane>(self, hi: V256<T>, lo: V256<T>) -> V256<T> {
        // Upper 128 bits from hi, lower 128 bits from lo
        V256::from_raw(unsafe { _mm256_blend_epi32(lo.raw, hi.raw, 0xF0) })
    }

    #[inline(always)]
    unsafe fn concat_lower_upper<T: Lane>(self, hi: V256<T>, lo: V256<T>) -> V256<T> {
        // Lower 128 bits from hi, upper 128 bits from lo
        V256::from_raw(unsafe { _mm256_blend_epi32(lo.raw, hi.raw, 0x0F) })
    }

    #[inline(always)]
    unsafe fn concat_even<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    // 8 lanes of 32-bit. Even lanes are 0,2,4,6.
                    // Take even lanes from a -> lower 4 lanes of result
                    // Take even lanes from b -> upper 4 lanes of result
                    // permutevar8x32 picks from a single source, so we need two permutes + blend
                    let even_idx = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
                    let a_even = _mm256_permutevar8x32_epi32(a.raw, even_idx);
                    let b_even = _mm256_permutevar8x32_epi32(b.raw, even_idx);
                    // a_even has [a0,a2,a4,a6, ...], b_even has [b0,b2,b4,b6, ...]
                    // We want lower 128 from a_even, upper 128 from b_even
                    V256::from_raw(_mm256_blend_epi32(a_even, b_even, 0xF0))
                }
                8 => {
                    // 4 lanes of 64-bit. Even lanes are 0,2.
                    // From a: lanes 0,2 -> result lanes 0,1
                    // From b: lanes 0,2 -> result lanes 2,3
                    let a_perm = _mm256_permute4x64_epi64(a.raw, 0b00_00_10_00); // [a0,a2,_,_]
                    let b_perm = _mm256_permute4x64_epi64(b.raw, 0b00_00_10_00); // [b0,b2,_,_]
                    // Data is in lower 128 bits of each; combine lower halves
                    V256::from_raw(_mm256_permute2x128_si256(a_perm, b_perm, 0x20))
                }
                2 => {
                    // 16 lanes of 16-bit. Even lanes: 0,2,4,6,8,10,12,14
                    // Use _mm256_shuffle_epi8 to pack even 16-bit lanes into lower bytes
                    // within each 128-bit lane, then combine.
                    // Even 16-bit lanes in lower 128-bit half: bytes 0-1, 4-5, 8-9, 12-13
                    let shuf_even = _mm256_setr_epi8(
                        0, 1, 4, 5, 8, 9, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 4, 5, 8, 9,
                        12, 13, -1, -1, -1, -1, -1, -1, -1, -1,
                    );
                    let a_shuf = _mm256_shuffle_epi8(a.raw, shuf_even);
                    let b_shuf = _mm256_shuffle_epi8(b.raw, shuf_even);
                    // a_shuf: lower 128 has [a0,a2,a4,a6, 0,0,0,0] (8 bytes of data)
                    //         upper 128 has [a8,a10,a12,a14, 0,0,0,0]
                    // We need to combine the 8 bytes from lower + 8 bytes from upper
                    // into a single 128-bit half. Use _mm256_permutevar8x32_epi32:
                    // Lower 64 bits of lower 128 = dwords 0,1
                    // Lower 64 bits of upper 128 = dwords 4,5
                    let a_packed = _mm256_permutevar8x32_epi32(
                        a_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    let b_packed = _mm256_permutevar8x32_epi32(
                        b_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    // a_packed lower 128 has all 8 even lanes from a
                    // b_packed lower 128 has all 8 even lanes from b
                    // Combine lower halves: a's lower 128 -> result lower, b's lower 128 -> result upper
                    V256::from_raw(_mm256_permute2x128_si256(a_packed, b_packed, 0x20))
                }
                1 => {
                    // 32 lanes of 8-bit. Even lanes: 0,2,4,...,30
                    let shuf_even = _mm256_setr_epi8(
                        0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 4, 6, 8,
                        10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
                    );
                    let a_shuf = _mm256_shuffle_epi8(a.raw, shuf_even);
                    let b_shuf = _mm256_shuffle_epi8(b.raw, shuf_even);
                    let a_packed = _mm256_permutevar8x32_epi32(
                        a_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    let b_packed = _mm256_permutevar8x32_epi32(
                        b_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    V256::from_raw(_mm256_permute2x128_si256(a_packed, b_packed, 0x20))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    unsafe fn concat_odd<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            match T::BYTES {
                4 => {
                    let odd_idx = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
                    let a_odd = _mm256_permutevar8x32_epi32(a.raw, odd_idx);
                    let b_odd = _mm256_permutevar8x32_epi32(b.raw, odd_idx);
                    V256::from_raw(_mm256_blend_epi32(a_odd, b_odd, 0xF0))
                }
                8 => {
                    // Odd lanes are 1,3
                    let a_perm = _mm256_permute4x64_epi64(a.raw, 0b00_00_11_01); // [a1,a3,_,_]
                    let b_perm = _mm256_permute4x64_epi64(b.raw, 0b00_00_11_01); // [b1,b3,_,_]
                    // Data is in lower 128 bits of each; combine lower halves
                    V256::from_raw(_mm256_permute2x128_si256(a_perm, b_perm, 0x20))
                }
                2 => {
                    // Odd 16-bit lanes: bytes 2-3, 6-7, 10-11, 14-15
                    let shuf_odd = _mm256_setr_epi8(
                        2, 3, 6, 7, 10, 11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 6, 7, 10,
                        11, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                    );
                    let a_shuf = _mm256_shuffle_epi8(a.raw, shuf_odd);
                    let b_shuf = _mm256_shuffle_epi8(b.raw, shuf_odd);
                    let a_packed = _mm256_permutevar8x32_epi32(
                        a_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    let b_packed = _mm256_permutevar8x32_epi32(
                        b_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    V256::from_raw(_mm256_permute2x128_si256(a_packed, b_packed, 0x20))
                }
                1 => {
                    let shuf_odd = _mm256_setr_epi8(
                        1, 3, 5, 7, 9, 11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1, 1, 3, 5, 7, 9,
                        11, 13, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                    );
                    let a_shuf = _mm256_shuffle_epi8(a.raw, shuf_odd);
                    let b_shuf = _mm256_shuffle_epi8(b.raw, shuf_odd);
                    let a_packed = _mm256_permutevar8x32_epi32(
                        a_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    let b_packed = _mm256_permutevar8x32_epi32(
                        b_shuf,
                        _mm256_setr_epi32(0, 1, 4, 5, 0, 0, 0, 0),
                    );
                    V256::from_raw(_mm256_permute2x128_si256(a_packed, b_packed, 0x20))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    unsafe fn odd_even<T: Lane>(self, odd: V256<T>, even: V256<T>) -> V256<T> {
        unsafe {
            // Blend: take even-indexed lanes from `even`, odd-indexed lanes from `odd`
            match T::BYTES {
                1 => {
                    let mask = _mm256_set1_epi16(0x00FFu16 as i16); // even bytes = 0xFF
                    V256::from_raw(_mm256_or_si256(
                        _mm256_and_si256(mask, even.raw),
                        _mm256_andnot_si256(mask, odd.raw),
                    ))
                }
                2 => {
                    let mask = _mm256_set1_epi32(0x0000FFFFu32 as i32); // even 16-bit lanes
                    V256::from_raw(_mm256_or_si256(
                        _mm256_and_si256(mask, even.raw),
                        _mm256_andnot_si256(mask, odd.raw),
                    ))
                }
                4 => {
                    // blend_epi32: bit i selects from b when set, a when clear
                    // Even lanes (0,2,4,6) from even, odd lanes (1,3,5,7) from odd
                    // 0b10101010 = 0xAA: bits 1,3,5,7 set => those come from odd
                    V256::from_raw(_mm256_blend_epi32(even.raw, odd.raw, 0xAA))
                }
                8 => {
                    // Even lanes (0,2) from even, odd lanes (1,3) from odd
                    // Use blend_epi32 with 4-i32-per-64-bit-lane awareness
                    // Lane 0 = i32 0..1, Lane 1 = i32 2..3, Lane 2 = i32 4..5, Lane 3 = i32 6..7
                    // Even i64 lanes: 0,2 => i32 0,1,4,5 => bits 0,1,4,5 = 0x33
                    // Odd i64 lanes: 1,3 => i32 2,3,6,7 => bits 2,3,6,7 = 0xCC
                    V256::from_raw(_mm256_blend_epi32(even.raw, odd.raw, 0xCC))
                }
                _ => unreachable!(),
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn slide_up_lanes<T: Lane>(self, v: V256<T>, n: usize) -> V256<T> {
        unsafe {
            if T::BYTES >= 4 {
                // SIMD approach Iota + And + cmpeq + permutevar.
                // For u64, treat as pairs of u32 lanes (amt * 2).
                let i32_amt = (n * (T::BYTES / 4)) as u32;
                let iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                let start = _mm256_set1_epi32(0u32.wrapping_sub(i32_amt) as i32);
                let idx = _mm256_add_epi32(iota, start);
                let max_idx = _mm256_set1_epi32(7);
                let masked_idx = _mm256_and_si256(idx, max_idx);
                let valid = _mm256_cmpeq_epi32(idx, masked_idx);
                let permuted = _mm256_permutevar8x32_epi32(v.raw, masked_idx);
                V256::from_raw(_mm256_and_si256(permuted, valid))
            } else {
                // u8/u16: cross-lane byte shift via store/load.
                let byte_shift = n * T::BYTES;
                if byte_shift >= 32 {
                    return self.zero();
                }
                let mut buf: Aligned<A32, [u8; 64]> = Aligned::new([0u8; 64]);
                _mm256_store_si256(buf.as_mut_ptr().add(32).cast(), v.raw);
                V256::from_raw(_mm256_loadu_si256(buf.as_ptr().add(32 - byte_shift).cast()))
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::needless_range_loop)]
    unsafe fn slide_down_lanes<T: Lane>(self, v: V256<T>, n: usize) -> V256<T> {
        unsafe {
            if T::BYTES >= 4 {
                // SIMD approach Iota + And + cmpeq + permutevar.
                let i32_amt = (n * (T::BYTES / 4)) as i32;
                let iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
                let idx = _mm256_add_epi32(iota, _mm256_set1_epi32(i32_amt));
                let max_idx = _mm256_set1_epi32(7);
                let masked_idx = _mm256_and_si256(idx, max_idx);
                let valid = _mm256_cmpeq_epi32(idx, masked_idx);
                let permuted = _mm256_permutevar8x32_epi32(v.raw, masked_idx);
                V256::from_raw(_mm256_and_si256(permuted, valid))
            } else {
                // u8/u16: cross-lane byte shift via store/load.
                let byte_shift = n * T::BYTES;
                if byte_shift >= 32 {
                    return self.zero();
                }
                let mut buf: Aligned<A32, [u8; 64]> = Aligned::new([0u8; 64]);
                _mm256_store_si256(buf.as_mut_ptr().cast(), v.raw);
                V256::from_raw(_mm256_loadu_si256(buf.as_ptr().add(byte_shift).cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn compress<T: Lane>(self, v: V256<T>, mask: M256<T>) -> V256<T> {
        unsafe {
            if T::BYTES == 4 {
                // LUT-based compress for 8 * u32/i32/f32 lanes.
                let mask_bits = _mm256_movemask_ps(_mm256_castsi256_ps(mask.raw)) as usize;
                let packed = _mm256_set1_epi32(COMPRESS_32X8_LUT.value[mask_bits] as i32);
                let shifts = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
                let indices = _mm256_srlv_epi32(packed, shifts);
                V256::from_raw(_mm256_permutevar8x32_epi32(v.raw, indices))
            } else if T::BYTES == 8 {
                // LUT-based compress for 4 * u64/i64/f64 lanes.
                let mask_bits = _mm256_movemask_pd(_mm256_castsi256_pd(mask.raw)) as usize;
                let idx_ptr = COMPRESS_64X4_LUT.value.as_ptr().add(mask_bits * 8);
                let indices = _mm256_load_si256(idx_ptr.cast());
                V256::from_raw(_mm256_permutevar8x32_epi32(v.raw, indices))
            } else {
                // u8/u16/i8/i16: LUT infeasible, use scalar fallback.
                let lanes = simd::lanes::<T, Avx2>();
                let mut data: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                let mut mask_arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                _mm256_store_si256(data.as_mut_ptr().cast(), v.raw);
                _mm256_store_si256(mask_arr.as_mut_ptr().cast(), mask.raw);
                let mut result: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
                let mut dst = 0usize;
                for i in 0..lanes {
                    let off = i * T::BYTES;
                    let mask_byte = mask_arr[off + T::BYTES - 1];
                    if mask_byte & 0x80 != 0 {
                        result[dst * T::BYTES..(dst + 1) * T::BYTES]
                            .copy_from_slice(data[off..off + T::BYTES].as_ref());
                        dst += 1;
                    }
                }
                V256::from_raw(_mm256_load_si256(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn compress_store<T: Lane>(self, v: V256<T>, mask: M256<T>, ptr: *mut T) -> usize {
        unsafe {
            let compressed = self.compress(v, mask);
            let count = self.count_true::<T>(mask);
            // Store count lanes from compressed result.
            // For simplicity, store full vector (caller only reads `count` lanes).
            _mm256_storeu_si256(ptr.cast(), compressed.raw);
            count
        }
    }

    #[inline(always)]
    unsafe fn dup_even<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm256_castps_si256(_mm256_moveldup_ps(_mm256_castsi256_ps(v.raw)))
            } else if is_type::<T, f64>() {
                _mm256_castpd_si256(_mm256_movedup_pd(_mm256_castsi256_pd(v.raw)))
            } else if T::BYTES == 4 {
                // u32/i32: [v0,v0,v2,v2] per 128-bit lane
                _mm256_shuffle_epi32(v.raw, 0xA0)
            } else if T::BYTES == 8 {
                // u64/i64: duplicate low 64 of each 128-bit lane
                _mm256_unpacklo_epi64(v.raw, v.raw)
            } else if T::BYTES == 2 {
                // u16/i16: duplicate even 16-bit lanes
                let even = _mm256_and_si256(v.raw, _mm256_set1_epi32(0x0000FFFFu32 as i32));
                _mm256_or_si256(even, _mm256_slli_epi32(even, 16))
            } else {
                // u8/i8: duplicate even bytes
                let even = _mm256_and_si256(v.raw, _mm256_set1_epi16(0x00FFu16 as i16));
                _mm256_or_si256(even, _mm256_slli_epi16(even, 8))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn dup_odd<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if is_type::<T, f32>() {
                _mm256_castps_si256(_mm256_movehdup_ps(_mm256_castsi256_ps(v.raw)))
            } else if is_type::<T, f64>() {
                _mm256_castpd_si256(_mm256_permute_pd::<0xF>(_mm256_castsi256_pd(v.raw)))
            } else if T::BYTES == 4 {
                // u32/i32: [v1,v1,v3,v3] per 128-bit lane
                _mm256_shuffle_epi32(v.raw, 0xF5)
            } else if T::BYTES == 8 {
                // u64/i64: duplicate high 64 of each 128-bit lane
                _mm256_unpackhi_epi64(v.raw, v.raw)
            } else if T::BYTES == 2 {
                // u16/i16: duplicate odd 16-bit lanes
                let odd = _mm256_srli_epi32(v.raw, 16);
                _mm256_or_si256(odd, _mm256_slli_epi32(odd, 16))
            } else {
                // u8/i8: duplicate odd bytes
                let odd = _mm256_srli_epi16(v.raw, 8);
                _mm256_or_si256(odd, _mm256_slli_epi16(odd, 8))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn concat_lower_lower<T: Lane>(
        self,
        hi: V256<T>,
        lo: V256<T>,
    ) -> V256<T> {
        // Lower 128 of lo in low, lower 128 of hi in high
        V256::from_raw(unsafe { _mm256_permute2x128_si256(lo.raw, hi.raw, 0x20) })
    }

    #[inline(always)]
    unsafe fn concat_upper_upper<T: Lane>(
        self,
        hi: V256<T>,
        lo: V256<T>,
    ) -> V256<T> {
        // Upper 128 of lo in low, upper 128 of hi in high
        V256::from_raw(unsafe { _mm256_permute2x128_si256(lo.raw, hi.raw, 0x31) })
    }

    #[inline(always)]
    unsafe fn slide_1_up<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            // Shift within each 128-bit lane
            let shifted = match T::BYTES {
                1 => _mm256_bslli_epi128(v.raw, 1),
                2 => _mm256_bslli_epi128(v.raw, 2),
                4 => _mm256_bslli_epi128(v.raw, 4),
                8 => _mm256_bslli_epi128(v.raw, 8),
                _ => unreachable!(),
            };
            // Handle cross-lane carry: top element of lower 128 -> bottom of upper 128
            // carry_lane = [zero, lower_128_of_v]
            let carry_lane = _mm256_permute2x128_si256(v.raw, _mm256_setzero_si256(), 0x03);
            let carry = match T::BYTES {
                1 => _mm256_bsrli_epi128(carry_lane, 15),
                2 => _mm256_bsrli_epi128(carry_lane, 14),
                4 => _mm256_bsrli_epi128(carry_lane, 12),
                8 => _mm256_bsrli_epi128(carry_lane, 8),
                _ => unreachable!(),
            };
            V256::from_raw(_mm256_or_si256(shifted, carry))
        }
    }

    #[inline(always)]
    unsafe fn slide_1_down<T: Lane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let shifted = match T::BYTES {
                1 => _mm256_bsrli_epi128(v.raw, 1),
                2 => _mm256_bsrli_epi128(v.raw, 2),
                4 => _mm256_bsrli_epi128(v.raw, 4),
                8 => _mm256_bsrli_epi128(v.raw, 8),
                _ => unreachable!(),
            };
            // Carry: lowest element of upper lane -> highest position of lower lane
            // carry_lane = [upper_128_of_v, zero]
            let carry_lane = _mm256_permute2x128_si256(v.raw, _mm256_setzero_si256(), 0x21);
            let carry = match T::BYTES {
                1 => _mm256_bslli_epi128(carry_lane, 15),
                2 => _mm256_bslli_epi128(carry_lane, 14),
                4 => _mm256_bslli_epi128(carry_lane, 12),
                8 => _mm256_bslli_epi128(carry_lane, 8),
                _ => unreachable!(),
            };
            V256::from_raw(_mm256_or_si256(shifted, carry))
        }
    }

    #[inline(always)]
    unsafe fn expand<T: Lane>(self, v: V256<T>, mask: M256<T>) -> V256<T> {
        unsafe {
            // No native AVX2 expand; scalar fallback
            let lanes = simd::lanes::<T, Avx2>();
            let mut data: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut mask_arr: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            _mm256_store_si256(data.as_mut_ptr().cast(), v.raw);
            _mm256_store_si256(mask_arr.as_mut_ptr().cast(), mask.raw);
            let mut result: Aligned<A32, [u8; 32]> = Aligned::new([0u8; 32]);
            let mut src = 0usize;
            for i in 0..lanes {
                let off = i * T::BYTES;
                let mask_byte = mask_arr[off + T::BYTES - 1];
                if mask_byte & 0x80 != 0 {
                    result[off..off + T::BYTES]
                        .copy_from_slice(&data[src * T::BYTES..(src + 1) * T::BYTES]);
                    src += 1;
                }
                // else: result stays zero
            }
            V256::from_raw(_mm256_load_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn combine_shift_right_bytes<T: Lane, const BYTES: usize>(
        self,
        hi: V256<T>,
        lo: V256<T>,
    ) -> V256<T> {
        // _mm256_alignr_epi8 is per-128-bit-lane PALIGNR.
        unsafe {
            let raw = match BYTES {
                0 => lo.raw,
                1 => _mm256_alignr_epi8::<1>(hi.raw, lo.raw),
                2 => _mm256_alignr_epi8::<2>(hi.raw, lo.raw),
                3 => _mm256_alignr_epi8::<3>(hi.raw, lo.raw),
                4 => _mm256_alignr_epi8::<4>(hi.raw, lo.raw),
                5 => _mm256_alignr_epi8::<5>(hi.raw, lo.raw),
                6 => _mm256_alignr_epi8::<6>(hi.raw, lo.raw),
                7 => _mm256_alignr_epi8::<7>(hi.raw, lo.raw),
                8 => _mm256_alignr_epi8::<8>(hi.raw, lo.raw),
                9 => _mm256_alignr_epi8::<9>(hi.raw, lo.raw),
                10 => _mm256_alignr_epi8::<10>(hi.raw, lo.raw),
                11 => _mm256_alignr_epi8::<11>(hi.raw, lo.raw),
                12 => _mm256_alignr_epi8::<12>(hi.raw, lo.raw),
                13 => _mm256_alignr_epi8::<13>(hi.raw, lo.raw),
                14 => _mm256_alignr_epi8::<14>(hi.raw, lo.raw),
                15 => _mm256_alignr_epi8::<15>(hi.raw, lo.raw),
                _ => hi.raw,
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn compress_blended_store<T: Lane>(
        self,
        v: V256<T>,
        mask: M256<T>,
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
    unsafe fn odd_even_blocks<T: Lane>(
        self,
        odd: V256<T>,
        even: V256<T>,
    ) -> V256<T> {
        // AVX2: 2 blocks. Block 0 (lower 128) from even, block 1 (upper 128) from odd.
        // _mm256_blend_epi32 with mask 0x0F: lower 4 i32 from even, upper 4 from odd.
        unsafe {
            V256::from_raw(_mm256_blend_epi32(odd.raw, even.raw, 0x0F))
        }
    }

    #[inline(always)]
    unsafe fn reverse_blocks<T: Lane>(self, v: V256<T>) -> V256<T> {
        // Swap the two 128-bit halves. _mm256_permute4x64_epi64(v, 0x4E) swaps
        // the two 128-bit lanes (01|23 -> 23|01).
        unsafe {
            V256::from_raw(_mm256_permute4x64_epi64(v.raw, 0x4E))
        }
    }

    #[inline(always)]
    unsafe fn compress_not<T: Lane>(self, v: V256<T>, mask: M256<T>) -> V256<T> {
        unsafe { self.compress(v, self.not_mask(mask)) }
    }

    #[inline(always)]
    unsafe fn compress_blocks_not(self, v: V256<u64>, mask: M256<u64>) -> V256<u64> {
        // 2 blocks: invert mask and compress
        unsafe { self.compress(v, self.not_mask(mask)) }
    }

    #[inline(always)]
    unsafe fn broadcast_block<T: Lane, const IDX: usize>(self, v: V256<T>) -> V256<T> {
        unsafe {
            if IDX == 0 {
                // Broadcast lower 128-bit block to both halves
                let lo = _mm256_castsi256_si128(v.raw);
                V256::from_raw(_mm256_broadcastsi128_si256(lo))
            } else {
                // IDX == 1: broadcast upper 128-bit block
                let hi = _mm256_extracti128_si256(v.raw, 1);
                V256::from_raw(_mm256_broadcastsi128_si256(hi))
            }
        }
    }

    #[inline(always)]
    unsafe fn compress_bits<T: Lane>(self, v: V256<T>, bits: *const u8) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut mask_arr = [0u8; 32];
            for i in 0..lanes {
                let byte_idx = i / 8;
                let bit_in_byte = i % 8;
                let b = bits.add(byte_idx).read();
                if (b >> bit_in_byte) & 1 != 0 {
                    for k in 0..T::BYTES {
                        mask_arr[i * T::BYTES + k] = 0xFF;
                    }
                }
            }
            let mask = M256::from_raw(_mm256_loadu_si256(mask_arr.as_ptr().cast()));
            self.compress(v, mask)
        }
    }

    #[inline(always)]
    unsafe fn compress_bits_store<T: Lane>(self, v: V256<T>, bits: *const u8, ptr: *mut T) -> usize {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut mask_arr = [0u8; 32];
            for i in 0..lanes {
                let byte_idx = i / 8;
                let bit_in_byte = i % 8;
                let b = bits.add(byte_idx).read();
                if (b >> bit_in_byte) & 1 != 0 {
                    for k in 0..T::BYTES {
                        mask_arr[i * T::BYTES + k] = 0xFF;
                    }
                }
            }
            let mask = M256::from_raw(_mm256_loadu_si256(mask_arr.as_ptr().cast()));
            self.compress_store(v, mask, ptr)
        }
    }

    #[inline(always)]
    unsafe fn lower_half<T: Lane>(self, v: V256<T>) -> crate::backend::sse2::V128<T> {
        unsafe {
            crate::backend::sse2::V128::from_raw(_mm256_castsi256_si128(v.raw))
        }
    }

    #[inline(always)]
    unsafe fn upper_half<T: Lane>(self, v: V256<T>) -> crate::backend::sse2::V128<T> {
        unsafe {
            crate::backend::sse2::V128::from_raw(_mm256_extracti128_si256(v.raw, 1))
        }
    }

    #[inline(always)]
    unsafe fn combine<T: Lane>(self, lo: crate::backend::sse2::V128<T>, hi: crate::backend::sse2::V128<T>) -> V256<T> {
        unsafe {
            let lo256 = _mm256_castsi128_si256(lo.raw());
            V256::from_raw(_mm256_inserti128_si256(lo256, hi.raw(), 1))
        }
    }

    #[inline(always)]
    unsafe fn insert_block<T: Lane, const IDX: usize>(self, v: V256<T>, blk: crate::backend::sse2::V128<T>) -> V256<T> {
        unsafe {
            if IDX == 0 {
                V256::from_raw(_mm256_inserti128_si256(v.raw, blk.raw(), 0))
            } else {
                V256::from_raw(_mm256_inserti128_si256(v.raw, blk.raw(), 1))
            }
        }
    }

    #[inline(always)]
    unsafe fn extract_block<T: Lane, const IDX: usize>(self, v: V256<T>) -> crate::backend::sse2::V128<T> {
        unsafe {
            if IDX == 0 {
                crate::backend::sse2::V128::from_raw(_mm256_castsi256_si128(v.raw))
            } else {
                crate::backend::sse2::V128::from_raw(_mm256_extracti128_si256(v.raw, 1))
            }
        }
    }

    #[inline(always)]
    unsafe fn interleave_whole_lower<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        // Cross-block interleave of lower halves
        // Result: interleave lanes from lower half of a and lower half of b
        unsafe {
            let il = self.interleave_lower(a, b);
            let iu = self.interleave_upper(a, b);
            self.concat_lower_lower(iu, il)
        }
    }

    #[inline(always)]
    unsafe fn interleave_whole_upper<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        // Cross-block interleave of upper halves
        unsafe {
            let il = self.interleave_lower(a, b);
            let iu = self.interleave_upper(a, b);
            self.concat_upper_upper(iu, il)
        }
    }

    #[inline(always)]
    unsafe fn interleave_even<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut arr_a = [0u8; 32];
            let mut arr_b = [0u8; 32];
            let mut result = [0u8; 32];
            _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
            _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);
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
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn interleave_odd<T: Lane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut arr_a = [0u8; 32];
            let mut arr_b = [0u8; 32];
            let mut result = [0u8; 32];
            _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
            _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);
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
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn two_tables_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        a: V256<T>,
        b: V256<T>,
        idx: V256<I>,
    ) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut arr_a = [0u8; 32];
            let mut arr_b = [0u8; 32];
            let mut arr_idx = [0u8; 32];
            let mut result = [0u8; 32];
            _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
            _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);
            _mm256_storeu_si256(arr_idx.as_mut_ptr().cast(), idx.raw);
            for i in 0..lanes {
                let idx_off = i * I::BYTES;
                let lane_idx: usize = match I::BYTES {
                    1 => arr_idx[idx_off] as usize,
                    2 => u16::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1]]) as usize,
                    4 => u32::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3]]) as usize,
                    _ => u64::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3], arr_idx[idx_off+4], arr_idx[idx_off+5], arr_idx[idx_off+6], arr_idx[idx_off+7]]) as usize,
                };
                let dst_off = i * T::BYTES;
                if lane_idx < lanes {
                    let src_off = lane_idx * T::BYTES;
                    result[dst_off..dst_off+T::BYTES].copy_from_slice(&arr_a[src_off..src_off+T::BYTES]);
                } else if lane_idx < 2 * lanes {
                    let src_off = (lane_idx - lanes) * T::BYTES;
                    result[dst_off..dst_off+T::BYTES].copy_from_slice(&arr_b[src_off..src_off+T::BYTES]);
                }
            }
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }

    #[inline(always)]
    unsafe fn table_lookup_lanes_or0<T: Lane, I: IntegerLane>(
        self,
        v: V256<T>,
        idx: V256<I>,
    ) -> V256<T> {
        unsafe {
            let lanes = 32 / T::BYTES;
            let mut arr_v = [0u8; 32];
            let mut arr_idx = [0u8; 32];
            let mut result = [0u8; 32];
            _mm256_storeu_si256(arr_v.as_mut_ptr().cast(), v.raw);
            _mm256_storeu_si256(arr_idx.as_mut_ptr().cast(), idx.raw);
            for i in 0..lanes {
                let idx_off = i * I::BYTES;
                let lane_idx_signed: i64 = match I::BYTES {
                    1 => arr_idx[idx_off] as i8 as i64,
                    2 => i16::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1]]) as i64,
                    4 => i32::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3]]) as i64,
                    _ => i64::from_le_bytes([arr_idx[idx_off], arr_idx[idx_off+1], arr_idx[idx_off+2], arr_idx[idx_off+3], arr_idx[idx_off+4], arr_idx[idx_off+5], arr_idx[idx_off+6], arr_idx[idx_off+7]]),
                };
                let dst_off = i * T::BYTES;
                if lane_idx_signed < 0 || lane_idx_signed as usize >= lanes {
                    for k in 0..T::BYTES {
                        result[dst_off + k] = 0;
                    }
                } else {
                    let src_off = (lane_idx_signed as usize) * T::BYTES;
                    result[dst_off..dst_off+T::BYTES].copy_from_slice(&arr_v[src_off..src_off+T::BYTES]);
                }
            }
            V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
        }
    }
}

// ---------------------------------------------------------------------------
// SimdReduce
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2.
unsafe impl SimdReduce for Avx2 {
    #[inline(always)]
    unsafe fn sum_of_lanes<T: Lane>(self, v: V256<T>) -> T {
        unsafe {
            // Step 1: reduce 256-bit -> 128-bit by adding upper and lower halves.
            let lo = _mm256_castsi256_si128(v.raw);
            let hi = _mm256_extracti128_si256(v.raw, 1);

            let mut r: __m128i;
            match T::BYTES {
                1 => {
                    r = _mm_add_epi8(lo, hi);
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 8));
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 4));
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 2));
                    r = _mm_add_epi8(r, _mm_srli_si128(r, 1));
                }
                2 => {
                    r = _mm_add_epi16(lo, hi);
                    r = _mm_add_epi16(r, _mm_srli_si128(r, 8));
                    r = _mm_add_epi16(r, _mm_srli_si128(r, 4));
                    r = _mm_add_epi16(r, _mm_srli_si128(r, 2));
                }
                4 => {
                    if is_type::<T, f32>() {
                        let flo = _mm_castsi128_ps(lo);
                        let fhi = _mm_castsi128_ps(hi);
                        let mut f = _mm_add_ps(flo, fhi);
                        f = _mm_add_ps(f, _mm_movehl_ps(f, f));
                        f = _mm_add_ss(f, _mm_shuffle_ps(f, f, 1));
                        r = _mm_castps_si128(f);
                    } else {
                        r = _mm_add_epi32(lo, hi);
                        r = _mm_add_epi32(r, _mm_srli_si128(r, 8));
                        r = _mm_add_epi32(r, _mm_srli_si128(r, 4));
                    }
                }
                8 => {
                    if is_type::<T, f64>() {
                        let flo = _mm_castsi128_pd(lo);
                        let fhi = _mm_castsi128_pd(hi);
                        let mut f = _mm_add_pd(flo, fhi);
                        f = _mm_add_pd(f, _mm_shuffle_pd(f, f, 1));
                        r = _mm_castpd_si128(f);
                    } else {
                        r = _mm_add_epi64(lo, hi);
                        r = _mm_add_epi64(r, _mm_srli_si128(r, 8));
                    }
                }
                _ => unreachable!(),
            }
            core::mem::transmute_copy(&_mm_cvtsi128_si64(r))
        }
    }

    #[inline(always)]
    unsafe fn min_of_lanes<T: Lane>(self, v: V256<T>) -> T {
        unsafe {
            // Tree reduction: swap 128-bit halves, min, then shift-and-min within lower half.
            // _mm256_permute2x128_si256(v, v, 0x01) swaps the two 128-bit halves.
            let swapped = V256::<T>::from_raw(_mm256_permute2x128_si256(v.raw, v.raw, 0x01));
            let mut r = self.min(v, swapped);
            // Now the answer is fully in the lower 128-bit lane.
            // _mm256_srli_si256 shifts within each 128-bit lane, which is what we want.
            match T::BYTES {
                1 => {
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                    r = self.min(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<4>(r.raw));
                    r = self.min(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<2>(r.raw));
                    r = self.min(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<1>(r.raw));
                    r = self.min(r, s);
                }
                2 => {
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                    r = self.min(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<4>(r.raw));
                    r = self.min(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<2>(r.raw));
                    r = self.min(r, s);
                }
                4 => {
                    if is_type::<T, f32>() {
                        // For f32, use shuffle within 128-bit lane for reduction
                        let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                        r = self.min(r, s);
                        let s = V256::<T>::from_raw(_mm256_srli_si256::<4>(r.raw));
                        r = self.min(r, s);
                    } else {
                        let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                        r = self.min(r, s);
                        let s = V256::<T>::from_raw(_mm256_srli_si256::<4>(r.raw));
                        r = self.min(r, s);
                    }
                }
                8 => {
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                    r = self.min(r, s);
                }
                _ => unreachable!(),
            }
            self.extract_lane(r, 0)
        }
    }

    #[inline(always)]
    unsafe fn max_of_lanes<T: Lane>(self, v: V256<T>) -> T {
        unsafe {
            let swapped = V256::<T>::from_raw(_mm256_permute2x128_si256(v.raw, v.raw, 0x01));
            let mut r = self.max(v, swapped);
            match T::BYTES {
                1 => {
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                    r = self.max(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<4>(r.raw));
                    r = self.max(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<2>(r.raw));
                    r = self.max(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<1>(r.raw));
                    r = self.max(r, s);
                }
                2 => {
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                    r = self.max(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<4>(r.raw));
                    r = self.max(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<2>(r.raw));
                    r = self.max(r, s);
                }
                4 => {
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                    r = self.max(r, s);
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<4>(r.raw));
                    r = self.max(r, s);
                }
                8 => {
                    let s = V256::<T>::from_raw(_mm256_srli_si256::<8>(r.raw));
                    r = self.max(r, s);
                }
                _ => unreachable!(),
            }
            self.extract_lane(r, 0)
        }
    }

    #[inline(always)]
    unsafe fn sums_of_8_abs_diff(
        self,
        a: V256<u8>,
        b: V256<u8>,
    ) -> V256<u64> {
        V256::from_raw(unsafe { _mm256_sad_epu8(a.raw, b.raw) })
    }

    #[inline(always)]
    unsafe fn sums_of_2<T: NarrowLane>(self, v: V256<T>) -> V256<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            let raw = if is_type::<T, i16>() {
                // i16 -> i32: madd with 1s
                _mm256_madd_epi16(v.raw, _mm256_set1_epi16(1))
            } else if is_type::<T, u8>() {
                // u8 -> u16: split even/odd, add as u16
                let mask = _mm256_set1_epi16(0x00FF);
                let even = _mm256_and_si256(v.raw, mask);
                let odd = _mm256_srli_epi16(v.raw, 8);
                _mm256_add_epi16(even, odd)
            } else if is_type::<T, u16>() {
                // u16 -> u32: mask low 16, shift high 16, add as u32
                let mask = _mm256_set1_epi32(0x0000FFFF);
                let even = _mm256_and_si256(v.raw, mask);
                let odd = _mm256_srli_epi32(v.raw, 16);
                _mm256_add_epi32(even, odd)
            } else if is_type::<T, i8>() {
                // i8 -> i16: sign-extend even/odd, add
                let even = _mm256_srai_epi16(_mm256_slli_epi16(v.raw, 8), 8);
                let odd = _mm256_srai_epi16(v.raw, 8);
                _mm256_add_epi16(even, odd)
            } else if is_type::<T, u32>() {
                // u32 -> u64: mask low 32, shift high 32, add as u64
                let mask = _mm256_set1_epi64x(0x00000000FFFFFFFFi64);
                let even = _mm256_and_si256(v.raw, mask);
                let odd = _mm256_srli_epi64(v.raw, 32);
                _mm256_add_epi64(even, odd)
            } else if is_type::<T, i32>() {
                // i32 -> i64: scalar fallback
                let mut arr: Aligned<A32, [i32; 8]> = Aligned::new([0i32; 8]);
                _mm256_store_si256(arr.as_mut_ptr().cast(), v.raw);
                let mut result: Aligned<A32, [i64; 4]> = Aligned::new([0i64; 4]);
                for i in 0..4 {
                    result.value[i] = arr.value[i * 2] as i64 + arr.value[i * 2 + 1] as i64;
                }
                _mm256_load_si256(result.as_ptr().cast())
            } else if is_type::<T, f32>() {
                // f32 -> f64: convert pairs to f64, add
                let ps = _mm256_castsi256_ps(v.raw);
                let lo128 = _mm256_castps256_ps128(ps);
                let hi128 = _mm256_extractf128_ps(ps, 1);
                // Even f32 lanes
                let even = _mm_shuffle_ps(lo128, hi128, 0x88); // [v0,v2,v4,v6]
                let odd = _mm_shuffle_ps(lo128, hi128, 0xDD); // [v1,v3,v5,v7]
                let even_pd = _mm256_cvtps_pd(even);
                let odd_pd = _mm256_cvtps_pd(odd);
                _mm256_castpd_si256(_mm256_add_pd(even_pd, odd_pd))
            } else {
                v.raw
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn sums_of_4<T: NarrowLane>(
        self,
        v: V256<T>,
    ) -> V256<<T::Wide as NarrowLane>::Wide>
    where
        T::Wide: NarrowLane + Lane,
        <T::Wide as NarrowLane>::Wide: Lane,
    {
        unsafe {
            let mid = self.sums_of_2(v);
            self.sums_of_2(mid)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdFloat
// ---------------------------------------------------------------------------

// SAFETY: All intrinsics require AVX2+FMA.
unsafe impl SimdFloat for Avx2 {
    #[inline(always)]
    unsafe fn sqrt<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_sqrt_ps(_mm256_castsi256_ps(v.raw)))
            } else {
                _mm256_castpd_si256(_mm256_sqrt_pd(_mm256_castsi256_pd(v.raw)))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn approx_reciprocal<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            if T::BYTES == 4 {
                V256::from_raw(_mm256_castps_si256(_mm256_rcp_ps(_mm256_castsi256_ps(
                    v.raw,
                ))))
            } else {
                let ones = _mm256_set1_pd(1.0);
                V256::from_raw(_mm256_castpd_si256(_mm256_div_pd(
                    ones,
                    _mm256_castsi256_pd(v.raw),
                )))
            }
        }
    }

    #[inline(always)]
    unsafe fn approx_reciprocal_sqrt<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            if T::BYTES == 4 {
                V256::from_raw(_mm256_castps_si256(_mm256_rsqrt_ps(_mm256_castsi256_ps(
                    v.raw,
                ))))
            } else {
                let ones = _mm256_set1_pd(1.0);
                let sq = _mm256_sqrt_pd(_mm256_castsi256_pd(v.raw));
                V256::from_raw(_mm256_castpd_si256(_mm256_div_pd(ones, sq)))
            }
        }
    }

    #[inline(always)]
    unsafe fn round<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_round_ps(
                    _mm256_castsi256_ps(v.raw),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                ))
            } else {
                _mm256_castpd_si256(_mm256_round_pd(
                    _mm256_castsi256_pd(v.raw),
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn trunc<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_round_ps(
                    _mm256_castsi256_ps(v.raw),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC,
                ))
            } else {
                _mm256_castpd_si256(_mm256_round_pd(
                    _mm256_castsi256_pd(v.raw),
                    _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC,
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn ceil<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_ceil_ps(_mm256_castsi256_ps(v.raw)))
            } else {
                _mm256_castpd_si256(_mm256_ceil_pd(_mm256_castsi256_pd(v.raw)))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn floor<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_floor_ps(_mm256_castsi256_ps(v.raw)))
            } else {
                _mm256_castpd_si256(_mm256_floor_pd(_mm256_castsi256_pd(v.raw)))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn mul_add<T: FloatLane>(self, a: V256<T>, b: V256<T>, c: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_fmadd_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _mm256_castsi256_ps(c.raw),
                ))
            } else {
                _mm256_castpd_si256(_mm256_fmadd_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _mm256_castsi256_pd(c.raw),
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn neg_mul_add<T: FloatLane>(self, a: V256<T>, b: V256<T>, c: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_fnmadd_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _mm256_castsi256_ps(c.raw),
                ))
            } else {
                _mm256_castpd_si256(_mm256_fnmadd_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _mm256_castsi256_pd(c.raw),
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn mul_sub<T: FloatLane>(self, a: V256<T>, b: V256<T>, c: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_fmsub_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _mm256_castsi256_ps(c.raw),
                ))
            } else {
                _mm256_castpd_si256(_mm256_fmsub_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _mm256_castsi256_pd(c.raw),
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn neg_mul_sub<T: FloatLane>(self, a: V256<T>, b: V256<T>, c: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                _mm256_castps_si256(_mm256_fnmsub_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                    _mm256_castsi256_ps(c.raw),
                ))
            } else {
                _mm256_castpd_si256(_mm256_fnmsub_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                    _mm256_castsi256_pd(c.raw),
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn copy_sign<T: FloatLane>(self, mag: V256<T>, sign: V256<T>) -> V256<T> {
        unsafe {
            if T::BYTES == 4 {
                let sign_mask = _mm256_set1_epi32(0x8000_0000u32 as i32);
                let abs_mag = _mm256_andnot_si256(sign_mask, mag.raw);
                let sign_bit = _mm256_and_si256(sign_mask, sign.raw);
                V256::from_raw(_mm256_or_si256(abs_mag, sign_bit))
            } else {
                let sign_mask = _mm256_set1_epi64x(0x8000_0000_0000_0000u64 as i64);
                let abs_mag = _mm256_andnot_si256(sign_mask, mag.raw);
                let sign_bit = _mm256_and_si256(sign_mask, sign.raw);
                V256::from_raw(_mm256_or_si256(abs_mag, sign_bit))
            }
        }
    }

    #[inline(always)]
    unsafe fn is_nan<T: FloatLane>(self, v: V256<T>) -> M256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                let ps = _mm256_castsi256_ps(v.raw);
                _mm256_castps_si256(_mm256_cmp_ps(ps, ps, _CMP_UNORD_Q))
            } else {
                let pd = _mm256_castsi256_pd(v.raw);
                _mm256_castpd_si256(_mm256_cmp_pd(pd, pd, _CMP_UNORD_Q))
            };
            M256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn is_inf<T: FloatLane>(self, v: V256<T>) -> M256<T> {
        unsafe {
            if T::BYTES == 4 {
                let abs_mask = _mm256_set1_epi32(0x7FFF_FFFFu32 as i32);
                let inf_bits = _mm256_set1_epi32(0x7F80_0000u32 as i32);
                let abs_v = _mm256_and_si256(v.raw, abs_mask);
                M256::from_raw(_mm256_cmpeq_epi32(abs_v, inf_bits))
            } else {
                let abs_mask = _mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64);
                let inf_bits = _mm256_set1_epi64x(0x7FF0_0000_0000_0000u64 as i64);
                let abs_v = _mm256_and_si256(v.raw, abs_mask);
                M256::from_raw(_mm256_cmpeq_epi64(abs_v, inf_bits))
            }
        }
    }

    #[inline(always)]
    unsafe fn zero_if_negative<T: FloatLane>(self, v: V256<T>) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                let sign = _mm256_srai_epi32(v.raw, 31);
                _mm256_andnot_si256(sign, v.raw)
            } else {
                // f64: broadcast sign bit from high dword
                let sign32 = _mm256_srai_epi32(v.raw, 31);
                let sign64 = _mm256_shuffle_epi32(sign32, 0xF5);
                _mm256_andnot_si256(sign64, v.raw)
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn is_finite<T: FloatLane>(self, v: V256<T>) -> M256<T> {
        unsafe {
            if T::BYTES == 4 {
                let shifted = _mm256_srli_epi32(_mm256_slli_epi32(v.raw, 1), 24);
                let max_exp = _mm256_set1_epi32(0xFF);
                // AVX2 has no unsigned cmplt; both values are small positive, so signed works.
                M256::from_raw(_mm256_cmpgt_epi32(max_exp, shifted))
            } else {
                // f64: check high dword exponent field < 0x7FF00000
                let abs_mask = _mm256_set1_epi64x(0x7FFF_FFFF_FFFF_FFFFu64 as i64);
                let abs_v = _mm256_and_si256(v.raw, abs_mask);
                let inf = _mm256_set1_epi64x(0x7FF0_0000_0000_0000u64 as i64);
                // finite iff high32(abs_v) < high32(inf). Use scalar-like approach.
                let hi32_abs = _mm256_srli_epi64(abs_v, 32);
                let hi32_inf = _mm256_srli_epi64(inf, 32);
                let lt = _mm256_cmpgt_epi32(hi32_inf, hi32_abs);
                M256::from_raw(_mm256_shuffle_epi32(lt, 0xF5))
            }
        }
    }

    #[inline(always)]
    unsafe fn add_sub<T: FloatLane>(
        self,
        a: V256<T>,
        b: V256<T>,
    ) -> V256<T> {
        unsafe {
            let raw = if T::BYTES == 4 {
                // AVX has _mm256_addsub_ps: [a0-b0, a1+b1, a2-b2, a3+b3, ...]
                _mm256_castps_si256(_mm256_addsub_ps(
                    _mm256_castsi256_ps(a.raw),
                    _mm256_castsi256_ps(b.raw),
                ))
            } else {
                // AVX has _mm256_addsub_pd
                _mm256_castpd_si256(_mm256_addsub_pd(
                    _mm256_castsi256_pd(a.raw),
                    _mm256_castsi256_pd(b.raw),
                ))
            };
            V256::from_raw(raw)
        }
    }

    #[inline(always)]
    unsafe fn min_number<T: FloatLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let nan_a = self.is_nan(a);
            let min_ab = self.min(a, b);
            self.if_then_else(nan_a, b, min_ab)
        }
    }

    #[inline(always)]
    unsafe fn max_number<T: FloatLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
            let nan_a = self.is_nan(a);
            let max_ab = self.max(a, b);
            self.if_then_else(nan_a, b, max_ab)
        }
    }

    #[inline(always)]
    unsafe fn min_magnitude<T: FloatLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
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
    unsafe fn max_magnitude<T: FloatLane>(self, a: V256<T>, b: V256<T>) -> V256<T> {
        unsafe {
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
    unsafe fn is_either_nan<T: FloatLane>(self, a: V256<T>, b: V256<T>) -> M256<T> {
        unsafe { self.or_mask(self.is_nan(a), self.is_nan(b)) }
    }
}

// ---------------------------------------------------------------------------
// SimdCrypto
// ---------------------------------------------------------------------------

// SAFETY: AES/CLMul intrinsics are guarded by runtime feature detection.
// Uses VAES/VPCLMULQDQ when available, otherwise splits to 2*128-bit operations.
unsafe impl crate::ops::SimdCrypto for Avx2 {
    #[inline(always)]
    unsafe fn aes_round(self, state: V256<u8>, round_key: V256<u8>) -> V256<u8> {
        unsafe {
            // Split into two 128-bit halves, apply AES-NI, recombine
            let lo_s = _mm256_castsi256_si128(state.raw);
            let hi_s = _mm256_extracti128_si256(state.raw, 1);
            let lo_k = _mm256_castsi256_si128(round_key.raw);
            let hi_k = _mm256_extracti128_si256(round_key.raw, 1);
            if is_x86_feature_detected!("aes") {
                let lo_r = _mm_aesenc_si128(lo_s, lo_k);
                let hi_r = _mm_aesenc_si128(hi_s, hi_k);
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            } else {
                let mut block0 = [0u8; 16];
                let mut block1 = [0u8; 16];
                let mut key0 = [0u8; 16];
                let mut key1 = [0u8; 16];
                _mm_storeu_si128(block0.as_mut_ptr().cast(), lo_s);
                _mm_storeu_si128(block1.as_mut_ptr().cast(), hi_s);
                _mm_storeu_si128(key0.as_mut_ptr().cast(), lo_k);
                _mm_storeu_si128(key1.as_mut_ptr().cast(), hi_k);
                super::crypto_soft::aes_round(&mut block0, &key0);
                super::crypto_soft::aes_round(&mut block1, &key1);
                let lo_r = _mm_loadu_si128(block0.as_ptr().cast());
                let hi_r = _mm_loadu_si128(block1.as_ptr().cast());
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            }
        }
    }

    #[inline(always)]
    unsafe fn aes_last_round(self, state: V256<u8>, round_key: V256<u8>) -> V256<u8> {
        unsafe {
            let lo_s = _mm256_castsi256_si128(state.raw);
            let hi_s = _mm256_extracti128_si256(state.raw, 1);
            let lo_k = _mm256_castsi256_si128(round_key.raw);
            let hi_k = _mm256_extracti128_si256(round_key.raw, 1);
            if is_x86_feature_detected!("aes") {
                let lo_r = _mm_aesenclast_si128(lo_s, lo_k);
                let hi_r = _mm_aesenclast_si128(hi_s, hi_k);
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            } else {
                let mut block0 = [0u8; 16];
                let mut block1 = [0u8; 16];
                let mut key0 = [0u8; 16];
                let mut key1 = [0u8; 16];
                _mm_storeu_si128(block0.as_mut_ptr().cast(), lo_s);
                _mm_storeu_si128(block1.as_mut_ptr().cast(), hi_s);
                _mm_storeu_si128(key0.as_mut_ptr().cast(), lo_k);
                _mm_storeu_si128(key1.as_mut_ptr().cast(), hi_k);
                super::crypto_soft::aes_last_round(&mut block0, &key0);
                super::crypto_soft::aes_last_round(&mut block1, &key1);
                let lo_r = _mm_loadu_si128(block0.as_ptr().cast());
                let hi_r = _mm_loadu_si128(block1.as_ptr().cast());
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            }
        }
    }

    #[inline(always)]
    unsafe fn aes_round_inv(self, state: V256<u8>, round_key: V256<u8>) -> V256<u8> {
        unsafe {
            let lo_s = _mm256_castsi256_si128(state.raw);
            let hi_s = _mm256_extracti128_si256(state.raw, 1);
            let lo_k = _mm256_castsi256_si128(round_key.raw);
            let hi_k = _mm256_extracti128_si256(round_key.raw, 1);
            if is_x86_feature_detected!("aes") {
                let lo_r = _mm_aesdec_si128(lo_s, lo_k);
                let hi_r = _mm_aesdec_si128(hi_s, hi_k);
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            } else {
                let mut block0 = [0u8; 16];
                let mut block1 = [0u8; 16];
                let mut key0 = [0u8; 16];
                let mut key1 = [0u8; 16];
                _mm_storeu_si128(block0.as_mut_ptr().cast(), lo_s);
                _mm_storeu_si128(block1.as_mut_ptr().cast(), hi_s);
                _mm_storeu_si128(key0.as_mut_ptr().cast(), lo_k);
                _mm_storeu_si128(key1.as_mut_ptr().cast(), hi_k);
                super::crypto_soft::aes_round_inv(&mut block0, &key0);
                super::crypto_soft::aes_round_inv(&mut block1, &key1);
                let lo_r = _mm_loadu_si128(block0.as_ptr().cast());
                let hi_r = _mm_loadu_si128(block1.as_ptr().cast());
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            }
        }
    }

    #[inline(always)]
    unsafe fn aes_last_round_inv(self, state: V256<u8>, round_key: V256<u8>) -> V256<u8> {
        unsafe {
            let lo_s = _mm256_castsi256_si128(state.raw);
            let hi_s = _mm256_extracti128_si256(state.raw, 1);
            let lo_k = _mm256_castsi256_si128(round_key.raw);
            let hi_k = _mm256_extracti128_si256(round_key.raw, 1);
            if is_x86_feature_detected!("aes") {
                let lo_r = _mm_aesdeclast_si128(lo_s, lo_k);
                let hi_r = _mm_aesdeclast_si128(hi_s, hi_k);
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            } else {
                let mut block0 = [0u8; 16];
                let mut block1 = [0u8; 16];
                let mut key0 = [0u8; 16];
                let mut key1 = [0u8; 16];
                _mm_storeu_si128(block0.as_mut_ptr().cast(), lo_s);
                _mm_storeu_si128(block1.as_mut_ptr().cast(), hi_s);
                _mm_storeu_si128(key0.as_mut_ptr().cast(), lo_k);
                _mm_storeu_si128(key1.as_mut_ptr().cast(), hi_k);
                super::crypto_soft::aes_last_round_inv(&mut block0, &key0);
                super::crypto_soft::aes_last_round_inv(&mut block1, &key1);
                let lo_r = _mm_loadu_si128(block0.as_ptr().cast());
                let hi_r = _mm_loadu_si128(block1.as_ptr().cast());
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            }
        }
    }

    #[inline(always)]
    unsafe fn cl_mul_lower(self, a: V256<u64>, b: V256<u64>) -> V256<u64> {
        unsafe {
            let lo_a = _mm256_castsi256_si128(a.raw);
            let hi_a = _mm256_extracti128_si256(a.raw, 1);
            let lo_b = _mm256_castsi256_si128(b.raw);
            let hi_b = _mm256_extracti128_si256(b.raw, 1);
            if is_x86_feature_detected!("pclmulqdq") {
                let lo_r = _mm_clmulepi64_si128(lo_a, lo_b, 0x00);
                let hi_r = _mm_clmulepi64_si128(hi_a, hi_b, 0x00);
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            } else {
                let mut arr_a = [0u64; 4];
                let mut arr_b = [0u64; 4];
                _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
                _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);
                // Block 0: lower u64 of first 128-bit block
                let (lo0, hi0) = super::crypto_soft::clmul_64(arr_a[0], arr_b[0]);
                // Block 1: lower u64 of second 128-bit block
                let (lo1, hi1) = super::crypto_soft::clmul_64(arr_a[2], arr_b[2]);
                let result = [lo0, hi0, lo1, hi1];
                V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn cl_mul_upper(self, a: V256<u64>, b: V256<u64>) -> V256<u64> {
        unsafe {
            let lo_a = _mm256_castsi256_si128(a.raw);
            let hi_a = _mm256_extracti128_si256(a.raw, 1);
            let lo_b = _mm256_castsi256_si128(b.raw);
            let hi_b = _mm256_extracti128_si256(b.raw, 1);
            if is_x86_feature_detected!("pclmulqdq") {
                let lo_r = _mm_clmulepi64_si128(lo_a, lo_b, 0x11);
                let hi_r = _mm_clmulepi64_si128(hi_a, hi_b, 0x11);
                V256::from_raw(_mm256_inserti128_si256(
                    _mm256_castsi128_si256(lo_r),
                    hi_r,
                    1,
                ))
            } else {
                let mut arr_a = [0u64; 4];
                let mut arr_b = [0u64; 4];
                _mm256_storeu_si256(arr_a.as_mut_ptr().cast(), a.raw);
                _mm256_storeu_si256(arr_b.as_mut_ptr().cast(), b.raw);
                // Block 0: upper u64 of first 128-bit block
                let (lo0, hi0) = super::crypto_soft::clmul_64(arr_a[1], arr_b[1]);
                // Block 1: upper u64 of second 128-bit block
                let (lo1, hi1) = super::crypto_soft::clmul_64(arr_a[3], arr_b[3]);
                let result = [lo0, hi0, lo1, hi1];
                V256::from_raw(_mm256_loadu_si256(result.as_ptr().cast()))
            }
        }
    }

    #[inline(always)]
    unsafe fn aes_key_gen_assist<const RCON: i32>(self, v: V256<u8>) -> V256<u8> {
        unsafe {
            let lo = _mm256_castsi256_si128(v.raw);
            let hi = _mm256_extracti128_si256(v.raw, 1);
            if is_x86_feature_detected!("aes") {
                let r_lo = _mm_aeskeygenassist_si128(lo, RCON);
                let r_hi = _mm_aeskeygenassist_si128(hi, RCON);
                V256::from_raw(_mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1))
            } else {
                let mut b_lo = [0u8; 16];
                let mut b_hi = [0u8; 16];
                _mm_storeu_si128(b_lo.as_mut_ptr().cast(), lo);
                _mm_storeu_si128(b_hi.as_mut_ptr().cast(), hi);
                let r_lo = super::crypto_soft::aes_key_gen_assist(&b_lo, RCON as u8);
                let r_hi = super::crypto_soft::aes_key_gen_assist(&b_hi, RCON as u8);
                let v_lo = _mm_loadu_si128(r_lo.as_ptr().cast());
                let v_hi = _mm_loadu_si128(r_hi.as_ptr().cast());
                V256::from_raw(_mm256_inserti128_si256(_mm256_castsi128_si256(v_lo), v_hi, 1))
            }
        }
    }

    #[inline(always)]
    unsafe fn aes_inv_mix_columns(self, v: V256<u8>) -> V256<u8> {
        unsafe {
            let lo = _mm256_castsi256_si128(v.raw);
            let hi = _mm256_extracti128_si256(v.raw, 1);
            if is_x86_feature_detected!("aes") {
                let r_lo = _mm_aesimc_si128(lo);
                let r_hi = _mm_aesimc_si128(hi);
                V256::from_raw(_mm256_inserti128_si256(_mm256_castsi128_si256(r_lo), r_hi, 1))
            } else {
                let mut b_lo = [0u8; 16];
                let mut b_hi = [0u8; 16];
                _mm_storeu_si128(b_lo.as_mut_ptr().cast(), lo);
                _mm_storeu_si128(b_hi.as_mut_ptr().cast(), hi);
                super::crypto_soft::aes_inv_mix_columns_block(&mut b_lo);
                super::crypto_soft::aes_inv_mix_columns_block(&mut b_hi);
                let v_lo = _mm_loadu_si128(b_lo.as_ptr().cast());
                let v_hi = _mm_loadu_si128(b_hi.as_ptr().cast());
                V256::from_raw(_mm256_inserti128_si256(_mm256_castsi128_si256(v_lo), v_hi, 1))
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
    use super::*;

    fn has_avx2() -> bool {
        is_x86_feature_detected!("avx2")
    }

    #[test]
    fn test_avx2_add_i32() {
        if !has_avx2() {
            return;
        }
        let s = Avx2;
        unsafe {
            let a = s.splat::<i32>(10);
            let b = s.splat::<i32>(32);
            let c: V256<i32> = s.add(a, b);
            for i in 0..8 {
                assert_eq!(s.extract_lane(c, i), 42);
            }
        }
    }

    #[test]
    fn test_avx2_load_store() {
        if !has_avx2() {
            return;
        }
        let s = Avx2;
        unsafe {
            let data: [i32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
            let v: V256<i32> = s.load_u(data.as_ptr());
            let mut out = [0i32; 8];
            s.store_u(v, out.as_mut_ptr());
            assert_eq!(out, data);
        }
    }

    #[test]
    fn test_avx2_float_add() {
        if !has_avx2() {
            return;
        }
        let s = Avx2;
        unsafe {
            let a = s.splat::<f32>(1.5);
            let b = s.splat::<f32>(2.5);
            let c: V256<f32> = s.add(a, b);
            let r: f32 = s.extract_lane(c, 0);
            assert!((r - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_avx2_fma() {
        if !has_avx2() || !is_x86_feature_detected!("fma") {
            return;
        }
        let s = Avx2;
        unsafe {
            let a = s.splat::<f32>(2.0);
            let b = s.splat::<f32>(3.0);
            let c = s.splat::<f32>(4.0);
            let r: V256<f32> = s.mul_add(a, b, c);
            let val: f32 = s.extract_lane(r, 0);
            assert!((val - 10.0).abs() < 1e-6);
        }
    }
}
