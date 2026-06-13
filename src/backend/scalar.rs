// All unsafe blocks in this module wrap transmute_copy for type-punning
// between generic Lane types and concrete primitives, with type identity
// verified via is_type() or size checks. Individual safety comments are on
// the outer unsafe impl blocks.
#![allow(clippy::undocumented_unsafe_blocks)]

/// Scalar (non-SIMD) backend.
///
/// Operates on single elements at a time. This serves as the reference
/// implementation and fallback when no SIMD features are available.
/// VECTOR_BYTES is set to 8 to allow operating on up to 8 bytes at once
/// conceptually, though each operation processes exactly one lane.
use crate::lane::{FloatLane, IntegerLane, Lane, NarrowLane, UnsignedLane, WideLane};
use crate::ops::{
    SimdArith, SimdBitwise, SimdCompare, SimdConvert, SimdCore, SimdFloat, SimdMask, SimdMemory,
    SimdReduce, SimdShuffle,
};
use crate::simd::Simd;

// ---------------------------------------------------------------------------
// Target type
// ---------------------------------------------------------------------------

/// The scalar (single-element) SIMD target.
#[derive(Clone, Copy, Debug)]
pub struct Scalar;

// SAFETY: Scalar vectors hold exactly one lane.
unsafe impl Simd for Scalar {
    type Vec<T: Lane> = Vec1<T>;
    type Mask<T: Lane> = Mask1<T>;
    // Half-width is identity for scalar (can't go below single element).
    type VecHalf<T: Lane> = Vec1<T>;
    type MaskHalf<T: Lane> = Mask1<T>;
    // Set to 1 so that lanes::<T>() returns 1 for u8 (the smallest type).
    const VECTOR_BYTES: usize = 1;
}

// ---------------------------------------------------------------------------
// Vector and Mask types
// ---------------------------------------------------------------------------

/// A single-element vector.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Vec1<T: Lane>(pub T);

/// A single-element mask. Stored as the unsigned lane type:
/// `0` = false, `!0` (all bits set) = true.
#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(transparent)]
pub struct Mask1<T: Lane>(pub T::Unsigned);

impl<T: Lane> Mask1<T> {
    /// Create a mask from a boolean.
    #[inline(always)]
    pub fn from_bool(b: bool) -> Self {
        if b {
            Self(all_ones::<T::Unsigned>())
        } else {
            Self(T::Unsigned::default())
        }
    }

    /// Convert to bool.
    #[inline(always)]
    pub fn to_bool(self) -> bool {
        self.0 != T::Unsigned::default()
    }
}

/// Return the all-ones bit pattern for an unsigned lane type.
#[inline(always)]
fn all_ones<U: UnsignedLane>() -> U {
    // SAFETY: All-ones is a valid bit pattern for any unsigned integer.
    unsafe {
        let mut v = core::mem::MaybeUninit::<U>::uninit();
        core::ptr::write_bytes(v.as_mut_ptr(), 0xFF, 1);
        v.assume_init()
    }
}

// ---------------------------------------------------------------------------
// Helper: type identity check
// ---------------------------------------------------------------------------

use crate::lane::is_type;

// ---------------------------------------------------------------------------
// SimdCore implementation
// ---------------------------------------------------------------------------

// SAFETY: All operations are plain Rust scalar ops. No CPU feature requirements.
unsafe impl SimdCore for Scalar {
    #[inline(always)]
    unsafe fn zero<T: Lane>(self) -> Vec1<T> {
        Vec1(T::default())
    }

    #[inline(always)]
    unsafe fn splat<T: Lane>(self, value: T) -> Vec1<T> {
        Vec1(value)
    }

    #[inline(always)]
    unsafe fn undefined<T: Lane>(self) -> Vec1<T> {
        Vec1(T::default())
    }

    #[inline(always)]
    unsafe fn bitcast<T: Lane, U: Lane>(self, v: Vec1<T>) -> Vec1<U> {
        debug_assert!(
            core::mem::size_of::<T>() == core::mem::size_of::<U>(),
            "bitcast: source and destination must have the same size"
        );
        let mut out = U::default();
        let copy_bytes = core::mem::size_of::<U>().min(core::mem::size_of::<T>());
        // SAFETY: Both T and U are Lane (Copy + 'static POD). We copy at most
        // min(size_of::<T>(), size_of::<U>()) bytes.
        // SAFETY: Both types have the same size (Lane invariant); copy is within bounds.
        unsafe {
            core::ptr::copy_nonoverlapping(
                core::ptr::from_ref(&v.0).cast::<u8>(),
                core::ptr::from_mut(&mut out).cast::<u8>(),
                copy_bytes,
            );
        }
        Vec1(out)
    }

    #[inline(always)]
    unsafe fn extract_lane<T: Lane>(self, v: Vec1<T>, index: usize) -> T {
        debug_assert!(index == 0, "Scalar: extract_lane index must be 0");
        let _ = index;
        v.0
    }

    #[inline(always)]
    unsafe fn insert_lane<T: Lane>(
        self,
        _v: Vec1<T>,
        index: usize,
        value: T,
    ) -> Vec1<T> {
        debug_assert!(index == 0, "Scalar: insert_lane index must be 0");
        let _ = index;
        Vec1(value)
    }

    #[inline(always)]
    unsafe fn iota<T: Lane>(self, base: T) -> Vec1<T> {
        // For scalar, there's only lane 0, which is base + 0 = base.
        Vec1(base)
    }
}

// ---------------------------------------------------------------------------
// SimdMemory implementation
// ---------------------------------------------------------------------------

// SAFETY: All operations are plain pointer reads/writes. No alignment requirement
// beyond what T itself requires for scalar operations.
unsafe impl SimdMemory for Scalar {
    #[inline(always)]
    unsafe fn load<T: Lane>(self, ptr: *const T) -> Vec1<T> {
        // SAFETY: Caller guarantees ptr is valid and aligned.
        Vec1(unsafe { ptr.read() })
    }

    #[inline(always)]
    unsafe fn load_u<T: Lane>(self, ptr: *const T) -> Vec1<T> {
        // SAFETY: Caller guarantees ptr is valid (unaligned read).
        Vec1(unsafe { ptr.read_unaligned() })
    }

    #[inline(always)]
    unsafe fn store<T: Lane>(self, v: Vec1<T>, ptr: *mut T) {
        // SAFETY: Caller guarantees ptr is valid and aligned.
        unsafe { ptr.write(v.0) }
    }

    #[inline(always)]
    unsafe fn store_u<T: Lane>(self, v: Vec1<T>, ptr: *mut T) {
        // SAFETY: Caller guarantees ptr is valid (unaligned write).
        unsafe { ptr.write_unaligned(v.0) }
    }

    #[inline(always)]
    unsafe fn stream<T: Lane>(self, v: Vec1<T>, ptr: *mut T) {
        // Scalar has no non-temporal stores; fall back to aligned store.
        // SAFETY: Caller guarantees ptr is valid and aligned.
        unsafe { ptr.write(v.0) }
    }

    #[inline(always)]
    unsafe fn load_dup128<T: Lane>(self, ptr: *const T) -> Vec1<T> {
        // Scalar: same as load (only 1 lane).
        Vec1(unsafe { ptr.read() })
    }

    #[inline(always)]
    unsafe fn masked_load<T: Lane>(self, mask: Mask1<T>, ptr: *const T) -> Vec1<T> {
        if mask.to_bool() {
            Vec1(unsafe { ptr.read() })
        } else {
            Vec1(T::default())
        }
    }

    #[inline(always)]
    unsafe fn blended_store<T: Lane>(self, v: Vec1<T>, mask: Mask1<T>, ptr: *mut T) {
        if mask.to_bool() {
            unsafe { ptr.write(v.0) }
        }
    }

    #[inline(always)]
    unsafe fn gather_index<T: Lane>(
        self,
        base: *const T,
        idx: Vec1<i32>,
    ) -> Vec1<T> {
        Vec1(unsafe { *base.offset(idx.0 as isize) })
    }

    #[inline(always)]
    unsafe fn scatter_index<T: Lane>(
        self,
        v: Vec1<T>,
        base: *mut T,
        idx: Vec1<i32>,
    ) {
        unsafe { *base.offset(idx.0 as isize) = v.0; }
    }

    #[inline(always)]
    unsafe fn load_interleaved_2<T: Lane>(
        self,
        ptr: *const T,
    ) -> (Vec1<T>, Vec1<T>) {
        (Vec1(unsafe { ptr.read() }), Vec1(unsafe { ptr.add(1).read() }))
    }

    #[inline(always)]
    unsafe fn load_interleaved_3<T: Lane>(
        self,
        ptr: *const T,
    ) -> (Vec1<T>, Vec1<T>, Vec1<T>) {
        (
            Vec1(unsafe { ptr.read() }),
            Vec1(unsafe { ptr.add(1).read() }),
            Vec1(unsafe { ptr.add(2).read() }),
        )
    }

    #[inline(always)]
    unsafe fn load_interleaved_4<T: Lane>(
        self,
        ptr: *const T,
    ) -> (Vec1<T>, Vec1<T>, Vec1<T>, Vec1<T>) {
        (
            Vec1(unsafe { ptr.read() }),
            Vec1(unsafe { ptr.add(1).read() }),
            Vec1(unsafe { ptr.add(2).read() }),
            Vec1(unsafe { ptr.add(3).read() }),
        )
    }

    #[inline(always)]
    unsafe fn store_interleaved_2<T: Lane>(
        self,
        v0: Vec1<T>,
        v1: Vec1<T>,
        ptr: *mut T,
    ) {
        unsafe {
            ptr.write(v0.0);
            ptr.add(1).write(v1.0);
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_3<T: Lane>(
        self,
        v0: Vec1<T>,
        v1: Vec1<T>,
        v2: Vec1<T>,
        ptr: *mut T,
    ) {
        unsafe {
            ptr.write(v0.0);
            ptr.add(1).write(v1.0);
            ptr.add(2).write(v2.0);
        }
    }

    #[inline(always)]
    unsafe fn store_interleaved_4<T: Lane>(
        self,
        v0: Vec1<T>,
        v1: Vec1<T>,
        v2: Vec1<T>,
        v3: Vec1<T>,
        ptr: *mut T,
    ) {
        unsafe {
            ptr.write(v0.0);
            ptr.add(1).write(v1.0);
            ptr.add(2).write(v2.0);
            ptr.add(3).write(v3.0);
        }
    }

    #[inline(always)]
    unsafe fn load_expand<T: Lane>(self, mask: Mask1<T>, ptr: *const T) -> Vec1<T> {
        // Single lane: if mask is true, load from ptr; else zero
        if mask.to_bool() {
            Vec1(unsafe { ptr.read() })
        } else {
            Vec1(T::default())
        }
    }
}

// ---------------------------------------------------------------------------
// SimdArith implementation
// ---------------------------------------------------------------------------

// SAFETY: All operations are plain Rust scalar arithmetic.
unsafe impl SimdArith for Scalar {
    #[inline(always)]
    unsafe fn add<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let va: f32 = core::mem::transmute_copy(&a.0);
                    let vb: f32 = core::mem::transmute_copy(&b.0);
                    let r = va + vb;
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else {
                // 8 bytes
                if is_type::<T, f64>() {
                    let va: f64 = core::mem::transmute_copy(&a.0);
                    let vb: f64 = core::mem::transmute_copy(&b.0);
                    let r = va + vb;
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u64>() {
                    let va: u64 = core::mem::transmute_copy(&a.0);
                    let vb: u64 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i64 = core::mem::transmute_copy(&a.0);
                    let vb: i64 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            }
        }
    }

    #[inline(always)]
    unsafe fn sub<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let va: f32 = core::mem::transmute_copy(&a.0);
                    let vb: f32 = core::mem::transmute_copy(&b.0);
                    let r = va - vb;
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, f64>() {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = va - vb;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = va.wrapping_sub(vb);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let r = va.wrapping_sub(vb);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn mul<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_mul(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_mul(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_mul(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_mul(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let va: f32 = core::mem::transmute_copy(&a.0);
                    let vb: f32 = core::mem::transmute_copy(&b.0);
                    let r = va * vb;
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_mul(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = va.wrapping_mul(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, f64>() {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = va * vb;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = va.wrapping_mul(vb);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let r = va.wrapping_mul(vb);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn div<T: FloatLane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let r = va / vb;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = va / vb;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn saturated_add<T: IntegerLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_add(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = va.saturating_add(vb);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let r = va.saturating_add(vb);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn saturated_sub<T: IntegerLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = va.saturating_sub(vb);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = va.saturating_sub(vb);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let r = va.saturating_sub(vb);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn abs<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    v // abs of unsigned is identity
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_abs();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    v
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_abs();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let val: f32 = core::mem::transmute_copy(&v.0);
                    let r = val.abs();
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    v
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_abs();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, f64>() {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val.abs();
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u64>() {
                v
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_abs();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn neg<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_neg();
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_neg();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_neg();
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_neg();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let val: f32 = core::mem::transmute_copy(&v.0);
                    let r = -val;
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_neg();
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_neg();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, f64>() {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = -val;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_neg();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_neg();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn min<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = if va < vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = if va < vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = if va < vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = if va < vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let va: f32 = core::mem::transmute_copy(&a.0);
                    let vb: f32 = core::mem::transmute_copy(&b.0);
                    let r = if va < vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = if va < vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = if va < vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, f64>() {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = if va < vb { va } else { vb };
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = if va < vb { va } else { vb };
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let r = if va < vb { va } else { vb };
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn max<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = if va > vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = if va > vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = if va > vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = if va > vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let va: f32 = core::mem::transmute_copy(&a.0);
                    let vb: f32 = core::mem::transmute_copy(&b.0);
                    let r = if va > vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = if va > vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = if va > vb { va } else { vb };
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, f64>() {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = if va > vb { va } else { vb };
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = if va > vb { va } else { vb };
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let r = if va > vb { va } else { vb };
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn mul_high<T: IntegerLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let wide = (va as u16) * (vb as u16);
                    let r = (wide >> 8) as u8;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let wide = (va as i16) * (vb as i16);
                    let r = (wide >> 8) as i8;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let wide = (va as u32) * (vb as u32);
                    let r = (wide >> 16) as u16;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let wide = (va as i32) * (vb as i32);
                    let r = (wide >> 16) as i16;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let wide = (va as u64) * (vb as u64);
                    let r = (wide >> 32) as u32;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let wide = (va as i64) * (vb as i64);
                    let r = (wide >> 32) as i32;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let wide = (va as u128) * (vb as u128);
                let r = (wide >> 64) as u64;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let wide = (va as i128) * (vb as i128);
                let r = (wide >> 64) as i64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn average_round<T: UnsignedLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                let va: u8 = core::mem::transmute_copy(&a.0);
                let vb: u8 = core::mem::transmute_copy(&b.0);
                let r = (((va as u16) + (vb as u16) + 1) >> 1) as u8;
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 2 {
                let va: u16 = core::mem::transmute_copy(&a.0);
                let vb: u16 = core::mem::transmute_copy(&b.0);
                let r = (((va as u32) + (vb as u32) + 1) >> 1) as u16;
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 4 {
                let va: u32 = core::mem::transmute_copy(&a.0);
                let vb: u32 = core::mem::transmute_copy(&b.0);
                let r = (((va as u64) + (vb as u64) + 1) >> 1) as u32;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = (((va as u128) + (vb as u128) + 1) >> 1) as u64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn abs_diff<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.abs_diff(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i8 = core::mem::transmute_copy(&a.0);
                    let vb: i8 = core::mem::transmute_copy(&b.0);
                    let r = (va.wrapping_sub(vb)).wrapping_abs();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let va: u16 = core::mem::transmute_copy(&a.0);
                    let vb: u16 = core::mem::transmute_copy(&b.0);
                    let r = va.abs_diff(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i16 = core::mem::transmute_copy(&a.0);
                    let vb: i16 = core::mem::transmute_copy(&b.0);
                    let r = (va.wrapping_sub(vb)).wrapping_abs();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, f32>() {
                    let va: f32 = core::mem::transmute_copy(&a.0);
                    let vb: f32 = core::mem::transmute_copy(&b.0);
                    let r = (va - vb).abs();
                    Vec1(core::mem::transmute_copy(&r))
                } else if is_type::<T, u32>() {
                    let va: u32 = core::mem::transmute_copy(&a.0);
                    let vb: u32 = core::mem::transmute_copy(&b.0);
                    let r = va.abs_diff(vb);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let va: i32 = core::mem::transmute_copy(&a.0);
                    let vb: i32 = core::mem::transmute_copy(&b.0);
                    let r = (va.wrapping_sub(vb)).wrapping_abs();
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, f64>() {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = (va - vb).abs();
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u64>() {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = va.abs_diff(vb);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: i64 = core::mem::transmute_copy(&a.0);
                let vb: i64 = core::mem::transmute_copy(&b.0);
                let r = (va.wrapping_sub(vb)).wrapping_abs();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn clamp<T: Lane>(
        self,
        v: Vec1<T>,
        lo: Vec1<T>,
        hi: Vec1<T>,
    ) -> Vec1<T> {
        unsafe { self.min(self.max(v, lo), hi) }
    }

    #[inline(always)]
    unsafe fn mul_even<T: NarrowLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T::Wide>
    where
        T::Wide: Lane,
    {
        unsafe {
            if is_type::<T, u8>() {
                let va: u8 = core::mem::transmute_copy(&a.0);
                let vb: u8 = core::mem::transmute_copy(&b.0);
                let r = (va as u16) * (vb as u16);
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, i8>() {
                let va: i8 = core::mem::transmute_copy(&a.0);
                let vb: i8 = core::mem::transmute_copy(&b.0);
                let r = (va as i16) * (vb as i16);
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u16>() {
                let va: u16 = core::mem::transmute_copy(&a.0);
                let vb: u16 = core::mem::transmute_copy(&b.0);
                let r = (va as u32) * (vb as u32);
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, i16>() {
                let va: i16 = core::mem::transmute_copy(&a.0);
                let vb: i16 = core::mem::transmute_copy(&b.0);
                let r = (va as i32) * (vb as i32);
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u32>() {
                let va: u32 = core::mem::transmute_copy(&a.0);
                let vb: u32 = core::mem::transmute_copy(&b.0);
                let r = (va as u64) * (vb as u64);
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, i32>() {
                let va: i32 = core::mem::transmute_copy(&a.0);
                let vb: i32 = core::mem::transmute_copy(&b.0);
                let r = (va as i64) * (vb as i64);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                // f32 -> f64
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let r = (va as f64) * (vb as f64);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn mul_odd<T: NarrowLane>(
        self,
        _a: Vec1<T>,
        _b: Vec1<T>,
    ) -> Vec1<T::Wide>
    where
        T::Wide: Lane,
    {
        // Scalar has only lane 0 (even), so mul_odd returns 0.
        Vec1(T::Wide::default())
    }

    #[inline(always)]
    unsafe fn widen_mul_pairwise_add_i16(
        self,
        a: Vec1<i16>,
        b: Vec1<i16>,
    ) -> Vec1<i32> {
        // Scalar has 1 i16 lane; the pairwise add reduces to a single multiply.
        Vec1((a.0 as i32) * (b.0 as i32))
    }

    #[inline(always)]
    unsafe fn sat_widen_mul_pairwise_add(
        self,
        a: Vec1<u8>,
        b: Vec1<i8>,
    ) -> Vec1<i16> {
        // u8 * i8 fits in i16 without saturation.
        Vec1((a.0 as i16) * (b.0 as i16))
    }

    #[inline(always)]
    unsafe fn mul_fixed_point_15(
        self,
        a: Vec1<i16>,
        b: Vec1<i16>,
    ) -> Vec1<i16> {
        // Fixed-point: ((a * b) + (1 << 14)) >> 15
        Vec1(((((a.0 as i32) * (b.0 as i32)) + 16384) >> 15) as i16)
    }

    #[inline(always)]
    unsafe fn reorder_widen_mul_accumulate(
        self,
        a: Vec1<i16>,
        b: Vec1<i16>,
        sum: Vec1<i32>,
    ) -> Vec1<i32> {
        Vec1(sum.0 + (a.0 as i32) * (b.0 as i32))
    }

    #[inline(always)]
    unsafe fn saturated_neg<T: IntegerLane>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe { self.saturated_sub(self.zero::<T>(), v) }
    }

    #[inline(always)]
    unsafe fn saturated_abs<T: IntegerLane>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe { self.max(v, self.saturated_neg(v)) }
    }

    #[inline(always)]
    unsafe fn masked_min_or<T: Lane>(
        self,
        no: Vec1<T>,
        mask: Mask1<T>,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        unsafe { self.if_then_else(mask, self.min(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_max_or<T: Lane>(
        self,
        no: Vec1<T>,
        mask: Mask1<T>,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        unsafe { self.if_then_else(mask, self.max(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_add_or<T: Lane>(
        self,
        no: Vec1<T>,
        mask: Mask1<T>,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        unsafe { self.if_then_else(mask, self.add(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_sub_or<T: Lane>(
        self,
        no: Vec1<T>,
        mask: Mask1<T>,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        unsafe { self.if_then_else(mask, self.sub(a, b), no) }
    }

    #[inline(always)]
    unsafe fn masked_mul_or<T: Lane>(
        self,
        no: Vec1<T>,
        mask: Mask1<T>,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        unsafe { self.if_then_else(mask, self.mul(a, b), no) }
    }
}

// ---------------------------------------------------------------------------
// SimdBitwise implementation
// ---------------------------------------------------------------------------

/// Perform bitwise AND on the raw bytes of two values of any Lane type.
#[inline(always)]
unsafe fn bitwise_and<T: Lane>(a: T, b: T) -> T {
    let size = core::mem::size_of::<T>();
    let mut result = T::default();
    let pa = core::ptr::from_ref(&a).cast::<u8>();
    let pb = core::ptr::from_ref(&b).cast::<u8>();
    let pr = core::ptr::from_mut(&mut result).cast::<u8>();
    for i in 0..size {
        // SAFETY: Pointer arithmetic within bounds of the Lane-sized value.
        unsafe { *pr.add(i) = *pa.add(i) & *pb.add(i) };
    }
    result
}

/// Perform bitwise OR on the raw bytes of two values of any Lane type.
#[inline(always)]
unsafe fn bitwise_or<T: Lane>(a: T, b: T) -> T {
    let size = core::mem::size_of::<T>();
    let mut result = T::default();
    let pa = core::ptr::from_ref(&a).cast::<u8>();
    let pb = core::ptr::from_ref(&b).cast::<u8>();
    let pr = core::ptr::from_mut(&mut result).cast::<u8>();
    for i in 0..size {
        // SAFETY: Pointer arithmetic within bounds of the Lane-sized value.
        unsafe { *pr.add(i) = *pa.add(i) | *pb.add(i) };
    }
    result
}

/// Perform bitwise XOR on the raw bytes of two values of any Lane type.
#[inline(always)]
unsafe fn bitwise_xor<T: Lane>(a: T, b: T) -> T {
    let size = core::mem::size_of::<T>();
    let mut result = T::default();
    let pa = core::ptr::from_ref(&a).cast::<u8>();
    let pb = core::ptr::from_ref(&b).cast::<u8>();
    let pr = core::ptr::from_mut(&mut result).cast::<u8>();
    for i in 0..size {
        // SAFETY: Pointer arithmetic within bounds of the Lane-sized value.
        unsafe { *pr.add(i) = *pa.add(i) ^ *pb.add(i) };
    }
    result
}

/// Perform bitwise NOT on the raw bytes of a value of any Lane type.
#[inline(always)]
unsafe fn bitwise_not<T: Lane>(a: T) -> T {
    let size = core::mem::size_of::<T>();
    let mut result = T::default();
    let pa = core::ptr::from_ref(&a).cast::<u8>();
    let pr = core::ptr::from_mut(&mut result).cast::<u8>();
    for i in 0..size {
        // SAFETY: Pointer arithmetic within bounds of the Lane-sized value.
        unsafe { *pr.add(i) = !*pa.add(i) };
    }
    result
}

// SAFETY: All operations are plain Rust scalar bitwise ops.
unsafe impl SimdBitwise for Scalar {
    #[inline(always)]
    unsafe fn and<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        Vec1(unsafe { bitwise_and(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn or<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        Vec1(unsafe { bitwise_or(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn xor<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        Vec1(unsafe { bitwise_xor(a.0, b.0) })
    }

    #[inline(always)]
    unsafe fn not<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        Vec1(unsafe { bitwise_not(v.0) })
    }

    #[inline(always)]
    unsafe fn and_not<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        Vec1(unsafe { bitwise_and(bitwise_not(a.0), b.0) })
    }

    #[inline(always)]
    unsafe fn shift_left<T: IntegerLane, const BITS: u32>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shl(BITS);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shl(BITS);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn shift_right<T: IntegerLane, const BITS: u32>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shr(BITS);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shr(BITS);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn rotate_right<T: IntegerLane, const BITS: u32>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.rotate_right(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    // Rotate via unsigned
                    let u = val as u8;
                    let r = u.rotate_right(BITS) as i8;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.rotate_right(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let u = val as u16;
                    let r = u.rotate_right(BITS) as i16;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.rotate_right(BITS);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let u = val as u32;
                    let r = u.rotate_right(BITS) as i32;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.rotate_right(BITS);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let u = val as u64;
                let r = u.rotate_right(BITS) as i64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn shift_left_same<T: IntegerLane>(self, v: Vec1<T>, bits: u32) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(bits);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(bits);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(bits);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(bits);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(bits);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shl(bits);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shl(bits);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shl(bits);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn shift_right_same<T: IntegerLane>(self, v: Vec1<T>, bits: u32) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(bits);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(bits);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(bits);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(bits);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(bits);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = val.wrapping_shr(bits);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shr(bits);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val.wrapping_shr(bits);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn shift_left_bytes<T: Lane, const BYTES: usize>(self, v: Vec1<T>) -> Vec1<T> {
        // Scalar: shifting bytes within a 1-lane vector.
        // If BYTES >= size_of::<T>(), result is zero.
        if BYTES >= core::mem::size_of::<T>() {
            return Vec1(T::default());
        }
        // Shift left by BYTES means move data toward more significant bytes.
        let mut buf = [0u8; 16];
        let size = core::mem::size_of::<T>();
        unsafe {
            core::ptr::copy_nonoverlapping(
                core::ptr::from_ref(&v.0).cast::<u8>(),
                buf.as_mut_ptr().add(BYTES),
                size - BYTES,
            );
            let mut result = T::default();
            core::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                core::ptr::from_mut(&mut result).cast::<u8>(),
                size,
            );
            Vec1(result)
        }
    }

    #[inline(always)]
    unsafe fn shift_right_bytes<T: Lane, const BYTES: usize>(self, v: Vec1<T>) -> Vec1<T> {
        if BYTES >= core::mem::size_of::<T>() {
            return Vec1(T::default());
        }
        let mut buf = [0u8; 16];
        let size = core::mem::size_of::<T>();
        unsafe {
            core::ptr::copy_nonoverlapping(
                core::ptr::from_ref(&v.0).cast::<u8>().add(BYTES),
                buf.as_mut_ptr(),
                size - BYTES,
            );
            let mut result = T::default();
            core::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                core::ptr::from_mut(&mut result).cast::<u8>(),
                size,
            );
            Vec1(result)
        }
    }

    #[inline(always)]
    unsafe fn population_count<T: IntegerLane>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.count_ones() as u8;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = (val as u8).count_ones() as i8;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.count_ones() as u16;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = (val as u16).count_ones() as i16;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.count_ones() as u32;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = (val as u32).count_ones() as i32;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.count_ones() as u64;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = (val as u64).count_ones() as i64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn leading_zero_count<T: IntegerLane>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.leading_zeros() as u8;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = (val as u8).leading_zeros() as i8;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.leading_zeros() as u16;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = (val as u16).leading_zeros() as i16;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.leading_zeros() as u32;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = (val as u32).leading_zeros() as i32;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.leading_zeros() as u64;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = (val as u64).leading_zeros() as i64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn trailing_zero_count<T: IntegerLane>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.trailing_zeros() as u8;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = (val as u8).trailing_zeros() as i8;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.trailing_zeros() as u16;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = (val as u16).trailing_zeros() as i16;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.trailing_zeros() as u32;
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = (val as u32).trailing_zeros() as i32;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.trailing_zeros() as u64;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = (val as u64).trailing_zeros() as i64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn reverse_lane_bytes<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        let size = core::mem::size_of::<T>();
        if size <= 1 {
            return v;
        }
        let mut buf = [0u8; 8];
        unsafe {
            core::ptr::copy_nonoverlapping(
                core::ptr::from_ref(&v.0).cast::<u8>(),
                buf.as_mut_ptr(),
                size,
            );
        }
        buf[..size].reverse();
        let mut result = T::default();
        unsafe {
            core::ptr::copy_nonoverlapping(
                buf.as_ptr(),
                core::ptr::from_mut(&mut result).cast::<u8>(),
                size,
            );
        }
        Vec1(result)
    }

    #[inline(always)]
    unsafe fn reverse_bits<T: IntegerLane>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let r = val.reverse_bits();
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let r = (val as u8).reverse_bits() as i8;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let r = val.reverse_bits();
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let r = (val as u16).reverse_bits() as i16;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let r = val.reverse_bits();
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let r = (val as u32).reverse_bits() as i32;
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.reverse_bits();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = (val as u64).reverse_bits() as i64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn shl<T: IntegerLane>(
        self,
        v: Vec1<T>,
        bits: Vec1<T>,
    ) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let shift: u8 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shl(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let shift: i8 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shl(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let shift: u16 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shl(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let shift: i16 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shl(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let shift: u32 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shl(shift);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let shift: i32 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shl(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let shift: u64 = core::mem::transmute_copy(&bits.0);
                let r = val.wrapping_shl(shift as u32);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let shift: i64 = core::mem::transmute_copy(&bits.0);
                let r = val.wrapping_shl(shift as u32);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn shr<T: IntegerLane>(
        self,
        v: Vec1<T>,
        bits: Vec1<T>,
    ) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if is_type::<T, u8>() {
                    let val: u8 = core::mem::transmute_copy(&v.0);
                    let shift: u8 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shr(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i8 = core::mem::transmute_copy(&v.0);
                    let shift: i8 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shr(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                if is_type::<T, u16>() {
                    let val: u16 = core::mem::transmute_copy(&v.0);
                    let shift: u16 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shr(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i16 = core::mem::transmute_copy(&v.0);
                    let shift: i16 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shr(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 4 {
                if is_type::<T, u32>() {
                    let val: u32 = core::mem::transmute_copy(&v.0);
                    let shift: u32 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shr(shift);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    let val: i32 = core::mem::transmute_copy(&v.0);
                    let shift: i32 = core::mem::transmute_copy(&bits.0);
                    let r = val.wrapping_shr(shift as u32);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if is_type::<T, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let shift: u64 = core::mem::transmute_copy(&bits.0);
                let r = val.wrapping_shr(shift as u32);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let shift: i64 = core::mem::transmute_copy(&bits.0);
                let r = val.wrapping_shr(shift as u32);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn ror<T: IntegerLane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                if crate::lane::is_type::<T, u8>() {
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.rotate_right(vb as u32 & 7);
                    Vec1(core::mem::transmute_copy(&r))
                } else {
                    // i8: treat as u8 rotation
                    let va: u8 = core::mem::transmute_copy(&a.0);
                    let vb: u8 = core::mem::transmute_copy(&b.0);
                    let r = va.rotate_right(vb as u32 & 7);
                    Vec1(core::mem::transmute_copy(&r))
                }
            } else if T::BYTES == 2 {
                let va: u16 = core::mem::transmute_copy(&a.0);
                let vb: u16 = core::mem::transmute_copy(&b.0);
                let r = va.rotate_right(vb as u32 & 15);
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 4 {
                let va: u32 = core::mem::transmute_copy(&a.0);
                let vb: u32 = core::mem::transmute_copy(&b.0);
                let r = va.rotate_right(vb & 31);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = va.rotate_right(vb as u32 & 63);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn rol<T: IntegerLane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                let va: u8 = core::mem::transmute_copy(&a.0);
                let vb: u8 = core::mem::transmute_copy(&b.0);
                let r = va.rotate_left(vb as u32 & 7);
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 2 {
                let va: u16 = core::mem::transmute_copy(&a.0);
                let vb: u16 = core::mem::transmute_copy(&b.0);
                let r = va.rotate_left(vb as u32 & 15);
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 4 {
                let va: u32 = core::mem::transmute_copy(&a.0);
                let vb: u32 = core::mem::transmute_copy(&b.0);
                let r = va.rotate_left(vb & 31);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: u64 = core::mem::transmute_copy(&a.0);
                let vb: u64 = core::mem::transmute_copy(&b.0);
                let r = va.rotate_left(vb as u32 & 63);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn rotate_left<T: IntegerLane, const BITS: u32>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe {
            if T::BYTES == 1 {
                let val: u8 = core::mem::transmute_copy(&v.0);
                let r = val.rotate_left(BITS);
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 2 {
                let val: u16 = core::mem::transmute_copy(&v.0);
                let r = val.rotate_left(BITS);
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 4 {
                let val: u32 = core::mem::transmute_copy(&v.0);
                let r = val.rotate_left(BITS);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.rotate_left(BITS);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn broadcast_sign_bit<T: IntegerLane>(self, v: Vec1<T>) -> Vec1<T> {
        // Arithmetic shift right by (bits - 1): all-ones if MSB set, else all-zeros.
        unsafe {
            if T::BYTES == 1 {
                let val: i8 = core::mem::transmute_copy(&v.0);
                let r = val >> 7;
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 2 {
                let val: i16 = core::mem::transmute_copy(&v.0);
                let r = val >> 15;
                Vec1(core::mem::transmute_copy(&r))
            } else if T::BYTES == 4 {
                let val: i32 = core::mem::transmute_copy(&v.0);
                let r = val >> 31;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val >> 63;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimdCompare implementation
// ---------------------------------------------------------------------------

// SAFETY: All operations are plain Rust scalar comparisons.
unsafe impl SimdCompare for Scalar {
    #[inline(always)]
    unsafe fn eq<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Mask1<T> {
        Mask1::from_bool(a.0 == b.0)
    }

    #[inline(always)]
    unsafe fn ne<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Mask1<T> {
        Mask1::from_bool(a.0 != b.0)
    }

    #[inline(always)]
    unsafe fn lt<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Mask1<T> {
        // Lane: PartialOrd, so we can use < directly
        Mask1::from_bool(a.0 < b.0)
    }

    #[inline(always)]
    unsafe fn le<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Mask1<T> {
        Mask1::from_bool(a.0 <= b.0)
    }

    #[inline(always)]
    unsafe fn gt<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Mask1<T> {
        Mask1::from_bool(a.0 > b.0)
    }

    #[inline(always)]
    unsafe fn ge<T: Lane>(self, a: Vec1<T>, b: Vec1<T>) -> Mask1<T> {
        Mask1::from_bool(a.0 >= b.0)
    }

    #[inline(always)]
    unsafe fn test_bit<T: IntegerLane>(
        self,
        v: Vec1<T>,
        bit: Vec1<T>,
    ) -> Mask1<T> {
        let anded = unsafe { bitwise_and(v.0, bit.0) };
        Mask1::from_bool(anded != T::default())
    }
}

// ---------------------------------------------------------------------------
// SimdMask implementation
// ---------------------------------------------------------------------------

// SAFETY: All operations are plain Rust scalar ops on mask bits.
unsafe impl SimdMask for Scalar {
    #[inline(always)]
    unsafe fn mask_from_vec<T: Lane>(self, v: Vec1<T>) -> Mask1<T> {
        Mask1::from_bool(v.0 != T::default())
    }

    #[inline(always)]
    unsafe fn vec_from_mask<T: Lane>(self, m: Mask1<T>) -> Vec1<T> {
        // Convert all-ones unsigned back to T via transmute.
        let mut result = T::default();
        // SAFETY: T::Unsigned and T have the same size (Lane invariant).
        unsafe {
            core::ptr::copy_nonoverlapping(
                core::ptr::from_ref(&m.0).cast::<u8>(),
                core::ptr::from_mut(&mut result).cast::<u8>(),
                core::mem::size_of::<T>(),
            );
        }
        Vec1(result)
    }

    #[inline(always)]
    unsafe fn first_n<T: Lane>(self, n: usize) -> Mask1<T> {
        // Scalar has 1 lane: true if n >= 1.
        Mask1::from_bool(n >= 1)
    }

    #[inline(always)]
    unsafe fn count_true<T: Lane>(self, m: Mask1<T>) -> usize {
        if m.to_bool() { 1 } else { 0 }
    }

    #[inline(always)]
    unsafe fn all_true<T: Lane>(self, m: Mask1<T>) -> bool {
        m.to_bool()
    }

    #[inline(always)]
    unsafe fn all_false<T: Lane>(self, m: Mask1<T>) -> bool {
        !m.to_bool()
    }

    #[inline(always)]
    unsafe fn find_first_true<T: Lane>(self, m: Mask1<T>) -> Option<usize> {
        if m.to_bool() { Some(0) } else { None }
    }

    #[inline(always)]
    unsafe fn if_then_else<T: Lane>(
        self,
        mask: Mask1<T>,
        yes: Vec1<T>,
        no: Vec1<T>,
    ) -> Vec1<T> {
        if mask.to_bool() { yes } else { no }
    }

    #[inline(always)]
    unsafe fn if_then_else_zero<T: Lane>(
        self,
        mask: Mask1<T>,
        yes: Vec1<T>,
    ) -> Vec1<T> {
        if mask.to_bool() { yes } else { Vec1(T::default()) }
    }

    #[inline(always)]
    unsafe fn if_then_zero_else<T: Lane>(
        self,
        mask: Mask1<T>,
        no: Vec1<T>,
    ) -> Vec1<T> {
        if mask.to_bool() { Vec1(T::default()) } else { no }
    }

    #[inline(always)]
    unsafe fn and_mask<T: Lane>(self, a: Mask1<T>, b: Mask1<T>) -> Mask1<T> {
        Mask1::from_bool(a.to_bool() && b.to_bool())
    }

    #[inline(always)]
    unsafe fn or_mask<T: Lane>(self, a: Mask1<T>, b: Mask1<T>) -> Mask1<T> {
        Mask1::from_bool(a.to_bool() || b.to_bool())
    }

    #[inline(always)]
    unsafe fn not_mask<T: Lane>(self, m: Mask1<T>) -> Mask1<T> {
        Mask1::from_bool(!m.to_bool())
    }

    #[inline(always)]
    unsafe fn xor_mask<T: Lane>(self, a: Mask1<T>, b: Mask1<T>) -> Mask1<T> {
        Mask1::from_bool(a.to_bool() ^ b.to_bool())
    }

    #[inline(always)]
    unsafe fn find_last_true<T: Lane>(self, m: Mask1<T>) -> Option<usize> {
        if m.to_bool() { Some(0) } else { None }
    }

    #[inline(always)]
    unsafe fn bits_from_mask<T: Lane>(self, m: Mask1<T>) -> u64 {
        if m.to_bool() { 1 } else { 0 }
    }

    #[inline(always)]
    unsafe fn exclusive_neither<T: Lane>(self, a: Mask1<T>, b: Mask1<T>) -> Mask1<T> {
        // NOR: true only where neither a nor b is set (C++ ExclusiveNeither).
        Mask1::from_bool(!a.to_bool() && !b.to_bool())
    }

    #[inline(always)]
    unsafe fn slide_mask_1_up<T: Lane>(self, _mask: Mask1<T>) -> Mask1<T> {
        // Single lane: shifting up fills with false, so result is always false
        Mask1::from_bool(false)
    }

    #[inline(always)]
    unsafe fn slide_mask_1_down<T: Lane>(self, _mask: Mask1<T>) -> Mask1<T> {
        // Single lane: shifting down fills with false, so result is always false
        Mask1::from_bool(false)
    }

    #[inline(always)]
    unsafe fn if_negative_then_else<T: Lane>(
        self,
        v: Vec1<T>,
        yes: Vec1<T>,
        no: Vec1<T>,
    ) -> Vec1<T> {
        // SAFETY: reinterpret raw bits as a signed integer of matching width to
        // test the sign bit (works for both signed integers and floats).
        if unsafe { is_negative_bits(v) } { yes } else { no }
    }

    #[inline(always)]
    unsafe fn if_negative_then_else_zero<T: Lane>(self, v: Vec1<T>, yes: Vec1<T>) -> Vec1<T> {
        if unsafe { is_negative_bits(v) } {
            yes
        } else {
            unsafe { self.zero::<T>() }
        }
    }

    #[inline(always)]
    unsafe fn if_negative_then_zero_else<T: Lane>(self, v: Vec1<T>, no: Vec1<T>) -> Vec1<T> {
        if unsafe { is_negative_bits(v) } {
            unsafe { self.zero::<T>() }
        } else {
            no
        }
    }
}

/// Test whether the sign bit (MSB) of a lane's raw bits is set.
/// Works for signed integers and floats (treats -0.0 as negative, matching Highway).
#[inline(always)]
unsafe fn is_negative_bits<T: Lane>(v: Vec1<T>) -> bool {
    unsafe {
        match T::BYTES {
            1 => {
                let x: i8 = core::mem::transmute_copy(&v.0);
                x < 0
            }
            2 => {
                let x: i16 = core::mem::transmute_copy(&v.0);
                x < 0
            }
            4 => {
                let x: i32 = core::mem::transmute_copy(&v.0);
                x < 0
            }
            _ => {
                let x: i64 = core::mem::transmute_copy(&v.0);
                x < 0
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimdConvert implementation
// ---------------------------------------------------------------------------

// SAFETY: All operations are plain Rust scalar conversions.
unsafe impl SimdConvert for Scalar {
    #[inline(always)]
    unsafe fn promote_to<N: NarrowLane>(
        self,
        v: Vec1<N>,
    ) -> Vec1<N::Wide>
    where
        N::Wide: Lane,
    {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            // Dispatch based on type to perform the correct widening conversion.
            if is_type::<N, u8>() {
                let val: u8 = core::mem::transmute_copy(&v.0);
                let r = val as u16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<N, u16>() {
                let val: u16 = core::mem::transmute_copy(&v.0);
                let r = val as u32;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<N, u32>() {
                let val: u32 = core::mem::transmute_copy(&v.0);
                let r = val as u64;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<N, i8>() {
                let val: i8 = core::mem::transmute_copy(&v.0);
                let r = val as i16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<N, i16>() {
                let val: i16 = core::mem::transmute_copy(&v.0);
                let r = val as i32;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<N, i32>() {
                let val: i32 = core::mem::transmute_copy(&v.0);
                let r = val as i64;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                // f32 -> f64
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val as f64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn demote_to<W: WideLane>(
        self,
        v: Vec1<W>,
    ) -> Vec1<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if is_type::<W, u16>() {
                let val: u16 = core::mem::transmute_copy(&v.0);
                let r = val.min(u8::MAX as u16) as u8;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, u32>() {
                let val: u32 = core::mem::transmute_copy(&v.0);
                let r = val.min(u16::MAX as u32) as u16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val.min(u32::MAX as u64) as u32;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, i16>() {
                let val: i16 = core::mem::transmute_copy(&v.0);
                let r = val.max(i8::MIN as i16).min(i8::MAX as i16) as i8;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, i32>() {
                let val: i32 = core::mem::transmute_copy(&v.0);
                let r = val.max(i16::MIN as i32).min(i16::MAX as i32) as i16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, i64>() {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val.max(i32::MIN as i64).min(i32::MAX as i64) as i32;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                // f64 -> f32
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val as f32;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn convert_to_int<F: FloatLane>(self, v: Vec1<F>) -> Vec1<F::Int> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if core::mem::size_of::<F>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val as i32;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val as i64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn convert_to_float<F: FloatLane>(self, v: Vec1<F::Int>) -> Vec1<F> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if core::mem::size_of::<F>() == 4 {
                let val: i32 = core::mem::transmute_copy(&v.0);
                let r = val as f32;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val as f64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn truncate_to<W: WideLane>(self, v: Vec1<W>) -> Vec1<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // Keep low bits (truncation, no saturation).
        unsafe {
            if is_type::<W, u16>() {
                let val: u16 = core::mem::transmute_copy(&v.0);
                let r = val as u8;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, u32>() {
                let val: u32 = core::mem::transmute_copy(&v.0);
                let r = val as u16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, u64>() {
                let val: u64 = core::mem::transmute_copy(&v.0);
                let r = val as u32;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, i16>() {
                let val: i16 = core::mem::transmute_copy(&v.0);
                let r = val as i8;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, i32>() {
                let val: i32 = core::mem::transmute_copy(&v.0);
                let r = val as i16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<W, i64>() {
                let val: i64 = core::mem::transmute_copy(&v.0);
                let r = val as i32;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                // f64 -> f32
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val as f32;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn ordered_demote_2_to<W: WideLane>(
        self,
        lo: Vec1<W>,
        _hi: Vec1<W>,
    ) -> Vec1<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // Scalar has only 1 lane. Demote lo; hi is ignored (no room).
        unsafe { self.demote_to(lo) }
    }

    #[inline(always)]
    unsafe fn nearest_int<F: FloatLane>(self, v: Vec1<F>) -> Vec1<F::Int> {
        unsafe {
            if core::mem::size_of::<F>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                // Clamp to avoid UB on overflow, then round-to-nearest-even.
                let clamped = if val >= i32::MAX as f32 {
                    i32::MAX
                } else {
                    val.round_ties_even() as i32
                };
                Vec1(core::mem::transmute_copy(&clamped))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let clamped = if val >= i64::MAX as f64 {
                    i64::MAX
                } else {
                    val.round_ties_even() as i64
                };
                Vec1(core::mem::transmute_copy(&clamped))
            }
        }
    }

    #[inline(always)]
    unsafe fn reorder_demote_2_to<W: WideLane>(
        self,
        a: Vec1<W>,
        _b: Vec1<W>,
    ) -> Vec1<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // Scalar: same as ordered_demote_2_to (just demote the first element)
        unsafe { self.demote_to(a) }
    }

    #[inline(always)]
    unsafe fn demote_in_range_to<W: WideLane>(self, v: Vec1<W>) -> Vec1<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // In-range: same as demote_to for scalar (truncation)
        unsafe { self.demote_to(v) }
    }

    #[inline(always)]
    unsafe fn convert_in_range_to_int<F: FloatLane>(self, v: Vec1<F>) -> Vec1<F::Int> {
        // Same as convert_to_int for scalar (truncation toward zero)
        unsafe { self.convert_to_int(v) }
    }

    #[inline(always)]
    unsafe fn promote_lower_to<N: NarrowLane>(self, v: Vec1<N>) -> Vec1<N::Wide>
    where
        N::Wide: Lane,
    {
        // Scalar: lower half is the entire vector, just promote
        unsafe { self.promote_to(v) }
    }

    #[inline(always)]
    unsafe fn promote_upper_to<N: NarrowLane>(self, _v: Vec1<N>) -> Vec1<N::Wide>
    where
        N::Wide: Lane,
    {
        // Scalar: no upper half, return zero
        Vec1(N::Wide::default())
    }

    #[inline(always)]
    unsafe fn ordered_truncate_2_to<W: WideLane>(
        self,
        lo: Vec1<W>,
        _hi: Vec1<W>,
    ) -> Vec1<W::Narrow>
    where
        W::Narrow: Lane,
    {
        // Scalar has only 1 lane. Truncate lo; hi is ignored (no room).
        unsafe { self.truncate_to(lo) }
    }
}

// ---------------------------------------------------------------------------
// SimdShuffle implementation
// ---------------------------------------------------------------------------

// SAFETY: Scalar "shuffle" operations are trivial identity operations on a single lane.
unsafe impl SimdShuffle for Scalar {
    #[inline(always)]
    unsafe fn reverse<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        v // 1 element -> already reversed
    }

    #[inline(always)]
    unsafe fn broadcast_lane<T: Lane, const IDX: usize>(self, v: Vec1<T>) -> Vec1<T> {
        debug_assert!(IDX == 0, "Scalar: broadcast_lane IDX must be 0");
        v
    }

    #[inline(always)]
    unsafe fn interleave_lower<T: Lane>(
        self,
        a: Vec1<T>,
        _b: Vec1<T>,
    ) -> Vec1<T> {
        a // 1 element: take from a
    }

    #[inline(always)]
    unsafe fn interleave_upper<T: Lane>(
        self,
        _a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        b // 1 element: take from b for "upper"
    }

    #[inline(always)]
    unsafe fn zip_lower<T: Lane>(
        self,
        a: Vec1<T>,
        _b: Vec1<T>,
    ) -> Vec1<T> {
        a
    }

    #[inline(always)]
    unsafe fn zip_upper<T: Lane>(
        self,
        _a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        b
    }

    #[inline(always)]
    unsafe fn table_lookup_bytes<T: Lane>(
        self,
        table: Vec1<T>,
        _idx: Vec1<T>,
    ) -> Vec1<T> {
        // Scalar: just return the single element. The index is meaningless
        // with only one element.
        table
    }

    #[inline(always)]
    unsafe fn table_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        v: Vec1<T>,
        _idx: Vec1<I>,
    ) -> Vec1<T> {
        v // only lane 0 exists
    }

    #[inline(always)]
    unsafe fn reverse2<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        v // 1 element
    }

    #[inline(always)]
    unsafe fn reverse4<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        v
    }

    #[inline(always)]
    unsafe fn reverse8<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        v
    }

    #[inline(always)]
    unsafe fn concat_upper_lower<T: Lane>(
        self,
        hi: Vec1<T>,
        _lo: Vec1<T>,
    ) -> Vec1<T> {
        // Scalar: "upper half" of hi is the only lane, "lower half" of lo
        // doesn't exist. Return hi.
        hi
    }

    #[inline(always)]
    unsafe fn concat_lower_upper<T: Lane>(
        self,
        hi: Vec1<T>,
        _lo: Vec1<T>,
    ) -> Vec1<T> {
        hi
    }

    #[inline(always)]
    unsafe fn concat_even<T: Lane>(
        self,
        a: Vec1<T>,
        _b: Vec1<T>,
    ) -> Vec1<T> {
        a // lane 0 of a is even
    }

    #[inline(always)]
    unsafe fn concat_odd<T: Lane>(
        self,
        _a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        b // no odd lane; return b
    }

    #[inline(always)]
    unsafe fn odd_even<T: Lane>(
        self,
        _odd: Vec1<T>,
        even: Vec1<T>,
    ) -> Vec1<T> {
        even // lane 0 is even
    }

    #[inline(always)]
    unsafe fn slide_up_lanes<T: Lane>(self, v: Vec1<T>, n: usize) -> Vec1<T> {
        if n > 0 { Vec1(T::default()) } else { v }
    }

    #[inline(always)]
    unsafe fn slide_down_lanes<T: Lane>(self, v: Vec1<T>, n: usize) -> Vec1<T> {
        if n > 0 { Vec1(T::default()) } else { v }
    }

    #[inline(always)]
    unsafe fn compress<T: Lane>(self, v: Vec1<T>, mask: Mask1<T>) -> Vec1<T> {
        if mask.to_bool() { v } else { Vec1(T::default()) }
    }

    #[inline(always)]
    unsafe fn compress_store<T: Lane>(
        self,
        v: Vec1<T>,
        mask: Mask1<T>,
        ptr: *mut T,
    ) -> usize {
        if mask.to_bool() {
            unsafe { ptr.write(v.0) };
            1
        } else {
            0
        }
    }

    #[inline(always)]
    unsafe fn dup_even<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        // Only 1 lane (lane 0 = even), identity.
        v
    }

    #[inline(always)]
    unsafe fn dup_odd<T: Lane>(self, _v: Vec1<T>) -> Vec1<T> {
        // No odd lane to duplicate; return zero.
        Vec1(T::default())
    }

    #[inline(always)]
    unsafe fn concat_lower_lower<T: Lane>(
        self,
        _hi: Vec1<T>,
        lo: Vec1<T>,
    ) -> Vec1<T> {
        // Scalar: the "lower half" of each is the entire single lane. Return lo.
        lo
    }

    #[inline(always)]
    unsafe fn concat_upper_upper<T: Lane>(
        self,
        _hi: Vec1<T>,
        _lo: Vec1<T>,
    ) -> Vec1<T> {
        // Scalar: upper half of a 1-lane vector is empty. Return zero.
        Vec1(T::default())
    }

    #[inline(always)]
    unsafe fn slide_1_up<T: Lane>(self, _v: Vec1<T>) -> Vec1<T> {
        // Shift up by 1: lane 0 gets zero (original was shifted to non-existent lane 1).
        Vec1(T::default())
    }

    #[inline(always)]
    unsafe fn slide_1_down<T: Lane>(self, _v: Vec1<T>) -> Vec1<T> {
        // Shift down by 1: lane 0 gets value from non-existent lane 1, so zero.
        Vec1(T::default())
    }

    #[inline(always)]
    unsafe fn expand<T: Lane>(self, v: Vec1<T>, mask: Mask1<T>) -> Vec1<T> {
        if mask.to_bool() { v } else { Vec1(T::default()) }
    }

    #[inline(always)]
    unsafe fn combine_shift_right_bytes<T: Lane, const BYTES: usize>(
        self,
        _hi: Vec1<T>,
        _lo: Vec1<T>,
    ) -> Vec1<T> {
        // Scalar has only 1 byte-lane at most; no valid BYTES exist for a 1-element vector.
        unreachable!("combine_shift_right_bytes: no valid BYTES for scalar")
    }

    #[inline(always)]
    unsafe fn compress_blended_store<T: Lane>(
        self,
        v: Vec1<T>,
        mask: Mask1<T>,
        ptr: *mut T,
    ) -> usize {
        if mask.to_bool() {
            unsafe { ptr.write(v.0) };
            1
        } else {
            0
        }
    }

    #[inline(always)]
    unsafe fn odd_even_blocks<T: Lane>(
        self,
        _odd: Vec1<T>,
        even: Vec1<T>,
    ) -> Vec1<T> {
        // Scalar: only 1 block (block 0 = even).
        even
    }

    #[inline(always)]
    unsafe fn reverse_blocks<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        // Scalar: only 1 block, nothing to reverse.
        v
    }

    #[inline(always)]
    unsafe fn compress_not<T: Lane>(self, v: Vec1<T>, mask: Mask1<T>) -> Vec1<T> {
        unsafe { self.compress(v, self.not_mask(mask)) }
    }

    #[inline(always)]
    unsafe fn compress_blocks_not(self, v: Vec1<u64>, _mask: Mask1<u64>) -> Vec1<u64> {
        // Single block, no-op
        v
    }

    #[inline(always)]
    unsafe fn broadcast_block<T: Lane, const IDX: usize>(self, v: Vec1<T>) -> Vec1<T> {
        // Single block, return as-is (IDX must be 0)
        v
    }

    #[inline(always)]
    unsafe fn compress_bits<T: Lane>(self, v: Vec1<T>, bits: *const u8) -> Vec1<T> {
        unsafe {
            let b = bits.read();
            if b & 1 != 0 { v } else { Vec1(T::default()) }
        }
    }

    #[inline(always)]
    unsafe fn compress_bits_store<T: Lane>(self, v: Vec1<T>, bits: *const u8, ptr: *mut T) -> usize {
        unsafe {
            let b = bits.read();
            if b & 1 != 0 {
                ptr.write(v.0);
                1
            } else {
                0
            }
        }
    }

    #[inline(always)]
    unsafe fn lower_half<T: Lane>(self, v: Vec1<T>) -> Vec1<T> {
        v
    }

    #[inline(always)]
    unsafe fn upper_half<T: Lane>(self, _v: Vec1<T>) -> Vec1<T> {
        Vec1(T::default())
    }

    #[inline(always)]
    unsafe fn combine<T: Lane>(self, lo: Vec1<T>, _hi: Vec1<T>) -> Vec1<T> {
        lo
    }

    #[inline(always)]
    unsafe fn insert_block<T: Lane, const IDX: usize>(self, _v: Vec1<T>, blk: Vec1<T>) -> Vec1<T> {
        blk
    }

    #[inline(always)]
    unsafe fn extract_block<T: Lane, const IDX: usize>(self, v: Vec1<T>) -> Vec1<T> {
        v
    }

    #[inline(always)]
    unsafe fn interleave_whole_lower<T: Lane>(self, a: Vec1<T>, _b: Vec1<T>) -> Vec1<T> {
        // Single element: same as interleave_lower
        a
    }

    #[inline(always)]
    unsafe fn interleave_whole_upper<T: Lane>(self, _a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        b
    }

    #[inline(always)]
    unsafe fn interleave_even<T: Lane>(self, a: Vec1<T>, _b: Vec1<T>) -> Vec1<T> {
        a
    }

    #[inline(always)]
    unsafe fn interleave_odd<T: Lane>(self, _a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        b
    }

    #[inline(always)]
    unsafe fn two_tables_lookup_lanes<T: Lane, I: IntegerLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
        idx: Vec1<I>,
    ) -> Vec1<T> {
        // Single lane: index 0 -> a, index 1 -> b
        unsafe {
            let i: u64 = match I::BYTES {
                1 => core::mem::transmute_copy::<I, u8>(&idx.0) as u64,
                2 => core::mem::transmute_copy::<I, u16>(&idx.0) as u64,
                4 => core::mem::transmute_copy::<I, u32>(&idx.0) as u64,
                _ => core::mem::transmute_copy::<I, u64>(&idx.0),
            };
            if i == 0 { a } else { b }
        }
    }

    #[inline(always)]
    unsafe fn table_lookup_lanes_or0<T: Lane, I: IntegerLane>(
        self,
        v: Vec1<T>,
        idx: Vec1<I>,
    ) -> Vec1<T> {
        unsafe {
            let i: i64 = match I::BYTES {
                1 => core::mem::transmute_copy::<I, i8>(&idx.0) as i64,
                2 => core::mem::transmute_copy::<I, i16>(&idx.0) as i64,
                4 => core::mem::transmute_copy::<I, i32>(&idx.0) as i64,
                _ => core::mem::transmute_copy::<I, i64>(&idx.0),
            };
            if i < 0 { Vec1(T::default()) } else if i == 0 { v } else { Vec1(T::default()) }
        }
    }
}

// ---------------------------------------------------------------------------
// SimdReduce implementation
// ---------------------------------------------------------------------------

// SAFETY: Scalar reductions are identity operations on a single lane.
unsafe impl SimdReduce for Scalar {
    #[inline(always)]
    unsafe fn sum_of_lanes<T: Lane>(self, v: Vec1<T>) -> T {
        v.0
    }

    #[inline(always)]
    unsafe fn min_of_lanes<T: Lane>(self, v: Vec1<T>) -> T {
        v.0
    }

    #[inline(always)]
    unsafe fn max_of_lanes<T: Lane>(self, v: Vec1<T>) -> T {
        v.0
    }

    #[inline(always)]
    unsafe fn sums_of_8_abs_diff(
        self,
        a: Vec1<u8>,
        b: Vec1<u8>,
    ) -> Vec1<u64> {
        // Scalar: only 1 byte, so the "sum of 8" is just 1 absolute difference.
        Vec1(a.0.abs_diff(b.0) as u64)
    }

    #[inline(always)]
    unsafe fn sums_of_2<T: NarrowLane>(self, v: Vec1<T>) -> Vec1<T::Wide>
    where
        T::Wide: Lane,
    {
        // Scalar: only 1 lane. Widen v[0] (v[1] doesn't exist, treated as 0).
        unsafe {
            if is_type::<T, u8>() {
                let val: u8 = core::mem::transmute_copy(&v.0);
                let r = val as u16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u16>() {
                let val: u16 = core::mem::transmute_copy(&v.0);
                let r = val as u32;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, u32>() {
                let val: u32 = core::mem::transmute_copy(&v.0);
                let r = val as u64;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, i8>() {
                let val: i8 = core::mem::transmute_copy(&v.0);
                let r = val as i16;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, i16>() {
                let val: i16 = core::mem::transmute_copy(&v.0);
                let r = val as i32;
                Vec1(core::mem::transmute_copy(&r))
            } else if is_type::<T, i32>() {
                let val: i32 = core::mem::transmute_copy(&v.0);
                let r = val as i64;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                // f32 -> f64
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val as f64;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn sums_of_4<T: NarrowLane>(
        self,
        v: Vec1<T>,
    ) -> Vec1<<T::Wide as NarrowLane>::Wide>
    where
        T::Wide: NarrowLane + Lane,
        <T::Wide as NarrowLane>::Wide: Lane,
    {
        // Scalar: sums_of_2(sums_of_2(v)). With 1 lane, this is just double-widen.
        unsafe {
            let mid = self.sums_of_2(v);
            self.sums_of_2(mid)
        }
    }
}

// ---------------------------------------------------------------------------
// SimdFloat implementation
// ---------------------------------------------------------------------------

// SAFETY: All operations delegate to scalar float math.
unsafe impl SimdFloat for Scalar {
    #[inline(always)]
    unsafe fn sqrt<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val.sqrt();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val.sqrt();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn approx_reciprocal<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = 1.0f32 / val;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = 1.0f64 / val;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn approx_reciprocal_sqrt<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = 1.0f32 / val.sqrt();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = 1.0f64 / val.sqrt();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn round<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val.round_ties_even();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val.round_ties_even();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn trunc<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val.trunc();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val.trunc();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn ceil<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val.ceil();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val.ceil();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn floor<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = val.floor();
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = val.floor();
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn mul_add<T: FloatLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
        c: Vec1<T>,
    ) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let vc: f32 = core::mem::transmute_copy(&c.0);
                let r = va.mul_add(vb, vc);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let vc: f64 = core::mem::transmute_copy(&c.0);
                let r = va.mul_add(vb, vc);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn neg_mul_add<T: FloatLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
        c: Vec1<T>,
    ) -> Vec1<T> {
        // -(a*b) + c
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let vc: f32 = core::mem::transmute_copy(&c.0);
                let r = (-va).mul_add(vb, vc);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let vc: f64 = core::mem::transmute_copy(&c.0);
                let r = (-va).mul_add(vb, vc);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn mul_sub<T: FloatLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
        c: Vec1<T>,
    ) -> Vec1<T> {
        // a*b - c
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let vc: f32 = core::mem::transmute_copy(&c.0);
                let r = va.mul_add(vb, -vc);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let vc: f64 = core::mem::transmute_copy(&c.0);
                let r = va.mul_add(vb, -vc);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn neg_mul_sub<T: FloatLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
        c: Vec1<T>,
    ) -> Vec1<T> {
        // -(a*b) - c
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let vc: f32 = core::mem::transmute_copy(&c.0);
                let r = (-va).mul_add(vb, -vc);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let vc: f64 = core::mem::transmute_copy(&c.0);
                let r = (-va).mul_add(vb, -vc);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn copy_sign<T: FloatLane>(
        self,
        mag: Vec1<T>,
        sign: Vec1<T>,
    ) -> Vec1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let vm: f32 = core::mem::transmute_copy(&mag.0);
                let vs: f32 = core::mem::transmute_copy(&sign.0);
                let r = vm.copysign(vs);
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let vm: f64 = core::mem::transmute_copy(&mag.0);
                let vs: f64 = core::mem::transmute_copy(&sign.0);
                let r = vm.copysign(vs);
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn is_nan<T: FloatLane>(self, v: Vec1<T>) -> Mask1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                Mask1::from_bool(val.is_nan())
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                Mask1::from_bool(val.is_nan())
            }
        }
    }

    #[inline(always)]
    unsafe fn is_inf<T: FloatLane>(self, v: Vec1<T>) -> Mask1<T> {
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                Mask1::from_bool(val.is_infinite())
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                Mask1::from_bool(val.is_infinite())
            }
        }
    }

    #[inline(always)]
    unsafe fn zero_if_negative<T: FloatLane>(self, v: Vec1<T>) -> Vec1<T> {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                let r = if val.is_sign_negative() { 0.0f32 } else { val };
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                let r = if val.is_sign_negative() { 0.0f64 } else { val };
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn is_finite<T: FloatLane>(self, v: Vec1<T>) -> Mask1<T> {
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let val: f32 = core::mem::transmute_copy(&v.0);
                Mask1::from_bool(val.is_finite())
            } else {
                let val: f64 = core::mem::transmute_copy(&v.0);
                Mask1::from_bool(val.is_finite())
            }
        }
    }

    #[inline(always)]
    unsafe fn add_sub<T: FloatLane>(
        self,
        a: Vec1<T>,
        b: Vec1<T>,
    ) -> Vec1<T> {
        // Scalar: lane 0 is even -> subtract.
        unsafe {
            if core::mem::size_of::<T>() == 4 {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let r = va - vb;
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = va - vb;
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn min_number<T: FloatLane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        unsafe {
            if crate::lane::is_type::<T, f32>() {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let r = if va.is_nan() { vb } else if vb.is_nan() { va } else { va.min(vb) };
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = if va.is_nan() { vb } else if vb.is_nan() { va } else { va.min(vb) };
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn max_number<T: FloatLane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        unsafe {
            if crate::lane::is_type::<T, f32>() {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let r = if va.is_nan() { vb } else if vb.is_nan() { va } else { va.max(vb) };
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let r = if va.is_nan() { vb } else if vb.is_nan() { va } else { va.max(vb) };
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn min_magnitude<T: FloatLane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        unsafe {
            if crate::lane::is_type::<T, f32>() {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let aa = va.abs();
                let ab = vb.abs();
                let r = if aa < ab { va } else if ab < aa { vb } else { va.min(vb) };
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let aa = va.abs();
                let ab = vb.abs();
                let r = if aa < ab { va } else if ab < aa { vb } else { va.min(vb) };
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn max_magnitude<T: FloatLane>(self, a: Vec1<T>, b: Vec1<T>) -> Vec1<T> {
        unsafe {
            if crate::lane::is_type::<T, f32>() {
                let va: f32 = core::mem::transmute_copy(&a.0);
                let vb: f32 = core::mem::transmute_copy(&b.0);
                let aa = va.abs();
                let ab = vb.abs();
                let r = if aa > ab { va } else if ab > aa { vb } else { va.max(vb) };
                Vec1(core::mem::transmute_copy(&r))
            } else {
                let va: f64 = core::mem::transmute_copy(&a.0);
                let vb: f64 = core::mem::transmute_copy(&b.0);
                let aa = va.abs();
                let ab = vb.abs();
                let r = if aa > ab { va } else if ab > aa { vb } else { va.max(vb) };
                Vec1(core::mem::transmute_copy(&r))
            }
        }
    }

    #[inline(always)]
    unsafe fn is_either_nan<T: FloatLane>(self, a: Vec1<T>, b: Vec1<T>) -> Mask1<T> {
        unsafe { self.or_mask(self.is_nan(a), self.is_nan(b)) }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_and_splat() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let z: Vec1<u32> = s.zero();
            assert_eq!(z.0, 0);

            let v: Vec1<u32> = s.splat(42);
            assert_eq!(v.0, 42);

            let vf: Vec1<f32> = s.splat(2.75);
            assert!((vf.0 - 2.75).abs() < 1e-6);
        }
    }

    #[test]
    fn test_add_sub_mul() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let a: Vec1<i32> = s.splat(10);
            let b: Vec1<i32> = s.splat(3);

            let sum: Vec1<i32> = s.add(a, b);
            assert_eq!(sum.0, 13);

            let diff: Vec1<i32> = s.sub(a, b);
            assert_eq!(diff.0, 7);

            let prod: Vec1<i32> = s.mul(a, b);
            assert_eq!(prod.0, 30);
        }
    }

    #[test]
    fn test_float_arith() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let a: Vec1<f32> = s.splat(10.0);
            let b: Vec1<f32> = s.splat(3.0);

            let sum: Vec1<f32> = s.add(a, b);
            assert!((sum.0 - 13.0).abs() < 1e-6);

            let diff: Vec1<f32> = s.sub(a, b);
            assert!((diff.0 - 7.0).abs() < 1e-6);

            let prod: Vec1<f32> = s.mul(a, b);
            assert!((prod.0 - 30.0).abs() < 1e-6);

            let quot: Vec1<f32> = s.div(a, b);
            assert!((quot.0 - 10.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_bitwise() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let a: Vec1<u32> = s.splat(0xFF00_FF00);
            let b: Vec1<u32> = s.splat(0x00FF_00FF);

            let and_result: Vec1<u32> = s.and(a, b);
            assert_eq!(and_result.0, 0);

            let or_result: Vec1<u32> = s.or(a, b);
            assert_eq!(or_result.0, 0xFFFF_FFFF);

            let xor_result: Vec1<u32> = s.xor(a, b);
            assert_eq!(xor_result.0, 0xFFFF_FFFF);

            let not_result: Vec1<u32> = s.not(a);
            assert_eq!(not_result.0, 0x00FF_00FF);
        }
    }

    #[test]
    fn test_shifts() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let v: Vec1<u32> = s.splat(0x0000_00FF);

            let shl: Vec1<u32> = s.shift_left::<u32, 8>(v);
            assert_eq!(shl.0, 0x0000_FF00);

            let shr: Vec1<u32> = s.shift_right::<u32, 8>(shl);
            assert_eq!(shr.0, 0x0000_00FF);
        }
    }

    #[test]
    fn test_comparisons() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let a: Vec1<i32> = s.splat(5);
            let b: Vec1<i32> = s.splat(10);

            assert!(s.lt(a, b).to_bool());
            assert!(!s.gt(a, b).to_bool());
            assert!(s.le(a, b).to_bool());
            assert!(!s.ge(a, b).to_bool());
            assert!(!s.eq(a, b).to_bool());
            assert!(s.ne(a, b).to_bool());

            let c: Vec1<i32> = s.splat(5);
            assert!(s.eq(a, c).to_bool());
        }
    }

    #[test]
    fn test_mask_ops() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let yes: Vec1<i32> = s.splat(42);
            let no: Vec1<i32> = s.splat(0);

            let m_true = Mask1::<i32>::from_bool(true);
            let m_false = Mask1::<i32>::from_bool(false);

            let result: Vec1<i32> = s.if_then_else(m_true, yes, no);
            assert_eq!(result.0, 42);

            let result: Vec1<i32> = s.if_then_else(m_false, yes, no);
            assert_eq!(result.0, 0);

            assert_eq!(s.count_true::<i32>(m_true), 1);
            assert_eq!(s.count_true::<i32>(m_false), 0);
            assert!(s.all_true::<i32>(m_true));
            assert!(s.all_false::<i32>(m_false));
        }
    }

    #[test]
    fn test_saturating_arith() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let a: Vec1<u8> = s.splat(200);
            let b: Vec1<u8> = s.splat(100);

            let sum: Vec1<u8> = s.saturated_add(a, b);
            assert_eq!(sum.0, 255); // clamped

            let diff: Vec1<u8> = s.saturated_sub(Vec1(50u8), Vec1(100u8));
            assert_eq!(diff.0, 0); // clamped
        }
    }

    #[test]
    fn test_load_store() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let data: [u32; 1] = [42];
            let v: Vec1<u32> = s.load(data.as_ptr());
            assert_eq!(v.0, 42);

            let mut out: [u32; 1] = [0];
            s.store(v, out.as_mut_ptr());
            assert_eq!(out[0], 42);
        }
    }

    #[test]
    fn test_float_ops() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let v: Vec1<f32> = s.splat(4.0);
            let sq: Vec1<f32> = s.sqrt(v);
            assert!((sq.0 - 2.0).abs() < 1e-6);

            let v2: Vec1<f32> = s.splat(2.5);
            let r: Vec1<f32> = s.round(v2);
            // rintf uses round-to-nearest-even (banker's rounding): 2.5 -> 2.0
            assert!((r.0 - 2.0).abs() < 1e-6);

            let t: Vec1<f32> = s.trunc(v2);
            assert!((t.0 - 2.0).abs() < 1e-6);

            let c: Vec1<f32> = s.ceil(v2);
            assert!((c.0 - 3.0).abs() < 1e-6);

            let f: Vec1<f32> = s.floor(v2);
            assert!((f.0 - 2.0).abs() < 1e-6);

            // mul_add: 2*3 + 4 = 10
            let a: Vec1<f32> = s.splat(2.0);
            let b: Vec1<f32> = s.splat(3.0);
            let c: Vec1<f32> = s.splat(4.0);
            let ma: Vec1<f32> = s.mul_add(a, b, c);
            assert!((ma.0 - 10.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_nan_inf() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let nan: Vec1<f32> = s.splat(f32::NAN);
            assert!(s.is_nan(nan).to_bool());

            let inf: Vec1<f32> = s.splat(f32::INFINITY);
            assert!(s.is_inf(inf).to_bool());

            let normal: Vec1<f32> = s.splat(1.0);
            assert!(!s.is_nan(normal).to_bool());
            assert!(!s.is_inf(normal).to_bool());
        }
    }

    #[test]
    fn test_promote_demote() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            // u8 -> u16
            let v8: Vec1<u8> = s.splat(200);
            let v16: Vec1<u16> = s.promote_to(v8);
            assert_eq!(v16.0, 200);

            // i32 -> i16 (saturating)
            let v32: Vec1<i32> = s.splat(40000);
            let v16s: Vec1<i16> = s.demote_to(v32);
            assert_eq!(v16s.0, i16::MAX);

            // f32 -> i32
            let vf: Vec1<f32> = s.splat(3.7);
            let vi: Vec1<i32> = s.convert_to_int(vf);
            assert_eq!(vi.0, 3);

            // i32 -> f32
            let vi2: Vec1<i32> = s.splat(42);
            let vf2: Vec1<f32> = s.convert_to_float(vi2);
            assert!((vf2.0 - 42.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reduce() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let v: Vec1<i32> = s.splat(42);
            assert_eq!(s.sum_of_lanes(v), 42);
            assert_eq!(s.min_of_lanes(v), 42);
            assert_eq!(s.max_of_lanes(v), 42);
        }
    }

    #[test]
    fn test_average_round() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let a: Vec1<u8> = s.splat(10);
            let b: Vec1<u8> = s.splat(21);
            let avg: Vec1<u8> = s.average_round(a, b);
            // (10 + 21 + 1) >> 1 = 16
            assert_eq!(avg.0, 16);
        }
    }

    #[test]
    fn test_abs_neg() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let v: Vec1<i32> = s.splat(-5);
            let a: Vec1<i32> = s.abs(v);
            assert_eq!(a.0, 5);

            let n: Vec1<i32> = s.neg(v);
            assert_eq!(n.0, 5);

            let vf: Vec1<f32> = s.splat(-2.75);
            let af: Vec1<f32> = s.abs(vf);
            assert!((af.0 - 2.75).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bitcast() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let v: Vec1<f32> = s.splat(1.0_f32);
            let u: Vec1<u32> = s.bitcast(v);
            assert_eq!(u.0, 1.0_f32.to_bits());
        }
    }

    #[test]
    fn test_mul_high() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let a: Vec1<u16> = s.splat(0x1234);
            let b: Vec1<u16> = s.splat(0x5678);
            let hi: Vec1<u16> = s.mul_high(a, b);
            let expected = ((0x1234u32 * 0x5678u32) >> 16) as u16;
            assert_eq!(hi.0, expected);
        }
    }

    #[test]
    fn test_copy_sign() {
        let s = Scalar;
        // SAFETY: transmute_copy between same-sized types; type identity verified via is_type or size check.
        unsafe {
            let mag: Vec1<f32> = s.splat(5.0);
            let sign: Vec1<f32> = s.splat(-1.0);
            let result: Vec1<f32> = s.copy_sign(mag, sign);
            assert!((result.0 - (-5.0)).abs() < 1e-6);
        }
    }
}
