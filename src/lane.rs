//! Lane types for SIMD operations.
//!
//! Each lane type represents a scalar element that can be stored in a SIMD vector.
//! The trait hierarchy captures relationships between types (signed/unsigned,
//! integer/float, narrow/wide) used to constrain operations at compile time.

/// A scalar type that can occupy a single SIMD lane.
///
/// # Safety
/// Implementations must be plain-old-data types with size equal to `BYTES`.
/// `Unsigned` must be the unsigned integer type of the same width, used for
/// bitwise/mask representations.
pub unsafe trait Lane:
    Copy + Default + PartialEq + PartialOrd + 'static
{
    /// Size of this lane type in bytes.
    const BYTES: usize = core::mem::size_of::<Self>();

    /// Unique numeric identifier for this lane type, used for compile-time
    /// type discrimination (since `TypeId` comparison is not yet const-stable).
    const LANE_ID: u8;

    /// The unsigned integer type with the same width, used for mask representation.
    type Unsigned: UnsignedLane;
}

/// Check if lane type `T` is the same as lane type `U` at compile time.
///
/// When monomorphized, this reduces to a constant `true`/`false` and the
/// optimizer eliminates dead branches — giving zero runtime cost.
#[inline(always)]
pub const fn is_type<T: Lane, U: Lane>() -> bool {
    T::LANE_ID == U::LANE_ID
}

/// An integer lane type (signed or unsigned).
///
/// # Safety
/// Must only be implemented for primitive integer types.
pub unsafe trait IntegerLane: Lane {}

/// A signed integer lane type.
///
/// # Safety
/// Must only be implemented for `i8`, `i16`, `i32`, `i64`.
pub unsafe trait SignedLane: IntegerLane {
    /// The unsigned counterpart of the same width.
    type UnsignedOf: UnsignedLane;
}

/// An unsigned integer lane type.
///
/// # Safety
/// Must only be implemented for `u8`, `u16`, `u32`, `u64`.
pub unsafe trait UnsignedLane: IntegerLane {
    /// The signed counterpart of the same width.
    type SignedOf: SignedLane;
}

/// A floating-point lane type.
///
/// # Safety
/// Must only be implemented for `f32`, `f64`.
pub unsafe trait FloatLane: Lane {
    /// The unsigned integer type of the same width (for bit manipulation).
    type Bits: UnsignedLane;
    /// The signed integer type of the same width (for conversions).
    type Int: SignedLane;
}

/// A lane type that can be promoted (widened) to a wider type.
///
/// # Safety
/// `Wide` must be exactly twice the width of `Self`.
pub unsafe trait NarrowLane: Lane {
    /// The wider type (double the width).
    type Wide: WideLane;
}

/// A lane type that can be demoted (narrowed) to a narrower type.
///
/// # Safety
/// `Narrow` must be exactly half the width of `Self`.
pub unsafe trait WideLane: Lane {
    /// The narrower type (half the width).
    type Narrow: NarrowLane;
}

// ---------------------------------------------------------------------------
// Implementations for primitive types
// ---------------------------------------------------------------------------

macro_rules! impl_lane {
    ($ty:ty, $unsigned:ty, $id:expr) => {
        // SAFETY: Primitive type, size matches, Unsigned type is correct.
        unsafe impl Lane for $ty {
            const LANE_ID: u8 = $id;
            type Unsigned = $unsigned;
        }
    };
}

impl_lane!(u8, u8, 0);
impl_lane!(u16, u16, 1);
impl_lane!(u32, u32, 2);
impl_lane!(u64, u64, 3);
impl_lane!(i8, u8, 4);
impl_lane!(i16, u16, 5);
impl_lane!(i32, u32, 6);
impl_lane!(i64, u64, 7);
impl_lane!(f32, u32, 8);
impl_lane!(f64, u64, 9);

// SAFETY: All of the following are primitive integer types.
macro_rules! impl_integer_lane {
    ($($ty:ty),+) => {
        $(
            // SAFETY: Primitive integer type.
            unsafe impl IntegerLane for $ty {}
        )+
    };
}

impl_integer_lane!(u8, u16, u32, u64, i8, i16, i32, i64);

// Unsigned lanes
macro_rules! impl_unsigned_lane {
    ($u:ty, $s:ty) => {
        // SAFETY: Primitive unsigned integer with correct signed counterpart.
        unsafe impl UnsignedLane for $u {
            type SignedOf = $s;
        }
    };
}

impl_unsigned_lane!(u8, i8);
impl_unsigned_lane!(u16, i16);
impl_unsigned_lane!(u32, i32);
impl_unsigned_lane!(u64, i64);

// Signed lanes
macro_rules! impl_signed_lane {
    ($s:ty, $u:ty) => {
        // SAFETY: Primitive signed integer with correct unsigned counterpart.
        unsafe impl SignedLane for $s {
            type UnsignedOf = $u;
        }
    };
}

impl_signed_lane!(i8, u8);
impl_signed_lane!(i16, u16);
impl_signed_lane!(i32, u32);
impl_signed_lane!(i64, u64);

// Float lanes
// SAFETY: f32 is 4 bytes, matching u32/i32.
unsafe impl FloatLane for f32 {
    type Bits = u32;
    type Int = i32;
}

// SAFETY: f64 is 8 bytes, matching u64/i64.
unsafe impl FloatLane for f64 {
    type Bits = u64;
    type Int = i64;
}

// Narrow/Wide relationships
macro_rules! impl_narrow_wide {
    ($narrow:ty, $wide:ty) => {
        // SAFETY: $wide is exactly twice the width of $narrow.
        unsafe impl NarrowLane for $narrow {
            type Wide = $wide;
        }
        // SAFETY: $narrow is exactly half the width of $wide.
        unsafe impl WideLane for $wide {
            type Narrow = $narrow;
        }
    };
}

impl_narrow_wide!(u8, u16);
impl_narrow_wide!(u16, u32);
impl_narrow_wide!(u32, u64);
impl_narrow_wide!(i8, i16);
impl_narrow_wide!(i16, i32);
impl_narrow_wide!(i32, i64);
impl_narrow_wide!(f32, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lane_bytes() {
        assert_eq!(<u8 as Lane>::BYTES, 1);
        assert_eq!(<u16 as Lane>::BYTES, 2);
        assert_eq!(<u32 as Lane>::BYTES, 4);
        assert_eq!(<u64 as Lane>::BYTES, 8);
        assert_eq!(<i8 as Lane>::BYTES, 1);
        assert_eq!(<i16 as Lane>::BYTES, 2);
        assert_eq!(<i32 as Lane>::BYTES, 4);
        assert_eq!(<i64 as Lane>::BYTES, 8);
        assert_eq!(<f32 as Lane>::BYTES, 4);
        assert_eq!(<f64 as Lane>::BYTES, 8);
    }
}
