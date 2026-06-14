//! Safety-invariant tests for the safe public API.
//!
//! These verify the *runtime* guarantees that make the value/slice API safe:
//! the bounds-checked slice loads/stores panic (rather than invoking UB) on
//! too-short or misaligned inputs. Token unforgeability is covered by the
//! `compile_fail` doctests in the crate root.

use highway::{SimdOps, TargetId, WithSimd, dispatch, dispatch_to};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Kernel: `load_slice` from `data` (expected to panic when too short).
struct LoadSlice<'a> {
    data: &'a [f32],
}
impl WithSimd for LoadSlice<'_> {
    type Output = f32;
    fn with_simd<S: SimdOps>(self, s: S) -> f32 {
        let v = s.load_slice(self.data);
        s.get_lane(v)
    }
}

/// Kernel: `store_slice` into `out` (expected to panic when too short).
struct StoreSlice<'a> {
    out: &'a mut [f32],
}
impl WithSimd for StoreSlice<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) {
        let v = s.splat::<f32>(1.0);
        s.store_slice(v, self.out);
    }
}

/// Kernel: `load_aligned_slice` from `data` (panics when too short or misaligned).
struct LoadAligned<'a> {
    data: &'a [f32],
}
impl WithSimd for LoadAligned<'_> {
    type Output = f32;
    fn with_simd<S: SimdOps>(self, s: S) -> f32 {
        let v = s.load_aligned_slice(self.data);
        s.get_lane(v)
    }
}

// ---------------------------------------------------------------------------
// Bounds checks — must panic, never read/write out of bounds (works on all
// targets because an empty slice is shorter than `lanes` for every target).
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "load_slice")]
fn load_slice_panics_on_short() {
    let empty: [f32; 0] = [];
    dispatch(LoadSlice { data: &empty });
}

#[test]
#[should_panic(expected = "store_slice")]
fn store_slice_panics_on_short() {
    let mut empty: [f32; 0] = [];
    dispatch(StoreSlice { out: &mut empty });
}

#[test]
#[should_panic(expected = "load_aligned_slice")]
fn load_aligned_slice_panics_on_short() {
    let empty: [f32; 0] = [];
    dispatch(LoadAligned { data: &empty });
}

// A length that satisfies the count check but is still too short by one for a
// wide target also must panic. Use 1 element: scalar(1 lane) succeeds, so guard.
#[test]
fn load_slice_exact_length_ok() {
    // Exactly `lanes` elements must succeed on every target.
    for target in [
        TargetId::Scalar,
        TargetId::Sse2,
        TargetId::Avx2,
        TargetId::Avx512,
    ] {
        // lanes for f32 on this target (forced; falls back to scalar if absent)
        struct Lanes;
        impl WithSimd for Lanes {
            type Output = usize;
            fn with_simd<S: SimdOps>(self, s: S) -> usize {
                s.lanes::<f32>()
            }
        }
        let lanes = dispatch_to(Lanes, target);
        let data: Vec<f32> = (0..lanes).map(|i| i as f32).collect();
        let first = dispatch_to(LoadSlice { data: &data }, target);
        assert_eq!(first, 0.0, "load_slice exact length on {target:?}");
    }
}

// ---------------------------------------------------------------------------
// Alignment check — `load_aligned_slice` must panic on a misaligned slice
// rather than executing an aligned load on an unaligned pointer (UB).
// Only meaningful on accelerated targets (scalar VECTOR_BYTES == 1).
// ---------------------------------------------------------------------------

#[test]
fn load_aligned_slice_panics_on_misaligned() {
    let best = highway::dispatch::detect_best_target();
    if !best.is_accelerated() {
        return; // scalar-only CPU: alignment is always satisfied (VECTOR_BYTES==1)
    }

    // Build a buffer and take a sub-slice offset by one f32 (4 bytes). For any
    // accelerated target (VECTOR_BYTES >= 16) this start is misaligned.
    let buf: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let misaligned = &buf[1..]; // +4 bytes from buf start

    let result =
        std::panic::catch_unwind(|| dispatch(LoadAligned { data: misaligned }));
    assert!(
        result.is_err(),
        "load_aligned_slice must panic on a misaligned slice (target {best:?})"
    );
}

// Sanity: an actually-aligned slice does NOT panic (uses AlignedVec).
#[test]
fn load_aligned_slice_ok_when_aligned() {
    use highway::aligned_vec_from_slice;
    let src: Vec<f32> = (0..256).map(|i| i as f32).collect();
    let aligned = aligned_vec_from_slice(&src);
    let first = dispatch(LoadAligned { data: &aligned });
    assert_eq!(first, 0.0);
}
