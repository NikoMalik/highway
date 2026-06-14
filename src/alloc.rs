/// Aligned memory allocator for SIMD operations.
///
/// Provides cache-line and vector-register aligned allocation via the
/// `allocator-api2` crate. Default alignment is 128 bytes, covering
/// both cache lines and AVX-512 vector widths.
use allocator_api2::alloc::{AllocError, Allocator, Layout};
use core::ptr::NonNull;

/// An allocator that guarantees at least `ALIGN`-byte alignment.
///
/// Default `ALIGN` is 128 bytes, which satisfies:
/// - Typical cache line size (64 bytes)
/// - AVX-512 vector width (64 bytes)
/// - Future 1024-bit vectors
///
/// # Example
/// ```ignore
/// use highway::alloc::{AlignedAlloc, AlignedVec};
/// let mut v: AlignedVec<f32> = AlignedVec::new_in(AlignedAlloc::<128>);
/// v.push(1.0);
/// assert!(v.as_ptr() as usize % 128 == 0);
/// ```
pub struct AlignedAlloc<const ALIGN: usize = 128>;

// SAFETY: We delegate to the global allocator but enforce at least ALIGN-byte
// alignment on every allocation. The global allocator is sound, and we never
// violate the Layout contract.
unsafe impl<const ALIGN: usize> Allocator for AlignedAlloc<ALIGN> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let align = layout.align().max(ALIGN);
        let layout = Layout::from_size_align(layout.size(), align)
            .map_err(|_| AllocError)?;
        // SAFETY: layout is valid (size > 0 guaranteed by caller or we handle zero-size).
        if layout.size() == 0 {
            // Return a dangling pointer for zero-size allocations.
            let ptr = NonNull::new(align as *mut u8).ok_or(AllocError)?;
            return Ok(NonNull::slice_from_raw_parts(ptr, 0));
        }
        // SAFETY: We use the global allocator with a valid, non-zero-size layout.
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or(AllocError)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        if layout.size() == 0 {
            return;
        }
        let align = layout.align().max(ALIGN);
        let layout = Layout::from_size_align(layout.size(), align)
            .expect("AlignedAlloc: invalid layout in deallocate");
        // SAFETY: Caller guarantees ptr was allocated by this allocator with this layout.
        unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) }
    }
}

/// A `Vec` type alias that uses `AlignedAlloc` for over-aligned allocation.
///
/// The default alignment is 128 bytes.
pub type AlignedVec<T, const ALIGN: usize = 128> =
    allocator_api2::vec::Vec<T, AlignedAlloc<ALIGN>>;

/// Create a new empty `AlignedVec` with default alignment.
pub fn aligned_vec<T>() -> AlignedVec<T, 128> {
    AlignedVec::new_in(AlignedAlloc::<128>)
}

/// Create a new empty `AlignedVec` with the specified capacity.
pub fn aligned_vec_with_capacity<T>(cap: usize) -> AlignedVec<T, 128> {
    AlignedVec::with_capacity_in(cap, AlignedAlloc::<128>)
}

/// Create an `AlignedVec` by copying from a slice.
///
/// The resulting vec is 128-byte aligned, making it safe to use with
/// aligned SIMD loads (`Simd::load`).
pub fn aligned_vec_from_slice<T: Clone>(src: &[T]) -> AlignedVec<T, 128> {
    let mut v = aligned_vec_with_capacity(src.len());
    v.extend_from_slice(src);
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment() {
        let mut v: AlignedVec<f32, 128> = aligned_vec_with_capacity(64);
        v.push(1.0);
        let ptr = v.as_ptr() as usize;
        assert_eq!(ptr % 128, 0, "pointer should be 128-byte aligned");
    }

    #[test]
    fn test_custom_alignment() {
        let mut v: AlignedVec<u8, 64> =
            AlignedVec::with_capacity_in(256, AlignedAlloc::<64>);
        v.push(42);
        let ptr = v.as_ptr() as usize;
        assert_eq!(ptr % 64, 0, "pointer should be 64-byte aligned");
    }

    #[test]
    fn test_push_and_read() {
        let mut v: AlignedVec<i32> = aligned_vec();
        for i in 0..100 {
            v.push(i);
        }
        for i in 0..100 {
            assert_eq!(v[i as usize], i);
        }
    }

    #[test]
    fn test_zero_size() {
        let v: AlignedVec<()> = aligned_vec_with_capacity(10);
        assert_eq!(v.len(), 0);
    }
}
