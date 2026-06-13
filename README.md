# highway

A Rust port of [Google's Highway](https://github.com/google/highway) SIMD library.

Portable SIMD operations with runtime CPU detection and dispatch to the best available target.

## Targets

| Target | Vector width | Requirement |
|--------|-------------|-------------|
| Scalar | 1 lane | All platforms |
| SSE2 | 128-bit | x86_64 |
| AVX2 | 256-bit | x86_64 + AVX2 + FMA |
| AVX-512 | 512-bit | x86_64 + AVX-512F/BW/DQ/VL |

Target selection is automatic at runtime via `dispatch()`. No compile-time feature flags needed -- all backends are compiled in and the best one is selected on first call.


### Writing a SIMD Kernel

Implement the `WithSimd` trait. The `dispatch()` function calls `with_simd` on the best available backend:

```rust
use highway::{dispatch, WithSimd, SimdOps};

struct SumKernel<'a> {
    data: &'a [f32],
}

impl WithSimd for SumKernel<'_> {
    type Output = f32;

    fn with_simd<S: SimdOps>(self, s: S) -> f32 {
        let lanes = s.lanes::<f32>();
        let mut i = 0;
        let mut acc = unsafe { s.zero::<f32>() };

        // Vectorized loop
        while i + lanes <= self.data.len() {
            unsafe {
                let v = s.load_u(self.data.as_ptr().add(i));
                acc = s.add(acc, v);
            }
            i += lanes;
        }

        // Horizontal reduction
        let mut total = unsafe { s.sum_of_lanes(acc) };

        // Scalar tail
        while i < self.data.len() {
            total += self.data[i];
            i += 1;
        }
        total
    }
}

let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
let sum = dispatch(SumKernel { data: &data });
assert_eq!(sum, (0..1024).sum::<i32>() as f32);
```

### `simd_fn!` — Inline Kernels Without Boilerplate

The `simd_fn!` macro lets you write SIMD code inline without defining a separate struct:

```rust
use highway::simd_fn;

let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
let ptr = data.as_ptr();
let len = data.len();

let sum: f32 = simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
    let lanes = s.lanes::<f32>();
    let mut acc = unsafe { s.zero::<f32>() };
    let mut i = 0;
    while i + lanes <= len {
        unsafe {
            let v = s.load_u(ptr.add(i));
            acc = s.add(acc, v);
        }
        i += lanes;
    }
    let mut total = unsafe { s.sum_of_lanes(acc) };
    while i < len {
        total += unsafe { *ptr.add(i) };
        i += 1;
    }
    total
});

assert_eq!(sum, 36.0);
```

Two forms:

| Syntax | When to use |
|--------|------------|
| `simd_fn!(\|s\| -> T { ... })` | No external data needed |
| `simd_fn!([a: A, b: B] \|s\| -> T { ... })` | Capture local variables with types |

Captures require explicit types because `macro_rules!` generates a struct internally and Rust struct fields need types. The macro expands to a `struct` + `impl WithSimd` + `dispatch()` call — zero overhead, full monomorphization for each backend.

### SIMD Methods on a Struct

The `simd_fn!` macro is designed for adding SIMD-accelerated methods to your types. Extract `self` fields into locals, then capture them:

```rust
use highway::{simd_fn, SimdOps};

struct AudioBuffer {
    samples: Vec<f32>,
}

impl AudioBuffer {
    fn peak(&self) -> f32 {
        let ptr = self.samples.as_ptr();
        let len = self.samples.len();
        simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
            let lanes = s.lanes::<f32>();
            let mut best = unsafe { s.splat(0.0f32) };
            let mut i = 0;
            while i + lanes <= len {
                unsafe {
                    let v = s.load_u(ptr.add(i));
                    let a = s.abs(v);
                    best = s.max(best, a);
                }
                i += lanes;
            }
            let mut peak = unsafe { s.max_of_lanes(best) };
            while i < len {
                let a = unsafe { *ptr.add(i) }.abs();
                if a > peak { peak = a; }
                i += 1;
            }
            peak
        })
    }

    fn sum(&self) -> f32 {
        let ptr = self.samples.as_ptr();
        let len = self.samples.len();
        simd_fn!([ptr: *const f32, len: usize] |s| -> f32 {
            let lanes = s.lanes::<f32>();
            let mut acc = unsafe { s.zero::<f32>() };
            let mut i = 0;
            while i + lanes <= len {
                unsafe {
                    let v = s.load_u(ptr.add(i));
                    acc = s.add(acc, v);
                }
                i += lanes;
            }
            let mut total = unsafe { s.sum_of_lanes(acc) };
            while i < len {
                total += unsafe { *ptr.add(i) };
                i += 1;
            }
            total
        })
    }

    fn find(&self, needle: f32) -> Option<usize> {
        let ptr = self.samples.as_ptr();
        let len = self.samples.len();
        simd_fn!([ptr: *const f32, len: usize, needle: f32] |s| -> Option<usize> {
            let lanes = s.lanes::<f32>();
            let needle_v = unsafe { s.splat(needle) };
            let mut i = 0;
            while i + lanes <= len {
                unsafe {
                    let v = s.load_u(ptr.add(i));
                    let mask = s.eq(v, needle_v);
                    if let Some(idx) = s.find_first_true(mask) {
                        return Some(i + idx);
                    }
                }
                i += lanes;
            }
            // Scalar tail
            while i < len {
                if unsafe { *ptr.add(i) } == needle {
                    return Some(i);
                }
                i += 1;
            }
            None
        })
    }
}

let buf = AudioBuffer {
    samples: vec![0.1, -0.5, 0.3, 0.8, -0.2, 0.5, 0.0, 0.4],
};
assert_eq!(buf.peak(), 0.8);
assert_eq!(buf.sum(), 1.4000001); // f32 precision
assert_eq!(buf.find(0.3), Some(2));
assert_eq!(buf.find(9.9), None);
```

Each method gets its own SIMD kernel — no separate top-level structs needed. For complex kernels with lifetimes or generics, fall back to a manual `struct` + `impl WithSimd`.

### `WithSimd` vs `simd_fn!` — When to Use What

| Use case | Approach |
|----------|---------|
| Quick one-off SIMD operation | `simd_fn!` |
| Struct methods (`.sum()`, `.find()`, ...) | `simd_fn!` with captures |
| Kernel reused in multiple places | `struct` + `impl WithSimd` |
| Generic over element type (`T: Lane`) | `struct` + `impl WithSimd` |
| Needs lifetimes on captures | `struct` + `impl WithSimd` |

### Explicit Target Selection

Use `dispatch_to()` to force a specific backend (useful for testing or benchmarking):

```rust
use highway::{dispatch_to, TargetId, WithSimd, SimdOps};

struct MyKernel;
impl WithSimd for MyKernel {
    type Output = &'static str;
    fn with_simd<S: SimdOps>(self, _s: S) -> &'static str {
        std::any::type_name::<S>()
    }
}

let name = dispatch_to(MyKernel, TargetId::Scalar);
assert!(name.contains("Scalar"));
```

### Available Operations (~190 functions)

Operations are grouped into sub-traits on the `SimdOps` supertrait.

All SIMD operations are `unsafe fn` because they use platform intrinsics internally.

**SimdCore** (8) -- vector construction and element access:
`zero`, `splat`, `undefined`, `bitcast`, `extract_lane`, `insert_lane`, `get_lane`, `iota`

**SimdMemory** (17) -- load, store, gather/scatter, interleaved memory access:
`load`, `load_u`, `store`, `store_u`, `stream`, `load_dup128`, `masked_load`, `blended_store`, `gather_index`, `scatter_index`, `load_interleaved_2`, `load_interleaved_3`, `load_interleaved_4`, `store_interleaved_2`, `store_interleaved_3`, `store_interleaved_4`, `load_expand`

**SimdArith** (27) -- arithmetic, saturation, widening multiply, masked variants:
`add`, `sub`, `mul`, `div`, `saturated_add`, `saturated_sub`, `saturated_neg`, `saturated_abs`, `abs`, `neg`, `min`, `max`, `mul_high`, `average_round`, `abs_diff`, `clamp`, `mul_even`, `mul_odd`, `widen_mul_pairwise_add_i16`, `sat_widen_mul_pairwise_add`, `mul_fixed_point_15`, `reorder_widen_mul_accumulate`, `masked_min_or`, `masked_max_or`, `masked_add_or`, `masked_sub_or`, `masked_mul_or`

**SimdBitwise** (23) -- bitwise ops, shifts, rotates, byte/bit reversal:
`and`, `or`, `xor`, `not`, `and_not`, `shift_left`, `shift_right`, `rotate_right`, `rotate_left`, `shift_left_same`, `shift_right_same`, `shift_left_bytes`, `shift_right_bytes`, `population_count`, `leading_zero_count`, `trailing_zero_count`, `reverse_lane_bytes`, `reverse_bits`, `broadcast_sign_bit`, `shl`, `shr`, `ror`, `rol`

**SimdCompare** (7) -- lane-wise comparisons:
`eq`, `ne`, `lt`, `le`, `gt`, `ge`, `test_bit`

**SimdMask** (22) -- mask creation, queries, boolean ops, sign-bit selects:
`mask_from_vec`, `vec_from_mask`, `first_n`, `count_true`, `all_true`, `all_false`, `find_first_true`, `find_last_true`, `if_then_else`, `if_then_else_zero`, `if_then_zero_else`, `and_mask`, `or_mask`, `not_mask`, `xor_mask`, `bits_from_mask`, `exclusive_neither`, `slide_mask_1_up`, `slide_mask_1_down`, `if_negative_then_else`, `if_negative_then_else_zero`, `if_negative_then_zero_else`

**SimdConvert** (13) -- type promotion, demotion, float/int conversion:
`promote_to`, `promote_lower_to`, `promote_upper_to`, `demote_to`, `demote_in_range_to`, `convert_to_int`, `convert_in_range_to_int`, `convert_to_float`, `truncate_to`, `ordered_demote_2_to`, `reorder_demote_2_to`, `ordered_truncate_2_to`, `nearest_int`

**SimdShuffle** (47) -- lane rearrangement, compress, half-vector & block ops:
`reverse`, `broadcast_lane`, `interleave_lower`, `interleave_upper`, `interleave_whole_lower`, `interleave_whole_upper`, `interleave_even`, `interleave_odd`, `zip_lower`, `zip_upper`, `table_lookup_bytes`, `table_lookup_lanes`, `table_lookup_lanes_or0`, `two_tables_lookup_lanes`, `reverse2`, `reverse4`, `reverse8`, `concat_upper_lower`, `concat_lower_upper`, `concat_lower_lower`, `concat_upper_upper`, `concat_even`, `concat_odd`, `odd_even`, `odd_even_blocks`, `reverse_blocks`, `slide_up_lanes`, `slide_down_lanes`, `slide_1_up`, `slide_1_down`, `dup_even`, `dup_odd`, `combine_shift_right_bytes`, `lower_half`, `upper_half`, `combine`, `insert_block`, `extract_block`, `broadcast_block`, `compress`, `compress_store`, `compress_not`, `compress_blocks_not`, `compress_blended_store`, `compress_bits`, `compress_bits_store`, `expand`

**SimdReduce** (6) -- horizontal reductions:
`sum_of_lanes`, `min_of_lanes`, `max_of_lanes`, `sums_of_8_abs_diff`, `sums_of_2`, `sums_of_4`

**SimdFloat** (22) -- float-specific ops (rounding, FMA, min/max variants, classification):
`sqrt`, `approx_reciprocal`, `approx_reciprocal_sqrt`, `round`, `trunc`, `ceil`, `floor`, `mul_add`, `neg_mul_add`, `mul_sub`, `neg_mul_sub`, `copy_sign`, `is_nan`, `is_inf`, `is_finite`, `is_either_nan`, `zero_if_negative`, `add_sub`, `min_number`, `max_number`, `min_magnitude`, `max_magnitude`

**SimdCrypto** (8) -- AES and carry-less multiply (separate trait, not part of `SimdOps`):
`aes_round`, `aes_last_round`, `aes_round_inv`, `aes_last_round_inv`, `aes_key_gen_assist`, `aes_inv_mix_columns`, `cl_mul_lower`, `cl_mul_upper`

### Not Yet Ported from C++ Highway

The following C++ Highway operations have no Rust equivalent yet:

**Type support**
- `float16_t` / `bfloat16_t` -- half-precision and bfloat16 lane types with all associated operations

**Resizing / construction**
- `ResizeBitCast` -- cast between different vector sizes
- `ConcatSubvec` -- concatenate sub-vectors
- `Dup128VecFromValues` -- create vector by duplicating a 128-bit pattern of individual values

**Conversion**
- `PromoteInRangeTo` / `PromoteInRangeUpperTo` -- promote with in-range assumption
- `ConvertTo` for additional type pairs -- u16/i16->f32, u32->f32, i64->f64, u64->f64

**Shuffle / permute**
- `Per4LaneBlockShuffle` -- shuffles within 4-lane blocks
- `MultiRotateRight` -- per-lane multi-rotate (AVX-512 VBMI2)
- `BitShuffle` -- per-lane bit gather

**Mask**
- `CombineMasks` / `UpperHalfOfMask` -- combine/split half-width masks

**Reduction / arithmetic**
- `SumsOfAdjQuadAbsDiff` -- sum of adjacent quad absolute differences (AVX-512 `dbsad`)
- `SumsOfShuffledQuadAbsDiff` -- shuffled quad SAD variant
- `HighestSetBitIndex` -- index of the highest set bit per lane
- `RearrangeToOddPlusEven` -- pairwise odd+even recombination

**Cryptography**
- `GaloisAffine` / `GaloisAffineInverse` -- GF(2^8) affine transforms (GFNI)
- AVX-512 full-width VAES / VPCLMULQDQ (current AES/CLMul split into 128-bit blocks)

### Aligned Memory with `AlignedVec`

Standard `Vec` does not guarantee SIMD-friendly alignment. Use `AlignedVec` for aligned loads (`load`) instead of unaligned loads (`load_u`):

```rust
use highway::{aligned_vec_from_slice, aligned_vec_with_capacity, dispatch, WithSimd, SimdOps};

// Create from existing data
let data = aligned_vec_from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
assert_eq!(data.as_ptr() as usize % 128, 0); // 128-byte aligned

// Create with capacity, then fill
let mut output = aligned_vec_with_capacity::<f32>(64);
output.resize(64, 0.0);

struct AlignedKernel<'a> {
    input: &'a [f32],
    output: &'a mut [f32],
}

impl WithSimd for AlignedKernel<'_> {
    type Output = ();
    fn with_simd<S: SimdOps>(self, s: S) -> () {
        let lanes = s.lanes::<f32>();
        let mut i = 0;
        while i + lanes <= self.input.len() {
            unsafe {
                // Aligned load -- requires pointer aligned to vector width.
                // AlignedVec guarantees 128-byte alignment, which exceeds
                // the requirement of any target (max 64 bytes for AVX-512).
                let v = s.load(self.input.as_ptr().add(i));
                let doubled = s.add(v, v);
                s.store(doubled, self.output.as_mut_ptr().add(i));
            }
            i += lanes;
        }
        // scalar tail
        while i < self.input.len() {
            self.output[i] = self.input[i] * 2.0;
            i += 1;
        }
    }
}
```

**When to use `load` vs `load_u`:**

- `load` (aligned) -- pointer must be aligned to the vector width (16/32/64 bytes). Use with `AlignedVec` or data you know is aligned. Faster on some architectures.
- `load_u` (unaligned) -- works with any pointer. Use with standard `Vec`, slices from arbitrary sources, or when alignment is unknown.

Same applies to `store` vs `store_u`.

### Supported Lane Types

`u8`, `u16`, `u32`, `u64`, `i8`, `i16`, `i32`, `i64`, `f32`, `f64`

Not all operations are available for all lane types. For instance, `div` is only for floats, `saturated_add` is for `u8`/`u16`/`i8`/`i16`.

## C++ Highway vs Rust highway

Side-by-side comparison using a real-world example: SIMD-accelerated Unicode codepoint width lookup (from [Ghostty](https://github.com/ghostty-org/ghostty)).

### C++ (Google Highway)

```cpp
template <class D, typename T = uint16_t>
int8_t CodepointWidth16(D d, uint16_t input) {
    const size_t N = hn::Lanes(d);
    const hn::Vec<D> input_vec = Set(d, input);

    HWY_ALIGN constexpr T gte_keys[] = { 0x2E3A, 0x3400, 0x4E00, 0xF900, /* ... */ };
    HWY_ALIGN constexpr T lte_keys[] = { 0x2E3A, 0x4DBF, 0x9FFF, 0xFAFF, /* ... */ };

    size_t i = 0;
    for (; i + N <= array_size(lte_keys) && lte_keys[i] != 0; i += N) {
        const hn::Vec<D> lte_vec = hn::Load(d, lte_keys + i);
        const hn::Vec<D> gte_vec = hn::Load(d, gte_keys + i);
        const intptr_t idx = hn::FindFirstTrue(
            d, hn::And(hn::Le(input_vec, lte_vec), hn::Ge(input_vec, gte_vec)));

        if (idx >= 5) return 0;       // zero-width
        else if (idx >= 0) return 2;  // wide
    }

    return 1; // normal width
}
```

### Rust (highway)

```rust
use highway::{dispatch, WithSimd, SimdOps};

struct WidthKernel { input: u16 }

impl WithSimd for WidthKernel {
    type Output = i8;

    fn with_simd<S: SimdOps>(self, s: S) -> i8 {
        let n = s.lanes::<u16>();
        let input_vec = unsafe { s.splat::<u16>(self.input) };

        static GTE_KEYS: &[u16] = &[0x2E3A, 0x3400, 0x4E00, 0xF900, /* ... */];
        static LTE_KEYS: &[u16] = &[0x2E3A, 0x4DBF, 0x9FFF, 0xFAFF, /* ... */];

        let mut i = 0;
        while i + n <= LTE_KEYS.len() && LTE_KEYS[i] != 0 {
            unsafe {
                let lte_vec = s.load_u(LTE_KEYS.as_ptr().add(i));
                let gte_vec = s.load_u(GTE_KEYS.as_ptr().add(i));
                let in_range = s.and_mask(s.le(input_vec, lte_vec),
                                          s.ge(input_vec, gte_vec));
                if let Some(idx) = s.find_first_true(in_range) {
                    if idx >= 5 { return 0; }      // zero-width
                    else        { return 2; }       // wide
                }
            }
            i += n;
        }

        1 // normal width
    }
}

let width = dispatch(WidthKernel { input: 0x4E00 }); // CJK Unified Ideograph
assert_eq!(width, 2); // wide
```

### API Mapping

| C++ Highway | Rust highway | Notes |
|---|---|---|
| `hn::Lanes(d)` | `s.lanes::<T>()` | Returns number of lanes for type `T` |
| `Set(d, val)` | `s.splat(val)` | Broadcast scalar to all lanes |
| `hn::Load(d, ptr)` | `s.load(ptr)` / `s.load_u(ptr)` | `load` = aligned, `load_u` = unaligned |
| `hn::Store(v, d, ptr)` | `s.store(v, ptr)` / `s.store_u(v, ptr)` | |
| `hn::Add(a, b)` | `s.add(a, b)` | All ops are methods on `s` |
| `hn::Le(a, b)` | `s.le(a, b)` | Returns `Mask<T>` |
| `hn::And(m1, m2)` | `s.and(a, b)` / `s.and_mask(m1, m2)` | `and` for vectors, `and_mask` for masks |
| `hn::FindFirstTrue(d, m)` | `s.find_first_true(m)` | Returns `Option<usize>` instead of `-1` |
| `hn::IfThenElse(m, a, b)` | `s.if_then_else(m, a, b)` | |
| `hn::SumOfLanes(d, v)` | `s.sum_of_lanes(v)` | Returns scalar `T`, not a vector |
| `hn::MulAdd(a, b, c)` | `s.mul_add(a, b, c)` | FMA: `a * b + c` |
| `HWY_ALIGN T arr[]` | `AlignedVec<T>` / `aligned_vec_from_slice()` | 128-byte alignment by default |
| Template `D d` | Trait bound `S: SimdOps` | `s: S` is the SIMD target value |
| `HWY_DYNAMIC_DISPATCH(func)` | `dispatch(kernel)` | Calls `kernel.with_simd(best_target)` |
| `intptr_t` (-1 = not found) | `Option<usize>` | Rust idiom for optional values |

### Key Differences

- **Free functions vs methods**: C++ Highway uses `hn::Op(d, args...)`. Rust highway uses `s.op(args...)` where `s` is the SIMD target.
- **Tag `d` vs target `s`**: C++ passes a tag type `D d` that selects vector width. Rust uses a trait-bounded target `S: SimdOps` that carries both the target and its operations.
- **Dispatch model**: C++ uses `HWY_DYNAMIC_DISPATCH` macros. Rust uses `dispatch(kernel)` where `kernel` implements `WithSimd`.
- **Masks are separate types**: C++ Highway masks are often the same type as vectors (except AVX-512). Rust separates `Vec<T>` and `Mask<T>` at the type level, with dedicated `and_mask`/`or_mask`/`not_mask` operations.
- **No macros needed**: C++ Highway relies on `HWY_ALIGN`, `HWY_DYNAMIC_DISPATCH`, `HWY_FOREACH_TARGET` macros. Rust highway uses traits, generics, and `AlignedVec` instead.

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `alloc` | Yes | Enables `AlignedVec` and aligned allocation via `allocator-api2` |


The `Simd` trait uses Generic Associated Types (GATs) so that each backend defines its own vector and mask types:

```rust
pub unsafe trait Simd: Copy + Sized + 'static {
    type Vec<T: Lane>: Copy;
    type Mask<T: Lane>: Copy;
    const VECTOR_BYTES: usize;
    fn lanes<T: Lane>(self) -> usize;
}
```

## License

Same as the original Highway library -- Apache 2.0.
