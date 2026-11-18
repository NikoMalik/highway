//! Optional CPU extensions a backend token can carry in its `F` parameter.
//!
//! A token such as `Sse2<{ SSSE3 | PCLMUL }>` is only handed out by
//! [`Cpu`](crate::Cpu) after the matching runtime check at startup, so
//! operations test `F` at compile time instead of probing the CPU per call.

/// SSSE3 (`pshufb`, `pmulhrsw`, `pmaddubsw`).
pub const SSSE3: u32 = 1 << 0;
/// SSE4.1 (`pmuldq` and friends).
pub const SSE41: u32 = 1 << 1;
/// AES-NI round instructions.
pub const AES: u32 = 1 << 2;
/// Carry-less multiply (`pclmulqdq`).
pub const PCLMUL: u32 = 1 << 3;
/// AVX-512 VBMI (byte permutes).
pub const VBMI: u32 = 1 << 4;
/// AVX-512 VBMI2 (byte and word compress).
pub const VBMI2: u32 = 1 << 5;
/// AVX-512 BITALG (byte and word popcount).
pub const BITALG: u32 = 1 << 6;
/// AVX-512 VPOPCNTDQ (dword and qword popcount).
pub const VPOPCNTDQ: u32 = 1 << 7;
