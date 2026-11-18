#![allow(clippy::undocumented_unsafe_blocks)]
//! `Cpu` detects features once and dispatches without per-call checks.

use highway::{Cpu, SimdOps, TargetId, WithSimd, dispatch, simd_fn};

struct SumKernel<'a> {
    data: &'a [u32],
}

impl WithSimd for SumKernel<'_> {
    type Output = (u32, usize);

    fn with_simd<S: SimdOps>(self, s: S) -> Self::Output {
        let lanes = s.lanes::<u32>();
        let mut acc = s.zero::<u32>();
        let mut chunks = self.data.chunks_exact(lanes);
        for chunk in &mut chunks {
            acc = s.add(acc, s.load_slice(chunk));
        }
        let total =
            s.sum_of_lanes(acc) + chunks.remainder().iter().sum::<u32>();
        (total, lanes)
    }
}

#[test]
fn detect_matches_the_cached_target() {
    let cpu = Cpu::detect();
    assert_eq!(cpu.target(), highway::dispatch::detect_best_target());
    assert_ne!(cpu.target(), TargetId::Uninitialized);
}

#[test]
fn tier_extensions_match_the_cpu() {
    let cpu = Cpu::detect();
    #[cfg(target_arch = "x86_64")]
    for (feature, flag) in [
        ("ssse3", highway::features::SSSE3),
        ("sse4.1", highway::features::SSE41),
        ("aes", highway::features::AES),
        ("pclmulqdq", highway::features::PCLMUL),
        ("avx512vbmi", highway::features::VBMI),
        ("avx512vbmi2", highway::features::VBMI2),
        ("avx512bitalg", highway::features::BITALG),
        ("avx512vpopcntdq", highway::features::VPOPCNTDQ),
    ] {
        let detected = match feature {
            "ssse3" => is_x86_feature_detected!("ssse3"),
            "sse4.1" => is_x86_feature_detected!("sse4.1"),
            "aes" => is_x86_feature_detected!("aes"),
            "pclmulqdq" => is_x86_feature_detected!("pclmulqdq"),
            "avx512vbmi" => is_x86_feature_detected!("avx512vbmi"),
            "avx512vbmi2" => is_x86_feature_detected!("avx512vbmi2"),
            "avx512bitalg" => is_x86_feature_detected!("avx512bitalg"),
            _ => is_x86_feature_detected!("avx512vpopcntdq"),
        };
        if cpu.has(flag) {
            assert!(
                detected,
                "tier {:?} claims {feature} but the cpu lacks it",
                cpu.tier()
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(cpu.tier(), highway::Tier::Scalar);
}

#[test]
fn capped_never_claims_more_than_the_cpu() {
    let cpu = Cpu::detect();
    for max in highway::Tier::ALL {
        let capped = cpu.capped(max);
        assert!(capped.tier() <= max, "{max:?} -> {:?}", capped.tier());
        assert!(
            cpu.has(capped.tier().extensions()),
            "{max:?} -> {:?}",
            capped.tier()
        );
        assert!(capped.target() as u8 <= cpu.target() as u8);
    }
    assert_eq!(
        cpu.capped(highway::Tier::Scalar).tier(),
        highway::Tier::Scalar
    );
    assert_eq!(cpu.capped(cpu.tier()), cpu);
}

#[test]
fn every_tier_runs_the_same_kernel() {
    let data: Vec<u32> = (0..1000).collect();
    let cpu = Cpu::detect();
    for max in highway::Tier::ALL {
        assert_eq!(
            cpu.capped(max).dispatch(SumKernel { data: &data }).0,
            (0..1000).sum::<u32>()
        );
    }
}

#[test]
fn cpu_dispatch_matches_dispatch() {
    let data: Vec<u32> = (0..1000).collect();
    let cpu = Cpu::detect();
    assert_eq!(
        cpu.dispatch(SumKernel { data: &data }),
        dispatch(SumKernel { data: &data })
    );
    assert_eq!(
        cpu.dispatch(SumKernel { data: &data }).0,
        (0..1000).sum::<u32>()
    );
}

#[test]
fn simd_fn_takes_a_startup_cpu() {
    let cpu = Cpu::detect();
    let plain: usize = simd_fn!(|s| -> usize { s.lanes::<u8>() });
    let with_cpu: usize = simd_fn!(cpu => |s| -> usize { s.lanes::<u8>() });
    assert_eq!(plain, with_cpu);
    let base = 40u32;
    let captured: u32 = simd_fn!(cpu => [base: u32] |s| -> u32 { base + u32::try_from(s.lanes::<u64>()).unwrap() });
    assert!(captured > base);
}

#[test]
fn resolved_entry_matches_dispatch_on_every_tier() {
    let data: Vec<u32> = (0..777).collect();
    let cpu = Cpu::detect();
    for max in highway::Tier::ALL {
        let capped = cpu.capped(max);
        let resolved = capped.resolve::<SumKernel<'_>>();
        assert_eq!(
            resolved.call(SumKernel { data: &data }),
            capped.dispatch(SumKernel { data: &data })
        );
    }
}
