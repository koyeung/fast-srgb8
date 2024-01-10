use super::TO_SRGB8_TABLE;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;
use core::mem::transmute;

const MAXV: float32x4_t = unsafe { transmute([0x3f7fffffu32; 4]) };
const MINV: float32x4_t = unsafe { transmute([0x39000000u32; 4]) };
const MANT_MASK: uint32x4_t = unsafe { transmute([0xffu32; 4]) };
const TOP_SCALE: uint32x4_t = unsafe { transmute([0x02000000u32; 4]) };

#[inline]
#[target_feature(enable = "neon")]
unsafe fn simd_to_srgb8_neon(input: float32x4_t) -> uint32x4_t {
    // force NaN with positive or negative sign to zero
    // https://stackoverflow.com/questions/13458372/efficiently-convert-nans-to-zero-using-arm-vfp-instructions
    let input_non_nan_mask = vceqq_f32(input, input);
    let input_u32x4 = vreinterpretq_u32_f32(input);
    let input_u32x4 = vandq_u32(input_u32x4, input_non_nan_mask);
    let input = vreinterpretq_f32_u32(input_u32x4);

    // clamp between minv/maxv
    let clamped = vminnmq_f32(vmaxnmq_f32(input, MINV), MAXV);
    // Table index
    let tab_index = vshrq_n_u32(vreinterpretq_u32_f32(clamped), 20);

    // without gather instructions (which might not be a good idea to use
    // anyway), we need to still do 4 separate lookups (despite this). This
    // reduces SIMD parallelism, but it could be a lot worse.
    let indices: [u32; 4] = transmute(tab_index);
    #[cfg(all(not(unstable_bench), test))]
    {
        for &i in &indices {
            debug_assert!(TO_SRGB8_TABLE
                .get(i.checked_sub((127 - 13) * 8).unwrap() as usize)
                .is_some());
        }
    }
    let loaded: [u32; 4] = [
        *TO_SRGB8_TABLE.get_unchecked(*indices.get_unchecked(0) as usize - (127 - 13) * 8),
        *TO_SRGB8_TABLE.get_unchecked(*indices.get_unchecked(1) as usize - (127 - 13) * 8),
        *TO_SRGB8_TABLE.get_unchecked(*indices.get_unchecked(2) as usize - (127 - 13) * 8),
        *TO_SRGB8_TABLE.get_unchecked(*indices.get_unchecked(3) as usize - (127 - 13) * 8),
    ];

    let entry: uint32x4_t = transmute(loaded);
    let tabmult1 = vshrq_n_u32(vreinterpretq_u32_f32(clamped), 12);
    let tabmult2 = vandq_u32(tabmult1, MANT_MASK);
    let tabmult3 = vorrq_u32(tabmult2, TOP_SCALE);

    // emulate: _mm_madd_epi16(entry, tabmult3)
    // https://stackoverflow.com/questions/69659665/neon-equivalent-of-mm-madd-epi16-and-mm-maddubs-epi16
    let entry_u16x8 = vreinterpretq_u16_u32(entry);
    let tabmult3_u16x8 = vreinterpretq_u16_u32(tabmult3);
    let pl = vmull_u16(vget_low_u16(entry_u16x8), vget_low_u16(tabmult3_u16x8));
    let ph = vmull_high_u16(entry_u16x8, tabmult3_u16x8);
    let tabprod = vpaddq_u32(pl, ph);

    vshrq_n_u32(tabprod, 16)
}

#[inline]
pub unsafe fn simd_to_srgb8(input: [f32; 4]) -> [u8; 4] {
    let res = simd_to_srgb8_neon(transmute(input));

    let [a, b, c, d]: [u32; 4] = transmute(res);
    #[cfg(all(not(unstable_bench), test))]
    {
        debug_assert!([a, b, c, d].iter().all(|v| *v < 256), "{:?}", [a, b, c, d]);
    }
    [a as u8, b as u8, c as u8, d as u8]
    // [vals[0] as u8, vals[1] as u8, vals[2] as u8, vals[3] as u8]
}
