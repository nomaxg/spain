// arithmetic functions copied over from arkworks to use
// in bigrsa.rs.
// arkworks doesn't export these directly, so we copy them here.

#![allow(dead_code)]

/// Sets a = a + b + carry, and returns the new carry.
#[inline(always)]
#[doc(hidden)]
pub const fn adc(a: &mut u64, b: u64, carry: u64) -> u64 {
    let tmp = *a as u128 + b as u128 + carry as u128;
    *a = tmp as u64;
    (tmp >> 64) as u64
}

/// Sets a = a + b + carry, and returns the new carry.
#[inline(always)]
#[doc(hidden)]
pub fn adc_for_add_with_carry(a: &mut u64, b: u64, carry: u8) -> u8 {
    #[cfg(all(target_arch = "x86_64", feature = "asm"))]
    #[allow(unsafe_code)]
    unsafe {
        use core::arch::x86_64::_addcarry_u64;
        _addcarry_u64(carry, *a, b, a)
    }
    #[cfg(not(all(target_arch = "x86_64", feature = "asm")))]
    {
        let tmp = *a as u128 + b as u128 + carry as u128;
        *a = tmp as u64;
        (tmp >> 64) as u8
    }
}

/// Calculate a + b + carry, returning the sum
#[inline(always)]
#[doc(hidden)]
pub const fn adc_no_carry(a: u64, b: u64, carry: &u64) -> u64 {
    let tmp = a as u128 + b as u128 + *carry as u128;
    tmp as u64
}

/// Sets a = a - b - borrow, and returns the borrow.
#[inline(always)]
pub(crate) const fn sbb(a: &mut u64, b: u64, borrow: u64) -> u64 {
    let tmp = (1u128 << 64) + (*a as u128) - (b as u128) - (borrow as u128);
    *a = tmp as u64;
    (tmp >> 64 == 0) as u64
}

/// Sets a = a - b - borrow, and returns the borrow.
#[inline(always)]
#[doc(hidden)]
pub fn sbb_for_sub_with_borrow(a: &mut u64, b: u64, borrow: u8) -> u8 {
    #[cfg(target_arch = "x86_64")]
    #[allow(unsafe_code)]
    {
        use core::arch::x86_64::_subborrow_u64;
        _subborrow_u64(borrow, *a, b, a)
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let tmp = (1u128 << 64) + (*a as u128) - (b as u128) - (borrow as u128);
        *a = tmp as u64;
        u8::from(tmp >> 64 == 0)
    }
}

#[inline(always)]
#[doc(hidden)]
pub const fn widening_mul(a: u64, b: u64) -> u128 {
    #[cfg(not(target_family = "wasm"))]
    {
        a as u128 * b as u128
    }
    #[cfg(target_family = "wasm")]
    {
        let a_lo = a as u32 as u64;
        let a_hi = a >> 32;
        let b_lo = b as u32 as u64;
        let b_hi = b >> 32;

        let lolo = (a_lo * b_lo) as u128;
        let lohi = ((a_lo * b_hi) as u128) << 32;
        let hilo = ((a_hi * b_lo) as u128) << 32;
        let hihi = ((a_hi * b_hi) as u128) << 64;
        (lolo | hihi) + (lohi + hilo)
    }
}

/// Calculate a + b * c, returning the lower 64 bits of the result and setting
/// `carry` to the upper 64 bits.
#[inline(always)]
#[doc(hidden)]
pub const fn mac(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + widening_mul(b, c);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}

/// Calculate a + b * c, discarding the lower 64 bits of the result and setting
/// `carry` to the upper 64 bits.
#[inline(always)]
#[doc(hidden)]
pub const fn mac_discard(a: u64, b: u64, c: u64, carry: &mut u64) {
    let tmp = (a as u128) + widening_mul(b, c);
    *carry = (tmp >> 64) as u64;
}

/// Calculate a + (b * c) + carry, returning the least significant digit
/// and setting carry to the most significant digit.
#[inline(always)]
#[doc(hidden)]
pub const fn mac_with_carry(a: u64, b: u64, c: u64, carry: &mut u64) -> u64 {
    let tmp = (a as u128) + widening_mul(b, c) + (*carry as u128);
    *carry = (tmp >> 64) as u64;
    tmp as u64
}
