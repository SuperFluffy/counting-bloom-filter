use bitvec::prelude::*;
use fasthash::{
    RandomState,
    murmur3,
    xx,
};
use std::{
    hash::{
        BuildHasher as _,
        Hash,
        Hasher as _,
    },
    marker::PhantomData,
};

/// Returns the optimal number of bags for a given false positive `rate` and a number of
/// `n_expected_items`.
pub fn optimal_num_bags(false_positive_rate: f64, n_expected_items: usize) -> usize {
    optimal_num_bags_f64(false_positive_rate, n_expected_items as f64) as usize
}

/// Returns the optimal number of bags for a given false positive `rate` and a number of
/// `n_expected_items`.
pub fn optimal_num_bags_f64(false_positive_rate: f64, n_expected_items: f64) -> f64 {
    assert!(false_positive_rate > 0.0);
    assert!(false_positive_rate < 1.0);
    use std::f64::consts::LN_2;
    f64::ceil(
        n_expected_items * f64::ln(1.0 / false_positive_rate) / LN_2.powi(2)
    )
}

/// Returns the optimal number of hash functions for a given number of `n_bags` and a number of
/// `n_expected_items`.
pub fn optimal_num_hash_functions(n_bags: usize, n_expected_items: usize) -> u64 {
    optimal_num_hash_functions_f64(n_bags as f64, n_expected_items as f64) as u64
}

/// Returns the optimal number of hash functions for a given number of `n_bags` and a number of
/// `n_expected_items`.
pub fn optimal_num_hash_functions_f64(n_bags: f64, n_expected_items: f64) -> f64 {
    f64::ceil((n_bags / n_expected_items) * std::f64::consts::LN_2)
}

pub struct CountingBloomFilter<T> {
    bitmap: BitVec,
    n_bags: usize,
    n_hash_functions: u64,
    n_count_bits: u8,
    max_count: u8,
    murmur_hasher: RandomState<murmur3::Hash128_x64>,
    xx_hasher: RandomState<xx::Hash64>,
    underlying: PhantomData<T>,
}

impl<T> CountingBloomFilter<T> {
    pub fn bitmap(&self) -> &BitVec {
        &self.bitmap
    }

    pub fn builder() -> CountingBloomFilterBuilder<T> {
        CountingBloomFilterBuilder {
            n_bags: None,
            n_hash_functions: None,
            n_count_bits: None,
            underlying: PhantomData,
        }
    }

    pub fn n_bags(&self) -> usize {
        self.n_bags
    }

    pub fn n_hash_functions(&self) -> u64 {
        self.n_hash_functions
    }

    pub fn n_count_bits(&self) -> u8 {
        self.n_count_bits
    }
}

impl<T: Hash> CountingBloomFilter<T> {
    /// Check if `item` has been inserted into the bloom filter.
    pub fn contains(&self, item: &T) -> bool {
        let mut murmur_hasher = self.murmur_hasher.build_hasher();
        let mut xx_hasher = self.xx_hasher.build_hasher();

        let murmur_hash = {
            item.hash(&mut murmur_hasher);
            murmur_hasher.finish()
        };
        let xx_hash = {
            item.hash(&mut xx_hasher);
            xx_hasher.finish()
        };

        let mut contained = true;
        for i in 0..self.n_hash_functions {
            let idx = murmur_hash
                .wrapping_add(i.wrapping_mul(xx_hash))
                .wrapping_add(i.pow(3)) % (self.n_bags as u64);
            let idx = idx as usize;
            if let Some(subslice) = self.bitmap.get(idx..(idx + self.n_count_bits as usize)) {
                contained &= subslice.any();
            }
        }
        contained
    }

    /// Returns an uper bound on the number of times `item` was inserted into the counting bloom
    /// filter.
    pub fn estimate_count(&self, item: &T) -> u8 {
        let mut murmur_hasher = self.murmur_hasher.build_hasher();
        let mut xx_hasher = self.xx_hasher.build_hasher();

        let murmur_hash = {
            item.hash(&mut murmur_hasher);
            murmur_hasher.finish()
        };
        let xx_hash = {
            item.hash(&mut xx_hasher);
            xx_hasher.finish()
        };

        let mut estimate = self.max_count;
        for i in 0..self.n_hash_functions {
            let idx = murmur_hash
                .wrapping_add(i.wrapping_mul(xx_hash))
                .wrapping_add(i.pow(3)) % (self.n_bags as u64);
            let idx = idx as usize;
            if let Some(subslice) = self.bitmap.get(idx..(idx + self.n_count_bits as usize)) {
                let count: u8 = subslice.load();
                estimate = std::cmp::min(estimate, count);
            }
        }
        estimate
    }

    /// Inserts `item` into the counting bloom filter, incrementing the buckets it is placed by
    /// one, up to the threshold `max_count`.
    pub fn insert(&mut self, item: &T) {
        let mut murmur_hasher = self.murmur_hasher.build_hasher();
        let mut xx_hasher = self.xx_hasher.build_hasher();

        // Murmur3 only has 32 and 128 bit versions. The implementation used here truncates the the
        // resulting u128 to a u64, which seems to be acceptable for non-cryptographic hash
        // functions:
        //
        // https://stackoverflow.com/questions/11475423/is-any-64-bit-portion-of-a-128-bit-hash-as-collision-proof-as-a-64-bit-hash
        let murmur_hash = {
            item.hash(&mut murmur_hasher);
            murmur_hasher.finish()
        };
        let xx_hash = {
            item.hash(&mut xx_hasher);
            xx_hasher.finish()
        };

        for i in 0..self.n_hash_functions {
            let idx = murmur_hash
                .wrapping_add(i.wrapping_mul(xx_hash))
                .wrapping_add(i.pow(3)) % (self.n_bags as u64);
            let idx = idx as usize;
            if let Some(subslice) = self.bitmap.get_mut(idx..(idx + self.n_count_bits as usize)) {
                // NOTE: We load the subslice into a u16. Because 
                let count: u16 = subslice.load();
                let new_count = count + 1;
                // TODO: Attempt to replace this by a branchfree version.
                subslice.store(if new_count > self.max_count as u16 {
                    self.max_count
                } else {
                    new_count as u8
                });
            }
        }
    }
}

pub struct CountingBloomFilterBuilder<T> {
    n_bags: Option<usize>,
    n_hash_functions: Option<u64>,
    n_count_bits: Option<u8>,
    underlying: PhantomData<T>,
}

impl<T> CountingBloomFilterBuilder<T> {
    pub fn build(self) -> Result<CountingBloomFilter<T>, &'static str> {
        let n_bags = self.n_bags.ok_or("n_bags field must be set")?;
        let n_hash_functions = self.n_hash_functions.ok_or("n_hash_functions field must be set")?;
        let n_count_bits = self.n_count_bits.ok_or("n_count_bits field must be set")?;

        if n_count_bits < 1 && n_count_bits > 8 {
            Err("n_count_bits must be at least 1 and at most 8")?;
        }

        let murmur_hasher = RandomState::<murmur3::Hash128_x64>::new();
        let xx_hasher = RandomState::<xx::Hash64>::new();

        Ok(CountingBloomFilter {
            bitmap: BitVec::repeat(false, n_bags * n_count_bits as usize),
            n_bags,
            n_hash_functions,
            n_count_bits,
            max_count: u8::MAX >> (8 - n_count_bits),
            murmur_hasher,
            xx_hasher,
            underlying: PhantomData,
        })
    }

    pub fn n_bags(self, n_bags: usize) -> Self {
        let mut this = self;
        this.n_bags.replace(n_bags);
        this
    }

    pub fn n_count_bits(self, n_count_bits: u8) -> Self {
        let mut this = self;
        this.n_count_bits.replace(n_count_bits);
        this
    }

    pub fn n_hash_functions(self, n_hash_functions: u64) -> Self {
        let mut this = self;
        this.n_hash_functions.replace(n_hash_functions);
        this
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CountingBloomFilter,
        optimal_num_bags,
        optimal_num_hash_functions,
    };

    #[test]
    #[should_panic]
    fn fast_positive_rate_too_low() {
        optimal_num_bags(0.0, 10);
    }

    #[test]
    #[should_panic]
    fn fast_positive_rate_too_high() {
        optimal_num_bags(1.0, 10);
    }

    #[test]
    fn inserted_once_is_contained() -> Result<(), Box<dyn std::error::Error>> {
        let n_expected_items = 100;
        let n_bags = optimal_num_bags(0.01, n_expected_items);
        let n_hash_functions = optimal_num_hash_functions(n_bags, n_expected_items);
        let mut filter: CountingBloomFilter<u32> = CountingBloomFilter::builder()
            .n_bags(n_bags)
            .n_count_bits(6)
            .n_hash_functions(n_hash_functions)
            .build()?;
        filter.insert(&1u32);
        assert!(filter.contains(&1u32));
        Ok(())
    }

    #[test]
    fn inserted_never_is_not_contained() -> Result<(), Box<dyn std::error::Error>> {
        let n_expected_items = 100;
        let n_bags = optimal_num_bags(0.01, n_expected_items);
        let n_hash_functions = optimal_num_hash_functions(n_bags, n_expected_items);
        let filter: CountingBloomFilter<u32> = CountingBloomFilter::builder()
            .n_bags(n_bags)
            .n_count_bits(6)
            .n_hash_functions(n_hash_functions)
            .build()?;
        assert!(!filter.contains(&1u32));
        Ok(())
    }

    #[test]
    fn inserted_never_has_upper_bound_0() -> Result<(), Box<dyn std::error::Error>> {
        let n_expected_items = 100;
        let n_bags = optimal_num_bags(0.01, n_expected_items);
        let n_hash_functions = optimal_num_hash_functions(n_bags, n_expected_items);
        let filter: CountingBloomFilter<u32> = CountingBloomFilter::builder()
            .n_bags(n_bags)
            .n_count_bits(6)
            .n_hash_functions(n_hash_functions)
            .build()?;
        assert_eq!(filter.estimate_count(&1u32), 0);
        Ok(())
    }

    #[test]
    fn inserted_twice_has_upper_bound_2() -> Result<(), Box<dyn std::error::Error>> {
        let n_expected_items = 100;
        let n_bags = optimal_num_bags(0.01, n_expected_items);
        let n_hash_functions = optimal_num_hash_functions(n_bags, n_expected_items);
        let mut filter: CountingBloomFilter<u32> = CountingBloomFilter::builder()
            .n_bags(n_bags)
            .n_count_bits(6)
            .n_hash_functions(n_hash_functions)
            .build()?;
        filter.insert(&1u32);
        filter.insert(&1u32);
        assert_eq!(filter.estimate_count(&1u32), 2);
        Ok(())
    }

    #[test]
    fn inserting_saturates() -> Result<(), Box<dyn std::error::Error>> {
        let n_expected_items = 100;
        let n_bags = optimal_num_bags(0.01, n_expected_items);
        let n_hash_functions = optimal_num_hash_functions(n_bags, n_expected_items);
        let n_count_bits = 6;
        let mut filter: CountingBloomFilter<u32> = CountingBloomFilter::builder()
            .n_bags(n_bags)
            .n_count_bits(n_count_bits)
            .n_hash_functions(n_hash_functions)
            .build()?;
        let max_count = u8::MAX >> (8 - n_count_bits);
        for _ in 0..max_count as u16 * 2 {
            filter.insert(&1u32);
        }
        assert_eq!(filter.estimate_count(&1u32), max_count);
        Ok(())
    }

    #[test]
    fn test_optimal_num_bits() {
        assert_eq!(optimal_num_bags(0.01, 10), 96);
        assert_eq!(optimal_num_bags(0.01, 5000), 47926);
        assert_eq!(optimal_num_bags(0.01, 100_000), 958506);
    }

    #[test]
    fn test_optimal_num_hashes() {
        assert_eq!(optimal_num_hash_functions(96, 10), 7);
        assert_eq!(optimal_num_hash_functions(47926, 5000), 7);
        assert_eq!(optimal_num_hash_functions(958506, 100000), 7);
    }
}
