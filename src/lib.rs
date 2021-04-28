use bitvec::prelude::*;
use fasthash::{
    FastHasher as _,
    Seed,
    murmur3,
};
use std::{
    cmp,
    hash::{
        Hash,
        Hasher as _,
    },
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

fn calculate_bucket_range(
    iter_idx: u64,
    murmur_hash: u64,
    xx_hash: u64,
    n_bags: usize,
    n_count_bits: u8,
) -> std::ops::Range<usize>
{
    let n_count_bits = n_count_bits as usize;
    let idx = murmur_hash
        .wrapping_add(iter_idx.wrapping_mul(xx_hash))
        .wrapping_add(iter_idx.pow(3)) % (n_bags as u64);
    let bucket_idx = idx as usize * n_count_bits;
    bucket_idx..bucket_idx + n_count_bits
}

struct BucketRanges {
    murmur_hash: u64,
    xx_hash: u64,
    n_bags: usize,
    n_count_bits: u8,
    n_hash_functions: u64,
    current_index: u64,
}

impl BucketRanges {
    fn new<T: Hash>(
        item: &T,
        murmur_seed: u32,
        xx_seed: u64,
        n_bags: usize,
        n_count_bits: u8,
        n_hash_functions: u64,
    ) -> Self
    {
        // Murmur3 only has 32 and 128 bit versions. The implementation used here truncates the the
        // resulting u128 to a u64, which seems to be acceptable for non-cryptographic hash
        // functions:
        //
        // https://stackoverflow.com/questions/11475423/is-any-64-bit-portion-of-a-128-bit-hash-as-collision-proof-as-a-64-bit-hash
        let mut murmur_hasher = murmur3::Hasher128_x64::with_seed(murmur_seed);
        let murmur_hash = {
            item.hash(&mut murmur_hasher);
            murmur_hasher.finish()
        };
        let mut xx_hasher = twox_hash::XxHash64::with_seed(xx_seed);
        let xx_hash = {
            item.hash(&mut xx_hasher);
            xx_hasher.finish()
        };

        Self {
            murmur_hash,
            xx_hash,
            n_bags,
            n_count_bits,
            n_hash_functions,
            current_index: 0,
        }
    }
}

impl Iterator for BucketRanges {
    type Item = std::ops::Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.n_hash_functions {
            None?;
        }

        let bucket_range = calculate_bucket_range(
            self.current_index,
            self.murmur_hash,
            self.xx_hash,
            self.n_bags,
            self.n_count_bits,
        );
        self.current_index += 1;
        Some(bucket_range)
    }
}


#[derive(Clone)]
pub struct CountingBloomFilter {
    bitmap: BitVec,
    n_bags: usize,
    n_hash_functions: u64,
    n_count_bits: u8,
    max_count: u8,
    murmur_seed: u32,
    xx_seed: u64,
}

impl CountingBloomFilter {
    pub fn bitmap(&self) -> &BitVec {
        &self.bitmap
    }

    pub fn builder() -> CountingBloomFilterBuilder {
        CountingBloomFilterBuilder {
            n_bags: None,
            n_hash_functions: None,
            n_count_bits: None,
            murmur_seed: None,
            xx_seed: None,
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

impl CountingBloomFilter {
    /// Check if `item` has been inserted into the bloom filter.
    pub fn contains<T: Hash>(&self, item: &T) -> bool {
        let mut contained = true;

        for bucket_range in BucketRanges::new(
            item,
            self.murmur_seed,
            self.xx_seed,
            self.n_bags,
            self.n_count_bits,
            self.n_hash_functions,
        ) {
            let subslice = self
                .bitmap
                .get(bucket_range)
                .expect("bucket has to lie in bitmap range");
            contained &= subslice.any();
        }
        contained
    }

    /// Returns an uper bound on the number of times `item` was inserted into the counting bloom
    /// filter.
    pub fn estimate_count<T: Hash>(&self, item: &T) -> u8 {
        let mut estimate = self.max_count;
        for bucket_range in BucketRanges::new(
            item,
            self.murmur_seed,
            self.xx_seed,
            self.n_bags,
            self.n_count_bits,
            self.n_hash_functions,
        ) {
            let subslice = self
                .bitmap
                .get(bucket_range)
                .expect("bucket has to lie in bitmap range");
            let count: u8 = subslice.load();
            estimate = std::cmp::min(estimate, count);
        }
        estimate
    }

    /// Inserts `item` into the counting bloom filter, incrementing the buckets it is placed by
    /// one, up to the threshold `max_count`.
    pub fn insert<T: Hash>(&mut self, item: &T) {
        for bucket_range in BucketRanges::new(
            item,
            self.murmur_seed,
            self.xx_seed,
            self.n_bags,
            self.n_count_bits,
            self.n_hash_functions,
        ) {
            let subslice = self
                .bitmap
                .get_mut(bucket_range)
                .expect("bucket has to lie in bitmap range");
            let count: u8 = subslice.load();
            // TODO: Attempt to replace this by a branchfree version.
            let new_count = cmp::min(self.max_count, count.saturating_add(1));
            subslice.store(new_count);
        }
    }

    /// Merge two counting bloom filters by adding their bags up to the max threshold.
    ///
    /// Returns an error if any of `n_bags`, `n_hash_functions`, `n_count_bits`, `max_count`,
    /// `murmur_seed` or `xx_seed` don't match.
    pub fn merge(&mut self, other: &Self) -> Result<(), Box<(dyn std::error::Error + Send + Sync)>> {
        if self.n_bags != other.n_bags {
            Err("number of bags don't match between counting bloom filters")?;
        }
        if self.n_hash_functions != other.n_hash_functions {
            Err("number of hash functions don't match between counting bloom filters")?;
        }
        if self.n_count_bits != other.n_count_bits {
            Err("number of count bits don't match between counting bloom filters")?;
        }
        if self.max_count != other.max_count {
            Err("max counts don't match between counting bloom filters")?;
        }
        if self.murmur_seed != other.murmur_seed {
            Err("murmur seeds don't match between counting bloom filters")?;
        }
        if self.xx_seed != other.xx_seed {
            Err("xx seeds don't match between counting bloom filters")?;
        }
        for i in 0..self.n_bags {
            let from = i;
            let to = i + self.n_count_bits as usize;
            if let Some(subslice) = self.bitmap.get_mut(from..to) {
                if let Some(other_subslice) = other.bitmap.get(from..to) {
                    let count: u8 = subslice.load();
                    let other_count: u8 = other_subslice.load();
                    // TODO: Attempt to replace this by a branchfree version.
                    let new_count = cmp::min(self.max_count, count.saturating_add(other_count));
                    subslice.store(new_count);
                }
            }
        }
        Ok(())
    }
}

pub struct CountingBloomFilterBuilder {
    n_bags: Option<usize>,
    n_hash_functions: Option<u64>,
    n_count_bits: Option<u8>,
    murmur_seed: Option<u32>,
    xx_seed: Option<u64>,
}

impl CountingBloomFilterBuilder {
    pub fn build(self) -> Result<CountingBloomFilter, &'static str> {
        let n_bags = self.n_bags.ok_or("n_bags field must be set")?;
        let n_hash_functions = self.n_hash_functions.ok_or("n_hash_functions field must be set")?;
        let n_count_bits = self.n_count_bits.ok_or("n_count_bits field must be set")?;

        if n_count_bits < 1 && n_count_bits > 8 {
            Err("n_count_bits must be at least 1 and at most 8")?;
        }

        let murmur_seed = self.murmur_seed.unwrap_or_else(|| Seed::gen().into());
        let xx_seed = self.xx_seed.unwrap_or_else(|| Seed::gen().into());

        Ok(CountingBloomFilter {
            bitmap: BitVec::repeat(false, n_bags * n_count_bits as usize),
            n_bags,
            n_hash_functions,
            n_count_bits,
            max_count: u8::MAX >> (8 - n_count_bits),
            murmur_seed,
            xx_seed,
        })
    }

    pub fn murmur_seed(self, murmur_seed: u32) -> Self {
        let mut this = self;
        this.murmur_seed.replace(murmur_seed);
        this
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

    pub fn xx_seed(self, xx_seed: u64) -> Self {
        let mut this = self;
        this.xx_seed.replace(xx_seed);
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
        let mut filter = CountingBloomFilter::builder()
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
        let filter = CountingBloomFilter::builder()
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
        let filter = CountingBloomFilter::builder()
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
        let mut filter = CountingBloomFilter::builder()
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
        let mut filter = CountingBloomFilter::builder()
            .n_bags(n_bags)
            .n_count_bits(n_count_bits)
            .n_hash_functions(n_hash_functions)
            .build()?;
        let max_count = u8::MAX >> (8 - n_count_bits);
        for i in 1..max_count * 2 {
            filter.insert(&1u32);
            assert_eq!(filter.estimate_count(&1u32), std::cmp::min(i, max_count));
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
