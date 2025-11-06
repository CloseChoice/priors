"""
ASV benchmarks for the priors library.
Measures performance of FP-Growth algorithm on various dataset sizes.
"""

import numpy as np


def generate_transactions(num_transactions, num_items, avg_size, seed=42):
    """Generate random transaction data."""
    np.random.seed(seed)
    transactions = []
    for _ in range(num_transactions):
        size = max(1, int(np.random.poisson(avg_size)))
        items = np.random.choice(num_items, size=min(size, num_items), replace=False)
        row = np.zeros(num_items, dtype=np.int32)
        row[items] = 1
        transactions.append(row)
    return np.array(transactions, dtype=np.int32)


class FPGrowthSmall:
    """Benchmark FP-Growth on small dataset (1K transactions)."""

    def setup(self):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(1000, 30, 8, seed=42)
        self.min_support = 0.05

    def time_fp_growth_small(self):
        """Time FP-Growth on small dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)

    def peakmem_fp_growth_small(self):
        """Measure peak memory usage on small dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)


class FPGrowthMedium:
    """Benchmark FP-Growth on medium dataset (5K transactions)."""

    def setup(self):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(5000, 50, 12, seed=42)
        self.min_support = 0.03

    def time_fp_growth_medium(self):
        """Time FP-Growth on medium dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)

    def peakmem_fp_growth_medium(self):
        """Measure peak memory usage on medium dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)


class FPGrowthLarge:
    """Benchmark FP-Growth on large dataset (10K transactions)."""

    def setup(self):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(10000, 80, 15, seed=42)
        self.min_support = 0.02

    def time_fp_growth_large(self):
        """Time FP-Growth on large dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)

    def peakmem_fp_growth_large(self):
        """Measure peak memory usage on large dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)


class FPGrowthXLarge:
    """Benchmark FP-Growth on extra large dataset (50K transactions)."""

    def setup(self):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(50000, 100, 20, seed=42)
        self.min_support = 0.01

    def time_fp_growth_xlarge(self):
        """Time FP-Growth on extra large dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)

    def peakmem_fp_growth_xlarge(self):
        """Measure peak memory usage on extra large dataset."""
        self.priors.fp_growth(self.transactions, self.min_support)


class FPGrowthStreamingSmall:
    """Benchmark FP-Growth Streaming on small dataset (1K transactions)."""

    def setup(self):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(1000, 30, 8, seed=42)
        self.min_support = 0.05

    def time_fp_growth_streaming_small(self):
        """Time FP-Growth Streaming on small dataset."""
        self.priors.fp_growth_streaming(self.transactions, self.min_support)

    def peakmem_fp_growth_streaming_small(self):
        """Measure peak memory usage on small dataset."""
        self.priors.fp_growth_streaming(self.transactions, self.min_support)


class TransactionScaling:
    """Benchmark how FP-Growth scales with transaction count."""

    param_names = ['num_transactions']
    params = [[1000, 5000, 10000, 25000, 50000]]

    def setup(self, num_transactions):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(num_transactions, 50, 12, seed=42)
        self.min_support = 0.03

    def time_scaling_transactions(self, num_transactions):
        """Time FP-Growth with varying transaction counts."""
        self.priors.fp_growth(self.transactions, self.min_support)


class ItemScaling:
    """Benchmark how FP-Growth scales with item count."""

    param_names = ['num_items']
    params = [[20, 50, 100, 200]]

    def setup(self, num_items):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(5000, num_items, 12, seed=42)
        self.min_support = 0.03

    def time_scaling_items(self, num_items):
        """Time FP-Growth with varying item counts."""
        self.priors.fp_growth(self.transactions, self.min_support)


class SupportThreshold:
    """Benchmark how FP-Growth performs with different support thresholds."""

    param_names = ['min_support']
    params = [[0.01, 0.03, 0.05, 0.10]]

    def setup(self, min_support):
        import priors
        self.priors = priors
        self.transactions = generate_transactions(10000, 80, 15, seed=42)

    def time_support_threshold(self, min_support):
        """Time FP-Growth with varying support thresholds."""
        self.priors.fp_growth(self.transactions, min_support)
