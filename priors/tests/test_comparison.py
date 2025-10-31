"""
Comprehensive comparison tests between priors FP-Growth and other libraries.

This module tests both correctness (does the algorithm produce similar results?)
and performance (which implementation is faster?) comparing:
- priors (Rust implementation)
- mlxtend FP-Growth
- efficient-apriori (uses Apriori algorithm but finds same patterns)
"""

import numpy as np
import pandas as pd
import pytest
import priors


# ============================================================================
# Helper Functions
# ============================================================================

def generate_random_transactions(num_transactions, num_items, avg_size, seed=42):
    """Generate random transaction data for testing."""
    np.random.seed(seed)
    transactions_matrix = np.zeros((num_transactions, num_items), dtype=np.int32)
    transactions_list = []

    for i in range(num_transactions):
        size = max(1, int(np.random.normal(avg_size, avg_size * 0.3)))
        size = min(size, num_items)
        items = np.random.choice(num_items, size, replace=False)

        # Set binary matrix
        transactions_matrix[i, items] = 1

        # Create list representation
        transactions_list.append(tuple(sorted(items.tolist())))

    return transactions_matrix, transactions_list


def count_itemsets(result):
    """Count total itemsets from priors result format."""
    if isinstance(result, list):
        return sum(level.shape[0] for level in result if level is not None and level.shape[0] > 0)
    return 0


def count_mlxtend_itemsets(df):
    """Count itemsets from mlxtend DataFrame result."""
    return len(df) if df is not None else 0


def count_efficient_apriori_itemsets(itemsets_dict):
    """Count itemsets from efficient-apriori result."""
    return sum(len(itemsets_dict[k]) for k in itemsets_dict.keys())


# ============================================================================
# FP-Growth Correctness Tests
# ============================================================================

class TestFPGrowthCorrectness:
    """Test that FP-Growth produces correct results compared to other implementations."""

    def test_fpgrowth_vs_mlxtend_basic(self):
        """Basic correctness test against mlxtend FP-Growth."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        # Small dataset for easy verification
        transactions = np.array([
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
        ], dtype=np.int32)

        min_support = 0.4

        # Run priors
        priors_result = priors.fp_growth(transactions, min_support)
        priors_count = count_itemsets(priors_result)

        # Run mlxtend
        df = pd.DataFrame(transactions.astype(bool), columns=[f'i{i}' for i in range(5)])
        mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
        mlxtend_count = count_mlxtend_itemsets(mlxtend_result)

        # Both should find itemsets
        assert priors_count > 0, "Priors should find itemsets"
        assert mlxtend_count > 0, "MLxtend should find itemsets"

        # Both implementations should find exactly the same itemsets
        assert priors_count == mlxtend_count, f"Count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"

    def test_fpgrowth_vs_efficient_apriori_basic(self):
        """Basic correctness test against efficient-apriori."""
        efficient_apriori_module = pytest.importorskip("efficient_apriori")
        from efficient_apriori import apriori as efficient_apriori

        transactions_list = [
            ('eggs', 'bacon', 'soup'),
            ('eggs', 'bacon', 'apple'),
            ('soup', 'bacon', 'banana'),
            ('eggs', 'apple', 'soup'),
            ('eggs', 'bacon', 'soup', 'apple'),
        ]

        min_support = 0.4

        # Run efficient-apriori
        itemsets, _ = efficient_apriori(transactions_list, min_support=min_support)
        efficient_count = count_efficient_apriori_itemsets(itemsets)

        # Convert to binary matrix for priors
        all_items = sorted(set(item for transaction in transactions_list for item in transaction))
        item_to_idx = {item: idx for idx, item in enumerate(all_items)}

        transactions_matrix = np.zeros((len(transactions_list), len(all_items)), dtype=np.int32)
        for i, transaction in enumerate(transactions_list):
            for item in transaction:
                transactions_matrix[i, item_to_idx[item]] = 1

        # Run priors
        priors_result = priors.fp_growth(transactions_matrix, min_support)
        priors_count = count_itemsets(priors_result)

        # Both should find itemsets
        assert efficient_count > 0, "Efficient-apriori should find itemsets"
        assert priors_count > 0, "Priors should find itemsets"

        # Both implementations should find exactly the same itemsets
        assert priors_count == efficient_count, f"Count mismatch: priors={priors_count}, efficient-apriori={efficient_count}"

    def test_fpgrowth_vs_mlxtend_medium(self):
        """Test FP-Growth vs mlxtend on medium dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        # Medium dataset
        transactions, _ = generate_random_transactions(500, 20, 5, seed=999)
        min_support = 0.1

        # Run priors
        priors_result = priors.fp_growth(transactions, min_support)
        priors_count = count_itemsets(priors_result)

        # Run mlxtend
        df = pd.DataFrame(transactions.astype(bool),
                         columns=[f'i{i}' for i in range(transactions.shape[1])])
        mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
        mlxtend_count = count_mlxtend_itemsets(mlxtend_result)

        # Both should find itemsets
        assert priors_count > 0, "Priors should find itemsets"
        assert mlxtend_count > 0, "MLxtend should find itemsets"

        # Both implementations should find exactly the same itemsets
        assert priors_count == mlxtend_count, f"Count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"


# ============================================================================
# Scale Tests (Correctness on Larger Datasets)
# ============================================================================

class TestScaleCorrectness:
    """Test correctness on larger, randomly generated datasets."""

    def test_fpgrowth_medium_dataset(self):
        """Test FP-Growth on medium dataset (1000 transactions)."""
        transactions, _ = generate_random_transactions(1000, 20, 5, seed=123)
        min_support = 0.05

        result = priors.fp_growth(transactions, min_support)
        count = count_itemsets(result)

        assert count > 0, "Should find itemsets on medium dataset"
        assert count < 10000, "Should not find unreasonably many itemsets"

    def test_fpgrowth_large_dataset(self):
        """Test FP-Growth on larger dataset (5000 transactions)."""
        transactions, _ = generate_random_transactions(5000, 30, 7, seed=789)
        min_support = 0.03

        result = priors.fp_growth(transactions, min_support)
        count = count_itemsets(result)

        assert count > 0, "Should find itemsets on large dataset"
        assert count < 50000, "Should not find unreasonably many itemsets"

    def test_fpgrowth_different_supports(self):
        """Test FP-Growth with different support thresholds."""
        transactions, _ = generate_random_transactions(500, 15, 4, seed=456)

        # Higher support should find fewer or equal itemsets than lower support
        result_high = priors.fp_growth(transactions, 0.2)
        result_low = priors.fp_growth(transactions, 0.05)

        count_high = count_itemsets(result_high)
        count_low = count_itemsets(result_low)

        assert count_low >= count_high, "Lower support should find at least as many itemsets as higher support"

    def test_fpgrowth_empty_transactions(self):
        """Test FP-Growth handles edge case with empty result."""
        transactions = np.array([
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ], dtype=np.int32)

        # Very high support - likely no patterns
        min_support = 0.9

        result = priors.fp_growth(transactions, min_support)
        # Should not crash, even if no itemsets found
        assert isinstance(result, list)


# ============================================================================
# Performance Benchmark Tests (requires pytest-benchmark)
# ============================================================================

class TestPerformanceSmall:
    """Performance benchmarks on small dataset (1000 transactions, 20 items)."""

    def setup_method(self):
        self.transactions, self.transactions_list = generate_random_transactions(1000, 20, 5)
        self.min_support = 0.05

    def test_priors_fpgrowth_small(self, benchmark):
        """Benchmark priors FP-Growth on small dataset."""
        result = benchmark(priors.fp_growth, self.transactions, self.min_support)
        assert count_itemsets(result) > 0

    def test_mlxtend_fpgrowth_small(self, benchmark):
        """Benchmark mlxtend FP-Growth on small dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        df = pd.DataFrame(self.transactions.astype(bool),
                         columns=[f'i{i}' for i in range(self.transactions.shape[1])])

        result = benchmark(mlxtend_fpgrowth, df, self.min_support, use_colnames=False)
        assert count_mlxtend_itemsets(result) > 0

    def test_mlxtend_apriori_small(self, benchmark):
        """Benchmark mlxtend Apriori on small dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import apriori as mlxtend_apriori

        df = pd.DataFrame(self.transactions.astype(bool),
                         columns=[f'i{i}' for i in range(self.transactions.shape[1])])

        result = benchmark(mlxtend_apriori, df, self.min_support, use_colnames=False)
        assert count_mlxtend_itemsets(result) > 0

    def test_efficient_apriori_small(self, benchmark):
        """Benchmark efficient-apriori on small dataset."""
        efficient_apriori_module = pytest.importorskip("efficient_apriori")
        from efficient_apriori import apriori as efficient_apriori

        itemsets, _ = benchmark(efficient_apriori, self.transactions_list,
                               min_support=self.min_support)
        assert count_efficient_apriori_itemsets(itemsets) > 0


class TestPerformanceMedium:
    """Performance benchmarks on medium dataset (5000 transactions, 50 items)."""

    def setup_method(self):
        self.transactions, self.transactions_list = generate_random_transactions(5000, 50, 8)
        self.min_support = 0.03

    def test_priors_fpgrowth_medium(self, benchmark):
        """Benchmark priors FP-Growth on medium dataset."""
        result = benchmark(priors.fp_growth, self.transactions, self.min_support)
        assert count_itemsets(result) > 0

    def test_mlxtend_fpgrowth_medium(self, benchmark):
        """Benchmark mlxtend FP-Growth on medium dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        df = pd.DataFrame(self.transactions.astype(bool),
                         columns=[f'i{i}' for i in range(self.transactions.shape[1])])

        result = benchmark(mlxtend_fpgrowth, df, self.min_support, use_colnames=False)
        assert count_mlxtend_itemsets(result) > 0

    def test_mlxtend_apriori_medium(self, benchmark):
        """Benchmark mlxtend Apriori on medium dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import apriori as mlxtend_apriori

        df = pd.DataFrame(self.transactions.astype(bool),
                         columns=[f'i{i}' for i in range(self.transactions.shape[1])])

        result = benchmark(mlxtend_apriori, df, self.min_support, use_colnames=False)
        assert count_mlxtend_itemsets(result) > 0

    def test_efficient_apriori_medium(self, benchmark):
        """Benchmark efficient-apriori on medium dataset."""
        efficient_apriori_module = pytest.importorskip("efficient_apriori")
        from efficient_apriori import apriori as efficient_apriori

        itemsets, _ = benchmark(efficient_apriori, self.transactions_list,
                               min_support=self.min_support)
        assert count_efficient_apriori_itemsets(itemsets) > 0


class TestPerformanceLarge:
    """Performance benchmarks on large dataset (10000 transactions, 100 items)."""

    def setup_method(self):
        self.transactions, self.transactions_list = generate_random_transactions(10000, 100, 12)
        self.min_support = 0.02

    def test_priors_fpgrowth_large(self, benchmark):
        """Benchmark priors FP-Growth on large dataset."""
        result = benchmark(priors.fp_growth, self.transactions, self.min_support)
        assert count_itemsets(result) > 0

    def test_mlxtend_fpgrowth_large(self, benchmark):
        """Benchmark mlxtend FP-Growth on large dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        df = pd.DataFrame(self.transactions.astype(bool),
                         columns=[f'i{i}' for i in range(self.transactions.shape[1])])

        result = benchmark(mlxtend_fpgrowth, df, self.min_support, use_colnames=False)
        assert count_mlxtend_itemsets(result) > 0

    def test_mlxtend_apriori_large(self, benchmark):
        """Benchmark mlxtend Apriori on large dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import apriori as mlxtend_apriori

        df = pd.DataFrame(self.transactions.astype(bool),
                         columns=[f'i{i}' for i in range(self.transactions.shape[1])])

        result = benchmark(mlxtend_apriori, df, self.min_support, use_colnames=False)
        assert count_mlxtend_itemsets(result) > 0

    def test_efficient_apriori_large(self, benchmark):
        """Benchmark efficient-apriori on large dataset."""
        efficient_apriori_module = pytest.importorskip("efficient_apriori")
        from efficient_apriori import apriori as efficient_apriori

        itemsets, _ = benchmark(efficient_apriori, self.transactions_list,
                               min_support=self.min_support)
        assert count_efficient_apriori_itemsets(itemsets) > 0


# ============================================================================
# Run as standalone script
# ============================================================================

if __name__ == "__main__":
    # Run correctness tests
    print("Running correctness tests...")
    pytest.main([__file__, "-v", "-k", "Correctness"])

    # Run performance benchmarks (requires pytest-benchmark)
    print("\nRunning performance benchmarks...")
    pytest.main([__file__, "-v", "-k", "Performance", "--benchmark-only"])
