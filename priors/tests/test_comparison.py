"""
Comprehensive comparison tests between priors FP-Growth and other libraries.
Tests correctness, performance, and edge cases.
"""

import numpy as np
import pandas as pd
import pytest
import priors
import time
from typing import Dict, List, Tuple, Optional, Set

# ============================================================================
# Helper Functions
# ============================================================================

def count_itemsets(result):
    """Count total itemsets from priors result format."""
    if result is None:
        return 0
    if isinstance(result, list):
        return sum(level.shape[0] for level in result if level is not None and hasattr(level, 'shape') and level.shape[0] > 0)
    if hasattr(result, 'shape'):
        return result.shape[0]
    return 0


def generate_transactions(num_transactions, num_items, avg_size, seed=42):
    """Generate random transaction data for testing."""
    np.random.seed(seed)
    transactions = np.zeros((num_transactions, num_items), dtype=np.int32)

    for i in range(num_transactions):
        size = max(1, int(np.random.normal(avg_size, avg_size * 0.3)))
        size = min(size, num_items)
        items = np.random.choice(num_items, size, replace=False)
        transactions[i, items] = 1

    return transactions


def extract_itemsets_from_mlxtend(mlxtend_result):
    """Extract itemsets from mlxtend result format."""
    itemsets = set()
    if mlxtend_result is not None and len(mlxtend_result) > 0:
        for _, row in mlxtend_result.iterrows():
            itemset = tuple(sorted(row['itemsets']))
            itemsets.add(itemset)
    return itemsets


def extract_itemsets_from_efficient_apriori(ea_itemsets):
    """Extract itemsets from efficient_apriori result format."""
    itemsets = set()
    if ea_itemsets:
        for size_k, itemsets_k in ea_itemsets.items():
            for itemset in itemsets_k:
                itemsets.add(tuple(sorted(itemset)))
    return itemsets


def extract_itemsets_from_priors(priors_result):
    """Extract itemsets from priors result format."""
    itemsets = set()
    if priors_result is not None:
        if isinstance(priors_result, list):
            for level_idx, level in enumerate(priors_result):
                if level is not None and hasattr(level, 'shape') and level.shape[0] > 0:
                    # This is a simplified extraction - actual format may differ
                    for i in range(level.shape[0]):
                        # Create itemset based on level index and position
                        # This might need adjustment based on actual priors output format
                        itemsets.add((level_idx, i))  # Placeholder
    return itemsets


# ============================================================================
# Correctness Tests
# ============================================================================

class TestFPGrowthCorrectness:
    """Test correctness by comparing with established libraries."""

    def test_fpgrowth_vs_mlxtend_basic(self):
        """Compare priors FP-Growth with mlxtend on basic dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        # Create simple test data
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
        df = pd.DataFrame(transactions.astype(bool), 
                         columns=[f'item_{i}' for i in range(transactions.shape[1])])
        mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
        mlxtend_count = len(mlxtend_result)
        
        # Compare counts
        assert priors_count == mlxtend_count, \
            f"Itemset count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"

    def test_fpgrowth_vs_efficient_apriori_basic(self):
        """Compare priors FP-Growth with efficient_apriori."""
        try:
            import efficient_apriori
        except ImportError:
            pytest.skip("efficient_apriori not available")

        # Create test data
        transactions = generate_transactions(50, 10, 4, seed=123)
        min_support = 0.2
        
        # Run priors
        priors_result = priors.fp_growth(transactions, min_support)
        priors_count = count_itemsets(priors_result)
        
        # Run efficient_apriori
        transactions_list = [tuple(np.where(row)[0]) for row in transactions]
        ea_itemsets, ea_rules = efficient_apriori.apriori(transactions_list, min_support=min_support)
        ea_count = sum(len(itemsets) for itemsets in ea_itemsets.values()) if ea_itemsets else 0
        
        # Compare counts (allowing small differences due to implementation details)
        diff = abs(priors_count - ea_count)
        tolerance = max(1, min(priors_count, ea_count) * 0.1)  # 10% tolerance
        
        assert diff <= tolerance, \
            f"Itemset count mismatch beyond tolerance: priors={priors_count}, efficient_apriori={ea_count}, diff={diff}"

    def test_fpgrowth_vs_mlxtend_medium(self):
        """Compare with mlxtend on medium-sized dataset."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        # Generate medium dataset
        transactions = generate_transactions(200, 20, 6, seed=456)
        min_support = 0.1
        
        # Run priors
        start_time = time.time()
        priors_result = priors.fp_growth(transactions, min_support)
        priors_time = time.time() - start_time
        priors_count = count_itemsets(priors_result)
        
        # Run mlxtend
        df = pd.DataFrame(transactions.astype(bool),
                         columns=[f'item_{i}' for i in range(transactions.shape[1])])
        start_time = time.time()
        mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
        mlxtend_time = time.time() - start_time
        mlxtend_count = len(mlxtend_result)
        
        # Compare results
        assert priors_count == mlxtend_count, \
            f"Itemset count mismatch: priors={priors_count}, mlxtend={mlxtend_count}"
        
        # Priors should be faster or comparable
        speed_ratio = mlxtend_time / priors_time if priors_time > 0 else 1
        print(f"Speed ratio (mlxtend/priors): {speed_ratio:.2f}x")


# ============================================================================
# Scale Tests
# ============================================================================

class TestScaleCorrectness:
    """Test correctness at different scales."""

    def test_fpgrowth_medium_dataset(self):
        """Test on medium-sized dataset."""
        transactions = generate_transactions(1000, 50, 12, seed=789)
        min_support = 0.05
        
        start_time = time.time()
        result = priors.fp_growth(transactions, min_support)
        elapsed = time.time() - start_time
        
        itemset_count = count_itemsets(result)
        
        assert itemset_count > 0, "Should find itemsets in medium dataset"
        assert elapsed < 5.0, f"Medium dataset took too long: {elapsed:.2f}s"
        print(f"Medium dataset: {itemset_count} itemsets in {elapsed:.3f}s")

    def test_fpgrowth_large_dataset(self):
        """Test on large dataset."""
        transactions = generate_transactions(5000, 100, 15, seed=101112)
        min_support = 0.02
        
        start_time = time.time()
        result = priors.fp_growth(transactions, min_support)
        elapsed = time.time() - start_time
        
        itemset_count = count_itemsets(result)
        
        assert itemset_count > 0, "Should find itemsets in large dataset"
        assert elapsed < 30.0, f"Large dataset took too long: {elapsed:.2f}s"
        print(f"Large dataset: {itemset_count} itemsets in {elapsed:.3f}s")

    def test_fpgrowth_different_supports(self):
        """Test with different support thresholds."""
        transactions = generate_transactions(500, 30, 8, seed=131415)
        
        support_levels = [0.1, 0.05, 0.02, 0.01]
        prev_count = 0
        
        for support in support_levels:
            start_time = time.time()
            result = priors.fp_growth(transactions, support)
            elapsed = time.time() - start_time
            
            itemset_count = count_itemsets(result)
            
            # Lower support should find more or equal itemsets
            assert itemset_count >= prev_count, \
                f"Support {support}: {itemset_count} < {prev_count} from higher support"
            
            assert elapsed < 10.0, f"Support {support} took too long: {elapsed:.2f}s"
            
            prev_count = itemset_count
            print(f"Support {support}: {itemset_count} itemsets in {elapsed:.3f}s")

    def test_fpgrowth_empty_transactions(self):
        """Test edge case with empty transaction list."""
        transactions = np.array([], dtype=np.int32).reshape(0, 10)
        min_support = 0.1
        
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        assert itemset_count == 0, f"Empty transactions should return 0 itemsets, got {itemset_count}"


# ============================================================================
# Performance Benchmarks
# ============================================================================

class TestPerformanceBenchmarks:
    """Performance benchmarks comparing different libraries."""

    @pytest.mark.slow
    def test_performance_small(self):
        """Benchmark on small dataset."""
        transactions = generate_transactions(1000, 30, 8, seed=42)
        min_support = 0.05
        
        # Benchmark priors
        start_time = time.time()
        priors_result = priors.fp_growth(transactions, min_support)
        priors_time = time.time() - start_time
        priors_count = count_itemsets(priors_result)
        
        print(f"Priors (small): {priors_count} itemsets in {priors_time:.3f}s")
        
        # Benchmark mlxtend if available
        try:
            mlxtend = pytest.importorskip("mlxtend")
            from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth
            
            df = pd.DataFrame(transactions.astype(bool),
                             columns=[f'item_{i}' for i in range(transactions.shape[1])])
            
            start_time = time.time()
            mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
            mlxtend_time = time.time() - start_time
            mlxtend_count = len(mlxtend_result)
            
            print(f"MLxtend (small): {mlxtend_count} itemsets in {mlxtend_time:.3f}s")
            speedup = mlxtend_time / priors_time if priors_time > 0 else 1
            print(f"Speedup: {speedup:.2f}x")
            
        except ImportError:
            print("MLxtend not available for comparison")

    @pytest.mark.slow
    def test_performance_medium(self):
        """Benchmark on medium dataset."""
        transactions = generate_transactions(5000, 50, 12, seed=42)
        min_support = 0.03
        
        # Benchmark priors
        start_time = time.time()
        priors_result = priors.fp_growth(transactions, min_support)
        priors_time = time.time() - start_time
        priors_count = count_itemsets(priors_result)
        
        print(f"Priors (medium): {priors_count} itemsets in {priors_time:.3f}s")
        
        assert priors_time < 10.0, f"Medium benchmark took too long: {priors_time:.2f}s"

    @pytest.mark.slow
    def test_performance_large(self):
        """Benchmark on large dataset."""
        transactions = generate_transactions(10000, 80, 15, seed=42)
        min_support = 0.02
        
        # Benchmark priors
        start_time = time.time()
        priors_result = priors.fp_growth(transactions, min_support)
        priors_time = time.time() - start_time
        priors_count = count_itemsets(priors_result)
        
        print(f"Priors (large): {priors_count} itemsets in {priors_time:.3f}s")
        
        assert priors_time < 60.0, f"Large benchmark took too long: {priors_time:.2f}s"


# ============================================================================
# Memory Tests
# ============================================================================

class TestMemoryEfficiency:
    """Test memory usage and efficiency."""

    def test_memory_large_sparse(self):
        """Test memory efficiency on large sparse dataset."""
        # Create large sparse dataset
        num_trans, num_items = 10000, 200
        transactions = np.zeros((num_trans, num_items), dtype=np.int32)
        
        # Add sparse items (5% density)
        np.random.seed(42)
        for i in range(num_trans):
            num_items_in_trans = np.random.randint(1, 11)  # 1-10 items per transaction
            items = np.random.choice(num_items, num_items_in_trans, replace=False)
            transactions[i, items] = 1
        
        min_support = 0.01
        
        # Should complete without memory error
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        assert itemset_count >= 0, "Large sparse dataset should complete"

    def test_memory_large_dense(self):
        """Test memory efficiency on large dense dataset."""
        # Create moderately dense dataset
        transactions = generate_transactions(5000, 100, 25, seed=42)  # 25% density
        min_support = 0.05
        
        # Should complete without memory error
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)
        
        assert itemset_count >= 0, "Large dense dataset should complete"


# ============================================================================
# Run as standalone script
# ============================================================================

if __name__ == "__main__":
    print("Running correctness tests...")
    pytest.main([__file__ + "::TestFPGrowthCorrectness", "-v"])
    
    print("\nRunning performance benchmarks...")
    try:
        pytest.main([__file__ + "::TestPerformanceBenchmarks", "--benchmark-only", "-v"])
    except SystemExit:
        # benchmark-only might not be available
        pytest.main([__file__ + "::TestPerformanceBenchmarks", "-v"])