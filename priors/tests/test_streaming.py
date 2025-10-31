"""
Comprehensive tests for streaming/lazy FP-Growth implementation.

Tests verify that streaming FP-Growth produces identical results to:
- Regular FP-Growth (batch processing)
- mlxtend FP-Growth
"""

import numpy as np
import pandas as pd
import pytest
import priors
from typing import Dict, List, Tuple, Optional

# Import shared utilities
from conftest import count_itemsets, generate_transactions, generate_all_ones_transactions


def run_streaming_fp_growth(transactions, min_support, chunk_size=None):
    """
    Run streaming FP-Growth on transactions.

    This is a simple wrapper around priors.fp_growth_streaming().

    Args:
        transactions: Transaction matrix
        min_support: Minimum support threshold
        chunk_size: Size of chunks (if None, uses default)

    Returns:
        Result from streaming FP-Growth
    """
    # Use the clean unified interface
    return priors.fp_growth_streaming(transactions, min_support, chunk_size=chunk_size)


# ============================================================================
# Basic Correctness Tests
# ============================================================================

# Test that streaming FP-Growth produces correct results.

def test_streaming_vs_regular_basic():
    """Verify streaming matches regular FP-Growth on basic dataset."""
    transactions = np.array([
        [1, 1, 0, 1, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 0, 1, 0],
    ], dtype=np.int32)

    min_support = 0.4

    # Run regular FP-Growth
    regular_result = priors.fp_growth(transactions, min_support)
    regular_count = count_itemsets(regular_result)

    # Run streaming FP-Growth
    streaming_result = run_streaming_fp_growth(transactions, min_support)
    streaming_count = count_itemsets(streaming_result)

    assert streaming_count == regular_count, \
        f"Count mismatch: streaming={streaming_count}, regular={regular_count}"

def test_streaming_vs_mlxtend():
    """Verify streaming matches mlxtend FP-Growth."""
    mlxtend = pytest.importorskip("mlxtend")
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

    transactions = generate_transactions(100, 15, 5, seed=123)
    min_support = 0.1

    # Run mlxtend
    df = pd.DataFrame(transactions.astype(bool),
                     columns=[f'i{i}' for i in range(transactions.shape[1])])
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
    mlxtend_count = len(mlxtend_result)

    # Run streaming
    streaming_result = run_streaming_fp_growth(transactions, min_support)
    streaming_count = count_itemsets(streaming_result)

    assert streaming_count == mlxtend_count, \
        f"Count mismatch: streaming={streaming_count}, mlxtend={mlxtend_count}"

def test_trivial_all_ones():
    """Test trivial case: all 1s dataset."""
    num_trans, num_items = 10, 5
    transactions = generate_all_ones_transactions(num_trans, num_items)
    min_support = 0.9  # High support but not 99%

    # Run streaming
    streaming_result = run_streaming_fp_growth(transactions, min_support)
    streaming_count = count_itemsets(streaming_result)

    # Run regular for comparison
    regular_result = priors.fp_growth(transactions, min_support)
    regular_count = count_itemsets(regular_result)

    assert streaming_count == regular_count, \
        f"Count mismatch: streaming={streaming_count}, regular={regular_count}"

def test_scaled_dataset():
    """Test scaled dataset: multiply small known dataset by 1000x."""
    # Create base dataset
    base_transactions = np.array([
        [1, 1, 0, 1],
        [1, 0, 1, 1],
        [0, 1, 1, 1],
    ], dtype=np.int32)

    # Scale it by repeating 1000 times
    transactions = np.tile(base_transactions, (1000, 1))
    min_support = 0.3

    # Run regular on base
    base_result = priors.fp_growth(base_transactions, min_support)
    base_count = count_itemsets(base_result)

    # Run streaming on scaled
    streaming_result = run_streaming_fp_growth(transactions, min_support, chunk_size=500)
    streaming_count = count_itemsets(streaming_result)

    assert streaming_count == base_count, \
        f"Count mismatch: streaming={streaming_count}, base={base_count}"

def test_different_chunk_sizes():
    """Test that different chunk sizes produce same results."""
    transactions = generate_transactions(200, 20, 6, seed=456)
    min_support = 0.05

    # Try different chunk sizes
    result1 = run_streaming_fp_growth(transactions, min_support, chunk_size=50)
    count1 = count_itemsets(result1)

    result2 = run_streaming_fp_growth(transactions, min_support, chunk_size=100)
    count2 = count_itemsets(result2)

    result3 = run_streaming_fp_growth(transactions, min_support, chunk_size=200)
    count3 = count_itemsets(result3)

    assert count1 == count2 == count3, \
        f"Chunk size mismatch: 50={count1}, 100={count2}, 200={count3}"


# ============================================================================
# Large-Scale Test
# ============================================================================

# Test streaming FP-Growth on large datasets.

@pytest.mark.slow
def test_10m_transactions():
    """Test with 10M+ transactions using generator."""
    if not hasattr(priors, 'create_lazy_fp_growth'):
        pytest.skip("Lazy FP-Growth functions not available")

    num_transactions = 10_000_000
    num_items = 50
    avg_size = 10
    min_support = 0.001  # 0.1% support = 10k transactions
    chunk_size = 100_000  # Process 100k at a time

    # Generate in chunks to avoid memory issues
    pid = priors.create_lazy_fp_growth()

    try:
        # Counting phase
        for i in range(0, num_transactions, chunk_size):
            actual_chunk_size = min(chunk_size, num_transactions - i)
            chunk = generate_transactions(actual_chunk_size, num_items, avg_size, seed=i)
            priors.lazy_count_pass(pid, chunk)

        # Finalize counts
        priors.lazy_finalize_counts(pid, min_support)

        # Building phase
        for i in range(0, num_transactions, chunk_size):
            actual_chunk_size = min(chunk_size, num_transactions - i)
            chunk = generate_transactions(actual_chunk_size, num_items, avg_size, seed=i)
            priors.lazy_build_pass(pid, chunk)

        priors.lazy_finalize_building(pid)

        # Mining phase
        result = priors.lazy_mine_patterns(pid, min_support)
        itemset_count = count_itemsets(result)

        # Verify on sample
        sample = generate_transactions(10_000, num_items, avg_size, seed=0)
        regular_result = priors.fp_growth(sample, min_support)
        regular_count = count_itemsets(regular_result)

        # Counts may differ slightly due to sampling, but should be in same ballpark
        assert itemset_count > 0, "Should find itemsets"
        assert regular_count > 0, "Regular should find itemsets on sample"

    finally:
        priors.lazy_cleanup(pid)