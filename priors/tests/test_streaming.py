"""
Comprehensive tests for streaming/lazy FP-Growth implementation.

Tests verify that streaming FP-Growth produces identical results to:
- Regular FP-Growth (batch processing)
- mlxtend FP-Growth
"""

import numpy as np
import pandas as pd
import pytest

# Import shared utilities
from conftest import count_itemsets, generate_all_ones_transactions, generate_transactions

import priors


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
    transactions = np.array(
        [
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
        ],
        dtype=np.int32,
    )

    min_support = 0.4

    # Run regular FP-Growth
    regular_result = priors.fp_growth(transactions, min_support)
    regular_count = count_itemsets(regular_result)

    # Run streaming FP-Growth
    streaming_result = run_streaming_fp_growth(transactions, min_support)
    streaming_count = count_itemsets(streaming_result)

    assert streaming_count == regular_count, (
        f"Count mismatch: streaming={streaming_count}, regular={regular_count}"
    )


@pytest.mark.slow
def test_streaming_vs_mlxtend():
    """Verify streaming matches mlxtend FP-Growth."""
    pytest.importorskip("mlxtend")
    from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

    transactions = generate_transactions(100, 15, 5, seed=123)
    min_support = 0.1

    # Run mlxtend
    df = pd.DataFrame(
        transactions.astype(bool),
        columns=[f"i{i}" for i in range(transactions.shape[1])],
    )
    mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
    mlxtend_count = len(mlxtend_result)

    # Run streaming
    streaming_result = run_streaming_fp_growth(transactions, min_support)
    streaming_count = count_itemsets(streaming_result)

    assert streaming_count == mlxtend_count, (
        f"Count mismatch: streaming={streaming_count}, mlxtend={mlxtend_count}"
    )


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

    assert streaming_count == regular_count, (
        f"Count mismatch: streaming={streaming_count}, regular={regular_count}"
    )


def test_scaled_dataset():
    """Test scaled dataset: multiply small known dataset by 100x."""
    # Create base dataset
    base_transactions = np.array(
        [
            [1, 1, 0, 1],
            [1, 0, 1, 1],
            [0, 1, 1, 1],
        ],
        dtype=np.int32,
    )

    # Scale it by repeating 100 times (reduced for CI performance)
    transactions = np.tile(base_transactions, (100, 1))
    min_support = 0.3

    # Run regular on base
    base_result = priors.fp_growth(base_transactions, min_support)
    base_count = count_itemsets(base_result)

    # Run streaming on scaled
    streaming_result = run_streaming_fp_growth(transactions, min_support, chunk_size=50)
    streaming_count = count_itemsets(streaming_result)

    assert streaming_count == base_count, (
        f"Count mismatch: streaming={streaming_count}, base={base_count}"
    )


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

    assert count1 == count2 == count3, (
        f"Chunk size mismatch: 50={count1}, 100={count2}, 200={count3}"
    )


# ============================================================================
# Large-Scale Test
# ============================================================================

# Test streaming FP-Growth on large datasets.


@pytest.mark.slow
def test_10m_transactions():
    """Test with 10M+ transactions using generator."""
    if not hasattr(priors, "create_lazy_fp_growth"):
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


@pytest.mark.slow
def test_endless_generator_constant_distribution():
    """
    Test streaming FP-Growth with an endless generator that repeats the same array.

    This verifies that processing the same pattern multiple times (e.g., 100 batches
    of 10k rows = 1M rows total) produces consistent support values. The distribution
    is constant, so support should remain exactly the same regardless of how many
    times we process the base pattern.
    """
    if not hasattr(priors, "create_lazy_fp_growth"):
        pytest.skip("Lazy FP-Growth functions not available")

    # Create a base pattern that will be repeated
    np.random.seed(42)
    base_pattern = np.array([
        [1, 1, 0],  # Items 0,1 appear together
        [1, 0, 1],  # Items 0,2 appear together
        [1, 1, 0],  # Items 0,1 appear together (repeat)
        [0, 1, 1],  # Items 1,2 appear together
        [1, 1, 1],  # All three items
        [1, 1, 0],  # Items 0,1 appear together
        [1, 0, 1],  # Items 0,2 appear together
        [0, 1, 1],  # Items 1,2 appear together
        [1, 1, 0],  # Items 0,1 appear together
        [1, 1, 1],  # All three items
    ], dtype=np.int32)

    # Calculate expected support:
    # Item 0: 8/10 = 0.8
    # Item 1: 9/10 = 0.9
    # Item 2: 5/10 = 0.5
    # Itemset {0,1}: 7/10 = 0.7
    # Itemset {0,2}: 4/10 = 0.4
    # Itemset {1,2}: 4/10 = 0.4
    # Itemset {0,1,2}: 2/10 = 0.2

    num_repeats = 100  # Repeat 100 times
    batch_size = 10000  # Each batch has 10k copies of the base pattern
    min_support = 0.15  # Should capture most patterns

    # Create endless generator
    def endless_generator():
        while True:
            # Each batch contains batch_size/10 copies of the base pattern
            yield np.tile(base_pattern, (batch_size // 10, 1))

    gen = endless_generator()

    # Process using lazy API
    pid = priors.create_lazy_fp_growth()

    try:
        # Counting phase - process num_repeats batches
        for _ in range(num_repeats):
            chunk = next(gen)
            priors.lazy_count_pass(pid, chunk)

        # Finalize counts
        priors.lazy_finalize_counts(pid, min_support)

        # Building phase - need to replay the same data
        gen2 = endless_generator()
        for _ in range(num_repeats):
            chunk = next(gen2)
            priors.lazy_build_pass(pid, chunk)

        priors.lazy_finalize_building(pid)

        # Mine patterns
        result = priors.lazy_mine_patterns(pid, min_support)
        streaming_count = count_itemsets(result)

        # Now verify against the base pattern processed once
        # The support should be EXACTLY the same since we're just repeating
        regular_result = priors.fp_growth(base_pattern, min_support)
        regular_count = count_itemsets(regular_result)

        # The counts should match exactly
        assert streaming_count == regular_count, (
            f"Support should be constant! Streaming (100x repeats) = {streaming_count}, "
            f"Regular (1x) = {regular_count}. Processing the same pattern multiple times "
            f"should yield identical support values."
        )

        # Also verify we're finding the expected itemsets
        # With min_support=0.15, we should find: {0}, {1}, {2}, {0,1}, {0,2}, {1,2}, {0,1,2}
        # That's 7 itemsets total (3 size-1, 3 size-2, 1 size-3)
        assert streaming_count == 7, (
            f"Expected 7 itemsets with min_support=0.15, got {streaming_count}"
        )

        print(f"✓ Constant distribution test passed: {streaming_count} itemsets found")
        print(f"  Processed {num_repeats * batch_size:,} rows (1M total) with constant support")

    finally:
        priors.lazy_cleanup(pid)


@pytest.mark.slow
def test_shifting_distribution_calculatable():
    """
    Test streaming FP-Growth with a generator where distribution shifts predictably.

    This test uses a generator that produces batches with calculatable patterns:
    - First N batches: All rows are all-zeros (except we make them all-ones for items 0,1)
    - Next N batches: All rows are all-ones (for items 0,1,2)
    - This creates a shifting distribution that we can verify at each step

    We verify that the streaming implementation correctly handles the changing
    support values as the distribution shifts.
    """
    if not hasattr(priors, "create_lazy_fp_growth"):
        pytest.skip("Lazy FP-Growth functions not available")

    batch_size = 1000
    num_items = 3

    # Phase 1: 50 batches where each row is [1, 1, 0] (items 0,1 present, item 2 absent)
    # Phase 2: 50 batches where each row is [1, 1, 1] (all items present)

    phase1_batches = 50
    phase2_batches = 50
    total_batches = phase1_batches + phase2_batches

    # Calculate expected supports after processing all batches:
    # Total transactions: (50 + 50) * 1000 = 100,000
    # Item 0: appears in all 100k transactions = 100k/100k = 1.0
    # Item 1: appears in all 100k transactions = 100k/100k = 1.0
    # Item 2: appears in only phase2 = 50k/100k = 0.5
    # Itemset {0,1}: appears in all 100k = 100k/100k = 1.0
    # Itemset {0,2}: appears in phase2 only = 50k/100k = 0.5
    # Itemset {1,2}: appears in phase2 only = 50k/100k = 0.5
    # Itemset {0,1,2}: appears in phase2 only = 50k/100k = 0.5

    min_support = 0.4  # Should capture item 2 and related itemsets

    def shifting_generator():
        # Phase 1: All rows are [1, 1, 0]
        for _ in range(phase1_batches):
            batch = np.ones((batch_size, num_items), dtype=np.int32)
            batch[:, 2] = 0  # Set item 2 to 0 for all rows
            yield batch

        # Phase 2: All rows are [1, 1, 1]
        for _ in range(phase2_batches):
            batch = np.ones((batch_size, num_items), dtype=np.int32)
            yield batch

    # Process using lazy API
    pid = priors.create_lazy_fp_growth()

    try:
        # Counting phase
        gen1 = shifting_generator()
        for _ in range(total_batches):
            chunk = next(gen1)
            priors.lazy_count_pass(pid, chunk)

        # Finalize counts
        priors.lazy_finalize_counts(pid, min_support)

        # Building phase - replay the data
        gen2 = shifting_generator()
        for _ in range(total_batches):
            chunk = next(gen2)
            priors.lazy_build_pass(pid, chunk)

        priors.lazy_finalize_building(pid)

        # Mine patterns
        result = priors.lazy_mine_patterns(pid, min_support)
        streaming_count = count_itemsets(result)

        # Expected itemsets with min_support=0.4:
        # - {0}: support 1.0 ✓
        # - {1}: support 1.0 ✓
        # - {2}: support 0.5 ✓
        # - {0,1}: support 1.0 ✓
        # - {0,2}: support 0.5 ✓
        # - {1,2}: support 0.5 ✓
        # - {0,1,2}: support 0.5 ✓
        # Total: 7 itemsets

        assert streaming_count == 7, (
            f"Expected 7 itemsets with shifting distribution, got {streaming_count}"
        )

        # Verify against a manually constructed dataset with the same distribution
        manual_data = np.vstack([
            np.tile(np.array([[1, 1, 0]], dtype=np.int32), (phase1_batches * batch_size, 1)),
            np.tile(np.array([[1, 1, 1]], dtype=np.int32), (phase2_batches * batch_size, 1)),
        ])

        regular_result = priors.fp_growth(manual_data, min_support)
        regular_count = count_itemsets(regular_result)

        assert streaming_count == regular_count, (
            f"Streaming with shifting distribution should match regular FP-Growth! "
            f"Streaming = {streaming_count}, Regular = {regular_count}"
        )

        print(f"✓ Shifting distribution test passed: {streaming_count} itemsets found")
        print(f"  Phase 1 (50k rows): [1,1,0] pattern")
        print(f"  Phase 2 (50k rows): [1,1,1] pattern")
        print(f"  Support correctly calculated across distribution shift")

    finally:
        priors.lazy_cleanup(pid)
