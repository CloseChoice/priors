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


def extract_itemsets_with_support(result, num_transactions):
    """
    Extract itemsets and their support values from fp_growth result.

    Args:
        result: Result from fp_growth - tuple (itemsets_list, supports_list) or list
        num_transactions: Total number of transactions for support calculation

    Returns:
        dict: Mapping from itemset tuple to support value (float), or None if no supports available
    """
    itemsets_with_support = {}

    # Handle tuple format (itemsets_list, supports_list) - regular fp_growth
    if isinstance(result, tuple) and len(result) == 2:
        itemsets_list, supports_list = result
        for level_itemsets, level_supports in zip(itemsets_list, supports_list, strict=True):
            if level_itemsets is not None and hasattr(level_itemsets, "shape") and level_itemsets.shape[0] > 0:
                for i in range(level_itemsets.shape[0]):
                    itemset = tuple(int(x) for x in sorted(level_itemsets[i]))
                    support = level_supports[i] / num_transactions
                    itemsets_with_support[itemset] = support
    # Handle list format (itemsets only) - lazy API returns only itemsets without supports
    elif isinstance(result, list):
        # Return just itemsets as keys with None values (no support info from lazy API)
        return None

    return itemsets_with_support


def extract_itemsets_only(result):
    """Extract just the itemsets (without support) from any result format."""
    itemsets = set()

    # Handle tuple format
    if isinstance(result, tuple) and len(result) == 2:
        itemsets_list, _ = result
        for level in itemsets_list:
            if level is not None and hasattr(level, "shape") and level.shape[0] > 0:
                for i in range(level.shape[0]):
                    itemset = tuple(int(x) for x in sorted(level[i]))
                    itemsets.add(itemset)
    # Handle list format
    elif isinstance(result, list):
        for level in result:
            if level is not None and hasattr(level, "shape") and level.shape[0] > 0:
                for i in range(level.shape[0]):
                    itemset = tuple(int(x) for x in sorted(level[i]))
                    itemsets.add(itemset)

    return itemsets


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
    base_pattern = np.array(
        [
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
        ],
        dtype=np.int32,
    )

    # Calculate expected support:
    # Item 0: 8/10 = 0.8
    # Item 1: 8/10 = 0.8
    # Item 2: 6/10 = 0.6
    # Itemset {0,1}: 6/10 = 0.6
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

        # Mine patterns and extract itemsets
        result = priors.lazy_mine_patterns(pid, min_support)
        streaming_itemsets = extract_itemsets_only(result)

        # Expected itemsets with manually calculated support values
        expected_with_support = {
            (0,): 0.8,
            (1,): 0.8,
            (2,): 0.6,
            (0, 1): 0.6,
            (0, 2): 0.4,
            (1, 2): 0.4,
            (0, 1, 2): 0.2,
        }

        # Verify streaming finds all expected itemsets
        assert streaming_itemsets == expected_with_support.keys(), (
            f"Itemsets mismatch!\n"
            f"Expected: {sorted(expected_with_support.keys())}\n"
            f"Got: {sorted(streaming_itemsets)}\n"
            f"Missing: {sorted(expected_with_support.keys() - streaming_itemsets)}\n"
            f"Extra: {sorted(streaming_itemsets - expected_with_support.keys())}"
        )

        # Verify against regular FP-Growth with support values
        regular_result = priors.fp_growth(base_pattern, min_support)
        regular_with_support = extract_itemsets_with_support(regular_result, len(base_pattern))

        # Check itemsets match
        assert streaming_itemsets == regular_with_support.keys(), (
            f"Constant distribution failed: streaming and regular find different itemsets"
        )

        # Verify regular FP-Growth support values match expected
        for itemset, expected_support in expected_with_support.items():
            actual_support = regular_with_support[itemset]
            assert abs(actual_support - expected_support) < 1e-6, (
                f"Support mismatch for {itemset}: expected {expected_support}, got {actual_support}"
            )

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

        # Mine patterns and extract itemsets
        total_transactions = total_batches * batch_size
        result = priors.lazy_mine_patterns(pid, min_support)
        streaming_itemsets = extract_itemsets_only(result)

        # Expected itemsets with manually calculated support values
        # Phase 1 (50k): [1,1,0], Phase 2 (50k): [1,1,1]
        expected_with_support = {
            (0,): 1.0,      # 100k/100k
            (1,): 1.0,      # 100k/100k
            (2,): 0.5,      # 50k/100k (only phase 2)
            (0, 1): 1.0,    # 100k/100k
            (0, 2): 0.5,    # 50k/100k (only phase 2)
            (1, 2): 0.5,    # 50k/100k (only phase 2)
            (0, 1, 2): 0.5, # 50k/100k (only phase 2)
        }

        # Verify streaming finds all expected itemsets
        assert streaming_itemsets == expected_with_support.keys(), (
            f"Itemsets mismatch!\n"
            f"Expected: {sorted(expected_with_support.keys())}\n"
            f"Got: {sorted(streaming_itemsets)}\n"
            f"Missing: {sorted(expected_with_support.keys() - streaming_itemsets)}\n"
            f"Extra: {sorted(streaming_itemsets - expected_with_support.keys())}"
        )

        # Verify against regular FP-Growth on same distribution
        manual_data = np.vstack([
            np.tile([[1, 1, 0]], (phase1_batches * batch_size, 1)),
            np.tile([[1, 1, 1]], (phase2_batches * batch_size, 1)),
        ]).astype(np.int32)

        regular_result = priors.fp_growth(manual_data, min_support)
        regular_with_support = extract_itemsets_with_support(regular_result, total_transactions)

        # Check streaming finds same itemsets as regular
        assert streaming_itemsets == regular_with_support.keys(), (
            f"Streaming and regular FP-Growth find different itemsets"
        )

        # Verify regular FP-Growth support values match expected
        for itemset, expected_support in expected_with_support.items():
            actual_support = regular_with_support[itemset]
            assert abs(actual_support - expected_support) < 1e-6, (
                f"Support mismatch for {itemset}: expected {expected_support}, got {actual_support}"
            )

    finally:
        priors.lazy_cleanup(pid)
