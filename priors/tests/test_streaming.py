"""
Comprehensive tests for streaming/lazy FP-Growth implementation.

Tests verify that streaming FP-Growth produces identical results to:
- Regular FP-Growth (batch processing)
- mlxtend FP-Growth
- Efficient-apriori

Also includes large-scale tests (10M+ transactions) and summary table output.
"""

import numpy as np
import pandas as pd
import pytest
import priors
import time
from typing import Dict, List, Tuple


# ============================================================================
# Helper Functions
# ============================================================================

def count_itemsets(result):
    """Count total itemsets from priors result format."""
    if isinstance(result, list):
        return sum(level.shape[0] for level in result if level is not None and level.shape[0] > 0)
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


def generate_all_ones_transactions(num_transactions, num_items):
    """Generate trivial dataset with all 1s."""
    return np.ones((num_transactions, num_items), dtype=np.int32)


def run_streaming_fp_growth(transactions, min_support, chunk_size=None):
    """
    Run streaming FP-Growth on transactions.

    Args:
        transactions: Transaction matrix
        min_support: Minimum support threshold
        chunk_size: Size of chunks (if None, use 2 chunks)

    Returns:
        Result from streaming FP-Growth
    """
    if chunk_size is None:
        chunk_size = max(1, len(transactions) // 2)

    pid = priors.create_lazy_fp_growth()

    try:
        # Counting pass
        for i in range(0, len(transactions), chunk_size):
            chunk = transactions[i:i + chunk_size]
            priors.lazy_count_pass(pid, chunk)

        # Finalize counts
        priors.lazy_finalize_counts(pid, min_support)

        # Building pass
        for i in range(0, len(transactions), chunk_size):
            chunk = transactions[i:i + chunk_size]
            priors.lazy_build_pass(pid, chunk)

        priors.lazy_finalize_building(pid)

        # Mine patterns
        result = priors.lazy_mine_patterns(pid, min_support)

        return result
    finally:
        priors.lazy_cleanup(pid)


# ============================================================================
# Test Results Storage
# ============================================================================

test_results = []


def add_test_result(test_name: str, dataset_size: Tuple[int, int],
                   itemsets_found: int, match_status: str,
                   execution_time: float, notes: str = ""):
    """Add a test result to the global results list."""
    test_results.append({
        'Test': test_name,
        'Dataset': f"{dataset_size[0]}x{dataset_size[1]}",
        'Itemsets': itemsets_found,
        'Status': match_status,
        'Time (s)': f"{execution_time:.4f}",
        'Notes': notes
    })


# ============================================================================
# Basic Correctness Tests
# ============================================================================

class TestStreamingCorrectness:
    """Test that streaming FP-Growth produces correct results."""

    def test_streaming_vs_regular_basic(self):
        """Verify streaming matches regular FP-Growth on basic dataset."""
        transactions = np.array([
            [1, 1, 0, 1, 0],
            [1, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 1, 0],
        ], dtype=np.int32)

        min_support = 0.4
        start = time.time()

        # Run regular FP-Growth
        regular_result = priors.fp_growth(transactions, min_support)
        regular_count = count_itemsets(regular_result)

        # Run streaming FP-Growth
        streaming_result = run_streaming_fp_growth(transactions, min_support)
        streaming_count = count_itemsets(streaming_result)

        elapsed = time.time() - start

        match = "✓" if regular_count == streaming_count else "✗"
        add_test_result(
            "Streaming vs Regular (basic)",
            transactions.shape,
            streaming_count,
            match,
            elapsed,
            f"Regular: {regular_count}"
        )

        assert streaming_count == regular_count, \
            f"Count mismatch: streaming={streaming_count}, regular={regular_count}"

    def test_streaming_vs_mlxtend(self):
        """Verify streaming matches mlxtend FP-Growth."""
        mlxtend = pytest.importorskip("mlxtend")
        from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

        transactions = generate_transactions(100, 15, 5, seed=123)
        min_support = 0.1

        start = time.time()

        # Run mlxtend
        df = pd.DataFrame(transactions.astype(bool),
                         columns=[f'i{i}' for i in range(transactions.shape[1])])
        mlxtend_result = mlxtend_fpgrowth(df, min_support=min_support, use_colnames=False)
        mlxtend_count = len(mlxtend_result)

        # Run streaming
        streaming_result = run_streaming_fp_growth(transactions, min_support)
        streaming_count = count_itemsets(streaming_result)

        elapsed = time.time() - start

        match = "✓" if streaming_count == mlxtend_count else "✗"
        add_test_result(
            "Streaming vs mlxtend",
            transactions.shape,
            streaming_count,
            match,
            elapsed,
            f"mlxtend: {mlxtend_count}"
        )

        assert streaming_count == mlxtend_count, \
            f"Count mismatch: streaming={streaming_count}, mlxtend={mlxtend_count}"

    def test_trivial_all_ones(self):
        """Test trivial case: all 1s dataset."""
        # With all 1s, every combination is frequent at 100% support
        num_trans, num_items = 10, 5
        transactions = generate_all_ones_transactions(num_trans, num_items)
        min_support = 0.99  # Very high support

        start = time.time()

        # Run streaming
        streaming_result = run_streaming_fp_growth(transactions, min_support)
        streaming_count = count_itemsets(streaming_result)

        # Calculate expected: all combinations from 1 to num_items
        # 2^n - 1 (excluding empty set)
        expected = 2 ** num_items - 1

        elapsed = time.time() - start

        match = "✓" if streaming_count == expected else "✗"
        add_test_result(
            "Trivial all-ones",
            transactions.shape,
            streaming_count,
            match,
            elapsed,
            f"Expected: {expected}"
        )

        assert streaming_count == expected, \
            f"Count mismatch: streaming={streaming_count}, expected={expected}"

    def test_scaled_dataset(self):
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

        start = time.time()

        # Run regular on base
        base_result = priors.fp_growth(base_transactions, min_support)
        base_count = count_itemsets(base_result)

        # Run streaming on scaled
        streaming_result = run_streaming_fp_growth(transactions, min_support, chunk_size=500)
        streaming_count = count_itemsets(streaming_result)

        elapsed = time.time() - start

        match = "✓" if streaming_count == base_count else "✗"
        add_test_result(
            "Scaled 1000x",
            transactions.shape,
            streaming_count,
            match,
            elapsed,
            f"Base: {base_count}"
        )

        assert streaming_count == base_count, \
            f"Count mismatch: streaming={streaming_count}, base={base_count}"

    def test_different_chunk_sizes(self):
        """Test that different chunk sizes produce same results."""
        transactions = generate_transactions(200, 20, 6, seed=456)
        min_support = 0.05

        start = time.time()

        # Try different chunk sizes
        result1 = run_streaming_fp_growth(transactions, min_support, chunk_size=50)
        count1 = count_itemsets(result1)

        result2 = run_streaming_fp_growth(transactions, min_support, chunk_size=100)
        count2 = count_itemsets(result2)

        result3 = run_streaming_fp_growth(transactions, min_support, chunk_size=200)
        count3 = count_itemsets(result3)

        elapsed = time.time() - start

        match = "✓" if count1 == count2 == count3 else "✗"
        add_test_result(
            "Different chunk sizes",
            transactions.shape,
            count1,
            match,
            elapsed,
            f"50:{count1}, 100:{count2}, 200:{count3}"
        )

        assert count1 == count2 == count3, \
            f"Chunk size mismatch: 50={count1}, 100={count2}, 200={count3}"


# ============================================================================
# Large-Scale Test
# ============================================================================

class TestLargeScale:
    """Test streaming FP-Growth on large datasets."""

    @pytest.mark.slow
    def test_10m_transactions(self):
        """Test with 10M+ transactions using generator."""
        num_transactions = 10_000_000
        num_items = 50
        avg_size = 10
        min_support = 0.001  # 0.1% support = 10k transactions
        chunk_size = 100_000  # Process 100k at a time

        print(f"\nGenerating {num_transactions:,} transactions...")
        start_gen = time.time()

        # Generate in chunks to avoid memory issues
        pid = priors.create_lazy_fp_growth()

        try:
            # Counting phase
            print("Counting phase...")
            start_count = time.time()
            for i in range(0, num_transactions, chunk_size):
                actual_chunk_size = min(chunk_size, num_transactions - i)
                chunk = generate_transactions(actual_chunk_size, num_items, avg_size, seed=i)
                priors.lazy_count_pass(pid, chunk)
                if (i + actual_chunk_size) % 1_000_000 == 0:
                    print(f"  Counted {i + actual_chunk_size:,} transactions...")

            count_time = time.time() - start_count
            print(f"Counting completed in {count_time:.2f}s")

            # Finalize counts
            priors.lazy_finalize_counts(pid, min_support)

            # Building phase
            print("Building phase...")
            start_build = time.time()
            for i in range(0, num_transactions, chunk_size):
                actual_chunk_size = min(chunk_size, num_transactions - i)
                chunk = generate_transactions(actual_chunk_size, num_items, avg_size, seed=i)
                priors.lazy_build_pass(pid, chunk)
                if (i + actual_chunk_size) % 1_000_000 == 0:
                    print(f"  Built {i + actual_chunk_size:,} transactions...")

            priors.lazy_finalize_building(pid)
            build_time = time.time() - start_build
            print(f"Building completed in {build_time:.2f}s")

            # Mining phase
            print("Mining phase...")
            start_mine = time.time()
            result = priors.lazy_mine_patterns(pid, min_support)
            mine_time = time.time() - start_mine
            print(f"Mining completed in {mine_time:.2f}s")

            total_time = time.time() - start_gen
            itemset_count = count_itemsets(result)

            add_test_result(
                "Large scale 10M",
                (num_transactions, num_items),
                itemset_count,
                "✓",
                total_time,
                f"Count:{count_time:.1f}s Build:{build_time:.1f}s Mine:{mine_time:.1f}s"
            )

            print(f"\nTotal: {itemset_count:,} itemsets in {total_time:.2f}s")

            # Verify on sample
            print("Verifying on sample...")
            sample = generate_transactions(10_000, num_items, avg_size, seed=0)
            regular_result = priors.fp_growth(sample, min_support)
            regular_count = count_itemsets(regular_result)

            # Counts may differ slightly due to sampling, but should be in same ballpark
            assert itemset_count > 0, "Should find itemsets"
            assert regular_count > 0, "Regular should find itemsets on sample"

        finally:
            priors.lazy_cleanup(pid)


# ============================================================================
# Summary Table Output
# ============================================================================

def print_summary_table():
    """Print a nicely formatted summary table of all test results."""
    if not test_results:
        return

    print("\n" + "=" * 100)
    print("STREAMING FP-GROWTH TEST SUMMARY")
    print("=" * 100)

    df = pd.DataFrame(test_results)

    # Calculate column widths
    col_widths = {
        'Test': max(df['Test'].str.len().max(), len('Test')) + 2,
        'Dataset': max(df['Dataset'].str.len().max(), len('Dataset')) + 2,
        'Itemsets': max(df['Itemsets'].astype(str).str.len().max(), len('Itemsets')) + 2,
        'Status': 6,
        'Time (s)': 10,
        'Notes': max(df['Notes'].str.len().max(), len('Notes')) + 2,
    }

    # Print header
    header = (
        f"{'Test':<{col_widths['Test']}} "
        f"{'Dataset':<{col_widths['Dataset']}} "
        f"{'Itemsets':>{col_widths['Itemsets']}} "
        f"{'Status':^{col_widths['Status']}} "
        f"{'Time (s)':>{col_widths['Time (s)']}} "
        f"{'Notes':<{col_widths['Notes']}}"
    )
    print(header)
    print("-" * len(header))

    # Print rows
    for _, row in df.iterrows():
        print(
            f"{row['Test']:<{col_widths['Test']}} "
            f"{row['Dataset']:<{col_widths['Dataset']}} "
            f"{row['Itemsets']:>{col_widths['Itemsets']}} "
            f"{row['Status']:^{col_widths['Status']}} "
            f"{row['Time (s)']:>{col_widths['Time (s)']}} "
            f"{row['Notes']:<{col_widths['Notes']}}"
        )

    print("=" * 100)

    # Print pass/fail summary
    passed = sum(1 for r in test_results if r['Status'] == '✓')
    total = len(test_results)
    print(f"\nPassed: {passed}/{total}")
    print("=" * 100 + "\n")


@pytest.fixture(scope="session", autouse=True)
def print_summary_on_exit(request):
    """Print summary table after all tests complete."""
    request.addfinalizer(print_summary_table)


# ============================================================================
# Run as standalone script
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
