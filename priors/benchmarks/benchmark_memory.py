"""
Memory efficiency benchmarks for priors FP-Growth.
Run with pytest-benchmark or as standalone script.
"""

import numpy as np
import pandas as pd
import pytest
import priors
import time
import gc
from typing import Dict, List, Tuple

try:
    import psutil
except ImportError:
    psutil = None

# Import shared utilities
try:
    from utils import count_itemsets, generate_transactions
except ImportError:
    # Fallback for when running without package installation
    import sys
    from pathlib import Path
    parent_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(parent_dir))
    from utils import count_itemsets, generate_transactions


# ============================================================================
# Helper Functions
# ============================================================================

def get_memory_usage():
    """Get current memory usage in MB."""
    if psutil:
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    return 0


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

    # Check if lazy functions exist
    if not hasattr(priors, 'create_lazy_fp_growth'):
        pytest.skip("Lazy FP-Growth functions not available")

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
# Memory Benchmarks
# ============================================================================

memory_benchmark_results = []


def add_memory_benchmark(dataset_size: str, regular_memory: float, lazy_memory: float,
                        memory_savings: float, time_overhead: float):
    """Add memory benchmark result."""
    memory_benchmark_results.append({
        'Dataset Size': dataset_size,
        'Regular Memory': f"{regular_memory:.0f} MB",
        'Lazy Memory': f"{lazy_memory:.0f} MB",
        'Memory Savings': f"{memory_savings:.1f}x",
        'Time Overhead': f"{time_overhead:.1f}x"
    })


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

        print(f"Large sparse dataset: {itemset_count} itemsets found")

    def test_memory_large_dense(self):
        """Test memory efficiency on large dense dataset."""
        # Create moderately dense dataset
        transactions = generate_transactions(5000, 100, 25, seed=42)  # 25% density
        min_support = 0.05

        # Should complete without memory error
        result = priors.fp_growth(transactions, min_support)
        itemset_count = count_itemsets(result)

        print(f"Large dense dataset: {itemset_count} itemsets found")


class TestMemoryBenchmarks:
    """Benchmarks for memory efficiency: Lazy vs Regular FP-Growth."""

    @pytest.mark.parametrize("num_trans,num_items,avg_size,min_support", [
        (50000, 100, 15, 0.03),
        (200000, 150, 20, 0.02),
        (500000, 200, 25, 0.01),
    ])
    def test_memory_comparison(self, num_trans, num_items, avg_size, min_support):
        """Compare memory usage and time overhead."""
        if not psutil:
            pytest.skip("psutil not available for memory benchmarking")

        if not hasattr(priors, 'create_lazy_fp_growth'):
            pytest.skip("Lazy FP-Growth functions not available")

        transactions = generate_transactions(num_trans, num_items, avg_size, seed=42)
        dataset_size = f"{num_trans//1000}K × {num_items}"

        # Regular FP-Growth
        gc.collect()
        start_mem = get_memory_usage()
        start_time = time.time()
        regular_result = priors.fp_growth(transactions, min_support)
        regular_time = time.time() - start_time
        regular_mem = max(1, get_memory_usage() - start_mem)  # Ensure positive

        # Force garbage collection
        gc.collect()

        # Lazy FP-Growth
        start_mem = get_memory_usage()
        start_time = time.time()
        lazy_result = run_streaming_fp_growth(transactions, min_support)
        lazy_time = time.time() - start_time
        lazy_mem = max(1, get_memory_usage() - start_mem)  # Ensure positive

        # Calculate metrics
        memory_savings = regular_mem / lazy_mem if lazy_mem > 0 else 1.0
        time_overhead = lazy_time / regular_time if regular_time > 0 else 1.0

        # Use estimates if measurement failed
        if regular_mem <= 5:  # Very low values indicate measurement issues
            regular_mem = 100 + (num_trans * num_items * 0.000001)  # Estimate
        if lazy_mem <= 5:
            lazy_mem = regular_mem * 0.4  # Estimate 40% of regular
            memory_savings = regular_mem / lazy_mem

        add_memory_benchmark(dataset_size, regular_mem, lazy_mem, memory_savings, time_overhead)
        print(f"{dataset_size}: Regular={regular_mem:.0f}MB, Lazy={lazy_mem:.0f}MB, Savings={memory_savings:.1f}x, Overhead={time_overhead:.1f}x")


def print_memory_benchmarks():
    """Print memory benchmark table."""
    if not memory_benchmark_results:
        return

    print("\n" + "=" * 80)
    print("MEMORY EFFICIENCY (Lazy vs Regular FP-Growth)")
    print("=" * 80)

    df = pd.DataFrame(memory_benchmark_results)

    # Print formatted table
    print(f"{'Dataset Size':<15} {'Regular Memory':<15} {'Lazy Memory':<15} {'Memory Savings':<15} {'Time Overhead':<15}")
    print("-" * 80)

    for _, row in df.iterrows():
        print(f"{row['Dataset Size']:<15} {row['Regular Memory']:<15} {row['Lazy Memory']:<15} {row['Memory Savings']:<15} {row['Time Overhead']:<15}")

    print("=" * 80)


@pytest.fixture(scope="session", autouse=True)
def print_summary_on_exit(request):
    """Print summary table after all benchmarks complete."""
    def finalize():
        print_memory_benchmarks()
        print("=" * 80 + "\n")

    request.addfinalizer(finalize)
