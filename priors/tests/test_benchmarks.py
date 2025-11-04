"""
Performance benchmarks for Priors FP-Growth.
Tests speed, memory efficiency, and scalability up to 500K transactions.
"""

import gc
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

import priors

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


def format_memory(mb):
    """Format memory in MB to human readable format."""
    if mb >= 1024:
        return f"{mb/1024:.1f} GB"
    else:
        return f"{mb:.0f} MB"


def format_time(seconds):
    """Format time in seconds to human readable format."""
    if seconds >= 60:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds:.2f}s"


# ============================================================================
# Benchmark Results Storage
# ============================================================================

speed_results = []
memory_results = []


def add_speed_result(
    dataset_size: str,
    num_trans: int,
    num_items: int,
    itemsets_found: int,
    execution_time: float,
    throughput: float,
    notes: str = "",
):
    """Add speed benchmark result."""
    speed_results.append(
        {
            "Dataset Size": dataset_size,
            "Transactions": f"{num_trans:,}",
            "Items": num_items,
            "Itemsets Found": f"{itemsets_found:,}",
            "Time": format_time(execution_time),
            "Throughput": f"{throughput:,.0f} trans/s",
            "Notes": notes,
        }
    )


def add_memory_result(
    dataset_size: str,
    peak_memory: float,
    dataset_memory: float,
    efficiency: float,
    notes: str = "",
):
    """Add memory benchmark result."""
    memory_results.append(
        {
            "Dataset Size": dataset_size,
            "Dataset Memory": format_memory(dataset_memory),
            "Peak Memory": format_memory(peak_memory),
            "Memory Efficiency": f"{efficiency:.1f}x",
            "Notes": notes,
        }
    )


# ============================================================================
# Speed Benchmarks
# ============================================================================


@pytest.mark.parametrize(
    "num_trans,num_items,avg_size,min_support",
    [
        (10_000, 50, 10, 0.05),
        (25_000, 60, 12, 0.04),
        (50_000, 80, 15, 0.03),
        (100_000, 100, 20, 0.025),
        (200_000, 120, 25, 0.02),
        (350_000, 150, 30, 0.015),
        (500_000, 200, 35, 0.01),
    ],
)
def test_speed_scaling(num_trans, num_items, avg_size, min_support):
    """Test Priors FP-Growth speed across different dataset sizes."""
    print(f"\n=== Testing {num_trans:,} transactions ===")

    # Generate data
    print("Generating transactions...")
    gen_start = time.time()
    transactions = generate_transactions(num_trans, num_items, avg_size, seed=42)
    gen_time = time.time() - gen_start
    print(f"Generation: {format_time(gen_time)}")

    # Calculate dataset size in memory
    dataset_memory = transactions.nbytes / 1024 / 1024

    # Measure memory before
    gc.collect()
    start_memory = get_memory_usage()

    # Run FP-Growth
    print("Running FP-Growth...")
    start_time = time.time()
    result = priors.fp_growth(transactions, min_support)
    execution_time = time.time() - start_time

    # Measure memory after
    peak_memory = get_memory_usage()
    memory_used = peak_memory - start_memory

    # Calculate metrics
    itemsets_found = count_itemsets(result)
    throughput = num_trans / execution_time if execution_time > 0 else 0
    memory_efficiency = dataset_memory / memory_used if memory_used > 0 else 1.0

    # Dataset size label
    dataset_size = f"{num_trans//1000}K × {num_items}"

    # Performance expectations
    max_time = min(60.0, num_trans / 5000)  # Scale expectations
    assert (
        execution_time < max_time
    ), f"Dataset {dataset_size} took {execution_time:.2f}s, expected < {max_time:.2f}s"

    assert itemsets_found >= 0, "Should find some itemsets or return 0"

    # Store results
    add_speed_result(
        dataset_size,
        num_trans,
        num_items,
        itemsets_found,
        execution_time,
        throughput,
        f"Support: {min_support}, Gen: {format_time(gen_time)}",
    )

    if psutil:
        add_memory_result(
            dataset_size,
            peak_memory,
            dataset_memory,
            memory_efficiency,
            f"Used: {format_memory(memory_used)}",
        )

    print(f"Results: {itemsets_found:,} itemsets in {format_time(execution_time)}")
    print(f"Throughput: {throughput:,.0f} transactions/second")
    if psutil:
        print(
            f"Memory: {format_memory(memory_used)} used, {memory_efficiency:.1f}x efficiency"
        )


# ============================================================================
# Memory Benchmarks
# ============================================================================


@pytest.mark.parametrize(
    "num_trans,num_items,density",
    [
        (50_000, 100, 0.15),  # Sparse
        (100_000, 150, 0.20),  # Medium
        (200_000, 200, 0.25),  # Dense
        (350_000, 250, 0.30),  # Very Dense
    ],
)
def test_memory_efficiency(num_trans, num_items, density):
    """Test memory efficiency across different data densities."""
    print(
        f"\n=== Memory Test: {num_trans:,} × {num_items} (density: {density:.0%}) ==="
    )

    # Generate data with specific density
    np.random.seed(42)
    transactions = np.zeros((num_trans, num_items), dtype=np.int32)

    for i in range(num_trans):
        num_items_in_trans = max(
            1, int(num_items * density * np.random.uniform(0.5, 1.5))
        )
        num_items_in_trans = min(num_items_in_trans, num_items)
        items = np.random.choice(num_items, num_items_in_trans, replace=False)
        transactions[i, items] = 1

    actual_density = np.mean(transactions)
    dataset_memory = transactions.nbytes / 1024 / 1024

    print(f"Actual density: {actual_density:.1%}")
    print(f"Dataset size: {format_memory(dataset_memory)}")

    # Test with different support levels
    support_levels = [0.05, 0.02, 0.01]

    for support in support_levels:
        print(f"\nTesting support {support}...")

        # Measure memory usage
        gc.collect()
        start_memory = get_memory_usage()

        start_time = time.time()
        result = priors.fp_growth(transactions, support)
        execution_time = time.time() - start_time

        peak_memory = get_memory_usage()
        memory_used = peak_memory - start_memory

        itemsets_found = count_itemsets(result)
        memory_efficiency = dataset_memory / memory_used if memory_used > 0 else 1.0

        print(
            f"  Support {support}: {itemsets_found:,} itemsets in {format_time(execution_time)}"
        )
        print(
            f"  Memory used: {format_memory(memory_used)} (efficiency: {memory_efficiency:.1f}x)"
        )

        # Reasonable memory usage expectations
        expected_max_memory = dataset_memory * 5  # At most 5x the dataset size
        if psutil and memory_used > expected_max_memory:
            print(
                f"  Warning: High memory usage ({format_memory(memory_used)} > {format_memory(expected_max_memory)})"
            )


# ============================================================================
# Scalability Tests
# ============================================================================


def test_transaction_scaling():
    """Test how performance scales with number of transactions."""
    print(f"\n=== Transaction Scaling Analysis ===")

    base_items = 100
    avg_size = 20
    min_support = 0.02

    sizes = [50_000, 100_000, 200_000, 400_000]
    times = []
    memories = []

    for size in sizes:
        print(f"\nTesting {size:,} transactions...")

        transactions = generate_transactions(size, base_items, avg_size, seed=42)

        gc.collect()
        start_memory = get_memory_usage()

        start_time = time.time()
        result = priors.fp_growth(transactions, min_support)
        execution_time = time.time() - start_time

        peak_memory = get_memory_usage()
        memory_used = peak_memory - start_memory

        itemsets_found = count_itemsets(result)

        times.append(execution_time)
        memories.append(memory_used)

        print(
            f"  Results: {itemsets_found:,} itemsets in {format_time(execution_time)}"
        )
        print(f"  Memory: {format_memory(memory_used)} used")

    # Analyze scaling
    print(f"\n=== Scaling Analysis ===")
    for i in range(1, len(sizes)):
        scale_factor = sizes[i] / sizes[i - 1]
        time_factor = times[i] / times[i - 1] if times[i - 1] > 0 else 1.0
        memory_factor = memories[i] / memories[i - 1] if memories[i - 1] > 0 else 1.0

        # Efficiency rating
        if time_factor <= scale_factor * 1.2:
            efficiency = "Excellent"
        elif time_factor <= scale_factor * 2.0:
            efficiency = "Good"
        elif time_factor <= scale_factor * 3.0:
            efficiency = "Fair"
        else:
            efficiency = "Poor"

        print(
            f"  {sizes[i-1]:,} → {sizes[i]:,}: "
            f"{scale_factor:.1f}x size → {time_factor:.1f}x time, {memory_factor:.1f}x memory ({efficiency})"
        )


def test_item_scaling():
    """Test how performance scales with number of items."""
    print(f"\n=== Item Scaling Analysis ===")

    base_transactions = 100_000
    avg_size = 15
    min_support = 0.02

    item_counts = [50, 100, 200, 400]

    for items in item_counts:
        print(f"\nTesting {items} items...")

        transactions = generate_transactions(
            base_transactions, items, avg_size, seed=42
        )

        start_time = time.time()
        result = priors.fp_growth(transactions, min_support)
        execution_time = time.time() - start_time

        itemsets_found = count_itemsets(result)

        print(
            f"  Results: {itemsets_found:,} itemsets in {format_time(execution_time)}"
        )

        # Should handle increasing item counts reasonably
        assert (
            execution_time < 30.0
        ), f"Item count {items} took too long: {execution_time:.2f}s"


# ============================================================================
# Stress Tests
# ============================================================================


@pytest.mark.slow
def test_maximum_dataset():
    """Test with maximum dataset size (500K transactions)."""
    print(f"\n=== Maximum Dataset Test (500K transactions) ===")

    num_trans = 500_000
    num_items = 200
    avg_size = 25
    min_support = 0.008  # Very low support for stress test

    print(f"Generating {num_trans:,} transactions with {num_items} items...")
    gen_start = time.time()
    transactions = generate_transactions(num_trans, num_items, avg_size, seed=42)
    gen_time = time.time() - gen_start

    dataset_memory = transactions.nbytes / 1024 / 1024
    print(
        f"Dataset: {format_memory(dataset_memory)}, Generation: {format_time(gen_time)}"
    )

    # Memory monitoring
    gc.collect()
    start_memory = get_memory_usage()

    print("Running FP-Growth...")
    start_time = time.time()
    result = priors.fp_growth(transactions, min_support)
    execution_time = time.time() - start_time

    peak_memory = get_memory_usage()
    memory_used = peak_memory - start_memory

    itemsets_found = count_itemsets(result)
    throughput = num_trans / execution_time if execution_time > 0 else 0

    print(f"\n=== Results ===")
    print(f"Itemsets found: {itemsets_found:,}")
    print(f"Execution time: {format_time(execution_time)}")
    print(f"Throughput: {throughput:,.0f} transactions/second")
    if psutil:
        print(f"Memory used: {format_memory(memory_used)}")
        print(f"Memory efficiency: {dataset_memory/memory_used:.1f}x")

    # Should complete in reasonable time (5 minutes max)
    assert (
        execution_time < 300.0
    ), f"Maximum dataset took too long: {execution_time:.2f}s"
    assert itemsets_found >= 0, "Should find itemsets or return 0"


@pytest.mark.slow
def test_very_low_support():
    """Test with very low support thresholds."""
    print(f"\n=== Very Low Support Test ===")

    transactions = generate_transactions(100_000, 100, 20, seed=42)

    support_levels = [0.001, 0.0005, 0.0001]  # Very low supports

    for support in support_levels:
        print(f"\nTesting support {support} ({support*100:.02f}%)...")

        start_time = time.time()
        result = priors.fp_growth(transactions, support)
        execution_time = time.time() - start_time

        itemsets_found = count_itemsets(result)

        print(f"  Found {itemsets_found:,} itemsets in {format_time(execution_time)}")

        # Should handle very low support without crashing
        assert (
            execution_time < 60.0
        ), f"Support {support} took too long: {execution_time:.2f}s"


# ============================================================================
# Summary Output
# ============================================================================


def print_speed_summary():
    """Print speed benchmark summary table."""
    if not speed_results:
        return

    print("\n" + "=" * 120)
    print("PRIORS FP-GROWTH SPEED BENCHMARKS")
    print("=" * 120)

    # Print header
    print(
        f"{'Dataset Size':<15} {'Transactions':<15} {'Items':<6} {'Itemsets Found':<15} "
        f"{'Time':<10} {'Throughput':<15} {'Notes':<30}"
    )
    print("-" * 120)

    # Print results
    for result in speed_results:
        print(
            f"{result['Dataset Size']:<15} {result['Transactions']:<15} "
            f"{result['Items']:<6} {result['Itemsets Found']:<15} "
            f"{result['Time']:<10} {result['Throughput']:<15} {result['Notes']:<30}"
        )

    print("=" * 120)


def print_memory_summary():
    """Print memory benchmark summary table."""
    if not memory_results:
        return

    print("\n" + "=" * 100)
    print("PRIORS FP-GROWTH MEMORY EFFICIENCY")
    print("=" * 100)

    # Print header
    print(
        f"{'Dataset Size':<15} {'Dataset Memory':<15} {'Peak Memory':<15} "
        f"{'Efficiency':<12} {'Notes':<30}"
    )
    print("-" * 100)

    # Print results
    for result in memory_results:
        print(
            f"{result['Dataset Size']:<15} {result['Dataset Memory']:<15} "
            f"{result['Peak Memory']:<15} {result['Memory Efficiency']:<12} "
            f"{result['Notes']:<30}"
        )

    print("=" * 100)


@pytest.fixture(scope="session", autouse=True)
def print_summary_on_exit(request):
    """Print summary tables after all tests complete."""

    def finalize():
        print_speed_summary()
        print_memory_summary()
        print("\n" + "=" * 120)
        print("BENCHMARK COMPLETE")
        print("=" * 120 + "\n")

    request.addfinalizer(finalize)


# ============================================================================
# Run as standalone script
# ============================================================================

if __name__ == "__main__":
    # Run all benchmarks
    pytest.main([__file__, "-v", "-s", "--tb=short"])
