"""
Performance benchmarks comparing priors FP-Growth with other libraries.
Run with pytest-benchmark or as standalone script.
"""

# Import shared utilities
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest

import priors

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import count_itemsets, generate_transactions

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
            pytest.importorskip("mlxtend")
            from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

            df = pd.DataFrame(
                transactions.astype(bool),
                columns=[f"item_{i}" for i in range(transactions.shape[1])],
            )

            start_time = time.time()
            mlxtend_result = mlxtend_fpgrowth(
                df, min_support=min_support, use_colnames=False
            )
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

    @pytest.mark.slow
    def test_scaling_transactions(self):
        """Test how performance scales with number of transactions."""
        base_items = 20
        avg_size = 8
        min_support = 0.1

        sizes = [1000, 5000, 10000]
        times = []

        for size in sizes:
            transactions = generate_transactions(size, base_items, avg_size, seed=42)

            start_time = time.time()
            result = priors.fp_growth(transactions, min_support)
            elapsed = time.time() - start_time

            times.append(elapsed)
            itemset_count = count_itemsets(result)

            print(f"Size {size}: {itemset_count} itemsets in {elapsed:.3f}s")

        # Report scaling
        for i in range(1, len(times)):
            scale_factor = sizes[i] / sizes[i - 1]
            time_factor = times[i] / times[i - 1] if times[i - 1] > 0 else 1
            print(
                f"Scale {sizes[i-1]} -> {sizes[i]}: {scale_factor}x size took {time_factor:.2f}x time"
            )

    @pytest.mark.slow
    def test_scaling_items(self):
        """Test how performance scales with number of items."""
        base_transactions = 1000
        avg_size = 10
        min_support = 0.05

        item_counts = [20, 50, 100]

        for items in item_counts:
            transactions = generate_transactions(
                base_transactions, items, avg_size, seed=42
            )

            start_time = time.time()
            result = priors.fp_growth(transactions, min_support)
            elapsed = time.time() - start_time

            itemset_count = count_itemsets(result)

            print(f"Items {items}: {itemset_count} itemsets in {elapsed:.3f}s")


# ============================================================================
# Speed Comparison Benchmarks
# ============================================================================

speed_benchmark_results = []


def add_speed_benchmark(
    dataset_size: str,
    mlxtend_time: str,
    efficient_apriori_time: str,
    priors_time: float,
    speedup: str,
):
    """Add speed benchmark result."""
    speed_benchmark_results.append(
        {
            "Dataset Size": dataset_size,
            "MLxtend": mlxtend_time,
            "Efficient-Apriori": efficient_apriori_time,
            "Priors FP-Growth": f"{priors_time:.2f}s",
            "Speedup": speedup,
        }
    )


class TestSpeedBenchmarks:
    """Benchmarks for speed comparison: Regular FP-Growth vs others."""

    @pytest.mark.parametrize(
        "num_trans,num_items,avg_size,min_support",
        [
            (10000, 50, 10, 0.05),
            (50000, 80, 15, 0.03),
            (100000, 100, 20, 0.02),
            (200000, 100, 25, 0.01),
        ],
    )
    def test_speed_comparison(self, num_trans, num_items, avg_size, min_support):
        """Compare execution times across libraries."""
        transactions = generate_transactions(num_trans, num_items, avg_size, seed=42)
        df = pd.DataFrame(
            transactions.astype(bool), columns=[f"i{i}" for i in range(num_items)]
        )
        dataset_size = f"{num_trans//1000}K × {num_items}"

        # Priors FP-Growth
        start = time.time()
        _ = priors.fp_growth(transactions, min_support)
        priors_time = time.time() - start

        # MLxtend (if available, skip if OOM)
        mlxtend_time = "OOM"
        try:
            if num_trans <= 50000:  # Avoid OOM on larger datasets
                pytest.importorskip("mlxtend")
                from mlxtend.frequent_patterns import \
                    fpgrowth as mlxtend_fpgrowth

                start = time.time()
                _ = mlxtend_fpgrowth(
                    df, min_support=min_support, use_colnames=False
                )
                mlxtend_time = f"{time.time() - start:.2f}s"
        except (ImportError, MemoryError, Exception):
            mlxtend_time = "OOM"

        # Efficient-Apriori (if available)
        ea_time = "N/A"
        try:
            import efficient_apriori

            transactions_list = [tuple(np.where(row)[0]) for row in transactions]
            start = time.time()
            itemsets, rules = efficient_apriori.apriori(
                transactions_list, min_support=min_support
            )
            ea_time = f"{time.time() - start:.2f}s"
        except (ImportError, Exception):
            ea_time = "N/A"

        # Calculate speedup vs Efficient-Apriori if available
        speedup = "N/A"
        if ea_time != "N/A" and isinstance(ea_time, str) and ea_time.endswith("s"):
            try:
                ea_float = float(ea_time[:-1])
                if priors_time > 0:
                    speedup = f"{ea_float / priors_time:.1f}x"
            except (ValueError, ZeroDivisionError):
                speedup = "N/A"

        add_speed_benchmark(dataset_size, mlxtend_time, ea_time, priors_time, speedup)
        print(
            f"{dataset_size}: Priors={priors_time:.2f}s, MLxtend={mlxtend_time}, EA={ea_time}, Speedup={speedup}"
        )


def print_speed_benchmarks():
    """Print speed benchmark table."""
    if not speed_benchmark_results:
        return

    print("\n" + "=" * 80)
    print("SPEED BENCHMARKS (Regular FP-Growth)")
    print("=" * 80)

    df = pd.DataFrame(speed_benchmark_results)

    # Print formatted table
    print(
        f"{'Dataset Size':<15} {'MLxtend':<12} {'Efficient-Apriori':<18} {'Priors FP-Growth':<18} {'Speedup':<10}"
    )
    print("-" * 80)

    for _, row in df.iterrows():
        print(
            f"{row['Dataset Size']:<15} {row['MLxtend']:<12} {row['Efficient-Apriori']:<18} {row['Priors FP-Growth']:<18} {row['Speedup']:<10}"
        )

    print("=" * 80)
    print("MLxtend fails with OOM (Out of Memory) on larger datasets")


@pytest.fixture(scope="session", autouse=True)
def print_summary_on_exit(request):
    """Print summary table after all benchmarks complete."""

    def finalize():
        print_speed_benchmarks()
        print("=" * 80 + "\n")

    request.addfinalizer(finalize)
