"""
Comprehensive output comparison test for all priors FP-Growth functions.
Prints all results to console for manual verification.
"""

import time
from typing import List

import numpy as np
import pandas as pd
from numpy.typing import NDArray

import priors


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 80)
    if title:
        print(f"  {title}")
        print("=" * 80)


def print_itemsets(result: List[NDArray], name: str):
    """Print itemsets in a readable format."""
    print(f"\n{name}:")
    print("-" * 40)

    if not result:
        print("  No itemsets found")
        return

    total_count = 0
    for level_idx, level in enumerate(result, 1):
        if len(level) == 0:
            continue
        print(f"  Level {level_idx} ({level.shape[1]}-itemsets): {len(level)} itemsets")
        for i, itemset in enumerate(level):
            print(f"    {i+1}. {set(itemset)}")
        total_count += len(level)

    print(f"\n  Total: {total_count} itemsets")


def generate_test_data(size: str = "small"):
    """Generate test datasets."""
    if size == "small":
        # Small dataset: 10 transactions, 5 items
        return np.array(
            [
                [1, 1, 0, 1, 0],
                [1, 0, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [1, 0, 1, 1, 1],
                [1, 1, 1, 0, 0],
                [1, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [1, 1, 1, 1, 0],
            ],
            dtype=np.int32,
        )
    elif size == "medium":
        # Medium dataset: random but reproducible
        np.random.seed(42)
        transactions = np.random.rand(100, 10) > 0.7
        return transactions.astype(np.int32)
    else:
        raise ValueError(f"Unknown size: {size}")


def test_all_functions_with_output():
    """Test all priors functions and print detailed output."""

    print_separator("PRIORS FP-GROWTH - COMPREHENSIVE OUTPUT TEST")

    # Test configurations
    test_configs = [
        ("Small Dataset", generate_test_data("small"), 0.4),
        ("Medium Dataset", generate_test_data("medium"), 0.1),
    ]

    for dataset_name, transactions, min_support in test_configs:
        print_separator(f"{dataset_name}")
        print(f"Transactions shape: {transactions.shape}")
        print(f"Min support: {min_support}")
        print(f"Min count: {int(min_support * transactions.shape[0])}")

        # ====================================================================
        # 1. Test fp_growth (normal version)
        # ====================================================================
        print_separator("1. fp_growth (Normal Version)")
        start = time.time()
        result_normal = priors.fp_growth(transactions, min_support)
        time_normal = time.time() - start

        print(f"Execution time: {time_normal:.4f}s")
        print_itemsets(result_normal, "Results")

        # ====================================================================
        # 2. Test fp_growth_streaming
        # ====================================================================
        print_separator("2. fp_growth_streaming")
        start = time.time()
        result_streaming = priors.fp_growth_streaming(transactions, min_support)
        time_streaming = time.time() - start

        print(f"Execution time: {time_streaming:.4f}s")
        print_itemsets(result_streaming, "Results")

        # ====================================================================
        # 3. Test Lazy API
        # ====================================================================
        print_separator("3. Lazy API (Manual Control)")
        start = time.time()

        # Create processor
        pid = priors.create_lazy_fp_growth()
        print(f"Created processor with ID: {pid}")

        # Phase 1: Count pass
        print("\nPhase 1: Count pass...")
        priors.lazy_count_pass(pid, transactions)

        # Phase 2: Finalize counts
        print("Phase 2: Finalize counts...")
        priors.lazy_finalize_counts(pid, min_support)

        # Phase 3: Build pass
        print("Phase 3: Build pass...")
        priors.lazy_build_pass(pid, transactions)

        # Phase 4: Finalize building
        print("Phase 4: Finalize building...")
        priors.lazy_finalize_building(pid)

        # Phase 5: Mine patterns
        print("Phase 5: Mine patterns...")
        result_lazy = priors.lazy_mine_patterns(pid, min_support)

        # Cleanup
        priors.lazy_cleanup(pid)
        print("Cleaned up processor")

        time_lazy = time.time() - start

        print(f"\nTotal execution time: {time_lazy:.4f}s")
        print_itemsets(result_lazy, "Results")

        # ====================================================================
        # 4. Compare with mlxtend
        # ====================================================================
        print_separator("4. mlxtend FP-Growth (Reference)")
        try:
            from mlxtend.frequent_patterns import fpgrowth as mlxtend_fpgrowth

            start = time.time()
            df = pd.DataFrame(
                transactions.astype(bool),
                columns=[f"item_{i}" for i in range(transactions.shape[1])],
            )
            mlxtend_result = mlxtend_fpgrowth(
                df, min_support=min_support, use_colnames=False
            )
            time_mlxtend = time.time() - start

            print(f"Execution time: {time_mlxtend:.4f}s")
            print(f"\nResults:")
            print("-" * 40)
            print(f"  Total itemsets: {len(mlxtend_result)}")

            # Group by itemset size
            if len(mlxtend_result) > 0:
                mlxtend_result["size"] = mlxtend_result["itemsets"].apply(
                    lambda x: len(x)
                )
                for size in sorted(mlxtend_result["size"].unique()):
                    size_df = mlxtend_result[mlxtend_result["size"] == size]
                    print(f"  Level {size} ({size}-itemsets): {len(size_df)} itemsets")
                    for _idx, row in size_df.iterrows():
                        itemset = sorted(row["itemsets"])
                        support = row["support"]
                        print(f"    {set(itemset)} (support: {support:.4f})")

        except ImportError:
            print("mlxtend not installed - skipping comparison")
            time_mlxtend = None

        # ====================================================================
        # 5. Comparison Summary
        # ====================================================================
        print_separator("5. Comparison Summary")

        def count_total_itemsets(result):
            return sum(len(level) for level in result if len(level) > 0)

        count_normal = count_total_itemsets(result_normal)
        count_streaming = count_total_itemsets(result_streaming)
        count_lazy = count_total_itemsets(result_lazy)

        print("\nItemset Counts:")
        print(f"  fp_growth:          {count_normal}")
        print(f"  fp_growth_streaming: {count_streaming}")
        print(f"  Lazy API:           {count_lazy}")
        if time_mlxtend is not None:
            print(f"  mlxtend:            {len(mlxtend_result)}")

        print("\nExecution Times:")
        print(f"  fp_growth:          {time_normal:.4f}s")
        print(f"  fp_growth_streaming: {time_streaming:.4f}s")
        print(f"  Lazy API:           {time_lazy:.4f}s")
        if time_mlxtend is not None:
            print(f"  mlxtend:            {time_mlxtend:.4f}s")

        print("\nConsistency Check:")
        all_match = count_normal == count_streaming == count_lazy
        if time_mlxtend is not None:
            all_match = all_match and (count_normal == len(mlxtend_result))

        if all_match:
            print("  ✓ All implementations produce the same number of itemsets")
        else:
            print("  ✗ MISMATCH DETECTED - Results differ!")

        # Check if actual itemsets are the same (not just counts)
        print("\nDetailed Itemset Comparison:")

        def extract_itemsets(result):
            """Extract all itemsets as frozen sets for comparison."""
            all_itemsets = set()
            for level in result:
                if len(level) > 0:
                    for itemset in level:
                        all_itemsets.add(frozenset(itemset))
            return all_itemsets

        itemsets_normal = extract_itemsets(result_normal)
        itemsets_streaming = extract_itemsets(result_streaming)
        itemsets_lazy = extract_itemsets(result_lazy)

        if itemsets_normal == itemsets_streaming:
            print("  ✓ fp_growth == fp_growth_streaming")
        else:
            print("  ✗ fp_growth != fp_growth_streaming")
            print(f"    Only in normal: {itemsets_normal - itemsets_streaming}")
            print(f"    Only in streaming: {itemsets_streaming - itemsets_normal}")

        if itemsets_normal == itemsets_lazy:
            print("  ✓ fp_growth == Lazy API")
        else:
            print("  ✗ fp_growth != Lazy API")
            print(f"    Only in normal: {itemsets_normal - itemsets_lazy}")
            print(f"    Only in lazy: {itemsets_lazy - itemsets_normal}")

        if time_mlxtend is not None:
            itemsets_mlxtend = set()
            for _, row in mlxtend_result.iterrows():
                itemsets_mlxtend.add(frozenset(row["itemsets"]))

            if itemsets_normal == itemsets_mlxtend:
                print("  ✓ fp_growth == mlxtend")
            else:
                print("  ✗ fp_growth != mlxtend")
                print(f"    Only in priors: {itemsets_normal - itemsets_mlxtend}")
                print(f"    Only in mlxtend: {itemsets_mlxtend - itemsets_normal}")

    print_separator("TEST COMPLETED")


if __name__ == "__main__":
    test_all_functions_with_output()
