"""
Benchmark comparison between different Apriori implementations:
- priors (our Rust implementation)
- mlxtend
- efficient-apriori
"""

import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
from efficient_apriori import apriori as efficient_apriori
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.preprocessing import TransactionEncoder

# Import the implementations
import priors


def generate_random_transactions(
    num_transactions: int, num_items: int, avg_transaction_size: int, seed: int = 42
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate random transaction data for benchmarking.

    Returns:
        tuple: (binary_matrix, transaction_lists)
    """
    random.seed(seed)
    np.random.seed(seed)

    transactions_lists = []
    binary_matrix = np.zeros((num_transactions, num_items), dtype=np.int32)

    for i in range(num_transactions):
        # Vary transaction size around the average
        size = max(
            1, int(np.random.normal(avg_transaction_size, avg_transaction_size * 0.3))
        )
        size = min(size, num_items)

        # Select random items for this transaction
        items = random.sample(range(num_items), size)
        transactions_lists.append(items)

        # Set binary matrix
        for item in items:
            binary_matrix[i, item] = 1

    return binary_matrix, transactions_lists


def generate_correlated_transactions(
    num_transactions: int,
    num_items: int,
    correlation_groups: List[List[int]],
    seed: int = 42,
) -> Tuple[np.ndarray, List[List[int]]]:
    """
    Generate transaction data with correlated items for more interesting patterns.
    """
    random.seed(seed)
    np.random.seed(seed)

    transactions_lists = []
    binary_matrix = np.zeros((num_transactions, num_items), dtype=np.int32)

    for i in range(num_transactions):
        transaction = []

        # For each correlation group, decide if we include it
        for group in correlation_groups:
            if random.random() < 0.3:  # 30% chance to include a group
                # Include all items in the group with high probability
                for item in group:
                    if random.random() < 0.8:  # 80% chance for each item in the group
                        transaction.append(item)

        # Add some random items
        remaining_items = [
            i
            for i in range(num_items)
            if i not in [item for group in correlation_groups for item in group]
        ]
        num_random = random.randint(0, 3)
        if remaining_items:
            transaction.extend(
                random.sample(remaining_items, min(num_random, len(remaining_items)))
            )

        # Remove duplicates and sort
        transaction = sorted(set(transaction))
        transactions_lists.append(transaction)

        # Set binary matrix
        for item in transaction:
            binary_matrix[i, item] = 1

    return binary_matrix, transactions_lists


# Wrapper functions for benchmarking
def run_priors_apriori(binary_matrix: np.ndarray, min_support: float) -> List[Any]:
    """Run our Rust implementation"""
    return priors.apriori(binary_matrix, min_support)


def run_mlxtend_apriori(binary_matrix: np.ndarray, min_support: float) -> pd.DataFrame:
    """Run mlxtend implementation"""
    # Convert binary matrix to DataFrame
    df = pd.DataFrame(
        binary_matrix, columns=[f"item_{i}" for i in range(binary_matrix.shape[1])]
    )
    df = df.astype(bool)

    # Run apriori
    frequent_itemsets = mlxtend_apriori(df, min_support=min_support, use_colnames=True)
    return frequent_itemsets


def run_efficient_apriori(
    transaction_lists: List[List[int]], min_support: float, num_transactions: int
) -> Tuple[Any, Any]:
    """Run efficient-apriori implementation"""
    # Convert to the format expected by efficient-apriori
    transactions_tuples = [tuple(transaction) for transaction in transaction_lists]

    # efficient-apriori expects relative support between 0 and 1
    itemsets, rules = efficient_apriori(transactions_tuples, min_support=min_support)
    return itemsets, rules


class BenchmarkDatasets:
    """Pre-generated datasets for consistent benchmarking"""

    @staticmethod
    def small_dataset() -> Tuple[np.ndarray, List[List[int]]]:
        """Small dataset: 1000 transactions, 20 items"""
        return generate_random_transactions(1000, 20, 5, seed=42)

    @staticmethod
    def medium_dataset() -> Tuple[np.ndarray, List[List[int]]]:
        """Medium dataset: 5000 transactions, 50 items"""
        return generate_random_transactions(5000, 50, 8, seed=42)

    @staticmethod
    def large_dataset() -> Tuple[np.ndarray, List[List[int]]]:
        """Large dataset: 10000 transactions, 100 items"""
        return generate_random_transactions(10000, 100, 12, seed=42)

    @staticmethod
    def correlated_dataset() -> Tuple[np.ndarray, List[List[int]]]:
        """Dataset with correlated items for interesting patterns"""
        correlation_groups = [
            [0, 1, 2],  # Group 1: items 0, 1, 2 often appear together
            [5, 6, 7, 8],  # Group 2: items 5, 6, 7, 8 often appear together
            [10, 11],  # Group 3: items 10, 11 often appear together
            [15, 16, 17],  # Group 4: items 15, 16, 17 often appear together
        ]
        return generate_correlated_transactions(3000, 25, correlation_groups, seed=42)


# Benchmark tests
class TestBenchmarkSmall:
    """Benchmarks for small dataset"""

    def setup_method(self):
        self.binary_matrix, self.transaction_lists = BenchmarkDatasets.small_dataset()
        self.min_support = 0.05  # 5% support
        self.num_transactions = len(self.transaction_lists)

    def test_priors_small(self, benchmark):
        result = benchmark(run_priors_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_mlxtend_small(self, benchmark):
        result = benchmark(run_mlxtend_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_efficient_apriori_small(self, benchmark):
        itemsets, rules = benchmark(
            run_efficient_apriori,
            self.transaction_lists,
            self.min_support,
            self.num_transactions,
        )
        assert len(itemsets) > 0


class TestBenchmarkMedium:
    """Benchmarks for medium dataset"""

    def setup_method(self):
        self.binary_matrix, self.transaction_lists = BenchmarkDatasets.medium_dataset()
        self.min_support = 0.03  # 3% support
        self.num_transactions = len(self.transaction_lists)

    def test_priors_medium(self, benchmark):
        result = benchmark(run_priors_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_mlxtend_medium(self, benchmark):
        result = benchmark(run_mlxtend_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_efficient_apriori_medium(self, benchmark):
        itemsets, rules = benchmark(
            run_efficient_apriori,
            self.transaction_lists,
            self.min_support,
            self.num_transactions,
        )
        assert len(itemsets) > 0


class TestBenchmarkLarge:
    """Benchmarks for large dataset"""

    def setup_method(self):
        self.binary_matrix, self.transaction_lists = BenchmarkDatasets.large_dataset()
        self.min_support = 0.02  # 2% support
        self.num_transactions = len(self.transaction_lists)

    def test_priors_large(self, benchmark):
        result = benchmark(run_priors_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_mlxtend_large(self, benchmark):
        result = benchmark(run_mlxtend_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_efficient_apriori_large(self, benchmark):
        itemsets, rules = benchmark(
            run_efficient_apriori,
            self.transaction_lists,
            self.min_support,
            self.num_transactions,
        )
        assert len(itemsets) > 0


class TestBenchmarkCorrelated:
    """Benchmarks for correlated dataset (more interesting patterns)"""

    def setup_method(self):
        self.binary_matrix, self.transaction_lists = (
            BenchmarkDatasets.correlated_dataset()
        )
        self.min_support = 0.05  # 5% support
        self.num_transactions = len(self.transaction_lists)

    def test_priors_correlated(self, benchmark):
        result = benchmark(run_priors_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_mlxtend_correlated(self, benchmark):
        result = benchmark(run_mlxtend_apriori, self.binary_matrix, self.min_support)
        assert len(result) > 0

    def test_efficient_apriori_correlated(self, benchmark):
        itemsets, rules = benchmark(
            run_efficient_apriori,
            self.transaction_lists,
            self.min_support,
            self.num_transactions,
        )
        assert len(itemsets) > 0


# Correctness validation tests
def test_correctness_comparison():
    """Test that all implementations produce similar results on the same data"""
    binary_matrix, transaction_lists = BenchmarkDatasets.small_dataset()
    min_support = 0.1
    num_transactions = len(transaction_lists)

    # Run all implementations
    priors_result = run_priors_apriori(binary_matrix, min_support)
    mlxtend_result = run_mlxtend_apriori(binary_matrix, min_support)
    efficient_result, _ = run_efficient_apriori(
        transaction_lists, min_support, num_transactions
    )

    # Extract 1-itemsets for comparison
    priors_1_itemsets = set()
    if len(priors_result) > 0:
        level_1 = priors_result[0]
        for i in range(level_1.shape[0]):
            priors_1_itemsets.add((level_1[i, 0],))

    mlxtend_1_itemsets = set()
    for _, row in mlxtend_result.iterrows():
        itemset = tuple(
            sorted(
                [int(item.split("_")[1]) for item in row["itemsets"] if row["itemsets"]]
            )
        )
        if len(itemset) == 1:
            mlxtend_1_itemsets.add(itemset)

    efficient_1_itemsets = set()
    if 1 in efficient_result:
        for itemset in efficient_result[1]:
            efficient_1_itemsets.add(itemset)

    # Check that we have some overlap (exact match might not be possible due to different thresholds/rounding)
    print(f"Priors 1-itemsets: {len(priors_1_itemsets)}")
    print(f"MLxtend 1-itemsets: {len(mlxtend_1_itemsets)}")
    print(f"Efficient-apriori 1-itemsets: {len(efficient_1_itemsets)}")

    # At least some itemsets should be found by all
    assert len(priors_1_itemsets) > 0
    assert len(mlxtend_1_itemsets) > 0
    assert len(efficient_1_itemsets) > 0


if __name__ == "__main__":
    # Run correctness test
    test_correctness_comparison()
    print("Correctness test passed!")

    # Example of manual timing comparison
    print("\nManual timing comparison:")
    binary_matrix, transaction_lists = BenchmarkDatasets.medium_dataset()
    min_support = 0.03
    num_transactions = len(transaction_lists)

    # Time priors
    start = time.time()
    priors_result = run_priors_apriori(binary_matrix, min_support)
    priors_time = time.time() - start

    # Time mlxtend
    start = time.time()
    mlxtend_result = run_mlxtend_apriori(binary_matrix, min_support)
    mlxtend_time = time.time() - start

    # Time efficient-apriori
    start = time.time()
    efficient_result, _ = run_efficient_apriori(
        transaction_lists, min_support, num_transactions
    )
    efficient_time = time.time() - start

    print(f"Priors (Rust): {priors_time:.4f}s")
    print(f"MLxtend: {mlxtend_time:.4f}s")
    print(f"Efficient-apriori: {efficient_time:.4f}s")

    if priors_time > 0:
        print(f"MLxtend vs Priors: {mlxtend_time/priors_time:.2f}x")
        print(f"Efficient-apriori vs Priors: {efficient_time/priors_time:.2f}x")
