"""
Shared utilities for priors library.

Used by both tests and benchmarks.
"""

import numpy as np


def count_itemsets(result):
    """
    Count total itemsets from priors result format.

    Args:
        result: Result from fp_growth() - can be list of arrays or single array

    Returns:
        int: Total number of itemsets found
    """
    if result is None:
        return 0
    if isinstance(result, list):
        return sum(level.shape[0] for level in result
                  if level is not None and hasattr(level, 'shape') and level.shape[0] > 0)
    if hasattr(result, 'shape'):
        return result.shape[0]
    return 0


def generate_transactions(num_transactions, num_items, avg_size, seed=42):
    """
    Generate random transaction data for testing/benchmarking.

    Args:
        num_transactions: Number of transactions to generate
        num_items: Number of unique items
        avg_size: Average number of items per transaction
        seed: Random seed for reproducibility

    Returns:
        np.ndarray: Binary matrix of shape (num_transactions, num_items)
    """
    np.random.seed(seed)
    transactions = np.zeros((num_transactions, num_items), dtype=np.int32)

    for i in range(num_transactions):
        size = max(1, int(np.random.normal(avg_size, avg_size * 0.3)))
        size = min(size, num_items)
        items = np.random.choice(num_items, size, replace=False)
        transactions[i, items] = 1

    return transactions


def generate_all_ones_transactions(num_transactions, num_items):
    """
    Generate trivial dataset with all 1s.

    Args:
        num_transactions: Number of transactions
        num_items: Number of items

    Returns:
        np.ndarray: Binary matrix filled with ones
    """
    return np.ones((num_transactions, num_items), dtype=np.int32)


def extract_itemsets_from_mlxtend(mlxtend_result):
    """
    Extract itemsets from mlxtend result format.

    Args:
        mlxtend_result: DataFrame from mlxtend.fpgrowth()

    Returns:
        set: Set of itemsets as tuples
    """
    itemsets = set()
    if mlxtend_result is not None and len(mlxtend_result) > 0:
        for _, row in mlxtend_result.iterrows():
            itemset = tuple(sorted(row['itemsets']))
            itemsets.add(itemset)
    return itemsets


def extract_itemsets_from_efficient_apriori(ea_itemsets):
    """
    Extract itemsets from efficient_apriori result format.

    Args:
        ea_itemsets: Dict from efficient_apriori.apriori()

    Returns:
        set: Set of itemsets as tuples
    """
    itemsets = set()
    if ea_itemsets:
        for size_k, itemsets_k in ea_itemsets.items():
            for itemset in itemsets_k:
                itemsets.add(tuple(sorted(itemset)))
    return itemsets


def extract_itemsets_from_priors(priors_result):
    """
    Extract itemsets from priors result format.

    Args:
        priors_result: Result from priors.fp_growth()

    Returns:
        set: Set of itemsets (simplified representation)
    """
    itemsets = set()
    if priors_result is not None:
        if isinstance(priors_result, list):
            for level_idx, level in enumerate(priors_result):
                if level is not None and hasattr(level, 'shape') and level.shape[0] > 0:
                    for i in range(level.shape[0]):
                        itemsets.add((level_idx, i))
    return itemsets


def extract_itemsets_from_result(result):
    """
    Extract itemset details from priors result for comparison.

    Args:
        result: Result from priors.fp_growth()

    Returns:
        set: Set of itemsets as tuples
    """
    itemsets = []
    if result is None:
        return set()

    if isinstance(result, list):
        for level_idx, level in enumerate(result):
            if level is not None and hasattr(level, 'shape') and level.shape[0] > 0:
                for i in range(level.shape[0]):
                    if hasattr(level, '__getitem__'):
                        itemsets.append(tuple(sorted(level[i])))
                    else:
                        itemsets.append(tuple(range(level_idx + 1)))

    return set(itemsets)
