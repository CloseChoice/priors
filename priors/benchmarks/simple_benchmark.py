#!/usr/bin/env python3
"""
Simple manual benchmark to test all implementations quickly
"""

import numpy as np
import time
from typing import List

# Import implementations
import priors
from mlxtend.frequent_patterns import apriori as mlxtend_apriori
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
from efficient_apriori import apriori as efficient_apriori


def create_test_data(num_transactions=1000, num_items=20, avg_size=5):
    """Create simple test data"""
    np.random.seed(42)
    
    # Create binary matrix
    binary_matrix = np.zeros((num_transactions, num_items), dtype=np.int32)
    transaction_lists = []
    
    for i in range(num_transactions):
        # Random transaction size
        size = max(1, min(num_items, int(np.random.normal(avg_size, 2))))
        
        # Random items
        items = np.random.choice(num_items, size, replace=False)
        
        # Fill binary matrix
        binary_matrix[i, items] = 1
        
        # Store as list
        transaction_lists.append(items.tolist())
    
    return binary_matrix, transaction_lists


def benchmark_implementation(name, func, *args, **kwargs):
    """Benchmark a single implementation"""
    print(f"\nTesting {name}...")
    
    try:
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        duration = end_time - start_time
        print(f"✓ {name}: {duration:.4f}s")
        
        # Try to get result size
        if hasattr(result, '__len__'):
            print(f"  Result size: {len(result)}")
        elif isinstance(result, tuple) and len(result) >= 1:
            if hasattr(result[0], '__len__'):
                print(f"  Result size: {len(result[0])}")
        
        return duration, result
        
    except Exception as e:
        print(f"✗ {name} failed: {e}")
        return None, None


def run_priors(binary_matrix, min_support):
    """Run our Rust implementation"""
    return priors.apriori(binary_matrix, min_support)


def run_mlxtend(binary_matrix, min_support):
    """Run MLxtend implementation"""
    df = pd.DataFrame(binary_matrix, columns=[f'item_{i}' for i in range(binary_matrix.shape[1])])
    df = df.astype(bool)
    return mlxtend_apriori(df, min_support=min_support, use_colnames=True)


def run_efficient_apriori(transaction_lists, min_support, num_transactions):
    """Run efficient-apriori implementation"""
    transactions_tuples = [tuple(tx) for tx in transaction_lists]
    # efficient-apriori expects relative support between 0 and 1
    itemsets, rules = efficient_apriori(transactions_tuples, min_support=min_support)
    return itemsets, rules


def main():
    """Run simple benchmark"""
    print("Simple Apriori Benchmark")
    print("=" * 50)
    
    # Create test data
    binary_matrix, transaction_lists = create_test_data(1000, 20, 5)
    min_support = 0.05  # 5%
    num_transactions = len(transaction_lists)
    
    print(f"Dataset: {num_transactions} transactions, {binary_matrix.shape[1]} items")
    print(f"Min support: {min_support} ({min_support * num_transactions:.0f} transactions)")
    
    # Benchmark all implementations
    results = {}
    
    # Our Rust implementation
    duration, result = benchmark_implementation(
        "Priors (Rust)",
        run_priors,
        binary_matrix, min_support
    )
    if duration is not None:
        results['priors'] = duration
    
    # MLxtend
    duration, result = benchmark_implementation(
        "MLxtend", 
        run_mlxtend,
        binary_matrix, min_support
    )
    if duration is not None:
        results['mlxtend'] = duration
    
    # Efficient-apriori
    duration, result = benchmark_implementation(
        "Efficient-Apriori",
        run_efficient_apriori,
        transaction_lists, min_support, num_transactions
    )
    if duration is not None:
        results['efficient_apriori'] = duration
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    if results:
        fastest = min(results.items(), key=lambda x: x[1])
        print(f"Fastest: {fastest[0]} ({fastest[1]:.4f}s)")
        
        if 'priors' in results:
            baseline = results['priors']
            print(f"\nSpeedup vs Priors (Rust):")
            for name, duration in results.items():
                if name != 'priors':
                    speedup = duration / baseline
                    if speedup > 1:
                        print(f"  {name}: {speedup:.2f}x slower")
                    else:
                        print(f"  {name}: {1/speedup:.2f}x faster")
    else:
        print("No successful benchmarks!")


if __name__ == "__main__":
    main()