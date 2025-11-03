#!/usr/bin/env python3
"""
Script to run comprehensive benchmarks and generate reports
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_pytest_benchmark():
    """Run pytest-benchmark and save results"""
    print("Running pytest-benchmark...")

    # Run benchmark with JSON output
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "benchmarks/benchmark_apriori.py",
        "--benchmark-json=benchmark_results.json",
        "--benchmark-min-rounds=3",
        "--benchmark-sort=mean",
        "-v",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Benchmark completed successfully!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Benchmark failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False


def analyze_benchmark_results():
    """Analyze and pretty-print benchmark results"""
    results_file = Path("benchmark_results.json")

    if not results_file.exists():
        print("No benchmark results found!")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    benchmarks = data["benchmarks"]

    # Create a summary table
    summary_data = []

    for bench in benchmarks:
        name = bench["name"]

        # Extract implementation and dataset size
        if "priors" in name:
            implementation = "Priors (Rust)"
        elif "mlxtend" in name:
            implementation = "MLxtend"
        elif "efficient_apriori" in name:
            implementation = "Efficient-Apriori"
        else:
            implementation = "Unknown"

        if "small" in name:
            dataset = "Small (1K tx, 20 items)"
        elif "medium" in name:
            dataset = "Medium (5K tx, 50 items)"
        elif "large" in name:
            dataset = "Large (10K tx, 100 items)"
        elif "correlated" in name:
            dataset = "Correlated (3K tx, 25 items)"
        else:
            dataset = "Unknown"

        stats = bench["stats"]
        summary_data.append(
            {
                "Implementation": implementation,
                "Dataset": dataset,
                "Mean (s)": f"{stats['mean']:.6f}",
                "Std (s)": f"{stats['stddev']:.6f}",
                "Min (s)": f"{stats['min']:.6f}",
                "Max (s)": f"{stats['max']:.6f}",
                "Rounds": stats["rounds"],
            }
        )

    # Create DataFrame and display
    df = pd.DataFrame(summary_data)

    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print(df.to_string(index=False))

    # Calculate speedups
    print("\n" + "=" * 80)
    print("SPEEDUP ANALYSIS (relative to Priors)")
    print("=" * 80)

    datasets = df["Dataset"].unique()
    for dataset in datasets:
        dataset_df = df[df["Dataset"] == dataset].copy()

        # Find Priors baseline
        priors_row = dataset_df[dataset_df["Implementation"] == "Priors (Rust)"]
        if len(priors_row) == 0:
            continue

        priors_time = float(priors_row["Mean (s)"].iloc[0])

        print(f"\nDataset: {dataset}")
        print("-" * 50)

        for _, row in dataset_df.iterrows():
            impl = row["Implementation"]
            time_val = float(row["Mean (s)"])

            if impl == "Priors (Rust)":
                speedup = 1.0
                status = "(baseline)"
            else:
                speedup = time_val / priors_time
                if speedup < 1.0:
                    status = f"({speedup:.2f}x faster than Priors)"
                else:
                    status = f"({speedup:.2f}x slower than Priors)"

            print(f"{impl:20s}: {time_val:.6f}s {status}")


def main():
    """Main function"""
    print("Starting comprehensive Apriori benchmark...")

    # First run the correctness test
    print("Running correctness validation...")
    try:
        from benchmark_apriori import test_correctness_comparison

        test_correctness_comparison()
        print("✓ Correctness validation passed!")
    except Exception as e:
        print(f"✗ Correctness validation failed: {e}")
        return

    # Run benchmarks
    if run_pytest_benchmark():
        analyze_benchmark_results()
    else:
        print("Benchmark execution failed!")


if __name__ == "__main__":
    main()
