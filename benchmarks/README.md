# Priors Benchmarks

This directory contains ASV (Airspeed Velocity) benchmarks for the priors library.

## 📊 View Results

Live benchmark results are available at: **https://closechoice.github.io/priors/**

## 🚀 Running Benchmarks Locally

### Quick Test
```bash
# Install ASV
pip install asv

# Run benchmarks on current code
asv run -E existing --quick

# View results
asv show
```

### Full Benchmark Suite
```bash
# Run benchmarks over multiple commits
asv run HEAD~10..HEAD

# Generate HTML report
asv publish

# View in browser
asv preview
```

## 📈 Benchmark Types

### Performance Benchmarks
- **FPGrowthSmall**: 1K transactions, 30 items
- **FPGrowthMedium**: 5K transactions, 50 items
- **FPGrowthLarge**: 10K transactions, 80 items
- **FPGrowthXLarge**: 50K transactions, 100 items
- **FPGrowthStreamingSmall**: Streaming algorithm on 1K transactions

### Scaling Benchmarks
- **TransactionScaling**: Tests performance across 1K-50K transactions
- **ItemScaling**: Tests performance across 20-200 items
- **SupportThreshold**: Tests with varying min_support values (0.01-0.10)

### Memory Benchmarks
Each performance benchmark also includes peak memory measurements.

## 🔧 Configuration

Benchmark configuration is in [`asv.conf.json`](../asv.conf.json):
- Python versions: 3.11, 3.12, 3.13
- Dependencies: numpy, pandas, mlxtend, efficient-apriori
- Build: Maturin (Rust extension)

## 🤖 Continuous Integration

Benchmarks run automatically via GitHub Actions on:
- Push to `ci/benchmarking` or `main` branches
- Pull requests to these branches
- Manual workflow dispatch

Results are automatically deployed to GitHub Pages.

## 📝 Adding New Benchmarks

Add new benchmark classes to [`benchmarks.py`](benchmarks.py):

```python
class MyNewBenchmark:
    def setup(self):
        # Setup code (runs once before benchmarks)
        import priors
        self.data = generate_test_data()

    def time_my_function(self):
        # Timing benchmark
        priors.my_function(self.data)

    def peakmem_my_function(self):
        # Memory benchmark
        priors.my_function(self.data)
```

## 📚 More Information

- [ASV Documentation](https://asv.readthedocs.io/)
- [Example: NumPy Benchmarks](https://pv.github.io/numpy-bench/)
- [Example: Pandas Benchmarks](https://pandas.pydata.org/speed/pandas/)
