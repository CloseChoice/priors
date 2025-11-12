# 🔧 Local ASV Benchmarking Guide

## Quick Start

### Option 1: Helper Script (Recommended)
```bash
./asv_helper.sh
```
Choose from:
1. Last 3 commits (fast, safe)
2. From 0.1.0 release onwards (complete)
3. Only HEAD (fastest)

### Option 2: Manual Commands

#### Benchmark only current commit (fastest)
```bash
asv run HEAD^! --quick
asv publish
asv preview
```

#### Benchmark last 2-3 commits
```bash
asv run HEAD~2..HEAD --quick
asv publish
asv preview
```

#### Benchmark from 0.1.0 release onwards (complete but slower)
```bash
asv run 5adfd57..HEAD --quick
asv publish
asv preview
```

## ⚠️ Common Issues

### Error: "library 'python3.10' not found"

**Problem**: ASV tries to build old commits that require Python 3.10, but you don't have it installed.

**Solution**: Use a limited commit range:
```bash
# ❌ Don't use: asv run HEAD~5..HEAD (goes too far back)
# ✅ Use instead:
asv run HEAD~2..HEAD --quick  # Only last 2 commits
# or
asv run 5adfd57..HEAD --quick  # From 0.1.0 release
```

### Slow benchmarks

**Solution**: Use `--quick` flag:
```bash
asv run HEAD^! --quick  # Much faster, less precise
```

### View results without running benchmarks

```bash
asv publish  # Generate HTML from existing results
asv preview  # Open in browser
```

## 📊 CI/CD vs Local

### GitHub Actions (Automatic)
- ✅ **First run**: Benchmarks all commits from 0.1.0 onwards
- ✅ **Subsequent runs**: Only benchmarks NEW commits (incremental)
- ✅ Caches previous results for speed
- ✅ Deploys to GitHub Pages automatically

### Local Development
- 📝 You control which commits to benchmark
- 💾 Results stored in `.asv/results/` (gitignored)
- 🔄 Use `--quick` for faster iteration
- 👀 Use `asv preview` to view locally

## 📈 Understanding ASV Commands

### `asv run` - Run benchmarks
```bash
asv run HEAD^!           # Only HEAD commit
asv run HEAD~5..HEAD     # Last 5 commits
asv run 5adfd57..HEAD    # From specific commit to HEAD
asv run NEW              # Only new commits (needs existing results)
asv run --quick          # Faster, less precise
asv run -E existing      # Use your current environment (fastest)
```

### `asv publish` - Generate HTML
```bash
asv publish              # Generate HTML from results
```

### `asv preview` - View results
```bash
asv preview              # Open browser with results
```

### `asv show` - Command line output
```bash
asv show HEAD            # Show results for HEAD
```

## 🎯 Workflow Recommendations

### Daily Development
```bash
# Quick benchmark of current work
asv run -E existing --quick
asv show HEAD
```

### Before PR
```bash
# Benchmark last few commits
asv run HEAD~3..HEAD --quick
asv publish
asv preview
```

### Full Analysis
```bash
# Complete benchmark from 0.1.0
asv run 5adfd57..HEAD
asv publish
asv preview
```

## 📚 More Info

- [ASV Documentation](https://asv.readthedocs.io/)
- [Main Benchmarks README](./README.md)
- [ASV Config](../asv.conf.json)
