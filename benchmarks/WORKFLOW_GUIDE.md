# 🔄 ASV Workflow Guide - Incremental Benchmarking

## 📊 How the CI/CD Benchmarking Works

### First Run (Initial Benchmark)
```
Push to ci/benchmarking or main
  ↓
GitHub Actions starts
  ↓
No cached results found
  ↓
Runs: asv run 5adfd57..HEAD --quick
  ↓
Benchmarks ALL commits from v0.1.0 to HEAD (~5-10 minutes)
  ↓
Caches results in .asv/results/
  ↓
Generates HTML and deploys to GitHub Pages
```

### Subsequent Runs (Incremental)
```
New commit pushed
  ↓
GitHub Actions starts
  ↓
Restores cached results from previous run
  ↓
Runs: asv run NEW --quick
  ↓
Only benchmarks NEW commits since last run (~30 seconds!)
  ↓
Updates cache
  ↓
Regenerates HTML and deploys
```

## 🚀 Performance Comparison

| Run Type | Commits Benchmarked | Time | Cache Used |
|----------|-------------------|------|------------|
| **First Run** | All from v0.1.0 (~20 commits) | ~5-10 min | ❌ No |
| **Incremental** | Only new (1-3 commits) | ~30-60 sec | ✅ Yes |
| **No New Commits** | Just HEAD verification | ~10 sec | ✅ Yes |

## 🎯 What Happens in Different Scenarios

### Scenario 1: First Push to Branch
```bash
# Workflow runs full benchmark
asv run 5adfd57..HEAD --quick

# Results:
# - Benchmarks ~20 commits
# - Takes ~5-10 minutes
# - Caches all results
```

### Scenario 2: New Commit on Same Branch
```bash
# Workflow detects cached results
asv run NEW --quick

# Results:
# - Only benchmarks 1 new commit
# - Takes ~30 seconds
# - Updates cache
```

### Scenario 3: Multiple New Commits
```bash
# You push 3 new commits
git push

# Workflow runs:
asv run NEW --quick

# Results:
# - Benchmarks 3 new commits
# - Takes ~1-2 minutes
# - Updates cache
```

### Scenario 4: PR from Feature Branch
```bash
# PR opened: feature-branch -> main

# Workflow runs:
# 1. Tries to use cache from main branch
# 2. If available: Only benchmarks new commits from feature
# 3. If not: Full benchmark from v0.1.0

# Results:
# - Comments on PR with benchmark link
# - No deployment (PRs don't deploy to gh-pages)
```

## 📈 Cache Strategy

### Cache Key Structure
```
asv-results-{branch_name}-{commit_sha}
```

### Cache Restore Order
1. Try: `asv-results-ci/benchmarking-abc123` (exact match)
2. Fallback: `asv-results-ci/benchmarking-` (any from this branch)
3. Fallback: `asv-results-main-` (any from main)
4. Fallback: `asv-results-` (any cached results)

This ensures:
- ✅ Fast rebuilds on same branch
- ✅ Reasonable performance on new branches
- ✅ No unnecessary full rebuilds

## 🔍 Monitoring in GitHub Actions

### Check Progress
1. Go to: **Actions** tab in GitHub
2. Click on **ASV Benchmarks** workflow
3. Click on latest run
4. Expand **Run benchmarks** step

You'll see:
```
🔍 Checking for existing benchmark results...
✅ Found cached results, running incremental benchmarks...
📊 This will only benchmark NEW commits since last run (much faster!)
✅ Incremental benchmark completed
✅ Benchmark run completed!
```

### View Summary
Click on **Summary** tab in the workflow run to see:
- 📊 Number of result files generated
- 📝 List of benchmarked commits
- 🔗 Link to live results

## ⚠️ Troubleshooting

### Problem: "No new commits to benchmark"
**Meaning**: All commits already benchmarked
**Action**: None needed - workflow verifies HEAD and continues

### Problem: "Some commits failed to build"
**Meaning**: Old commits with Python 3.10 encountered
**Action**: Workflow continues, benchmarks remaining commits

### Problem: Cache too large
**Meaning**: Results cache exceeds GitHub's 10GB limit
**Action**: Cache automatically rotates, keeps most recent

## 🎛️ Configuration

### Change Starting Commit
Edit `.github/workflows/asv-benchmarks.yml`:
```yaml
# Currently: 5adfd57 (v0.1.0)
asv run 5adfd57..HEAD --quick

# To change:
asv run YOUR_COMMIT..HEAD --quick
```

### Disable Incremental Benchmarks
```yaml
# Always run full benchmarks (slower but complete)
asv run 5adfd57..HEAD --quick
# Remove the "asv run NEW" line
```

### Benchmark More/Fewer Commits
```yaml
# More commits (slower but more history)
asv run 0240d7e..HEAD --quick  # From very first commit

# Fewer commits (faster)
asv run HEAD~5..HEAD --quick  # Only last 5 commits
```

## 📚 Best Practices

### For Contributors
- ✅ Trust the incremental system - it's fast!
- ✅ Check workflow logs if benchmarks fail
- ✅ PRs automatically get benchmark results commented

### For Maintainers
- ✅ First run takes time - be patient
- ✅ Monitor cache size in Actions settings
- ✅ Consider manual runs for important releases

### Manual Trigger
You can manually trigger benchmarks:
1. Go to **Actions** → **ASV Benchmarks**
2. Click **Run workflow**
3. Select branch
4. Click **Run workflow**

## 🔗 Related Files
- [Workflow Config](.github/workflows/asv-benchmarks.yml)
- [ASV Config](../asv.conf.json)
- [Local Guide](./LOCAL_GUIDE.md)
- [Main README](./README.md)
