#!/bin/bash
# Test script to simulate the workflow logic

echo "Testing incremental benchmark logic..."
echo ""

# Simulate the workflow
if [ -f .asv/results/benchmarks.json ] && [ -d .asv/results/MacBook-Pro-von-Cedric.local/ ]; then
  echo "✅ Found cached results - would run: asv run NEW --quick"
  echo "📊 This is the FAST path (only new commits)"
else
  echo "🆕 No cached results - would run: asv run 5adfd57..HEAD --quick"
  echo "⏱️  This is the SLOW path (first run)"
fi

echo ""
echo "Results directory contents:"
ls -la .asv/results/ 2>/dev/null || echo "  (directory doesn't exist)"
