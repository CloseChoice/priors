#!/bin/bash
# Helper script to run ASV benchmarks locally

echo "🔧 ASV Benchmark Helper"
echo ""
echo "Options:"
echo "  1) Benchmark last 3 commits (safe)"
echo "  2) Benchmark from 0.1.0 release onwards"
echo "  3) Benchmark only HEAD (fastest)"
echo ""
read -p "Choose option (1-3): " choice

case $choice in
  1)
    echo "Running: asv run HEAD~3..HEAD --quick"
    asv run HEAD~3..HEAD --quick --show-stderr
    ;;
  2)
    echo "Running: asv run 5adfd57..HEAD --quick"
    asv run 5adfd57..HEAD --quick --show-stderr
    ;;
  3)
    echo "Running: asv run HEAD^! --quick"
    asv run HEAD^! --quick --show-stderr
    ;;
  *)
    echo "Invalid option"
    exit 1
    ;;
esac

echo ""
echo "✅ Done! Generate HTML with:"
echo "   asv publish"
echo "   asv preview"
