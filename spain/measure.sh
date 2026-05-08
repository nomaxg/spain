#!/bin/bash
# measure.sh — cross-platform (macOS + Linux)
# Reports peak maximum resident set size in GB, while preserving stdout.

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <command> [args...]" >&2
  exit 1
fi

tmp=$(mktemp)

# Detect platform
os=$(uname -s)
case "$os" in
  Darwin)
    # macOS: /usr/bin/time -l (reports in bytes)
    /usr/bin/time -l -o "$tmp" "$@"
    mem_bytes=$(awk '/maximum resident set size/ {print $1}' "$tmp")
    mem_gb=$(awk -v b="$mem_bytes" 'BEGIN {printf("%.2f", b/1024/1024/1024)}')
    ;;
  Linux)
    # Linux: GNU time %M (reports in KB)
    /usr/bin/time -f "MAX_RSS_KB %M" -o "$tmp" "$@"
    mem_kb=$(awk '/MAX_RSS_KB/ {print $2}' "$tmp")
    mem_gb=$(awk -v kb="$mem_kb" 'BEGIN {printf("%.2f", kb/1024/1024)}')
    ;;
  *)
    echo "Unsupported OS: $os" >&2
    rm -f "$tmp"
    exit 1
    ;;
esac

echo "Peak memory: ${mem_gb} GB" >&2
rm -f "$tmp"
