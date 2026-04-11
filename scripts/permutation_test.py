#!/usr/bin/env python3
"""
Thin CLI wrapper for permutation tests.

Usage:
    python scripts/permutation_test.py --help
    python scripts/permutation_test.py --observed-metrics metrics.json --n-permutations 100
"""

import subprocess
import sys


def main():
    cmd = [sys.executable, "-m", "app.statistical_tests.run_permutation_test"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
