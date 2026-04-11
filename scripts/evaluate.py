#!/usr/bin/env python3
"""
Thin CLI wrapper for model evaluation / prediction generation.

Usage:
    python scripts/evaluate.py --model /path/to/model --input /path/to/data
"""

import subprocess
import sys


def main():
    cmd = [sys.executable, "-m", "app.generate_predictions"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
