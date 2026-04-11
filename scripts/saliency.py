#!/usr/bin/env python3
"""
Thin CLI wrapper for saliency map generation.

Usage:
    python scripts/saliency.py --input_dir /path/to/input --model_path /path/to/model
"""

import subprocess
import sys


def main():
    cmd = [sys.executable, "-m", "app.visualization.run_saliency_maps"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
