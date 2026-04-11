#!/usr/bin/env python3
"""
Thin CLI wrapper for Siamese network training.

Usage:
    python scripts/train.py --bat-type r --data-source video --background random
    python scripts/train.py --bat-type m --data-source video --background green
"""

import subprocess
import sys


def main():
    cmd = [sys.executable, "-m", "app.siamese_training.train_siamese"] + sys.argv[1:]
    sys.exit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
