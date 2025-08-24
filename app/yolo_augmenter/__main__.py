#!/usr/bin/env python3
"""
Entry point for YOLO Augmenter module when run as:
python -m app.yolo_augmenter
"""

from .cli import cli

if __name__ == "__main__":
    cli()