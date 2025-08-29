#!/usr/bin/env python3
"""
Main entry point for the yolo_augmenter module.

This allows running the module with:
    python -m yolo_augmenter
"""

from .cli import cli

if __name__ == "__main__":
    cli()