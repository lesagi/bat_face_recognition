#!/usr/bin/env python3
"""
Main entry point for the yolo_augmenter module.

This allows running the module with:
    python -m app.yolo_augmenter.square_crop_pipeline
"""

from .square_crop_pipeline import main

if __name__ == "__main__":
    main()