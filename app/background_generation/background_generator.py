"""
Background generation utilities for image processing.

This module provides static methods for creating various types of backgrounds
that can be used for image processing and background replacement.
"""

import numpy as np
from typing import Tuple

# Try absolute imports first (when run as module), fall back to relative imports (when run directly)
try:
    from app.image_processor.image_utils import create_blur_image, get_random_cropped_image
except ImportError:
    # Fallback to relative imports when running script directly
    # Add parent directory to path for relative imports
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from image_processor.image_utils import create_blur_image, get_random_cropped_image


class BackgroundGenerator:
    """Factory class for creating different types of backgrounds."""

    @staticmethod
    def solid_color(
        height: int, width: int, color: Tuple[int, int, int] = (0, 255, 0)
    ) -> np.ndarray:
        """Create a solid color background.

        Args:
            height: Image height
            width: Image width
            color: BGR color tuple (default: green)

        Returns:
            Background image as numpy array
        """
        background = np.zeros((height, width, 3), dtype=np.uint8)
        background[:] = color
        return background

    @staticmethod
    def noise(height: int, width: int, intensity: int = 255) -> np.ndarray:
        """Create a random noise background.

        Args:
            height: Image height
            width: Image width
            intensity: Maximum noise intensity (0-255)

        Returns:
            Background image as numpy array
        """
        return np.random.randint(0, intensity + 1, (height, width, 3), dtype=np.uint8)

    @staticmethod
    def blur(height: int, width: int) -> np.ndarray:
        """Create a blurred random background.

        Args:
            height: Image height
            width: Image width

        Returns:
            Background image as numpy array
        """
        return create_blur_image(height, width)

    @staticmethod
    def picsum(
        height: int,
        width: int,
        max_retries: int = 3,
        timeout: float = 30,
        fallback_to_generated: bool = True,
    ) -> np.ndarray:
        """Create a background from Picsum Photos with retry mechanism.

        Args:
            height: Image height
            width: Image width
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Timeout for each request in seconds (default: 10)
            fallback_to_generated: If True, generates a fallback image when all retries fail

        Returns:
            Background image as numpy array
        """
        return get_random_cropped_image(
            height, width, max_retries, timeout, fallback_to_generated
        )

    @staticmethod
    def gradient(
        height: int,
        width: int,
        start_color: Tuple[int, int, int] = (0, 0, 0),
        end_color: Tuple[int, int, int] = (255, 255, 255),
        direction: str = "vertical",
    ) -> np.ndarray:
        """Create a gradient background.

        Args:
            height: Image height
            width: Image width
            start_color: Starting BGR color
            end_color: Ending BGR color
            direction: 'vertical', 'horizontal', or 'diagonal'

        Returns:
            Background image as numpy array
        """
        background = np.zeros((height, width, 3), dtype=np.uint8)

        if direction == "vertical":
            for i in range(height):
                ratio = i / height
                color = [
                    int(start_color[j] + (end_color[j] - start_color[j]) * ratio)
                    for j in range(3)
                ]
                background[i, :] = color
        elif direction == "horizontal":
            for i in range(width):
                ratio = i / width
                color = [
                    int(start_color[j] + (end_color[j] - start_color[j]) * ratio)
                    for j in range(3)
                ]
                background[:, i] = color
        elif direction == "diagonal":
            for i in range(height):
                for j in range(width):
                    ratio = (i + j) / (height + width)
                    color = [
                        int(start_color[k] + (end_color[k] - start_color[k]) * ratio)
                        for k in range(3)
                    ]
                    background[i, j] = color

        return background
