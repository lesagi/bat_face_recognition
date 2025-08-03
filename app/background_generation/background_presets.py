"""
Predefined background presets for common image processing use cases.

This module provides convenient preset generators for commonly used background types.
"""

from typing import Callable, Tuple
import numpy as np

from .background_generator import BackgroundGenerator


class BackgroundPresets:
    """Predefined background generators for common use cases."""

    @staticmethod
    def green() -> Callable[[int, int], np.ndarray]:
        """Green background generator."""
        return lambda h, w: BackgroundGenerator.solid_color(h, w, (0, 255, 0))

    @staticmethod
    def white() -> Callable[[int, int], np.ndarray]:
        """White background generator."""
        return lambda h, w: BackgroundGenerator.solid_color(h, w, (255, 255, 255))

    @staticmethod
    def black() -> Callable[[int, int], np.ndarray]:
        """Black background generator."""
        return lambda h, w: BackgroundGenerator.solid_color(h, w, (0, 0, 0))

    @staticmethod
    def blue() -> Callable[[int, int], np.ndarray]:
        """Blue background generator."""
        return lambda h, w: BackgroundGenerator.solid_color(h, w, (255, 0, 0))

    @staticmethod
    def red() -> Callable[[int, int], np.ndarray]:
        """Red background generator."""
        return lambda h, w: BackgroundGenerator.solid_color(h, w, (0, 0, 255))

    @staticmethod
    def solid_color(
        color: Tuple[int, int, int] = (0, 255, 0)
    ) -> Callable[[int, int], np.ndarray]:
        """Solid color background generator with custom color.

        Args:
            color: BGR color tuple (default: green)

        Returns:
            Background generator function
        """
        return lambda h, w: BackgroundGenerator.solid_color(h, w, color)

    @staticmethod
    def noise(intensity: int = 255) -> Callable[[int, int], np.ndarray]:
        """Random noise background generator with custom intensity.

        Args:
            intensity: Maximum noise intensity (0-255)

        Returns:
            Background generator function
        """
        return lambda h, w: BackgroundGenerator.noise(h, w, intensity)

    @staticmethod
    def blur() -> Callable[[int, int], np.ndarray]:
        """Blurred random background generator."""
        return BackgroundGenerator.blur

    @staticmethod
    def picsum(
        max_retries: int = 3, timeout: float = 10, fallback_to_generated: bool = True
    ) -> Callable[[int, int], np.ndarray]:
        """Random Picsum photo background generator with retry mechanism.

        Args:
            max_retries: Maximum number of retry attempts (default: 3)
            timeout: Timeout for each request in seconds (default: 10)
            fallback_to_generated: If True, generates a fallback image when all retries fail

        Returns:
            Background generator function
        """
        return lambda h, w: BackgroundGenerator.picsum(
            h, w, max_retries, timeout, fallback_to_generated
        )

    @staticmethod
    def gradient(
        start_color: Tuple[int, int, int] = (0, 0, 0),
        end_color: Tuple[int, int, int] = (255, 255, 255),
        direction: str = "vertical",
    ) -> Callable[[int, int], np.ndarray]:
        """Gradient background generator.

        Args:
            start_color: Starting BGR color
            end_color: Ending BGR color
            direction: 'vertical', 'horizontal', or 'diagonal'

        Returns:
            Background generator function
        """
        return lambda h, w: BackgroundGenerator.gradient(
            h, w, start_color, end_color, direction
        )
