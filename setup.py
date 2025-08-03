#!/usr/bin/env python3
"""
Setup script for Bat Face Recognition Project

This package provides a comprehensive solution for bat face recognition
using deep learning techniques including Siamese networks, YOLO models,
and advanced image processing.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
def read_requirements(filename):
    """Read requirements from file, filtering out comments and empty lines."""
    with open(filename, 'r') as f:
        return [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#') and not line.startswith('-r')
        ]

setup(
    name="bat-face-recognition",
    version="1.0.0",
    author="Bat Face Recognition Research Team",
    description="A comprehensive machine learning system for individual bat identification using computer vision",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    
    # Package discovery
    packages=find_packages(where="app"),
    package_dir={"": "app"},
    
    # Dependencies
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("requirements-dev.txt"),
        "exact": read_requirements("requirements-exact.txt"),
    },
    
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "bat-face-train=main:main",
            "bat-face-predict=generate_predictions:main",
            "bat-face-web=web_app.app:main",
            "bat-face-augment=face_annotation_eyes_nose.create_augmented_dataset:main",
        ],
    },
    
    # Package metadata
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    
    # Project URLs
    project_urls={
        "Documentation": "https://github.com/your-org/bat-face-recognition/docs",
        "Source": "https://github.com/your-org/bat-face-recognition",
        "Tracker": "https://github.com/your-org/bat-face-recognition/issues",
    },
    
    # Additional metadata
    keywords="computer-vision deep-learning machine-learning bat-recognition siamese-networks yolo wildlife-conservation",
    license="MIT",
    
    # Include additional files
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt", "*.md"],
    },
    
    # Minimum versions for Python and key dependencies
    python_requires=">=3.8",
    zip_safe=False,
)