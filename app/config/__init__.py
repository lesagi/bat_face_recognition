"""
Configuration package initialization.
"""

# Core configs
from .general import *

# Module-specific configs
from siamese_network.paths import *
from siamese_network.config import *

# from background_replacement.config import *  # File doesn't exist
from video_processing.config import *
from data_augmentation.config import *
