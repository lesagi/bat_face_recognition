"""
Model Factory Exception Classes

This module defines exception classes for model configuration and creation errors.
These exceptions provide clear error messages and help with graceful error handling.
"""


class ModelConfigurationError(Exception):
    """Exception raised for model configuration issues.
    
    This exception is raised when there are problems with the model configuration
    data, such as missing required fields, invalid paths, or unsupported model types.
    """
    pass


class ModelCreationError(Exception):
    """Exception raised for model creation issues.
    
    This exception is raised when model instantiation fails, such as when
    dependencies are missing, model files are corrupted, or there are
    runtime errors during model loading.
    """
    pass