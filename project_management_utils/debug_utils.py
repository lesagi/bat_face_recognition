#!/usr/bin/env python3
"""
Debug utilities for the Bat Face Recognition project.

Provides a centralized debug system with global debug flag and print_debug method.
"""

import os
import sys
from typing import Any

# Global debug flag
_DEBUG_ENABLED = False

def set_debug(enabled: bool):
    """Set global debug mode.
    
    Args:
        enabled: Whether to enable debug output
    """
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled

def is_debug_enabled() -> bool:
    """Check if debug mode is enabled.
    
    Returns:
        True if debug is enabled, False otherwise
    """
    return _DEBUG_ENABLED

def print_debug(*args, **kwargs):
    """Print debug message only if debug is enabled.
    
    This function will return immediately without doing anything
    if the debug flag is False.
    
    Args:
        *args: Arguments to pass to print()
        **kwargs: Keyword arguments to pass to print()
    """
    if not _DEBUG_ENABLED:
        return
    
    print(*args, **kwargs)

def debug_section(title: str, enabled: bool = None):
    """Context manager for debug sections.
    
    Args:
        title: Title of the debug section
        enabled: Override debug flag for this section (optional)
    
    Usage:
        with debug_section("Processing Image"):
            print_debug("Step 1: Loading image...")
            print_debug("Step 2: Processing...")
    """
    if enabled is None:
        enabled = _DEBUG_ENABLED
    
    if enabled:
        print(f"\n{'='*20} {title} {'='*20}")
    
    return enabled

def debug_function(func):
    """Decorator to automatically enable debug for a function.
    
    Args:
        func: Function to decorate
        
    Usage:
        @debug_function
        def my_function():
            print_debug("This will only print if debug is enabled")
    """
    def wrapper(*args, **kwargs):
        # Enable debug for this function call
        original_debug = _DEBUG_ENABLED
        set_debug(True)
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore original debug state
            set_debug(original_debug)
    
    return wrapper

def debug_class(cls):
    """Decorator to automatically enable debug for a class.
    
    Args:
        cls: Class to decorate
        
    Usage:
        @debug_class
        class MyClass:
            def my_method(self):
                print_debug("This will only print if debug is enabled")
    """
    # Store original methods
    original_methods = {}
    
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith('_'):
            original_methods[attr_name] = attr
            
            def make_debug_method(method):
                def debug_method(self, *args, **kwargs):
                    print_debug(f"🔍 {cls.__name__}.{method.__name__}: Called with {len(args)} args")
                    result = method(self, *args, **kwargs)
                    print_debug(f"🔍 {cls.__name__}.{method.__name__}: Completed")
                    return result
                return debug_method
            
            setattr(cls, attr_name, make_debug_method(attr))
    
    return cls

# Environment variable support
def init_debug_from_env():
    """Initialize debug mode from environment variable DEBUG.
    
    Set DEBUG=1 or DEBUG=true to enable debug mode.
    """
    debug_env = os.environ.get('DEBUG', '').lower()
    if debug_env in ('1', 'true', 'yes', 'on'):
        set_debug(True)
        print_debug("🔧 Debug mode enabled via environment variable")

# Auto-initialize from environment
init_debug_from_env()

if __name__ == "__main__":
    # Test the debug system
    print("Testing debug utilities...")
    
    print_debug("This should not print (debug disabled)")
    
    set_debug(True)
    print_debug("This should print (debug enabled)")
    
    with debug_section("Test Section"):
        print_debug("Inside debug section")
    
    set_debug(False)
    print_debug("This should not print again (debug disabled)")
    
    print("Debug utility test completed!")
