"""
Pair class information data structures for Siamese network training.

This module provides DTOs for representing class information in training pairs,
supporting both single-class (positive) and multi-class (negative) pairs.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class ClassInfo:
    """
    Represents a single class in a training pair.
    
    Attributes:
        name: The class identifier (e.g., "bat_A", "bat_B")
        weight: Placeholder weight value (default 1.0). 
                Note: Actual weights are computed by ClassWeightCalculator.
                This field exists for potential future use or pre-computed values.
    """
    name: str
    weight: float = 1.0


@dataclass  
class PairClassInfo:
    """
    Represents all classes involved in a training pair.
    
    For positive pairs (same class): contains 1 ClassInfo
    For negative pairs (different classes): contains 2+ ClassInfo objects
    
    This DTO is serialized to strings for TensorFlow dataset compatibility.
    """
    classes: List[ClassInfo]
    
    def to_string(self) -> str:
        """
        Serialize to string for TensorFlow dataset compatibility.
        
        Format: "class_a:1.0|class_b:0.5|class_c:2.0"
        
        Returns:
            Serialized string representation
            
        Example:
            >>> pair = PairClassInfo.create_dual_class("bat_A", "bat_B")
            >>> pair.to_string()
            "bat_A:1.0|bat_B:1.0"
        """
        return "|".join([f"{c.name}:{c.weight}" for c in self.classes])
    
    @staticmethod
    def from_string(s: str) -> 'PairClassInfo':
        """
        Deserialize from string.
        
        Args:
            s: Serialized string in format "class_a:1.0|class_b:0.5"
            
        Returns:
            PairClassInfo object
            
        Raises:
            ValueError: If string format is invalid
            
        Example:
            >>> pair = PairClassInfo.from_string("bat_A:1.0|bat_B:1.0")
            >>> len(pair.classes)
            2
        """
        if not s or not isinstance(s, str):
            raise ValueError(f"Invalid input: expected non-empty string, got {type(s)}")
        
        classes = []
        for class_str in s.split("|"):
            parts = class_str.split(":")
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid class format: '{class_str}'. "
                    f"Expected format 'class_name:weight'"
                )
            name, weight_str = parts
            
            if not name:
                raise ValueError("Class name cannot be empty")
            
            try:
                weight = float(weight_str)
            except ValueError:
                raise ValueError(
                    f"Invalid weight value: '{weight_str}'. "
                    f"Expected numeric value."
                )
            
            classes.append(ClassInfo(name=name, weight=weight))
        
        return PairClassInfo(classes=classes)
    
    @staticmethod
    def create_single_class(class_name: str) -> 'PairClassInfo':
        """
        Helper to create PairClassInfo for positive pairs (single class).
        
        Args:
            class_name: Name of the class
            
        Returns:
            PairClassInfo with one ClassInfo
            
        Example:
            >>> pair = PairClassInfo.create_single_class("bat_A")
            >>> pair.to_string()
            "bat_A:1.0"
        """
        return PairClassInfo(classes=[ClassInfo(name=class_name)])
    
    @staticmethod
    def create_dual_class(class_a: str, class_b: str) -> 'PairClassInfo':
        """
        Helper to create PairClassInfo for negative pairs (two classes).
        
        Args:
            class_a: Name of first class
            class_b: Name of second class
            
        Returns:
            PairClassInfo with two ClassInfo objects
            
        Example:
            >>> pair = PairClassInfo.create_dual_class("bat_A", "bat_B")
            >>> pair.to_string()
            "bat_A:1.0|bat_B:1.0"
        """
        return PairClassInfo(classes=[
            ClassInfo(name=class_a),
            ClassInfo(name=class_b)
        ])

