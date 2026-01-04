#!/usr/bin/env python3
"""
Test suite for PairClassInfo DTOs.
Tests serialization, deserialization, and edge cases.
"""

import os
import sys
import unittest

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from siamese_data.pair_class_info import ClassInfo, PairClassInfo


class TestClassInfo(unittest.TestCase):
    """Test ClassInfo dataclass."""
    
    def test_create_with_defaults(self):
        """ClassInfo should default weight to 1.0."""
        info = ClassInfo(name="bat_A")
        self.assertEqual(info.name, "bat_A")
        self.assertEqual(info.weight, 1.0)
    
    def test_create_with_custom_weight(self):
        """ClassInfo should accept custom weight."""
        info = ClassInfo(name="bat_B", weight=0.5)
        self.assertEqual(info.name, "bat_B")
        self.assertEqual(info.weight, 0.5)
    
    def test_different_instances_are_independent(self):
        """Multiple ClassInfo instances should be independent."""
        info1 = ClassInfo(name="bat_A", weight=0.3)
        info2 = ClassInfo(name="bat_B", weight=0.7)
        self.assertNotEqual(info1.name, info2.name)
        self.assertNotEqual(info1.weight, info2.weight)


class TestPairClassInfoSerialization(unittest.TestCase):
    """Test PairClassInfo serialization to strings."""
    
    def test_single_class_serialization(self):
        """Single class should serialize correctly."""
        pair = PairClassInfo.create_single_class("bat_A")
        serialized = pair.to_string()
        self.assertEqual(serialized, "bat_A:1.0")
    
    def test_dual_class_serialization(self):
        """Two classes should serialize with pipe delimiter."""
        pair = PairClassInfo.create_dual_class("bat_A", "bat_B")
        serialized = pair.to_string()
        self.assertEqual(serialized, "bat_A:1.0|bat_B:1.0")
    
    def test_triple_class_serialization(self):
        """Three classes should serialize correctly (generic support)."""
        pair = PairClassInfo(classes=[
            ClassInfo(name="A"),
            ClassInfo(name="B"),
            ClassInfo(name="C")
        ])
        serialized = pair.to_string()
        self.assertEqual(serialized, "A:1.0|B:1.0|C:1.0")
    
    def test_custom_weights_serialization(self):
        """Custom weights should be preserved in serialization."""
        pair = PairClassInfo(classes=[
            ClassInfo(name="A", weight=0.25),
            ClassInfo(name="B", weight=0.75)
        ])
        serialized = pair.to_string()
        self.assertEqual(serialized, "A:0.25|B:0.75")
    
    def test_class_names_with_underscores(self):
        """Class names with underscores should serialize correctly."""
        pair = PairClassInfo.create_single_class("bat_A_123")
        serialized = pair.to_string()
        self.assertEqual(serialized, "bat_A_123:1.0")
    
    def test_class_names_with_hyphens(self):
        """Class names with hyphens should serialize correctly."""
        pair = PairClassInfo.create_dual_class("bat-A", "bat-B")
        serialized = pair.to_string()
        self.assertEqual(serialized, "bat-A:1.0|bat-B:1.0")


class TestPairClassInfoDeserialization(unittest.TestCase):
    """Test PairClassInfo deserialization from strings."""
    
    def test_single_class_deserialization(self):
        """Single class string should deserialize correctly."""
        pair = PairClassInfo.from_string("bat_A:1.0")
        self.assertEqual(len(pair.classes), 1)
        self.assertEqual(pair.classes[0].name, "bat_A")
        self.assertEqual(pair.classes[0].weight, 1.0)
    
    def test_dual_class_deserialization(self):
        """Dual class string should deserialize correctly."""
        pair = PairClassInfo.from_string("bat_A:1.0|bat_B:1.0")
        self.assertEqual(len(pair.classes), 2)
        self.assertEqual(pair.classes[0].name, "bat_A")
        self.assertEqual(pair.classes[1].name, "bat_B")
        self.assertEqual(pair.classes[0].weight, 1.0)
        self.assertEqual(pair.classes[1].weight, 1.0)
    
    def test_triple_class_deserialization(self):
        """Triple class string should deserialize correctly."""
        pair = PairClassInfo.from_string("A:1.0|B:1.0|C:1.0")
        self.assertEqual(len(pair.classes), 3)
        self.assertEqual(pair.classes[0].name, "A")
        self.assertEqual(pair.classes[1].name, "B")
        self.assertEqual(pair.classes[2].name, "C")
    
    def test_custom_weights_deserialization(self):
        """Custom weights should be preserved in deserialization."""
        pair = PairClassInfo.from_string("A:0.25|B:0.75")
        self.assertEqual(len(pair.classes), 2)
        self.assertAlmostEqual(pair.classes[0].weight, 0.25)
        self.assertAlmostEqual(pair.classes[1].weight, 0.75)
    
    def test_integer_weights_deserialization(self):
        """Integer weights should be converted to float."""
        pair = PairClassInfo.from_string("A:2|B:3")
        self.assertEqual(len(pair.classes), 2)
        self.assertEqual(pair.classes[0].weight, 2.0)
        self.assertEqual(pair.classes[1].weight, 3.0)
    
    def test_scientific_notation_weights(self):
        """Scientific notation weights should deserialize correctly."""
        pair = PairClassInfo.from_string("A:1e-3|B:2.5e2")
        self.assertEqual(len(pair.classes), 2)
        self.assertAlmostEqual(pair.classes[0].weight, 0.001)
        self.assertAlmostEqual(pair.classes[1].weight, 250.0)


class TestPairClassInfoRoundTrip(unittest.TestCase):
    """Test round-trip serialization/deserialization."""
    
    def test_single_class_round_trip(self):
        """Single class should survive round-trip."""
        original = PairClassInfo.create_single_class("bat_A")
        serialized = original.to_string()
        deserialized = PairClassInfo.from_string(serialized)
        
        self.assertEqual(len(deserialized.classes), 1)
        self.assertEqual(deserialized.classes[0].name, "bat_A")
        self.assertEqual(deserialized.classes[0].weight, 1.0)
    
    def test_dual_class_round_trip(self):
        """Dual class should survive round-trip."""
        original = PairClassInfo.create_dual_class("bat_A", "bat_B")
        serialized = original.to_string()
        deserialized = PairClassInfo.from_string(serialized)
        
        self.assertEqual(len(deserialized.classes), 2)
        self.assertEqual(deserialized.classes[0].name, "bat_A")
        self.assertEqual(deserialized.classes[1].name, "bat_B")
        self.assertEqual(deserialized.to_string(), serialized)
    
    def test_custom_weights_round_trip(self):
        """Custom weights should survive round-trip."""
        original = PairClassInfo(classes=[
            ClassInfo(name="A", weight=0.123),
            ClassInfo(name="B", weight=0.456),
            ClassInfo(name="C", weight=0.789)
        ])
        serialized = original.to_string()
        deserialized = PairClassInfo.from_string(serialized)
        
        self.assertEqual(len(deserialized.classes), 3)
        self.assertAlmostEqual(deserialized.classes[0].weight, 0.123, places=3)
        self.assertAlmostEqual(deserialized.classes[1].weight, 0.456, places=3)
        self.assertAlmostEqual(deserialized.classes[2].weight, 0.789, places=3)


class TestPairClassInfoErrorHandling(unittest.TestCase):
    """Test error handling for invalid inputs."""
    
    def test_empty_string_raises_error(self):
        """Empty string should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairClassInfo.from_string("")
        self.assertIn("non-empty string", str(ctx.exception))
    
    def test_none_input_raises_error(self):
        """None input should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairClassInfo.from_string(None)
        self.assertIn("non-empty string", str(ctx.exception))
    
    def test_invalid_format_no_colon_raises_error(self):
        """Missing colon should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairClassInfo.from_string("bat_A")
        self.assertIn("Invalid class format", str(ctx.exception))
    
    def test_invalid_format_multiple_colons_raises_error(self):
        """Multiple colons should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairClassInfo.from_string("bat_A:1.0:extra")
        self.assertIn("Invalid class format", str(ctx.exception))
    
    def test_invalid_weight_raises_error(self):
        """Non-numeric weight should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairClassInfo.from_string("bat_A:not_a_number")
        self.assertIn("Invalid weight value", str(ctx.exception))
    
    def test_empty_class_name_raises_error(self):
        """Empty class name should raise ValueError."""
        with self.assertRaises(ValueError) as ctx:
            PairClassInfo.from_string(":1.0")
        self.assertIn("Class name cannot be empty", str(ctx.exception))
    
    def test_partial_invalid_multi_class_raises_error(self):
        """Partially invalid multi-class string should raise ValueError."""
        with self.assertRaises(ValueError):
            PairClassInfo.from_string("bat_A:1.0|bat_B:invalid|bat_C:1.0")


class TestPairClassInfoHelpers(unittest.TestCase):
    """Test helper factory methods."""
    
    def test_create_single_class_helper(self):
        """create_single_class helper should work correctly."""
        pair = PairClassInfo.create_single_class("test_class")
        self.assertEqual(len(pair.classes), 1)
        self.assertEqual(pair.classes[0].name, "test_class")
        self.assertEqual(pair.classes[0].weight, 1.0)
    
    def test_create_dual_class_helper(self):
        """create_dual_class helper should work correctly."""
        pair = PairClassInfo.create_dual_class("class_a", "class_b")
        self.assertEqual(len(pair.classes), 2)
        self.assertEqual(pair.classes[0].name, "class_a")
        self.assertEqual(pair.classes[1].name, "class_b")
        self.assertEqual(pair.classes[0].weight, 1.0)
        self.assertEqual(pair.classes[1].weight, 1.0)
    
    def test_helpers_produce_serializable_output(self):
        """Helper-created objects should be serializable."""
        single = PairClassInfo.create_single_class("A")
        dual = PairClassInfo.create_dual_class("A", "B")
        
        self.assertIsInstance(single.to_string(), str)
        self.assertIsInstance(dual.to_string(), str)
        self.assertEqual(single.to_string(), "A:1.0")
        self.assertEqual(dual.to_string(), "A:1.0|B:1.0")


if __name__ == '__main__':
    unittest.main()

