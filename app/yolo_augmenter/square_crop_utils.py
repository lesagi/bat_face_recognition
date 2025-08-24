#!/usr/bin/env python3
"""
Utility functions for Square Crop Quality Assurance

This module provides tools for validating and verifying the quality
of square crop processing results.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


def validate_yolo_label_format(label_path: Path) -> Dict[str, Any]:
    """
    Validate YOLO segmentation label format.
    
    Args:
        label_path: Path to the label file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "object_count": 0,
        "total_coordinates": 0
    }
    
    try:
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Check minimum length (class_id + at least 3 points = 7 parts)
                if len(parts) < 7:
                    result["errors"].append(f"Line {line_num}: Insufficient coordinates (need at least 3 points)")
                    result["valid"] = False
                    continue
                
                # Check class_id
                try:
                    class_id = int(parts[0])
                    if class_id < 0:
                        result["warnings"].append(f"Line {line_num}: Negative class_id {class_id}")
                except ValueError:
                    result["errors"].append(f"Line {line_num}: Invalid class_id '{parts[0]}'")
                    result["valid"] = False
                    continue
                
                # Check coordinates
                coords = []
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        try:
                            x = float(parts[i])
                            y = float(parts[i + 1])
                            
                            # Check coordinate range [0, 1]
                            if not (0.0 <= x <= 1.0):
                                result["warnings"].append(f"Line {line_num}: X coordinate {x} outside [0,1] range")
                            if not (0.0 <= y <= 1.0):
                                result["warnings"].append(f"Line {line_num}: Y coordinate {y} outside [0,1] range")
                            
                            coords.extend([x, y])
                        except ValueError:
                            result["errors"].append(f"Line {line_num}: Invalid coordinate '{parts[i]}' or '{parts[i+1]}'")
                            result["valid"] = False
                            continue
                
                # Check minimum polygon points
                if len(coords) < 6:  # At least 3 points
                    result["errors"].append(f"Line {line_num}: Insufficient polygon points ({len(coords)//2})")
                    result["valid"] = False
                    continue
                
                result["object_count"] += 1
                result["total_coordinates"] += len(coords)
    
    except Exception as e:
        result["errors"].append(f"File read error: {e}")
        result["valid"] = False
    
    return result


def validate_image_format(image_path: Path) -> Dict[str, Any]:
    """
    Validate image format and properties.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "width": 0,
        "height": 0,
        "channels": 0,
        "file_size": 0
    }
    
    try:
        # Check file exists and is readable
        if not image_path.exists():
            result["errors"].append("Image file does not exist")
            result["valid"] = False
            return result
        
        # Get file size
        result["file_size"] = image_path.stat().st_size
        
        # Try to load image
        image = cv2.imread(str(image_path))
        if image is None:
            result["errors"].append("Could not load image with OpenCV")
            result["valid"] = False
            return result
        
        # Get image properties
        height, width = image.shape[:2]
        result["height"] = height
        result["width"] = width
        result["channels"] = image.shape[2] if len(image.shape) > 2 else 1
        
        # Check if image is square
        if width != height:
            result["warnings"].append(f"Image is not square: {width}x{height}")
        
        # Check reasonable dimensions
        if width < 64 or height < 64:
            result["warnings"].append(f"Image dimensions are very small: {width}x{height}")
        if width > 4096 or height > 4096:
            result["warnings"].append(f"Image dimensions are very large: {width}x{height}")
        
        # Check file size vs image dimensions
        expected_size = width * height * result["channels"]
        if result["file_size"] < expected_size * 0.1:  # Allow for compression
            result["warnings"].append("File size seems too small for image dimensions")
        if result["file_size"] > expected_size * 10:  # Allow for compression
            result["warnings"].append("File size seems too large for image dimensions")
    
    except Exception as e:
        result["errors"].append(f"Validation error: {e}")
        result["valid"] = False
    
    return result


def validate_coordinate_consistency(image_path: Path, label_path: Path) -> Dict[str, Any]:
    """
    Validate that label coordinates are consistent with image dimensions.
    
    Args:
        image_path: Path to the image file
        label_path: Path to the label file
        
    Returns:
        Dictionary with validation results
    """
    result = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "objects_checked": 0,
        "objects_with_issues": 0
    }
    
    try:
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            result["errors"].append("Could not load image")
            result["valid"] = False
            return result
        
        img_height, img_width = image.shape[:2]
        
        # Parse label
        label_validation = validate_yolo_label_format(label_path)
        if not label_validation["valid"]:
            result["errors"].extend(label_validation["errors"])
            result["valid"] = False
            return result
        
        # Check each object's coordinates
        with open(label_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 7:
                    continue
                
                result["objects_checked"] += 1
                has_issues = False
                
                # Check coordinates
                for i in range(1, len(parts), 2):
                    if i + 1 < len(parts):
                        try:
                            x_norm = float(parts[i])
                            y_norm = float(parts[i + 1])
                            
                            # Convert to absolute coordinates
                            x_abs = x_norm * img_width
                            y_abs = y_norm * img_height
                            
                            # Check if coordinates are within reasonable bounds
                            if x_abs < -img_width * 0.1 or x_abs > img_width * 1.1:
                                result["warnings"].append(f"Line {line_num}: X coordinate {x_norm} ({x_abs:.1f}px) may be outside reasonable bounds")
                                has_issues = True
                            
                            if y_abs < -img_height * 0.1 or y_abs > img_height * 1.1:
                                result["warnings"].append(f"Line {line_num}: Y coordinate {y_norm} ({y_abs:.1f}px) may be outside reasonable bounds")
                                has_issues = True
                        
                        except ValueError:
                            continue
                
                if has_issues:
                    result["objects_with_issues"] += 1
    
    except Exception as e:
        result["errors"].append(f"Coordinate validation error: {e}")
        result["valid"] = False
    
    return result


def generate_processing_statistics(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Generate comprehensive statistics about the processing results.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        
    Returns:
        Dictionary with processing statistics
    """
    stats = {
        "input_summary": {},
        "output_summary": {},
        "processing_summary": {},
        "quality_metrics": {}
    }
    
    try:
        # Input summary
        input_images = list((input_dir / "images").glob("*"))
        input_labels = list((input_dir / "labels").glob("*.txt"))
        
        stats["input_summary"] = {
            "total_images": len(input_images),
            "total_labels": len(input_labels),
            "image_formats": list(set(f.suffix.lower() for f in input_images)),
            "label_formats": list(set(f.suffix.lower() for f in input_labels))
        }
        
        # Output summary
        output_images = list((output_dir / "images").glob("*"))
        output_labels = list((output_dir / "labels").glob("*.txt"))
        
        stats["output_summary"] = {
            "total_images": len(output_images),
            "total_labels": len(output_labels),
            "image_formats": list(set(f.suffix.lower() for f in output_images)),
            "label_formats": list(set(f.suffix.lower() for f in input_labels))
        }
        
        # Processing summary
        stats["processing_summary"] = {
            "success_rate": len(output_images) / len(input_images) if input_images else 0,
            "images_processed": len(output_images),
            "labels_processed": len(output_labels),
            "processing_efficiency": len(output_images) / len(input_images) if input_images else 0
        }
        
        # Quality metrics
        quality_issues = 0
        total_objects = 0
        
        for label_path in output_labels:
            validation = validate_yolo_label_format(label_path)
            if not validation["valid"]:
                quality_issues += 1
            total_objects += validation["object_count"]
        
        stats["quality_metrics"] = {
            "total_objects": total_objects,
            "labels_with_issues": quality_issues,
            "quality_score": (len(output_labels) - quality_issues) / len(output_labels) if output_labels else 0
        }
    
    except Exception as e:
        logger.error(f"Error generating statistics: {e}")
    
    return stats


def create_validation_report(input_dir: Path, output_dir: Path, 
                           detailed: bool = False) -> str:
    """
    Create a comprehensive validation report.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        detailed: Whether to include detailed validation for each file
        
    Returns:
        Formatted validation report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("SQUARE CROP PROCESSING VALIDATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    try:
        # Generate statistics
        stats = generate_processing_statistics(input_dir, output_dir)
        
        # Summary section
        report_lines.append("📊 PROCESSING SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Input Images: {stats['input_summary']['total_images']}")
        report_lines.append(f"Output Images: {stats['output_summary']['total_images']}")
        report_lines.append(f"Success Rate: {stats['processing_summary']['success_rate']:.1%}")
        report_lines.append(f"Quality Score: {stats['quality_metrics']['quality_score']:.1%}")
        report_lines.append("")
        
        # Quality metrics
        report_lines.append("🔍 QUALITY METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Objects: {stats['quality_metrics']['total_objects']}")
        report_lines.append(f"Labels with Issues: {stats['quality_metrics']['labels_with_issues']}")
        report_lines.append("")
        
        if detailed:
            # Detailed validation
            report_lines.append("📋 DETAILED VALIDATION")
            report_lines.append("-" * 40)
            
            output_images = list((output_dir / "images").glob("*"))
            output_labels = list((output_dir / "labels").glob("*.txt"))
            
            # Validate a sample of files
            sample_size = min(10, len(output_images))
            sample_images = output_images[:sample_size]
            sample_labels = output_labels[:sample_size]
            
            for i, (img_path, label_path) in enumerate(zip(sample_images, sample_labels)):
                report_lines.append(f"Sample {i+1}: {img_path.name}")
                
                # Image validation
                img_validation = validate_image_format(img_path)
                if img_validation["valid"]:
                    report_lines.append(f"  ✅ Image: {img_validation['width']}x{img_validation['height']}")
                else:
                    report_lines.append(f"  ❌ Image: {len(img_validation['errors'])} errors")
                
                # Label validation
                label_validation = validate_yolo_label_format(label_path)
                if label_validation["valid"]:
                    report_lines.append(f"  ✅ Label: {label_validation['object_count']} objects")
                else:
                    report_lines.append(f"  ❌ Label: {len(label_validation['errors'])} errors")
                
                # Coordinate consistency
                coord_validation = validate_coordinate_consistency(img_path, label_path)
                if coord_validation["valid"]:
                    report_lines.append(f"  ✅ Coordinates: Consistent")
                else:
                    report_lines.append(f"  ⚠️ Coordinates: {len(coord_validation['warnings'])} warnings")
                
                report_lines.append("")
        
        # Recommendations
        report_lines.append("💡 RECOMMENDATIONS")
        report_lines.append("-" * 40)
        
        if stats['processing_summary']['success_rate'] < 0.95:
            report_lines.append("⚠️ Success rate is below 95%. Check for processing errors.")
        
        if stats['quality_metrics']['quality_score'] < 0.9:
            report_lines.append("⚠️ Quality score is below 90%. Validate label formats.")
        
        if stats['quality_metrics']['labels_with_issues'] > 0:
            report_lines.append("⚠️ Some labels have validation issues. Review output quality.")
        
        if not report_lines[-1].startswith("⚠️"):
            report_lines.append("✅ Processing quality is good. No major issues detected.")
    
    except Exception as e:
        report_lines.append(f"❌ Error generating report: {e}")
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    return "\n".join(report_lines)


def verify_processing_completeness(input_dir: Path, output_dir: Path) -> Dict[str, Any]:
    """
    Verify that all input files were processed correctly.
    
    Args:
        input_dir: Input directory path
        output_dir: Output directory path
        
    Returns:
        Dictionary with verification results
    """
    result = {
        "complete": True,
        "missing_files": [],
        "extra_files": [],
        "mismatched_counts": False,
        "verification_score": 0.0
    }
    
    try:
        # Get file lists
        input_images = set(f.stem for f in (input_dir / "images").glob("*"))
        input_labels = set(f.stem for f in (input_dir / "labels").glob("*.txt"))
        
        output_images = set(f.stem.replace("_cropped", "") for f in (output_dir / "images").glob("*"))
        output_labels = set(f.stem.replace("_cropped", "") for f in (output_dir / "labels").glob("*.txt"))
        
        # Check for missing files
        missing_images = input_images - output_images
        missing_labels = input_labels - output_labels
        
        if missing_images:
            result["missing_files"].extend([f"{name}.jpg" for name in missing_images])
            result["complete"] = False
        
        if missing_labels:
            result["missing_files"].extend([f"{name}.txt" for name in missing_labels])
            result["complete"] = False
        
        # Check for extra files
        extra_images = output_images - input_images
        extra_labels = output_labels - input_labels
        
        if extra_images:
            result["extra_files"].extend([f"{name}_cropped.jpg" for name in extra_images])
        
        if extra_labels:
            result["extra_files"].extend([f"{name}_cropped.txt" for name in extra_labels])
        
        # Check counts
        if len(input_images) != len(output_images) or len(input_labels) != len(output_labels):
            result["mismatched_counts"] = True
            result["complete"] = False
        
        # Calculate verification score
        total_expected = len(input_images) + len(input_labels)
        total_processed = len(output_images) + len(output_labels)
        result["verification_score"] = total_processed / total_expected if total_expected > 0 else 0.0
    
    except Exception as e:
        logger.error(f"Error verifying processing completeness: {e}")
        result["complete"] = False
    
    return result
