#!/usr/bin/env python3
"""
CVAT XML to YOLO Format Converter

Converts CVAT for Images XML annotations to YOLO format for training.
Supports multiple output formats:
1. YOLO Keypoint Detection (face + 3 keypoints)
2. Multi-class Detection (separate classes)
3. Face Detection only

Usage:
    python convert_cvat_to_yolo.py --format keypoints
    python convert_cvat_to_yolo.py --format multiclass
    python convert_cvat_to_yolo.py --format faceonly
"""

import os
import sys
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import shutil


class CVATToYOLOConverter:
    """Converts CVAT XML annotations to YOLO format."""

    def __init__(self, base_dir: str):
        """
        Initialize converter.

        Args:
            base_dir: Base directory containing train and val folders
        """
        self.base_dir = Path(base_dir)
        self.train_dir = self.base_dir / "train"
        self.val_dir = self.base_dir / "val"

        # Class mappings for different formats
        self.keypoint_classes = {"face": 0}  # Single class with keypoints

        self.multiclass_classes = {"face": 0, "left_eye": 1, "right_eye": 2, "nose": 3}

        self.face_only_classes = {"face": 0}

    def parse_xml_annotations(self, xml_file: Path) -> List[Dict]:
        """
        Parse CVAT XML file and extract annotations.

        Args:
            xml_file: Path to XML annotation file

        Returns:
            List of annotation dictionaries
        """
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            annotations = []

            for image in root.findall("image"):
                image_name = image.get("name")
                image_width = int(image.get("width"))
                image_height = int(image.get("height"))

                # Extract all annotations for this image
                image_annotations = {
                    "image_name": image_name,
                    "image_width": image_width,
                    "image_height": image_height,
                    "face_box": None,
                    "nose": None,
                    "left_eye": None,
                    "right_eye": None,
                }

                # Extract face bounding box
                for box in image.findall("box"):
                    if box.get("label") == "face":
                        xtl = float(box.get("xtl"))
                        ytl = float(box.get("ytl"))
                        xbr = float(box.get("xbr"))
                        ybr = float(box.get("ybr"))

                        # Convert to center coordinates and normalize
                        center_x = (xtl + xbr) / 2 / image_width
                        center_y = (ytl + ybr) / 2 / image_height
                        width = (xbr - xtl) / image_width
                        height = (ybr - ytl) / image_height

                        image_annotations["face_box"] = (
                            center_x,
                            center_y,
                            width,
                            height,
                        )
                        break

                # Extract point annotations
                for points in image.findall("points"):
                    label = points.get("label")
                    coords = points.get("points")

                    if coords and label in ["nose", "left_eye", "right_eye"]:
                        x, y = map(float, coords.split(","))
                        # Normalize coordinates
                        norm_x = x / image_width
                        norm_y = y / image_height
                        image_annotations[label] = (norm_x, norm_y)

                annotations.append(image_annotations)

            return annotations

        except Exception as e:
            print(f"Error parsing XML file {xml_file}: {e}")
            return []

    def convert_to_keypoint_format(self, annotations: List[Dict], output_dir: Path):
        """
        Convert to YOLO keypoint detection format.
        Format: class_id center_x center_y width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v kp3_x kp3_y kp3_v
        """
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        for ann in annotations:
            if ann["face_box"] is None:
                print(f"Warning: No face box found for {ann['image_name']}")
                continue

            # Create label file
            label_file = labels_dir / f"{Path(ann['image_name']).stem}.txt"

            with open(label_file, "w") as f:
                center_x, center_y, width, height = ann["face_box"]

                # Start with face bounding box
                line = f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"

                # Add keypoints in order: left_eye, right_eye, nose
                keypoints = ["left_eye", "right_eye", "nose"]

                for kp in keypoints:
                    if ann[kp] is not None:
                        kp_x, kp_y = ann[kp]
                        visibility = 2  # visible
                        line += f" {kp_x:.6f} {kp_y:.6f} {visibility}"
                    else:
                        # Missing keypoint
                        line += " 0.0 0.0 0"

                f.write(line + "\n")

    def convert_to_multiclass_format(self, annotations: List[Dict], output_dir: Path):
        """
        Convert to multi-class detection format.
        Each annotation type becomes a separate class.
        """
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        for ann in annotations:
            label_file = labels_dir / f"{Path(ann['image_name']).stem}.txt"

            with open(label_file, "w") as f:
                # Face bounding box
                if ann["face_box"] is not None:
                    center_x, center_y, width, height = ann["face_box"]
                    f.write(
                        f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"
                    )

                # Point annotations as small bounding boxes
                point_size = 0.02  # 2% of image size for point boxes

                for label, class_id in [("left_eye", 1), ("right_eye", 2), ("nose", 3)]:
                    if ann[label] is not None:
                        px, py = ann[label]
                        f.write(
                            f"{class_id} {px:.6f} {py:.6f} {point_size:.6f} {point_size:.6f}\n"
                        )

    def convert_to_face_only_format(self, annotations: List[Dict], output_dir: Path):
        """
        Convert to face detection only format.
        Only includes face bounding boxes.
        """
        labels_dir = output_dir / "labels"
        labels_dir.mkdir(exist_ok=True)

        for ann in annotations:
            if ann["face_box"] is None:
                continue

            label_file = labels_dir / f"{Path(ann['image_name']).stem}.txt"

            with open(label_file, "w") as f:
                center_x, center_y, width, height = ann["face_box"]
                f.write(f"0 {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n")

    def copy_images(self, source_dir: Path, output_dir: Path):
        """Copy images to output directory."""
        images_source = source_dir / "images"
        images_dest = output_dir / "images"

        if images_source.exists():
            if images_dest.exists():
                shutil.rmtree(images_dest)
            shutil.copytree(images_source, images_dest)
            print(f"Copied images from {images_source} to {images_dest}")

    def create_yaml_config(self, output_dir: Path, format_type: str):
        """Create YOLO dataset configuration file."""
        config_content = f"""# Bat Face Dataset Configuration
# Generated from CVAT annotations

path: {output_dir.absolute()}
train: train/images
val: val/images

"""

        if format_type == "keypoints":
            config_content += """# Keypoint detection format
nc: 1  # number of classes
names: ['face']

# Keypoint configuration
kpt_shape: [3, 3]  # [number_of_keypoints, coordinates_dimensions]
flip_idx: [1, 0, 2]  # left_eye <-> right_eye when flipping, nose stays same
"""

        elif format_type == "multiclass":
            config_content += """# Multi-class detection format
nc: 4  # number of classes
names: ['face', 'left_eye', 'right_eye', 'nose']
"""

        elif format_type == "faceonly":
            config_content += """# Face detection only
nc: 1  # number of classes
names: ['face']
"""

        config_file = output_dir / "dataset.yaml"
        with open(config_file, "w") as f:
            f.write(config_content)

        print(f"Created dataset config: {config_file}")

    def convert(self, format_type: str = "keypoints", output_dir: Optional[str] = None):
        """
        Main conversion method.

        Args:
            format_type: Output format ('keypoints', 'multiclass', 'faceonly')
            output_dir: Output directory (default: current_dir/yolo_dataset_{format})
        """
        if output_dir is None:
            output_dir = self.base_dir.parent / f"yolo_dataset_{format_type}"
        else:
            output_dir = Path(output_dir)

        print(f"Converting CVAT annotations to YOLO {format_type} format...")
        print(f"Output directory: {output_dir}")

        # Create output directories
        train_output = output_dir / "train"
        val_output = output_dir / "val"
        train_output.mkdir(parents=True, exist_ok=True)
        val_output.mkdir(parents=True, exist_ok=True)

        # Process train and val sets
        for split_name, split_dir, split_output in [
            ("train", self.train_dir, train_output),
            ("val", self.val_dir, val_output),
        ]:
            xml_file = split_dir / "annotations_cvat_images.xml"

            if not xml_file.exists():
                print(f"Warning: No XML file found at {xml_file}")
                continue

            print(f"Processing {split_name} annotations...")
            annotations = self.parse_xml_annotations(xml_file)

            if not annotations:
                print(f"No annotations found in {xml_file}")
                continue

            # Convert based on format type
            if format_type == "keypoints":
                self.convert_to_keypoint_format(annotations, split_output)
            elif format_type == "multiclass":
                self.convert_to_multiclass_format(annotations, split_output)
            elif format_type == "faceonly":
                self.convert_to_face_only_format(annotations, split_output)
            else:
                raise ValueError(f"Unknown format type: {format_type}")

            # Copy images
            self.copy_images(split_dir, split_output)

            print(f"Processed {len(annotations)} images for {split_name}")

        # Create YAML configuration
        self.create_yaml_config(output_dir, format_type)

        print(f"\n✅ Conversion complete!")
        print(f"📁 Output directory: {output_dir}")
        print(f"🏷️ Format: {format_type}")

        return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="Convert CVAT XML annotations to YOLO format"
    )
    parser.add_argument(
        "--format",
        choices=["keypoints", "multiclass", "faceonly"],
        default="keypoints",
        help="Output format type (default: keypoints)",
    )
    parser.add_argument(
        "--input_dir",
        default="face_annotation_eyes_nose/images_for_cvat",
        help="Input directory containing train and val folders",
    )
    parser.add_argument(
        "--output_dir", help="Output directory (default: auto-generated)"
    )

    args = parser.parse_args()

    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1

    # Initialize converter
    converter = CVATToYOLOConverter(args.input_dir)

    try:
        output_dir = converter.convert(args.format, args.output_dir)

        print(f"\n🎯 Next steps:")
        print(f"1. Use the dataset config: {output_dir}/dataset.yaml")
        print(f"2. Train with: yolo train data={output_dir}/dataset.yaml")

        if args.format == "keypoints":
            print(f"3. Use model: yolov8n-pose.pt (for keypoint detection)")
        else:
            print(f"3. Use model: yolov8n.pt (for detection)")

        return 0

    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
