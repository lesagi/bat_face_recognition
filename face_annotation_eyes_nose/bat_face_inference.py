#!/usr/bin/env python3
"""
Bat Face Keypoint Detection Inference Script

Uses the trained YOLO model to detect bat faces and keypoints (eyes, nose).

Usage:
    python bat_face_inference.py --input image.jpg
    python bat_face_inference.py --input images_folder/
    python bat_face_inference.py --input image.jpg --save_crops
"""

import argparse
import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    exit(1)


class BatFaceKeypointDetector:
    """Bat face and keypoint detector using trained YOLO model."""

    def __init__(self, model_path: str, confidence: float = 0.3):
        """
        Initialize the detector.

        Args:
            model_path: Path to trained YOLO model (.pt file)
            confidence: Confidence threshold for detections
        """
        self.model_path = model_path
        self.confidence = confidence

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        print("Model loaded successfully!")

    def detect_faces_and_keypoints(self, image_path: str) -> List[dict]:
        """
        Detect faces and keypoints in an image.

        Args:
            image_path: Path to input image

        Returns:
            List of detection dictionaries with face box and keypoints
        """
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            return []

        # Run inference
        results = self.model.predict(image_path, conf=self.confidence, verbose=False)

        if not results or len(results) == 0:
            return []

        result = results[0]
        detections = []

        # Process each detection
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()

            # Get keypoints if available
            keypoints = None
            if hasattr(result, "keypoints") and result.keypoints is not None:
                keypoints = result.keypoints.xy.cpu().numpy()  # [N, num_keypoints, 2]

            for i, (box, conf) in enumerate(zip(boxes, confidences)):
                detection = {
                    "face_box": box.astype(int),  # [x1, y1, x2, y2]
                    "confidence": float(conf),
                    "keypoints": None,
                }

                if keypoints is not None and i < len(keypoints):
                    kpts = keypoints[i]  # [num_keypoints, 2]
                    detection["keypoints"] = {
                        "left_eye": kpts[0] if len(kpts) > 0 else None,
                        "right_eye": kpts[1] if len(kpts) > 1 else None,
                        "nose": kpts[2] if len(kpts) > 2 else None,
                    }

                detections.append(detection)

        return detections

    def visualize_detections(
        self, image_path: str, detections: List[dict], output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize detections on the image.

        Args:
            image_path: Path to input image
            detections: List of detection results
            output_path: Optional path to save visualization

        Returns:
            Image with visualizations
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        # Colors for visualization
        face_color = (0, 255, 0)  # Green for face box
        keypoint_colors = {
            "left_eye": (255, 0, 0),  # Blue
            "right_eye": (0, 0, 255),  # Red
            "nose": (255, 255, 0),  # Cyan
        }

        for detection in detections:
            # Draw face bounding box
            x1, y1, x2, y2 = detection["face_box"]
            cv2.rectangle(image, (x1, y1), (x2, y2), face_color, 2)

            # Add confidence text
            conf = detection["confidence"]
            cv2.putText(
                image,
                f"Face: {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                face_color,
                2,
            )

            # Draw keypoints
            if detection["keypoints"]:
                for name, point in detection["keypoints"].items():
                    if point is not None and len(point) == 2:
                        x, y = int(point[0]), int(point[1])
                        if x > 0 and y > 0:  # Valid keypoint
                            color = keypoint_colors.get(name, (255, 255, 255))
                            cv2.circle(image, (x, y), 5, color, -1)
                            cv2.putText(
                                image,
                                name,
                                (x + 10, y),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.4,
                                color,
                                1,
                            )

        # Save if output path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to: {output_path}")

        return image

    def crop_faces(
        self,
        image_path: str,
        detections: List[dict],
        output_dir: str,
        padding: float = 0.2,
    ) -> List[str]:
        """
        Crop detected faces and save them.

        Args:
            image_path: Path to input image
            detections: Detection results
            output_dir: Directory to save face crops
            padding: Padding around face box (0.2 = 20%)

        Returns:
            List of saved crop file paths
        """
        image = cv2.imread(image_path)
        if image is None:
            return []

        img_height, img_width = image.shape[:2]
        image_name = Path(image_path).stem

        os.makedirs(output_dir, exist_ok=True)
        saved_crops = []

        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection["face_box"]

            # Add padding
            width = x2 - x1
            height = y2 - y1
            padding_w = int(width * padding)
            padding_h = int(height * padding)

            # Expand box with padding
            x1_pad = max(0, x1 - padding_w)
            y1_pad = max(0, y1 - padding_h)
            x2_pad = min(img_width, x2 + padding_w)
            y2_pad = min(img_height, y2 + padding_h)

            # Crop face
            face_crop = image[y1_pad:y2_pad, x1_pad:x2_pad]

            # Save crop
            crop_filename = f"{image_name}_face_{i+1}.jpg"
            crop_path = os.path.join(output_dir, crop_filename)
            cv2.imwrite(crop_path, face_crop)
            saved_crops.append(crop_path)

            print(f"Face crop saved: {crop_path}")

        return saved_crops

    def process_image(
        self,
        image_path: str,
        output_dir: str = "output",
        visualize: bool = True,
        save_crops: bool = False,
    ) -> dict:
        """
        Process a single image: detect, visualize, and optionally crop faces.

        Args:
            image_path: Path to input image
            output_dir: Output directory for results
            visualize: Whether to save visualization
            save_crops: Whether to save face crops

        Returns:
            Dictionary with results
        """
        print(f"Processing: {os.path.basename(image_path)}")

        # Detect faces and keypoints
        detections = self.detect_faces_and_keypoints(image_path)

        if not detections:
            print(f"No faces detected in {os.path.basename(image_path)}")
            return {"detections": [], "visualization": None, "crops": []}

        print(f"Found {len(detections)} face(s)")

        results = {"detections": detections}

        # Create visualization
        if visualize:
            vis_path = os.path.join(output_dir, f"vis_{os.path.basename(image_path)}")
            self.visualize_detections(image_path, detections, vis_path)
            results["visualization"] = vis_path

        # Save face crops
        if save_crops:
            crops_dir = os.path.join(output_dir, "crops")
            crops = self.crop_faces(image_path, detections, crops_dir)
            results["crops"] = crops

        return results


def main():
    parser = argparse.ArgumentParser(description="Bat Face Keypoint Detection")
    parser.add_argument("--input", required=True, help="Input image or directory")
    parser.add_argument(
        "--model",
        default="/runs/pose/augmented_train2/weights/best.pt",
        help="Path to trained model",
    )
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument(
        "--confidence", type=float, default=0.3, help="Confidence threshold"
    )
    parser.add_argument("--save_crops", action="store_true", help="Save cropped faces")
    parser.add_argument(
        "--no_visualize", action="store_true", help="Skip visualization"
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        print("Make sure you have trained the model first!")
        return 1

    # Initialize detector
    try:
        detector = BatFaceKeypointDetector(args.model, args.confidence)
    except Exception as e:
        print(f"Error loading model: {e}")
        return 1

    # Get input files
    input_path = Path(args.input)

    if input_path.is_file():
        # Single image
        image_files = [str(input_path)]
    elif input_path.is_dir():
        # Directory of images
        extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
        image_files = []
        for ext in extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        image_files = [str(f) for f in image_files]
    else:
        print(f"Error: Input path not found: {args.input}")
        return 1

    if not image_files:
        print(f"No image files found in: {args.input}")
        return 1

    print(f"Processing {len(image_files)} image(s)...")

    # Process images
    total_faces = 0
    for image_path in image_files:
        try:
            results = detector.process_image(
                image_path,
                args.output,
                visualize=not args.no_visualize,
                save_crops=args.save_crops,
            )
            total_faces += len(results["detections"])
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"\nProcessing complete!")
    print(f"Total faces detected: {total_faces}")
    print(f"Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
