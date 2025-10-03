#!/usr/bin/env python3
"""
Face Landmark Analyzer

This standalone script processes images in a directory to:
1. Apply face alignment using align_face_landmarks
2. Load pose model from config
3. Mark distances between eyes and from eyes to top of image
4. Output the maximum distance between eyes after alignment

Usage:
    python face_landmark_analyzer.py --input-dir /path/to/images --output-dir /path/to/output
"""

import os
import sys
import cv2
import numpy as np
import argparse
from typing import List, Tuple, Optional
from pathlib import Path

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from image_processor.transforms import ImageTransforms
from app.config.loader import load_config


class FaceLandmarkAnalyzer:
    """Analyzes face landmarks in images and measures distances."""
    
    def __init__(self):
        # Load configuration
        config = load_config()
        
        # Pose model config
        pose_config = config.models.pose
        self.pose_model_path = pose_config.model_path
        self.confidence_threshold = pose_config.confidence_threshold
        
        # Siamese network preprocessing config
        siamese_config = config.siamese_network.training.get('data_preprocessing', {})
        self.face_outer_margin_ratio = siamese_config.get('face_outer_margin_ratio', 0.15)
        
        if not self.pose_model_path or not os.path.exists(self.pose_model_path):
            raise ValueError(f"Pose model not found at: {self.pose_model_path}")
        
        print(f"🔧 Using pose model: {self.pose_model_path}")
        print(f"🔧 Confidence threshold: {self.confidence_threshold}")
        print(f"🔧 Face margin ratio: {self.face_outer_margin_ratio}")
    
    def analyze_single_image(self, image_path: str, output_path: str) -> Optional[float]:
        """
        Analyze a single image for face landmarks and distances.
        
        Args:
            image_path: Path to input image
            output_path: Path to save annotated output image
            
        Returns:
            Eye distance if successful, None otherwise
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return None
            
            print(f"📷 Processing: {os.path.basename(image_path)}")
            print(f"   Original shape: {image.shape}")
            
            # Step 1: Apply face alignment
            print("   🔄 Aligning face...")
            aligned_image = ImageTransforms.align_face_landmarks(
                image,
                face_detector_type="yolo_pose",
                debug=False
            )
            
            if aligned_image is None:
                print("   ⚠️ Face alignment failed, using original image")
                aligned_image = image
            else:
                print("   ✅ Face alignment successful")
            
            # Step 2: Save the aligned image
            print("   💾 Saving aligned image...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            success = cv2.imwrite(output_path, aligned_image)
            if success:
                print(f"   💾 Saved aligned image: {output_path}")
            else:
                print(f"   ❌ Failed to save: {output_path}")
            
            # For now, just return a dummy value since we're only doing alignment
            return 0.0
            
        except Exception as e:
            print(f"   ❌ Error processing {image_path}: {e}")
            return None
    
    def _detect_pose_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect pose landmarks using YOLO pose model.
        
        Args:
            image: Input image
            
        Returns:
            Landmarks array [N, 3] with [x, y, confidence] or None
        """
        try:
            from ultralytics import YOLO
            
            model = YOLO(self.pose_model_path)
            results = model.predict(
                source=image, 
                conf=self.confidence_threshold, 
                verbose=False, 
                save=False
            )
            
            if not results or len(results) == 0:
                return None
            
            result = results[0]
            if not hasattr(result, "keypoints") or result.keypoints is None:
                return None
            
            # Get keypoints data [K, 3] where K is number of keypoints
            # Each keypoint has [x, y, confidence]
            landmarks = result.keypoints.data.cpu().numpy()[0]
            
            # Filter by confidence
            valid_landmarks = landmarks[landmarks[:, 2] >= self.confidence_threshold]
            
            if len(valid_landmarks) < 3:  # Need at least left eye, right eye, nose
                print(f"   ⚠️ Only {len(valid_landmarks)} landmarks detected, need at least 3")
                return None
            
            print(f"   ✅ Detected {len(valid_landmarks)} landmarks")
            return valid_landmarks
            
        except Exception as e:
            print(f"   ⚠️ Pose detection error: {e}")
            return None
    
    def _calculate_square_from_landmarks(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around pose landmarks.
        
        Args:
            image: Input image
            landmarks: Landmarks array [N, 3] with [x, y, confidence]
            
        Returns:
            Tuple of (x1, y1, x2, y2) coordinates for bounding box
        """
        if len(landmarks) < 3:
            print("   ⚠️ Need at least 3 landmarks (eyes + nose)")
            return (0, 0, image.shape[1], image.shape[0])
        
        # Get eye and nose positions (first 3 landmarks)
        left_eye = landmarks[0][:2]
        right_eye = landmarks[1][:2]
        nose = landmarks[2][:2]
        
        print(f"     Eye positions: ({left_eye[0]:.1f}, {left_eye[1]:.1f}) and ({right_eye[0]:.1f}, {right_eye[1]:.1f})")
        print(f"     Nose position: ({nose[0]:.1f}, {nose[1]:.1f})")
        
        # Calculate bounding box around the landmarks
        min_x = int(min(left_eye[0], right_eye[0], nose[0]))
        max_x = int(max(left_eye[0], right_eye[0], nose[0]))
        min_y = int(min(left_eye[1], right_eye[1], nose[1]))
        max_y = int(max(left_eye[1], right_eye[1], nose[1]))
        
        # Add margin around the landmarks
        margin = int(max(max_x - min_x, max_y - min_y) * 0.2)  # 20% margin
        
        x1 = max(0, min_x - margin)
        y1 = max(0, min_y - margin)
        x2 = min(image.shape[1], max_x + margin)
        y2 = min(image.shape[0], max_y + margin)
        
        print(f"     Bounding box: [{x1}, {y1}, {x2}, {y2}]")
        
        return (x1, y1, x2, y2)
    
    def _mark_distances(self, image: np.ndarray, landmarks: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
        """
        Mark distances on the image and calculate eye distance.
        
        Args:
            image: Input image
            landmarks: Landmarks array [N, 3] with [x, y, confidence]
            
        Returns:
            Tuple of (annotated_image, eye_distance)
        """
        annotated_image = image.copy()
        h, w = image.shape[:2]
        
        # Assuming first 3 landmarks are: left eye, right eye, nose
        # (This may need adjustment based on your specific model's keypoint order)
        if len(landmarks) < 2:
            return annotated_image, None
        
        # Get eye positions (first two landmarks)
        left_eye = landmarks[0][:2].astype(int)
        right_eye = landmarks[1][:2].astype(int)
        
        # Calculate eye distance
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Calculate distances from eyes to top of image
        left_eye_to_top = left_eye[1]  # y-coordinate (distance from top)
        right_eye_to_top = right_eye[1]  # y-coordinate (distance from top)
        
        # Draw landmarks
        cv2.circle(annotated_image, tuple(left_eye), 5, (0, 255, 0), -1)  # Green
        cv2.circle(annotated_image, tuple(right_eye), 5, (0, 255, 0), -1)  # Green
        
        # Draw line between eyes
        cv2.line(annotated_image, tuple(left_eye), tuple(right_eye), (255, 0, 0), 2)  # Blue
        
        # Draw lines from eyes to top
        cv2.line(annotated_image, tuple(left_eye), (left_eye[0], 0), (0, 0, 255), 2)   # Red
        cv2.line(annotated_image, tuple(right_eye), (right_eye[0], 0), (0, 0, 255), 2)  # Red
        
        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Eye distance
        mid_point = ((left_eye + right_eye) // 2).astype(int)
        cv2.putText(annotated_image, f"Eye dist: {eye_distance:.1f}px", 
                   (mid_point[0] - 50, mid_point[1] - 10), 
                   font, font_scale, (255, 255, 0), thickness)
        
        # Left eye to top
        cv2.putText(annotated_image, f"L->Top: {left_eye_to_top}px", 
                   (left_eye[0] - 30, left_eye[1] - 20), 
                   font, font_scale, (0, 255, 255), thickness)
        
        # Right eye to top
        cv2.putText(annotated_image, f"R->Top: {right_eye_to_top}px", 
                   (right_eye[0] - 30, right_eye[1] - 20), 
                   font, font_scale, (0, 255, 255), thickness)
        
        return annotated_image, eye_distance
    
    def _mark_distances_with_square(self, image: np.ndarray, landmarks: np.ndarray, square_bbox: Tuple[int, int, int, int]) -> Tuple[np.ndarray, Optional[float]]:
        """
        Mark distances on the image with square overlay and calculate eye distance.
        
        Args:
            image: Input image
            landmarks: Landmarks array [N, 3] with [x, y, confidence]
            square_bbox: Square bounding box (x1, y1, x2, y2)
            
        Returns:
            Tuple of (annotated_image, eye_distance)
        """
        x1, y1, x2, y2 = square_bbox
        annotated_image = image.copy()
        
        # Draw the bounding box around the landmarks
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 255, 0), 3)  # Yellow rectangle
        
        # Assuming first 3 landmarks are: left eye, right eye, nose
        if len(landmarks) < 2:
            return annotated_image, None
        
        # Get eye positions (first two landmarks)
        left_eye = landmarks[0][:2].astype(int)
        right_eye = landmarks[1][:2].astype(int)
        
        # Calculate eye distance
        eye_distance = np.linalg.norm(right_eye - left_eye)
        
        # Calculate distances from eyes to top of square
        left_eye_to_square_top = left_eye[1] - y1
        right_eye_to_square_top = right_eye[1] - y1
        
        # Calculate percentage from top of square
        square_height = y2 - y1
        left_eye_percent = (left_eye_to_square_top / square_height) * 100
        right_eye_percent = (right_eye_to_square_top / square_height) * 100
        
        # Calculate vertical distribution
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Get nose position
        if len(landmarks) > 2:
            nose = landmarks[2][:2].astype(int)
            nose_y = nose[1]
        else:
            nose_y = eye_center_y
        
        # Calculate actual percentages from top of square
        square_height = y2 - y1
        eye_percent_from_top = ((eye_center_y - y1) / square_height) * 100 if square_height > 0 else 0
        nose_percent_from_top = ((nose_y - y1) / square_height) * 100 if square_height > 0 else 0
        
        print(f"     Actual positioning - Eyes: {eye_center_y}px ({eye_percent_from_top:.1f}%), Nose: {nose_y}px ({nose_percent_from_top:.1f}%)")
        
        # Draw landmarks
        cv2.circle(annotated_image, tuple(left_eye), 5, (0, 255, 0), -1)  # Green
        cv2.circle(annotated_image, tuple(right_eye), 5, (0, 255, 0), -1)  # Green
        
        # Draw nose if available
        if len(landmarks) > 2:
            # nose variable is already defined and adjusted above
            cv2.circle(annotated_image, tuple(nose), 5, (255, 0, 255), -1)  # Magenta
        
        # Draw line between eyes
        cv2.line(annotated_image, tuple(left_eye), tuple(right_eye), (255, 0, 0), 2)  # Blue
        
        # Draw lines from eyes to top of square
        cv2.line(annotated_image, tuple(left_eye), (left_eye[0], y1), (0, 0, 255), 2)   # Red
        cv2.line(annotated_image, tuple(right_eye), (right_eye[0], y1), (0, 0, 255), 2)  # Red
        
        # Add text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        
        # Eye distance
        mid_point = ((left_eye + right_eye) // 2).astype(int)
        cv2.putText(annotated_image, f"Eye dist: {eye_distance:.1f}px", 
                   (mid_point[0] - 50, mid_point[1] - 10), 
                   font, font_scale, (255, 255, 0), thickness)
        
        # Left eye to square top
        cv2.putText(annotated_image, f"L->Square: {left_eye_percent:.1f}%", 
                   (left_eye[0] - 40, left_eye[1] - 20), 
                   font, font_scale, (0, 255, 255), thickness)
        
        # Right eye to square top
        cv2.putText(annotated_image, f"R->Square: {right_eye_percent:.1f}%", 
                   (right_eye[0] - 40, right_eye[1] - 20), 
                   font, font_scale, (0, 255, 255), thickness)
        
        # Bounding box info
        cv2.putText(annotated_image, f"Pose Bounding Box", 
                   (x1 + 10, y1 - 10), 
                   font, font_scale, (255, 255, 0), thickness)
        
        # Landmark positioning info
        cv2.putText(annotated_image, f"Landmark Bounding Box: [{x1}, {y1}, {x2}, {y2}]", 
                   (x1 + 10, y2 + 20), 
                   font, font_scale, (255, 255, 0), thickness)
        cv2.putText(annotated_image, f"Eyes: {eye_center_y}px, Nose: {nose_y}px", 
                   (x1 + 10, y2 + 40), 
                   font, font_scale, (255, 255, 0), thickness)
        
        return annotated_image, eye_distance
    
    def analyze_directory(self, input_dir: str, output_dir: str) -> List[float]:
        """
        Analyze all images in a directory.
        
        Args:
            input_dir: Input directory with images
            output_dir: Output directory for annotated images
            
        Returns:
            List of eye distances for successful analyses
        """
        if not os.path.exists(input_dir):
            raise ValueError(f"Input directory does not exist: {input_dir}")
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for file in os.listdir(input_dir):
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(file)
        
        if not image_files:
            print(f"⚠️ No image files found in {input_dir}")
            return []
        
        print(f"🔄 Found {len(image_files)} images to process")
        
        # Process each image
        eye_distances = []
        successful = 0
        
        for filename in image_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"annotated_{filename}")
            
            result = self.analyze_single_image(input_path, output_path)
            
            if result is not None:
                successful += 1
            
            print()  # Empty line for readability
        
        print(f"🎯 Processing complete!")
        print(f"✅ Successfully processed: {successful}/{len(image_files)} images")
        
        return []


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze face landmarks and measure distances in images"
    )
    parser.add_argument(
        "--input-dir", "-i", 
        type=str, 
        required=True,
        help="Input directory containing images"
    )
    parser.add_argument(
        "--output-dir", "-o", 
        type=str, 
        required=True,
        help="Output directory for annotated images"
    )
    
    args = parser.parse_args()
    
    try:
        # Create analyzer
        analyzer = FaceLandmarkAnalyzer()
        
        # Analyze directory
        analyzer.analyze_directory(args.input_dir, args.output_dir)
        
        print(f"\n🏆 Face alignment processing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
