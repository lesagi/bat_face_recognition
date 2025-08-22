#!/usr/bin/env python3
"""
Guide for Training a Custom Bat Face Detector using YOLO.

This guide shows how to train a YOLO model specifically for bat face landmark detection
to replace OpenCV's human face detector in the pipeline.
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO


class BatFaceLandmarkTrainer:
    """Guide for training a custom bat face landmark detector."""

    def __init__(self):
        self.approaches = {
            "yolo_landmarks": "YOLO for landmark detection (RECOMMENDED)",
            "yolo_keypoints": "YOLO pose estimation adapted for faces",
            "dlib_custom": "Custom dlib landmark detector",
            "opencv_cascade": "Custom Haar/LBP cascade",
        }

    def show_training_approaches(self):
        """Show different approaches for training a custom face detector."""
        print("🦇 Custom Bat Face Detector Training Approaches")
        print("=" * 60)

        print("1. 🎯 YOLO Landmark Detection (RECOMMENDED)")
        print("   • Use your existing YOLO training setup")
        print("   • Train to detect eye and nose landmarks")
        print(
            "   • Outputs: [left_eye_x, left_eye_y, right_eye_x, right_eye_y, nose_x, nose_y]"
        )
        print("   • Pros: Uses your existing workflow, accurate, fast")
        print("   • Cons: Need to annotate landmarks")
        print()

        print("2. 🎯 YOLO Pose Estimation (ALTERNATIVE)")
        print("   • Use YOLOv8 pose model adapted for bat faces")
        print("   • Define keypoints for bat facial features")
        print("   • Outputs: Multiple facial keypoints")
        print("   • Pros: More detailed landmarks, proven approach")
        print("   • Cons: More complex annotation")
        print()

        print("3. 🔧 Custom dlib Detector")
        print("   • Train shape predictor for bat faces")
        print("   • Good for precise landmark detection")
        print("   • Pros: Very accurate landmarks")
        print("   • Cons: Complex training process, requires many annotations")
        print()

        print("4. 📊 OpenCV Cascade Classifier")
        print("   • Train Haar or LBP cascade for bat faces")
        print("   • Good for face detection (not landmarks)")
        print("   • Pros: Lightweight, fast")
        print("   • Cons: Only detection, no landmarks, outdated approach")

    def create_yolo_landmark_dataset_structure(self):
        """Create the dataset structure for YOLO landmark training."""
        print("\n📁 YOLO Landmark Dataset Structure")
        print("=" * 40)

        dataset_structure = """
        bat_face_landmarks/
        ├── data.yaml
        ├── train/
        │   ├── images/
        │   │   ├── charlie_001.jpg
        │   │   ├── fidu_001.jpg
        │   │   └── ...
        │   └── labels/
        │       ├── charlie_001.txt
        │       ├── fidu_001.txt
        │       └── ...
        ├── val/
        │   ├── images/
        │   └── labels/
        └── test/
            ├── images/
            └── labels/
        """

        print(dataset_structure)

        print("📝 Label Format (YOLO format):")
        print("   class_id x_center y_center width height")
        print("   0 0.5 0.3 0.8 0.6  # Normalized coordinates")
        print()

        print("🎯 For Landmarks, Use Multiple Classes:")
        print("   0: face_region")
        print("   1: left_eye")
        print("   2: right_eye")
        print("   3: nose")
        print()

        print("📄 data.yaml Example:")
        yaml_content = """
        path: ./bat_face_landmarks
        train: train/images
        val: val/images
        test: test/images
        
        nc: 4  # number of classes
        names: ['face', 'left_eye', 'right_eye', 'nose']
        """
        print(yaml_content)

    def show_annotation_process(self):
        """Show how to annotate bat faces for landmark detection."""
        print("\n🖊️  Annotation Process")
        print("=" * 30)

        print("1. 📊 Use Annotation Tools:")
        print("   • CVAT (Computer Vision Annotation Tool) - FREE")
        print("   • Label Studio - FREE")
        print("   • VGG Image Annotator (VIA) - FREE")
        print("   • Roboflow - PAID but excellent")
        print()

        print("2. 🎯 What to Annotate:")
        print("   • Face bounding box (overall face region)")
        print("   • Left eye center point")
        print("   • Right eye center point")
        print("   • Nose tip or center")
        print()

        print("3. 📏 Annotation Guidelines:")
        print("   • Use consistent landmark definitions")
        print("   • Annotate 200-500 images minimum")
        print("   • Include various angles, lighting, poses")
        print("   • Quality > Quantity")
        print()

        print("4. 🔄 Annotation Workflow:")
        print("   a. Select subset of your best bat face images")
        print("   b. Upload to annotation tool")
        print("   c. Define classes (face, left_eye, right_eye, nose)")
        print("   d. Annotate consistently")
        print("   e. Export in YOLO format")
        print("   f. Split into train/val/test")

    def show_training_code_example(self):
        """Show example code for training the YOLO landmark detector."""
        print("\n💻 Training Code Example")
        print("=" * 30)

        training_code = """
from ultralytics import YOLO

# Initialize model
model = YOLO('yolov8n.pt')  # Start with pretrained weights

# Train the model
results = model.train(
    data='bat_face_landmarks/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='bat_face_landmarks',
    patience=20,
    save=True,
    cache=True
)

# Validate the model
validation_results = model.val()

# Export the model
model.export(format='onnx')  # Optional: for deployment
        """

        print(training_code)

    def show_integration_example(self):
        """Show how to integrate the custom detector into the pipeline."""
        print("\n🔧 Integration into Pipeline")
        print("=" * 35)

        integration_code = '''
class CustomBatFaceDetector:
    """Custom bat face landmark detector using trained YOLO model."""
    
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_landmarks(self, image):
        """Detect bat face landmarks.
        
        Returns:
            landmarks: dict with 'left_eye', 'right_eye', 'nose' coordinates
            None if no face detected
        """
        results = self.model(image)
        
        if not results or not results[0].boxes:
            return None
        
        landmarks = {}
        for result in results[0]:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls)
                confidence = float(box.conf)
                
                if confidence < 0.5:
                    continue
                
                # Get center point of detected region
                x1, y1, x2, y2 = box.xyxy[0]
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Map class IDs to landmark names
                if class_id == 1:  # left_eye
                    landmarks['left_eye'] = (int(center_x), int(center_y))
                elif class_id == 2:  # right_eye  
                    landmarks['right_eye'] = (int(center_x), int(center_y))
                elif class_id == 3:  # nose
                    landmarks['nose'] = (int(center_x), int(center_y))
        
        # Need at least eyes for alignment
        if 'left_eye' in landmarks and 'right_eye' in landmarks:
            return landmarks
        return None

# Update ImageTransforms to use custom detector
@staticmethod
def align_face_landmarks_custom(original_image, custom_detector, debug=False):
    """Align face using custom bat face detector."""
    landmarks = custom_detector.detect_landmarks(original_image)
    
    if not landmarks:
        return None
    
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    
    # Calculate rotation angle
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0] 
    angle = np.degrees(np.arctan2(dy, dx))
    
    # Calculate rotation center
    center_x = (left_eye[0] + right_eye[0]) // 2
    center_y = (left_eye[1] + right_eye[1]) // 2
    center = (center_x, center_y)
    
    # Apply rotation
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned_image = cv2.warpAffine(
        original_image, rotation_matrix, 
        (original_image.shape[1], original_image.shape[0])
    )
    
    return aligned_image
        '''

        print(integration_code)

    def show_quick_start_checklist(self):
        """Show a quick start checklist for training."""
        print("\n✅ Quick Start Checklist")
        print("=" * 30)

        checklist = [
            "1. 📸 Select 200-500 best bat face images",
            "2. 🖊️  Set up CVAT or Label Studio for annotation",
            "3. 🎯 Define 4 classes: face, left_eye, right_eye, nose",
            "4. 📝 Annotate images consistently",
            "5. 📁 Export in YOLO format",
            "6. 🔄 Split into train/val/test (70/20/10)",
            "7. 💻 Train YOLO model (100+ epochs)",
            "8. 📊 Validate and test model performance",
            "9. 🔧 Integrate into your pipeline",
            "10. 🧪 Test on real bat face processing",
        ]

        for item in checklist:
            print(f"   {item}")

        print()
        print("⏱️  Estimated Timeline:")
        print("   • Annotation: 2-5 days (depending on dataset size)")
        print("   • Training: 2-6 hours (depending on GPU)")
        print("   • Integration: 1-2 days")
        print("   • Total: 1-2 weeks")

    def show_alternative_approaches(self):
        """Show alternative approaches if YOLO training is not feasible."""
        print("\n🔄 Alternative Approaches")
        print("=" * 30)

        print("1. 🎯 Simplified Landmark Detection:")
        print("   • Use template matching for eye detection")
        print("   • Find bright spots (eyes) in cropped face region")
        print("   • Simple but might work for consistent bat poses")
        print()

        print("2. 🤖 Pre-trained Model Adaptation:")
        print("   • Use MediaPipe Face Mesh as starting point")
        print("   • Fine-tune on bat faces")
        print("   • Requires deep learning expertise")
        print()

        print("3. 🔧 Hybrid Approach:")
        print("   • Use your YOLO for face detection")
        print("   • Use traditional CV for eye detection within face")
        print("   • Combine strengths of both approaches")
        print()

        print("4. ⚡ Skip Landmark-based Alignment:")
        print("   • Use only segmentation-based cropping")
        print("   • Rely on consistent face positioning in data")
        print("   • Fastest to implement")


def main():
    """Run the training guide."""
    trainer = BatFaceLandmarkTrainer()

    print("🦇 Custom Bat Face Detector Training Guide")
    print("=" * 60)
    print()

    # Show all sections
    trainer.show_training_approaches()
    trainer.create_yolo_landmark_dataset_structure()
    trainer.show_annotation_process()
    trainer.show_training_code_example()
    trainer.show_integration_example()
    trainer.show_quick_start_checklist()
    trainer.show_alternative_approaches()

    print("\n🎯 RECOMMENDATION:")
    print("   Start with YOLO landmark detection approach")
    print("   You already have the infrastructure and expertise")
    print("   200-300 well-annotated images should be sufficient")
    print()
    print("💡 NEXT STEPS:")
    print("   1. Select your best 200-300 bat face images")
    print("   2. Set up CVAT for annotation")
    print("   3. Begin annotating landmarks consistently")


if __name__ == "__main__":
    main()
