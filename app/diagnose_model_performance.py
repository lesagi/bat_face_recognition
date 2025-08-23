#!/usr/bin/env python3
"""
Diagnostic script to analyze pose model and segmentation model performance
on a specific image without running the full pipeline.
"""

import os
import sys
import cv2
import numpy as np

# Add app directory to path
app_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, app_dir)

from image_processor.processor import ImageProcessor
from ultralytics import YOLO


def test_pose_model(image_path: str, pose_model_path: str, output_dir: str):
    """Test YOLO pose model performance on the image."""
    print(f"\n🎯 TESTING POSE MODEL PERFORMANCE")
    print(f"=" * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📁 Image loaded: {image_rgb.shape}")

    # Load pose model
    try:
        pose_model = YOLO(pose_model_path)
        print(f"✅ Pose model loaded: {pose_model_path}")
    except Exception as e:
        print(f"❌ Failed to load pose model: {e}")
        return

    # Test with different confidence thresholds
    confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print(f"\n🔍 Testing with different confidence thresholds:")

    best_result = None
    best_confidence = None

    for conf in confidence_thresholds:
        try:
            results = pose_model(image_rgb, conf=conf, verbose=False)

            if results and len(results) > 0:
                result = results[0]

                # Check for detections
                if (
                    hasattr(result, "boxes")
                    and result.boxes is not None
                    and len(result.boxes) > 0
                ):
                    num_detections = len(result.boxes)
                    boxes_conf = (
                        result.boxes.conf.cpu().numpy()
                        if result.boxes.conf is not None
                        else []
                    )

                    # Check for keypoints
                    keypoints_info = ""
                    if hasattr(result, "keypoints") and result.keypoints is not None:
                        keypoints = result.keypoints.xy.cpu().numpy()
                        num_keypoints = len(keypoints[0]) if len(keypoints) > 0 else 0
                        keypoints_info = f", {num_keypoints} keypoints"

                    print(
                        f"   ✅ Confidence {conf}: {num_detections} detections (conf: {boxes_conf}){keypoints_info}"
                    )

                    if best_result is None:
                        best_result = result
                        best_confidence = conf
                else:
                    print(f"   ❌ Confidence {conf}: No detections")
            else:
                print(f"   ❌ Confidence {conf}: No results")

        except Exception as e:
            print(f"   ❌ Confidence {conf}: Error - {e}")

    # Save best result if any
    if best_result is not None:
        print(f"\n✅ Best pose detection at confidence {best_confidence}")

        # Save annotated image with green keypoints
        try:
            # Create custom visualization with green keypoints
            custom_img = image_rgb.copy()

            # Get keypoints and draw them manually in different shades of green
            if hasattr(best_result, "keypoints") and best_result.keypoints is not None:
                keypoints = best_result.keypoints.xy.cpu().numpy()

                # Define green shades for different keypoints
                green_colors = [
                    (0, 255, 0),  # Bright green for left eye
                    (0, 200, 0),  # Medium green for right eye
                    (0, 150, 0),  # Dark green for nose
                ]

                # Draw keypoints with labels
                keypoint_labels = ["Left Eye", "Right Eye", "Nose"]

                for detection_idx, kpts in enumerate(keypoints):
                    for kpt_idx, (x, y) in enumerate(kpts):
                        if x > 0 and y > 0:  # Valid keypoint
                            color = green_colors[kpt_idx % len(green_colors)]
                            # Draw larger circles for visibility
                            cv2.circle(custom_img, (int(x), int(y)), 12, color, -1)
                            cv2.circle(
                                custom_img, (int(x), int(y)), 15, (255, 255, 255), 2
                            )  # White border

                            # Add label
                            label = keypoint_labels[kpt_idx % len(keypoint_labels)]
                            cv2.putText(
                                custom_img,
                                label,
                                (int(x) + 20, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                color,
                                2,
                            )

            # Draw bounding box in green
            if hasattr(best_result, "boxes") and best_result.boxes is not None:
                boxes = best_result.boxes.xyxy.cpu().numpy()
                confidences = best_result.boxes.conf.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(custom_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Add confidence label
                    conf_text = f"Face: {confidences[i]:.3f}"
                    cv2.putText(
                        custom_img,
                        conf_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            annotated_path = os.path.join(
                output_dir, f"pose_detection_green_conf_{best_confidence}.jpg"
            )
            cv2.imwrite(annotated_path, cv2.cvtColor(custom_img, cv2.COLOR_RGB2BGR))
            print(f"💾 Green keypoints image saved: {annotated_path}")

        except Exception as e:
            print(f"⚠️  Could not save green keypoints image: {e}")
    else:
        print(f"\n❌ No pose detections found at any confidence level")


def test_segmentation_model(
    image_path: str, seg_model_path: str, model_name: str, output_dir: str
):
    """Test segmentation model performance on the image."""
    print(f"\n🔍 TESTING {model_name.upper()} SEGMENTATION MODEL")
    print(f"=" * 60)

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Failed to load image: {image_path}")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    print(f"📁 Image loaded: {image_rgb.shape}")

    # Test with different confidence thresholds
    confidence_thresholds = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    print(f"🔍 Testing with different confidence thresholds:")

    best_mask = None
    best_confidence = None
    best_pixel_count = 0

    for conf in confidence_thresholds:
        try:
            # Create processor with current confidence
            processor = ImageProcessor(model=seg_model_path, confidence_threshold=conf)

            # Try segmentation
            segmentation_result = processor.segment_image(image_rgb)

            if segmentation_result is not None:
                segmented_image, mask = segmentation_result
                pixel_count = np.sum(mask > 0)
                total_pixels = mask.size
                percentage = (pixel_count / total_pixels) * 100

                print(
                    f"   ✅ Confidence {conf}: {pixel_count} foreground pixels ({percentage:.1f}%)"
                )

                if pixel_count > best_pixel_count:
                    best_mask = mask
                    best_confidence = conf
                    best_pixel_count = pixel_count
            else:
                print(f"   ❌ Confidence {conf}: No segmentation")

        except Exception as e:
            print(f"   ❌ Confidence {conf}: Error - {e}")

    # Save best result if any
    if best_mask is not None:
        print(f"\n✅ Best {model_name} segmentation at confidence {best_confidence}")
        print(
            f"   📊 {best_pixel_count} foreground pixels ({(best_pixel_count/best_mask.size)*100:.1f}%)"
        )

        # Save mask
        mask_path = os.path.join(
            output_dir, f"{model_name}_segmentation_mask_conf_{best_confidence}.jpg"
        )
        cv2.imwrite(mask_path, (best_mask * 255).astype(np.uint8))
        print(f"💾 Best {model_name} mask saved: {mask_path}")

        # Save mask overlay
        try:
            overlay = image_rgb.copy()
            overlay[best_mask == 1] = [255, 0, 0]  # Red overlay for mask
            blended = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
            overlay_path = os.path.join(
                output_dir,
                f"{model_name}_segmentation_overlay_conf_{best_confidence}.jpg",
            )
            cv2.imwrite(overlay_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            print(f"💾 {model_name} overlay saved: {overlay_path}")
        except Exception as e:
            print(f"⚠️  Could not save overlay: {e}")
    else:
        print(f"\n❌ No {model_name} segmentation found at any confidence level")


def main():
    """Main diagnostic function."""

    # Configuration
    config = {
        "image_path": "/Users/MAC/Documents/bat_face_rec/face_recognition-original_background_20_06_24/data/training/20230831_143044/20230831_143044.1382.jpg",
        "pose_model_path": "/Users/MAC/Documents/bat_face_rec/face_annotation_eyes_nose/runs/weights/best.pt",
        "rousettus_seg_model_path": "/Users/MAC/Documents/bat_face_rec/app/yolo_segmentation_trainer/rousesttus/training_results/runs/segment/bat_face_seg/weights/best.pt",
        "mauritius_seg_model_path": "/Users/MAC/Documents/bat_face_rec/face_segmentation_mauritius/chosen_model/best.pt",
        "output_dir": "/Users/MAC/Documents/bat_face_rec/model_performance_diagnosis_face_recognition_data",
    }

    image_name = os.path.basename(config["image_path"])

    print(f"🔬 MODEL PERFORMANCE DIAGNOSIS")
    print(f"=" * 80)
    print(f"🖼️  Target image: {image_name}")
    print(f"📁 Image path: {config['image_path']}")
    print(f"🎯 Pose model: {config['pose_model_path']}")
    print(f"🔍 Rousettus segmentation: {config['rousettus_seg_model_path']}")
    print(f"🔍 Mauritius segmentation: {config['mauritius_seg_model_path']}")
    print(f"📤 Output directory: {config['output_dir']}")

    # Create output directory
    os.makedirs(config["output_dir"], exist_ok=True)

    # Check if image exists
    if not os.path.exists(config["image_path"]):
        print(f"❌ Image not found: {config['image_path']}")
        return

    # Test pose model
    test_pose_model(
        config["image_path"], config["pose_model_path"], config["output_dir"]
    )

    # Test rousettus segmentation model
    test_segmentation_model(
        config["image_path"],
        config["rousettus_seg_model_path"],
        "rousettus",
        config["output_dir"],
    )

    # Test mauritius segmentation model
    test_segmentation_model(
        config["image_path"],
        config["mauritius_seg_model_path"],
        "mauritius",
        config["output_dir"],
    )

    print(f"\n🎉 DIAGNOSIS COMPLETE!")
    print(f"=" * 80)
    print(f"📁 Results saved in: {config['output_dir']}")
    print(f"🔍 Check the saved images to understand model performance")


if __name__ == "__main__":
    main()
