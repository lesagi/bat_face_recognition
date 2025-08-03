"""
Image processing transforms module.

This module contains all image processing functions as static methods.
Functions are organized into:
- Mask-based transforms: Require segmentation masks from models
- Plain transforms: Work with images directly without models
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union


class ImageTransforms:
    """Static methods for all image processing transforms."""

    # =========================================================================
    # MASK-BASED TRANSFORMS (Require prediction masks)
    # =========================================================================

    @staticmethod
    def apply_background_replacement(
        original_image: np.ndarray,
        mask: np.ndarray,
        background_source: callable,
        **generator_kwargs,
    ) -> np.ndarray:
        """Replace background using segmentation mask with generated background.

        Args:
            original_image: Original image array
            mask: Binary segmentation mask (foreground = 1, background = 0)
            background_source: Generator function that takes (height, width, **kwargs)
            **generator_kwargs: Additional arguments for generator function

        Returns:
            Image with replaced background
        """
        # Ensure mask is 3D for broadcasting
        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        # Generate background using the provided generator
        height, width = original_image.shape[:2]
        background_image = background_source(height, width, **generator_kwargs)

        # Ensure background matches image size (should already match, but safety check)
        if background_image.shape != original_image.shape:
            background_image = cv2.resize(
                background_image, (original_image.shape[1], original_image.shape[0])
            )

        # Apply mask: keep foreground, replace background
        result = background_image.copy()
        result[mask_3d == 1] = original_image[mask_3d == 1]

        return result

    @staticmethod
    def create_blurred_background(
        original_image: np.ndarray, mask: np.ndarray, blur_strength: int = 15
    ) -> np.ndarray:
        """Create blurred background effect using segmentation mask.

        Args:
            original_image: Original image array
            mask: Binary segmentation mask (foreground = 1, background = 0)
            blur_strength: Blur kernel size (should be odd)

        Returns:
            Image with blurred background
        """
        # Ensure blur strength is odd
        if blur_strength % 2 == 0:
            blur_strength += 1

        # Create blurred version of the image
        blurred = cv2.GaussianBlur(original_image, (blur_strength, blur_strength), 0)

        # Ensure mask is 3D for broadcasting
        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        # Keep original foreground, use blurred background
        result = blurred.copy()
        result[mask_3d == 1] = original_image[mask_3d == 1]

        return result

    @staticmethod
    def create_color_background(
        original_image: np.ndarray,
        mask: np.ndarray,
        background_color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """Create solid color background using segmentation mask.

        Args:
            original_image: Original image array
            mask: Binary segmentation mask (foreground = 1, background = 0)
            background_color: RGB color for background (B, G, R for OpenCV)

        Returns:
            Image with solid color background
        """
        height, width = original_image.shape[:2]

        # Create solid color background
        background = np.full((height, width, 3), background_color, dtype=np.uint8)

        # Ensure mask is 3D for broadcasting
        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        # Keep original foreground, use color background
        result = background.copy()
        result[mask_3d == 1] = original_image[mask_3d == 1]

        return result

    # =========================================================================
    # PLAIN IMAGE TRANSFORMS (No model required)
    # =========================================================================

    @staticmethod
    def resize_image(
        original_image: np.ndarray,
        target_size: Tuple[int, int],
        interpolation: str = "bilinear",
    ) -> np.ndarray:
        """Resize image to target dimensions.

        Args:
            original_image: Input image array
            target_size: Target (width, height) dimensions
            interpolation: Interpolation method ('bilinear', 'nearest', 'cubic', 'area')

        Returns:
            Resized image array
        """
        # Map interpolation methods
        methods = {
            "bilinear": cv2.INTER_LINEAR,
            "nearest": cv2.INTER_NEAREST,
            "cubic": cv2.INTER_CUBIC,
            "area": cv2.INTER_AREA,
        }
        method = methods.get(interpolation, cv2.INTER_LINEAR)

        return cv2.resize(original_image, target_size, interpolation=method)

    @staticmethod
    def normalize_image(
        original_image: np.ndarray,
        scale: float = 255.0,
        mean_subtract: Optional[Union[float, Tuple[float, float, float]]] = None,
        std_divide: Optional[Union[float, Tuple[float, float, float]]] = None,
        clip_range: Optional[Tuple[float, float]] = None,
    ) -> np.ndarray:
        """Normalize image pixel values with advanced options.

        Args:
            original_image: Input image array
            scale: Scale factor for normalization (default: 255.0 for [0,1] range)
            mean_subtract: Mean to subtract (single value or per-channel)
            std_divide: Standard deviation to divide by (single value or per-channel)
            clip_range: Range to clip values to (min, max)

        Returns:
            Normalized image array
        """
        # Convert to float and apply basic scaling
        normalized = original_image.astype(np.float32) / scale

        # Mean subtraction (for models trained with specific means)
        if mean_subtract is not None:
            if isinstance(mean_subtract, (list, tuple)):
                # Per-channel means [R, G, B]
                for i, mean in enumerate(mean_subtract):
                    normalized[:, :, i] -= mean
            else:
                # Single mean for all channels
                normalized -= mean_subtract

        # Standard deviation division
        if std_divide is not None:
            if isinstance(std_divide, (list, tuple)):
                # Per-channel stds [R, G, B]
                for i, std in enumerate(std_divide):
                    normalized[:, :, i] /= std
            else:
                # Single std for all channels
                normalized /= std_divide

        # Clip to range
        if clip_range is not None:
            normalized = np.clip(normalized, clip_range[0], clip_range[1])

        return normalized

    @staticmethod
    def center_crop_square(
        original_image: np.ndarray, size: Optional[int] = None, center_crop_coordinates: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """Crop image to centered square.

        Args:
            original_image: Input image array
            size: Target size for square (if None, uses minimum of width/height)
            center_crop_coordinates: Center crop coordinates (x, y)
        Returns:
            Square cropped image
        """
        height, width = original_image.shape[:2]

        if size is None:
            size = min(height, width)

        # Calculate center crop coordinates
        if center_crop_coordinates is None:
            center_x, center_y = width // 2, height // 2
        else:
            center_x, center_y = center_crop_coordinates
        half_size = size // 2

        x1 = max(0, center_x - half_size)
        x2 = min(width, center_x + half_size)
        y1 = max(0, center_y - half_size)
        y2 = min(height, center_y + half_size)

        return original_image[y1:y2, x1:x2]

    @staticmethod
    def adjust_brightness_contrast(
        original_image: np.ndarray, brightness: float = 1.0, contrast: float = 1.0
    ) -> np.ndarray:
        """Adjust image brightness and contrast.

        Args:
            original_image: Input image array
            brightness: Brightness multiplier (1.0 = no change)
            contrast: Contrast multiplier (1.0 = no change)

        Returns:
            Adjusted image array
        """
        # Apply brightness and contrast
        # Formula: new_image = contrast * original_image + brightness_offset
        brightness_offset = (brightness - 1.0) * 30  # Scale brightness for OpenCV
        adjusted = cv2.convertScaleAbs(
            original_image, alpha=contrast, beta=brightness_offset
        )

        return adjusted

    @staticmethod
    def apply_gamma_correction(
        original_image: np.ndarray, gamma: float = 1.0
    ) -> np.ndarray:
        """Apply gamma correction to image.

        Args:
            original_image: Input image array
            gamma: Gamma value (< 1.0 = brighter, > 1.0 = darker, 1.0 = no change)

        Returns:
            Gamma corrected image array
        """
        if gamma == 1.0:
            return original_image

        # Build gamma correction lookup table
        gamma_table = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in np.arange(0, 256)]
        ).astype("uint8")

        # Apply gamma correction using lookup table
        return cv2.LUT(original_image, gamma_table)

    @staticmethod
    def apply_gaussian_blur(
        original_image: np.ndarray, kernel_size: int = 15, sigma: float = 0
    ) -> np.ndarray:
        """Apply Gaussian blur to image.

        Args:
            original_image: Input image array
            kernel_size: Size of the Gaussian kernel (should be odd)
            sigma: Standard deviation for Gaussian kernel (0 = auto-calculate)

        Returns:
            Blurred image array
        """
        # Ensure kernel size is odd
        if kernel_size % 2 == 0:
            kernel_size += 1

        return cv2.GaussianBlur(original_image, (kernel_size, kernel_size), sigma)

    @staticmethod
    def convert_color_space(
        original_image: np.ndarray, conversion: str = "BGR2RGB"
    ) -> np.ndarray:
        """Convert image between different color spaces.

        Args:
            original_image: Input image array
            conversion: Color space conversion ('BGR2RGB', 'BGR2GRAY', 'BGR2HSV', etc.)

        Returns:
            Converted image array
        """
        # Map common conversions
        conversions = {
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "GRAY2BGR": cv2.COLOR_GRAY2BGR,
            "BGR2HSV": cv2.COLOR_BGR2HSV,
            "HSV2BGR": cv2.COLOR_HSV2BGR,
        }

        if conversion in conversions:
            return cv2.cvtColor(original_image, conversions[conversion])
        else:
            # Try to use conversion string directly with OpenCV
            try:
                color_code = getattr(cv2, f"COLOR_{conversion}")
                return cv2.cvtColor(original_image, color_code)
            except AttributeError:
                raise ValueError(f"Unknown color conversion: {conversion}")

    @staticmethod
    def crop_square_around_segmentation(
        original_image: np.ndarray, mask: np.ndarray, buffer_factor: float = 1.2
    ) -> np.ndarray:
        """Crop square around segmented region from mask with buffer padding.

        Args:
            original_image: Original image array
            mask: Binary segmentation mask (foreground = 1, background = 0)
            buffer_factor: Factor to expand bounding box (1.0 = no buffer, 1.2 = 20% buffer)

        Returns:
            Cropped square image around segmented region with buffer
        """
        # Find bounding box of segmented region
        # Get coordinates where mask is positive
        y_coords, x_coords = np.where(mask > 0)

        if len(y_coords) == 0:
            # If no segmented region found, return center crop of entire image
            return ImageTransforms.center_crop_square(original_image)

        # Calculate tight bounding box from segmented coordinates
        min_x1 = float(np.min(x_coords))
        min_y1 = float(np.min(y_coords))
        max_x2 = float(np.max(x_coords))
        max_y2 = float(np.max(y_coords))

        # Add buffer to bounding box
        box_width = max_x2 - min_x1
        box_height = max_y2 - min_y1

        # Calculate buffer amounts
        width_buffer = (box_width * (buffer_factor - 1.0)) / 2
        height_buffer = (box_height * (buffer_factor - 1.0)) / 2

        # Expand bounding box with buffer
        buffered_min_x1 = min_x1 - width_buffer
        buffered_min_y1 = min_y1 - height_buffer
        buffered_max_x2 = max_x2 + width_buffer
        buffered_max_y2 = max_y2 + height_buffer

        # Clamp to image boundaries
        img_height, img_width = original_image.shape[:2]
        buffered_min_x1 = max(0, buffered_min_x1)
        buffered_min_y1 = max(0, buffered_min_y1)
        buffered_max_x2 = min(img_width, buffered_max_x2)
        buffered_max_y2 = min(img_height, buffered_max_y2)

        bbox = (buffered_min_x1, buffered_min_y1, buffered_max_x2, buffered_max_y2)

        # Use the bbox cropping method
        return ImageTransforms._crop_square_from_bbox(original_image, bbox)

    @staticmethod
    def _crop_square_from_bbox(
        original_image: np.ndarray, bbox: Tuple[float, float, float, float]
    ) -> np.ndarray:
        """Crop image to centered square based on bounding box coordinates.

        This function replicates the square cropping logic from main.py:
        1. Calculate center and dimensions from bounding box
        2. Create square crop using minimum dimension
        3. Return centered square region

        Args:
            original_image: Input image array
            bbox: Bounding box coordinates (min_x1, min_y1, max_x2, max_y2)

        Returns:
            Cropped square image
        """
        min_x1, min_y1, max_x2, max_y2 = bbox

        # Calculate center and dimensions
        box_width = max_x2 - min_x1
        box_height = max_y2 - min_y1
        x_center = (min_x1 + max_x2) / 2
        y_center = (min_y1 + max_y2) / 2

        # Create square crop using minimum dimension
        edge_length = min(box_width, box_height)
        x1_rect = int(x_center - edge_length / 2)
        x2_rect = int(x_center + edge_length / 2)
        y1_rect = int(y_center - edge_length / 2)
        y2_rect = int(y_center + edge_length / 2)

        return original_image[y1_rect:y2_rect, x1_rect:x2_rect]

    @staticmethod
    def pad_to_square(
        original_image: np.ndarray,
        pad_color: Tuple[int, int, int] = (0, 0, 0),
        target_size: Optional[int] = None,
    ) -> np.ndarray:
        """Pad image to square dimensions.

        Args:
            original_image: Input image array
            pad_color: Color for padding (B, G, R)
            target_size: Target size for square (if None, uses max of width/height)

        Returns:
            Padded square image
        """
        height, width = original_image.shape[:2]

        if target_size is None:
            target_size = max(height, width)

        # Calculate padding
        pad_h = (target_size - height) // 2
        pad_w = (target_size - width) // 2

        # Pad image
        padded = cv2.copyMakeBorder(
            original_image,
            pad_h,
            target_size - height - pad_h,  # top, bottom
            pad_w,
            target_size - width - pad_w,  # left, right
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

        return padded

    @staticmethod
    def align_face_landmarks(
        original_image: np.ndarray,
        target_eye_distance: Optional[float] = None,
        face_detector_type: str = "yolo_pose",
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        yolo_model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Align face by rotating so eye line is horizontal to the top border.

        This method detects facial landmarks (eyes and nose) and rotates the image
        so that the line connecting the eyes becomes parallel to the top of the image.

        Args:
            original_image: Input image array
            target_eye_distance: Optional target distance between eyes in pixels
            face_detector_type: Type of face detector ('opencv', 'mediapipe', 'yolo_pose')
            scale_factor: Scale factor for face detection (OpenCV)
            min_neighbors: Minimum neighbors for face detection (OpenCV)
            yolo_model_path: Path to YOLO pose model (.pt file) - required for 'yolo_pose'
            confidence_threshold: Confidence threshold for YOLO pose detection
            debug: If True, return debug information

        Returns:
            Aligned image array, or None if alignment fails

        Note:
            For 'yolo_pose' mode, a trained YOLO pose model is required that can detect
            bat face keypoints (left_eye, right_eye, nose).
        """
        try:
            if face_detector_type == "opencv":
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                return ImageTransforms._align_face_opencv(
                    original_image,
                    gray,
                    target_eye_distance,
                    scale_factor,
                    min_neighbors,
                    debug,
                )
            elif face_detector_type == "mediapipe":
                return ImageTransforms._align_face_mediapipe(
                    original_image, target_eye_distance, debug
                )
            elif face_detector_type == "yolo_pose":
                if yolo_model_path is None:
                    raise ValueError(
                        "yolo_model_path is required for yolo_pose face detection"
                    )
                return ImageTransforms._align_face_yolo_pose(
                    original_image,
                    yolo_model_path,
                    target_eye_distance,
                    confidence_threshold,
                    debug,
                )
            else:
                raise ValueError(f"Unknown face_detector_type: {face_detector_type}")

        except Exception as e:
            if debug:
                print(f"Face alignment failed: {e}")
            return None

    @staticmethod
    def _align_face_opencv(
        original_image: np.ndarray,
        gray: np.ndarray,
        target_eye_distance: Optional[float] = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Align face using OpenCV's cascade classifiers."""
        try:
            import cv2

            # Load face and eye cascade classifiers
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)

            if len(faces) == 0:
                if debug:
                    print("No faces detected")
                return None

            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Extract face region for eye detection
            face_gray = gray[y : y + h, x : x + w]

            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)

            if len(eyes) < 2:
                if debug:
                    print(f"Found {len(eyes)} eyes, need at least 2")
                return None

            # Sort eyes by area (largest first) and take top 2
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

            # Calculate eye centers (relative to face region)
            eye_centers = []
            for ex, ey, ew, eh in eyes:
                center_x = ex + ew // 2
                center_y = ey + eh // 2
                eye_centers.append((center_x, center_y))

            # Convert to absolute coordinates
            eye_centers = [(x + cx, y + cy) for cx, cy in eye_centers]

            # Sort eyes left to right
            eye_centers = sorted(eye_centers, key=lambda p: p[0])
            left_eye, right_eye = eye_centers

            # Calculate rotation angle
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            if debug:
                print(f"Eye centers: {left_eye}, {right_eye}")
                print(f"Rotation angle: {angle:.2f} degrees")

            # Calculate rotation center (midpoint between eyes)
            center_x = (left_eye[0] + right_eye[0]) // 2
            center_y = (left_eye[1] + right_eye[1]) // 2
            center = (center_x, center_y)

            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation
            aligned_image = cv2.warpAffine(
                original_image,
                rotation_matrix,
                (original_image.shape[1], original_image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            # Optional: Scale to target eye distance
            if target_eye_distance is not None:
                current_distance = np.linalg.norm(
                    np.array(right_eye) - np.array(left_eye)
                )
                scale = target_eye_distance / current_distance

                if scale != 1.0:
                    new_width = int(aligned_image.shape[1] * scale)
                    new_height = int(aligned_image.shape[0] * scale)
                    aligned_image = cv2.resize(aligned_image, (new_width, new_height))

            return aligned_image

        except Exception as e:
            if debug:
                print(f"OpenCV face alignment failed: {e}")
            return None

    @staticmethod
    def _align_face_mediapipe(
        original_image: np.ndarray,
        target_eye_distance: Optional[float] = None,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Align face using MediaPipe Face Mesh (more accurate for landmarks)."""
        try:
            import mediapipe as mp

            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                if debug:
                    print("No face landmarks detected with MediaPipe")
                return None

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = original_image.shape[:2]

            # MediaPipe landmark indices for eyes
            # Left eye: landmarks around 33, 133, 157, 158, 159, 160, 161, 173
            # Right eye: landmarks around 362, 398, 384, 385, 386, 387, 388, 466
            # Eye centers (approximate): left eye center ≈ 468, right eye center ≈ 473

            # Use eye corner landmarks for more stability
            left_eye_outer = face_landmarks.landmark[33]  # Left eye outer corner
            left_eye_inner = face_landmarks.landmark[133]  # Left eye inner corner
            right_eye_inner = face_landmarks.landmark[362]  # Right eye inner corner
            right_eye_outer = face_landmarks.landmark[398]  # Right eye outer corner

            # Calculate eye centers
            left_eye_center = (
                int((left_eye_outer.x + left_eye_inner.x) * w / 2),
                int((left_eye_outer.y + left_eye_inner.y) * h / 2),
            )
            right_eye_center = (
                int((right_eye_inner.x + right_eye_outer.x) * w / 2),
                int((right_eye_inner.y + right_eye_outer.y) * h / 2),
            )

            # Calculate rotation angle
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            if debug:
                print(f"Eye centers: {left_eye_center}, {right_eye_center}")
                print(f"Rotation angle: {angle:.2f} degrees")

            # Calculate rotation center (midpoint between eyes)
            center_x = (left_eye_center[0] + right_eye_center[0]) // 2
            center_y = (left_eye_center[1] + right_eye_center[1]) // 2
            center = (center_x, center_y)

            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation
            aligned_image = cv2.warpAffine(
                original_image,
                rotation_matrix,
                (original_image.shape[1], original_image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            # Optional: Scale to target eye distance
            if target_eye_distance is not None:
                current_distance = np.linalg.norm(
                    np.array(right_eye_center) - np.array(left_eye_center)
                )
                scale = target_eye_distance / current_distance

                if scale != 1.0:
                    new_width = int(aligned_image.shape[1] * scale)
                    new_height = int(aligned_image.shape[0] * scale)
                    aligned_image = cv2.resize(aligned_image, (new_width, new_height))

            return aligned_image

        except ImportError:
            if debug:
                print("MediaPipe not available. Install with: pip install mediapipe")
            return None
        except Exception as e:
            if debug:
                print(f"MediaPipe face alignment failed: {e}")
            return None

    @staticmethod
    def _align_face_yolo_pose(
        original_image: np.ndarray,
        yolo_model_path: str,
        target_eye_distance: Optional[float] = None,
        confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Align face using trained YOLO pose model for bat face landmarks."""
        try:
            from ultralytics import YOLO

            # Load YOLO pose model
            yolo_model = YOLO(yolo_model_path)

            # Perform inference on original image (YOLO handles color conversion internally)
            results = yolo_model.predict(
                source=original_image,
                conf=confidence_threshold,
                verbose=False,
                save=False,
            )

            if not results or len(results) == 0:
                if debug:
                    print("No detections from YOLO pose model")
                return None

            result = results[0]

            # Check if we have keypoints and boxes
            if not hasattr(result, "keypoints") or result.keypoints is None:
                if debug:
                    print("No keypoints detected by YOLO pose model")
                return None

            if (
                not hasattr(result, "boxes")
                or result.boxes is None
                or len(result.boxes) == 0
            ):
                if debug:
                    print("No face detections from YOLO pose model")
                return None

            # Get keypoints from the first (highest confidence) detection
            keypoints = result.keypoints.xy.cpu().numpy()  # [N, num_keypoints, 2]

            if len(keypoints) == 0:
                if debug:
                    print("No keypoints found in detection")
                return None

            # Get keypoints for the first detection [num_keypoints, 2]
            kpts = keypoints[0]

            # Your trained model should output keypoints in order: [left_eye, right_eye, nose]
            if len(kpts) < 2:
                if debug:
                    print(
                        "Not enough keypoints detected (need at least left_eye and right_eye)"
                    )
                return None

            # Extract eye coordinates
            left_eye = kpts[0]  # [x, y] for left eye
            right_eye = kpts[1]  # [x, y] for right eye

            # Check if keypoints are valid (not zeros or NaN)
            if (
                np.any(np.isnan(left_eye))
                or np.any(np.isnan(right_eye))
                or np.allclose(left_eye, 0)
                or np.allclose(right_eye, 0)
            ):
                if debug:
                    print("Invalid eye keypoints detected")
                return None

            # Convert to integer coordinates
            left_eye_center = (int(left_eye[0]), int(left_eye[1]))
            right_eye_center = (int(right_eye[0]), int(right_eye[1]))

            # Calculate rotation angle
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))

            if debug:
                print(f"YOLO Pose - Eye centers: {left_eye_center}, {right_eye_center}")
                print(f"YOLO Pose - Rotation angle: {angle:.2f} degrees")

            # Calculate rotation center (midpoint between eyes)
            center_x = (left_eye_center[0] + right_eye_center[0]) // 2
            center_y = (left_eye_center[1] + right_eye_center[1]) // 2
            center = (center_x, center_y)

            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            # Apply rotation
            aligned_image = cv2.warpAffine(
                original_image,
                rotation_matrix,
                (original_image.shape[1], original_image.shape[0]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            # Optional: Scale to target eye distance
            if target_eye_distance is not None:
                current_distance = np.linalg.norm(
                    np.array(right_eye_center) - np.array(left_eye_center)
                )
                if current_distance > 0:  # Avoid division by zero
                    scale = target_eye_distance / current_distance

                    if scale != 1.0:
                        new_width = int(aligned_image.shape[1] * scale)
                        new_height = int(aligned_image.shape[0] * scale)
                        aligned_image = cv2.resize(
                            aligned_image, (new_width, new_height)
                        )

            return aligned_image

        except ImportError:
            if debug:
                print(
                    "YOLO (ultralytics) not available. Install with: pip install ultralytics"
                )
            return None
        except Exception as e:
            if debug:
                print(f"YOLO pose face alignment failed: {e}")
            return None

    @staticmethod
    def center_face_landmarks(
        original_image: np.ndarray,
        target_face_size: int = 224,
        eye_y_ratio: float = 0.35,
        face_width_ratio: float = 0.8,
        face_detector_type: str = "opencv",
        yolo_model_path: Optional[str] = None,
        confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Center and standardize face position based on facial landmarks.

        This method ensures that eyes and nose are positioned consistently across images
        by cropping around facial landmarks at standardized positions.

        Args:
            original_image: Input image array
            target_face_size: Target size for the standardized face crop
            eye_y_ratio: Ratio from top where eye line should be positioned (0.35 = 35% from top)
            face_width_ratio: Ratio of face width to total crop width (0.8 = face takes 80% of width)
            face_detector_type: Type of face detector ('opencv', 'mediapipe', 'yolo_pose')
            yolo_model_path: Path to YOLO pose model (.pt file) - required for 'yolo_pose'
            confidence_threshold: Confidence threshold for YOLO pose detection
            debug: If True, show debug information

        Returns:
            Standardized face crop with consistent landmark positions, or None if failed
        """
        try:
            if face_detector_type == "opencv":
                # Convert to grayscale for face detection
                gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                return ImageTransforms._center_face_opencv(
                    original_image,
                    gray,
                    target_face_size,
                    eye_y_ratio,
                    face_width_ratio,
                    debug,
                )
            elif face_detector_type == "mediapipe":
                return ImageTransforms._center_face_mediapipe(
                    original_image,
                    target_face_size,
                    eye_y_ratio,
                    face_width_ratio,
                    debug,
                )
            elif face_detector_type == "yolo_pose":
                if yolo_model_path is None:
                    raise ValueError(
                        "yolo_model_path is required for yolo_pose face detection"
                    )
                return ImageTransforms._center_face_yolo_pose(
                    original_image,
                    yolo_model_path,
                    target_face_size,
                    eye_y_ratio,
                    face_width_ratio,
                    confidence_threshold,
                    debug,
                )
            else:
                raise ValueError(f"Unknown face_detector_type: {face_detector_type}")

        except Exception as e:
            if debug:
                print(f"Face centering failed: {e}")
            return None

    @staticmethod
    def _center_face_opencv(
        original_image: np.ndarray,
        gray: np.ndarray,
        target_face_size: int = 224,
        eye_y_ratio: float = 0.35,
        face_width_ratio: float = 0.8,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Center face using OpenCV's cascade classifiers."""
        try:
            # Load face and eye cascade classifiers
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )

            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)

            if len(faces) == 0:
                if debug:
                    print("No faces detected for centering")
                return None

            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face

            # Extract face region for eye detection
            face_gray = gray[y : y + h, x : x + w]

            # Detect eyes within the face region
            eyes = eye_cascade.detectMultiScale(face_gray, 1.1, 5)

            if len(eyes) < 2:
                if debug:
                    print(
                        f"Found {len(eyes)} eyes for centering, using face center as fallback"
                    )
                # Fallback: use face center
                eye_center_x = x + w // 2
                eye_center_y = y + h // 3  # Eyes are typically in upper third of face
            else:
                # Sort eyes by area and take top 2
                eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

                # Calculate eye centers (relative to face region)
                eye_centers = []
                for ex, ey, ew, eh in eyes:
                    center_x = ex + ew // 2
                    center_y = ey + eh // 2
                    eye_centers.append((center_x, center_y))

                # Convert to absolute coordinates
                eye_centers = [(x + cx, y + cy) for cx, cy in eye_centers]

                # Calculate midpoint between eyes
                eye_center_x = (eye_centers[0][0] + eye_centers[1][0]) // 2
                eye_center_y = (eye_centers[0][1] + eye_centers[1][1]) // 2

            # Calculate crop region to center the face
            # We want the eye line to be at eye_y_ratio from the top of our crop
            crop_top = int(eye_center_y - (target_face_size * eye_y_ratio))
            crop_bottom = crop_top + target_face_size

            # For horizontal centering, center on the eye midpoint
            crop_left = int(eye_center_x - (target_face_size // 2))
            crop_right = crop_left + target_face_size

            # Ensure crop region is within image bounds
            img_height, img_width = original_image.shape[:2]

            # Adjust if crop goes outside image bounds
            if crop_top < 0:
                offset = -crop_top
                crop_top = 0
                crop_bottom = min(img_height, crop_bottom + offset)
            elif crop_bottom > img_height:
                offset = crop_bottom - img_height
                crop_bottom = img_height
                crop_top = max(0, crop_top - offset)

            if crop_left < 0:
                offset = -crop_left
                crop_left = 0
                crop_right = min(img_width, crop_right + offset)
            elif crop_right > img_width:
                offset = crop_right - img_width
                crop_right = img_width
                crop_left = max(0, crop_left - offset)

            if debug:
                print(f"Eye center: ({eye_center_x}, {eye_center_y})")
                print(
                    f"Crop region: ({crop_left}, {crop_top}) to ({crop_right}, {crop_bottom})"
                )
                print(f"Crop size: {crop_right - crop_left} x {crop_bottom - crop_top}")

            # Extract the standardized face crop
            face_crop = original_image[crop_top:crop_bottom, crop_left:crop_right]

            # Resize to exact target size if needed
            if face_crop.shape[:2] != (target_face_size, target_face_size):
                face_crop = cv2.resize(face_crop, (target_face_size, target_face_size))

            return face_crop

        except Exception as e:
            if debug:
                print(f"OpenCV face centering failed: {e}")
            return None

    @staticmethod
    def _center_face_mediapipe(
        original_image: np.ndarray,
        target_face_size: int = 224,
        eye_y_ratio: float = 0.35,
        face_width_ratio: float = 0.8,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Center face using MediaPipe Face Mesh."""
        try:
            import mediapipe as mp

            mp_face_mesh = mp.solutions.face_mesh
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )

            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_image)

            if not results.multi_face_landmarks:
                if debug:
                    print("No face landmarks detected for centering with MediaPipe")
                return None

            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            h, w = original_image.shape[:2]

            # Get eye landmarks
            left_eye_outer = face_landmarks.landmark[33]  # Left eye outer corner
            left_eye_inner = face_landmarks.landmark[133]  # Left eye inner corner
            right_eye_inner = face_landmarks.landmark[362]  # Right eye inner corner
            right_eye_outer = face_landmarks.landmark[398]  # Right eye outer corner

            # Calculate eye centers and midpoint
            left_eye_center_x = int((left_eye_outer.x + left_eye_inner.x) * w / 2)
            left_eye_center_y = int((left_eye_outer.y + left_eye_inner.y) * h / 2)
            right_eye_center_x = int((right_eye_inner.x + right_eye_outer.x) * w / 2)
            right_eye_center_y = int((right_eye_inner.y + right_eye_outer.y) * h / 2)

            # Midpoint between eyes
            eye_center_x = (left_eye_center_x + right_eye_center_x) // 2
            eye_center_y = (left_eye_center_y + right_eye_center_y) // 2

            # Calculate crop region to center the face
            crop_top = int(eye_center_y - (target_face_size * eye_y_ratio))
            crop_bottom = crop_top + target_face_size
            crop_left = int(eye_center_x - (target_face_size // 2))
            crop_right = crop_left + target_face_size

            # Ensure crop region is within image bounds
            img_height, img_width = original_image.shape[:2]

            if crop_top < 0:
                offset = -crop_top
                crop_top = 0
                crop_bottom = min(img_height, crop_bottom + offset)
            elif crop_bottom > img_height:
                offset = crop_bottom - img_height
                crop_bottom = img_height
                crop_top = max(0, crop_top - offset)

            if crop_left < 0:
                offset = -crop_left
                crop_left = 0
                crop_right = min(img_width, crop_right + offset)
            elif crop_right > img_width:
                offset = crop_right - img_width
                crop_right = img_width
                crop_left = max(0, crop_left - offset)

            if debug:
                print(f"Eye center: ({eye_center_x}, {eye_center_y})")
                print(
                    f"Crop region: ({crop_left}, {crop_top}) to ({crop_right}, {crop_bottom})"
                )

            # Extract the standardized face crop
            face_crop = original_image[crop_top:crop_bottom, crop_left:crop_right]

            # Resize to exact target size if needed
            if face_crop.shape[:2] != (target_face_size, target_face_size):
                face_crop = cv2.resize(face_crop, (target_face_size, target_face_size))

            return face_crop

        except ImportError:
            if debug:
                print("MediaPipe not available for face centering")
            return None
        except Exception as e:
            if debug:
                print(f"MediaPipe face centering failed: {e}")
            return None

    @staticmethod
    def _center_face_yolo_pose(
        original_image: np.ndarray,
        yolo_model_path: str,
        target_face_size: int = 224,
        eye_y_ratio: float = 0.35,
        face_width_ratio: float = 0.8,
        confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Center face using YOLO pose model for bat face landmarks."""
        try:
            from ultralytics import YOLO

            # Load YOLO pose model
            yolo_model = YOLO(yolo_model_path)

            # Perform inference
            results = yolo_model.predict(
                source=original_image,
                conf=confidence_threshold,
                verbose=False,
                save=False,
            )

            if not results or len(results) == 0:
                if debug:
                    print("No detections from YOLO pose model for centering")
                return None

            result = results[0]

            # Check if we have keypoints and boxes
            if not hasattr(result, "keypoints") or result.keypoints is None:
                if debug:
                    print("No keypoints detected by YOLO pose model for centering")
                return None

            if (
                not hasattr(result, "boxes")
                or result.boxes is None
                or len(result.boxes) == 0
            ):
                if debug:
                    print("No face detections from YOLO pose model for centering")
                return None

            # Get keypoints from the first detection
            keypoints = result.keypoints.xy.cpu().numpy()

            if len(keypoints) == 0:
                if debug:
                    print("No keypoints found in detection for centering")
                return None

            # Get keypoints for the first detection
            kpts = keypoints[0]

            if len(kpts) < 2:
                if debug:
                    print("Not enough keypoints detected for centering")
                return None

            # Extract eye coordinates
            left_eye = kpts[0]  # [x, y] for left eye
            right_eye = kpts[1]  # [x, y] for right eye

            # Check if keypoints are valid
            if (
                np.any(np.isnan(left_eye))
                or np.any(np.isnan(right_eye))
                or np.allclose(left_eye, 0)
                or np.allclose(right_eye, 0)
            ):
                if debug:
                    print("Invalid eye keypoints detected for centering")
                return None

            # Calculate midpoint between eyes
            eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
            eye_center_y = int((left_eye[1] + right_eye[1]) / 2)

            # Calculate crop region to center the face
            crop_top = int(eye_center_y - (target_face_size * eye_y_ratio))
            crop_bottom = crop_top + target_face_size
            crop_left = int(eye_center_x - (target_face_size // 2))
            crop_right = crop_left + target_face_size

            # Ensure crop region is within image bounds
            img_height, img_width = original_image.shape[:2]

            if crop_top < 0:
                offset = -crop_top
                crop_top = 0
                crop_bottom = min(img_height, crop_bottom + offset)
            elif crop_bottom > img_height:
                offset = crop_bottom - img_height
                crop_bottom = img_height
                crop_top = max(0, crop_top - offset)

            if crop_left < 0:
                offset = -crop_left
                crop_left = 0
                crop_right = min(img_width, crop_right + offset)
            elif crop_right > img_width:
                offset = crop_right - img_width
                crop_right = img_width
                crop_left = max(0, crop_left - offset)

            if debug:
                print(f"YOLO Pose - Eye center: ({eye_center_x}, {eye_center_y})")
                print(
                    f"YOLO Pose - Crop region: ({crop_left}, {crop_top}) to ({crop_right}, {crop_bottom})"
                )

            # Extract the standardized face crop
            face_crop = original_image[crop_top:crop_bottom, crop_left:crop_right]

            # Resize to exact target size if needed
            if face_crop.shape[:2] != (target_face_size, target_face_size):
                face_crop = cv2.resize(face_crop, (target_face_size, target_face_size))

            return face_crop

        except ImportError:
            if debug:
                print("YOLO (ultralytics) not available for face centering")
            return None
        except Exception as e:
            if debug:
                print(f"YOLO pose face centering failed: {e}")
            return None

    @staticmethod
    def crop_face_box_with_margin(
        original_image: np.ndarray,
        margin_ratio: float = 0.2,
        make_square: bool = True,
        yolo_model_path: str = None,
        confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Crop around face bounding box using YOLO pose model with configurable margin.

        This method uses the YOLO pose model to detect the face bounding box,
        then crops around it with a configurable margin to include more context.

        Args:
            original_image: Input image array
            margin_ratio: Ratio of margin to add around the detected face box (0.2 = 20% on each side)
            make_square: If True, make the crop square by using the larger dimension
            yolo_model_path: Path to YOLO pose model (.pt file) - required
            confidence_threshold: Confidence threshold for YOLO pose detection
            debug: If True, print debug information

        Returns:
            Cropped image around face with margin, or None if no face detected
        """
        if yolo_model_path is None:
            if debug:
                print("YOLO model path is required for face box cropping")
            return None

        try:
            from ultralytics import YOLO
            import torch

            # Load YOLO pose model
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = YOLO(yolo_model_path)

            # Run inference
            results = model(original_image, device=device, verbose=False)

            # Extract detections
            detections = []
            for result in results:
                if hasattr(result, "boxes") and result.boxes is not None:
                    boxes = (
                        result.boxes.xyxy.cpu().numpy()
                    )  # [N, 4] format: x1, y1, x2, y2
                    confidences = result.boxes.conf.cpu().numpy()  # [N]

                    # Get keypoints if available
                    keypoints = None
                    if hasattr(result, "keypoints") and result.keypoints is not None:
                        keypoints = (
                            result.keypoints.xy.cpu().numpy()
                        )  # [N, num_keypoints, 2]

                    for i in range(len(boxes)):
                        if confidences[i] >= confidence_threshold:
                            detections.append(
                                {
                                    "box": boxes[i],  # [x1, y1, x2, y2]
                                    "confidence": confidences[i],
                                    "keypoints": (
                                        keypoints[i] if keypoints is not None else None
                                    ),
                                }
                            )

            if not detections:
                if debug:
                    print("No face detections found above confidence threshold")
                return None

            # Use the highest confidence detection
            best_detection = max(detections, key=lambda x: x["confidence"])
            face_box = best_detection["box"]  # [x1, y1, x2, y2]

            if debug:
                print(f"Face detection confidence: {best_detection['confidence']:.3f}")
                print(
                    f"Face box: [{face_box[0]:.0f}, {face_box[1]:.0f}, {face_box[2]:.0f}, {face_box[3]:.0f}]"
                )

            # Calculate face box dimensions
            x1, y1, x2, y2 = face_box
            face_width = x2 - x1
            face_height = y2 - y1

            # Calculate face center
            face_center_x = (x1 + x2) / 2
            face_center_y = (y1 + y2) / 2

            if make_square:
                # For square: use max edge of original box as base, apply margin uniformly
                max_edge = max(face_width, face_height)

                # Calculate final square size (0 margin = perfect square using max edge)
                square_size = max_edge * (
                    1 + 2 * margin_ratio
                )  # margin applied to both sides
                half_square = square_size / 2

                # Check if target square fits in image, if not reduce the size
                max_possible_width = original_image.shape[1]  # image width
                max_possible_height = original_image.shape[0]  # image height
                max_possible_square = min(max_possible_width, max_possible_height)

                # If target square is larger than what fits, reduce it
                if square_size > max_possible_square:
                    square_size = max_possible_square
                    half_square = square_size / 2

                # Center the square around face center
                square_x1 = face_center_x - half_square
                square_y1 = face_center_y - half_square
                square_x2 = face_center_x + half_square
                square_y2 = face_center_y + half_square

                # Adjust position if we go outside image boundaries (maintaining square size)
                if square_x1 < 0:
                    # Shift right
                    shift = -square_x1
                    square_x1 = 0
                    square_x2 = square_size
                elif square_x2 > original_image.shape[1]:
                    # Shift left
                    shift = square_x2 - original_image.shape[1]
                    square_x2 = original_image.shape[1]
                    square_x1 = original_image.shape[1] - square_size

                if square_y1 < 0:
                    # Shift down
                    shift = -square_y1
                    square_y1 = 0
                    square_y2 = square_size
                elif square_y2 > original_image.shape[0]:
                    # Shift up
                    shift = square_y2 - original_image.shape[0]
                    square_y2 = original_image.shape[0]
                    square_y1 = original_image.shape[0] - square_size

                crop_x1, crop_y1, crop_x2, crop_y2 = (
                    square_x1,
                    square_y1,
                    square_x2,
                    square_y2,
                )
            else:
                # For rectangle: apply margin separately to width and height
                margin_x = face_width * margin_ratio
                margin_y = face_height * margin_ratio

                # Expand the box with margin
                crop_x1 = max(0, x1 - margin_x)
                crop_y1 = max(0, y1 - margin_y)
                crop_x2 = min(original_image.shape[1], x2 + margin_x)
                crop_y2 = min(original_image.shape[0], y2 + margin_y)

            # Convert to integers
            crop_x1, crop_y1, crop_x2, crop_y2 = map(
                int, [crop_x1, crop_y1, crop_x2, crop_y2]
            )

            if debug:
                original_box_size = f"{face_width:.0f}x{face_height:.0f}"
                crop_size = f"{crop_x2-crop_x1}x{crop_y2-crop_y1}"
                print(f"Original face box: {original_box_size}")

                if make_square:
                    max_edge = max(face_width, face_height)
                    original_square_size = max_edge * (1 + 2 * margin_ratio)
                    margin_pixels = max_edge * margin_ratio
                    actual_square_size = crop_x2 - crop_x1  # The actual size used
                    print(f"Max edge: {max_edge:.0f}px")
                    print(
                        f"Margin ratio: {margin_ratio} ({margin_pixels:.0f}px on each side)"
                    )
                    print(
                        f"Target square size: {original_square_size:.0f}x{original_square_size:.0f}"
                    )
                    if actual_square_size != original_square_size:
                        print(
                            f"Reduced to fit image: {actual_square_size:.0f}x{actual_square_size:.0f}"
                        )
                    else:
                        print(
                            f"Actual square size: {actual_square_size:.0f}x{actual_square_size:.0f}"
                        )
                else:
                    margin_x = face_width * margin_ratio
                    margin_y = face_height * margin_ratio
                    print(
                        f"Margin ratio: {margin_ratio} ({margin_x:.0f}px x {margin_y:.0f}px)"
                    )

                print(
                    f"Final crop: {crop_size} at [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}]"
                )
                print(f"Square crop: {make_square}")

            # Crop the image
            cropped_image = original_image[crop_y1:crop_y2, crop_x1:crop_x2]

            if cropped_image.size == 0:
                if debug:
                    print("Cropped image is empty")
                return None

            return cropped_image

        except Exception as e:
            if debug:
                print(f"Face box cropping failed: {e}")
            return None

    @staticmethod
    def crop_square_around_segmentation_mask(
        original_image: np.ndarray,
        segmentation_mask: np.ndarray,
        margin_ratio: float = 0.2,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Crop a square around segmentation mask ensuring all mask points are included.

        This method finds the bounding box of the segmentation mask and creates a square
        crop that includes all mask points with optional margin.

        Args:
            original_image: Input image array
            segmentation_mask: Binary segmentation mask (0s and 1s or 0s and 255s)
            margin_ratio: Ratio of margin to add around the mask bounding box (0.2 = 20% on each side)
            debug: If True, print debug information

        Returns:
            Cropped square image that includes all mask points, or None if no mask found
        """
        try:
            # Ensure mask is binary
            if segmentation_mask.max() > 1:
                binary_mask = (segmentation_mask > 127).astype(np.uint8)
            else:
                binary_mask = segmentation_mask.astype(np.uint8)

            # Find all non-zero (mask) points
            mask_points = np.where(binary_mask > 0)

            if len(mask_points[0]) == 0:
                if debug:
                    print("No mask points found")
                return None

            # Get bounding box of mask
            min_y, max_y = mask_points[0].min(), mask_points[0].max()
            min_x, max_x = mask_points[1].min(), mask_points[1].max()

            # Calculate mask dimensions
            mask_width = max_x - min_x + 1
            mask_height = max_y - min_y + 1
            mask_center_x = (min_x + max_x) / 2
            mask_center_y = (min_y + max_y) / 2

            if debug:
                print(f"Mask bounding box: [{min_x}, {min_y}, {max_x}, {max_y}]")
                print(f"Mask dimensions: {mask_width}x{mask_height}")
                print(f"Mask center: ({mask_center_x:.1f}, {mask_center_y:.1f})")

            # Calculate square size based on max dimension + margin
            max_mask_dim = max(mask_width, mask_height)
            square_size = max_mask_dim * (1 + 2 * margin_ratio)
            half_square = square_size / 2

            if debug:
                print(f"Max mask dimension: {max_mask_dim}")
                print(
                    f"Margin ratio: {margin_ratio} ({max_mask_dim * margin_ratio:.0f}px on each side)"
                )
                print(f"Target square size: {square_size:.0f}x{square_size:.0f}")

            # Check if target square fits in image
            max_possible_width = original_image.shape[1]
            max_possible_height = original_image.shape[0]
            max_possible_square = min(max_possible_width, max_possible_height)

            # If target square is larger than what fits, reduce it
            original_square_size = square_size
            if square_size > max_possible_square:
                square_size = max_possible_square
                half_square = square_size / 2
                if debug:
                    print(
                        f"Reduced square size to fit image: {square_size:.0f}x{square_size:.0f}"
                    )

            # Center the square around mask center
            square_x1 = mask_center_x - half_square
            square_y1 = mask_center_y - half_square
            square_x2 = mask_center_x + half_square
            square_y2 = mask_center_y + half_square

            # Adjust position if we go outside image boundaries (maintaining square size)
            if square_x1 < 0:
                # Shift right
                shift = -square_x1
                square_x1 = 0
                square_x2 = square_size
            elif square_x2 > original_image.shape[1]:
                # Shift left
                shift = square_x2 - original_image.shape[1]
                square_x2 = original_image.shape[1]
                square_x1 = original_image.shape[1] - square_size

            if square_y1 < 0:
                # Shift down
                shift = -square_y1
                square_y1 = 0
                square_y2 = square_size
            elif square_y2 > original_image.shape[0]:
                # Shift up
                shift = square_y2 - original_image.shape[0]
                square_y2 = original_image.shape[0]
                square_y1 = original_image.shape[0] - square_size

            # Convert to integers
            crop_x1, crop_y1, crop_x2, crop_y2 = map(
                int, [square_x1, square_y1, square_x2, square_y2]
            )

            if debug:
                final_size = f"{crop_x2-crop_x1}x{crop_y2-crop_y1}"
                print(
                    f"Final crop: {final_size} at [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}]"
                )

                # Verify all mask points are included
                mask_included = (
                    crop_x1 <= min_x
                    and crop_x2 >= max_x
                    and crop_y1 <= min_y
                    and crop_y2 >= max_y
                )
                print(f"All mask points included: {mask_included}")

            # Crop the image
            cropped_image = original_image[crop_y1:crop_y2, crop_x1:crop_x2]

            if cropped_image.size == 0:
                if debug:
                    print("Cropped image is empty")
                return None

            return cropped_image

        except Exception as e:
            if debug:
                print(f"Segmentation mask cropping failed: {e}")
            return None

    @staticmethod
    def resize_square_image(
        original_image: np.ndarray,
        target_size: int,
        interpolation: str = "bilinear",
    ) -> np.ndarray:
        """Resize a square image to specified dimension.

        Args:
            original_image: Input image array (should be square)
            target_size: Target size for both width and height
            interpolation: Interpolation method ('bilinear', 'nearest', 'cubic', 'area')

        Returns:
            Resized square image array
        """
        return ImageTransforms.resize_image(
            original_image,
            target_size=(target_size, target_size),
            interpolation=interpolation,
        )
