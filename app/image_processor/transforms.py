"""
Image processing transforms module.

This module contains all image processing functions as static methods.
Functions are organized into:
- Mask-based transforms: Require segmentation masks from models
- Plain transforms: Work with images directly without models
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Union, Any
import os


class ImageTransforms:
    """Static methods for all image processing transforms."""

    # =========================================================================
    # MASK-BASED TRANSFORMS (Require prediction masks)
    # =========================================================================

    @staticmethod
    def _get_model_from_config(model_type: str):
        """Get model configuration and create model instance using factory pattern.
        
        Args:
            model_type: Type of model to create ("segmentation" or "pose")
            
        Returns:
            Model instance created by factory, or None if creation fails
        """
        try:
            from config.loader import load_config
            from models.yolo_factory import create_yolo_model
            
            config = load_config()
            
            if model_type == "segmentation":
                model_config = config.models.segmentation
            elif model_type == "pose":
                model_config = config.models.pose
            else:
                raise ValueError(f"Invalid model type: {model_type}")
            
            if not model_config:
                print(f"⚠️  No {model_type} model configuration found")
                return None
            
            # Create model using factory (heavy import happens here)
            model = create_yolo_model(model_config)
            return model
            
        except Exception as e:
            print(f"⚠️  Error creating model: {e}")
            return None

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
        # Ensure mask matches image size and is binary 0/1
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        if mask.shape[:2] != original_image.shape[:2]:
            import cv2
            mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.uint8)
        # Broadcast to 3 channels
        if mask.ndim == 2:
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
        foreground = mask_3d > 0
        result[foreground] = original_image[foreground]

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

        # Ensure mask is 3D for broadcasting and binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        # Keep original foreground, use blurred background
        result = blurred.copy()
        foreground = mask_3d > 0
        result[foreground] = original_image[foreground]

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

        # Ensure mask is 3D for broadcasting and binary
        if mask.max() > 1:
            mask = (mask > 127).astype(np.uint8)
        if len(mask.shape) == 2:
            mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        else:
            mask_3d = mask

        # Keep original foreground, use color background
        result = background.copy()
        foreground = mask_3d > 0
        result[foreground] = original_image[foreground]

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
        debug: bool = False,
        debug_dir: Optional[str] = None,
        base_filename: Optional[str] = None
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

        # Debug output if enabled
        if debug and debug_dir and base_filename:
            try:
                import cv2
                os.makedirs(debug_dir, exist_ok=True)
                
                print(f"🔍 Final image shape: {normalized.shape}, dtype: {normalized.dtype}")
                print(f"🔍 Value range: [{normalized.min():.3f}, {normalized.max():.3f}]")
                
                # Save final normalized image (convert back to uint8 for saving)
                final_image_uint8 = (normalized * 255).astype(np.uint8)
                debug_path = os.path.join(debug_dir, f"{base_filename}_step7_final_normalized.jpg")
                cv2.imwrite(debug_path, final_image_uint8)
                print(f"🔍 Saved debug final image: {debug_path}")
            except Exception as e:
                print(f"⚠️ Debug output failed: {e}")

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
    def segment_image(
        original_image: np.ndarray,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        base_filename: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Segment image using YOLO model created via factory pattern."""
        # Get model instance using factory (heavy work happens here, not during import)
        model = ImageTransforms._get_model_from_config("segmentation")
        if model is None:
            return None
        
        # Get confidence threshold from config
        try:
            from config.loader import load_config
            config = load_config()
            confidence_threshold = config.models.segmentation.confidence_threshold
        except Exception:
            confidence_threshold = 0.3

        try:
            results = model(original_image, conf=confidence_threshold)
            if not results or len(results) == 0:
                return None
            result = results[0]
            if result.masks is None or len(result.masks) == 0:
                return None
            masks = result.masks.data.cpu().numpy()  # shape [N, h, w] at model scale
            areas = [np.sum(m) for m in masks]
            mask = masks[int(np.argmax(areas))].astype(np.uint8)  # 0/1
            # Resize mask to original image size
            orig_h, orig_w = original_image.shape[:2]
            if mask.shape[0] != orig_h or mask.shape[1] != orig_w:
                import cv2
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
                mask = (mask > 0).astype(np.uint8)
            
            # Debug output if enabled
            if debug and debug_dir and base_filename:
                try:
                    import cv2
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    # Save mask visualization
                    mask_overlay = original_image.copy()
                    mask_colored = np.zeros_like(original_image)
                    mask_colored[:, :, 1] = mask * 255  # Green channel
                    mask_overlay = cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0)
                    
                    debug_path = os.path.join(debug_dir, f"{base_filename}_step2_mask_overlay.jpg")
                    cv2.imwrite(debug_path, mask_overlay)
                    print(f"🔍 Saved debug mask overlay: {debug_path}")
                    
                    # Save pure mask
                    debug_path = os.path.join(debug_dir, f"{base_filename}_step2_mask.jpg")
                    cv2.imwrite(debug_path, mask * 255)
                    print(f"🔍 Saved debug mask: {debug_path}")
                    
                    print(f"🔍 Segmentation mask shape: {mask.shape}")
                except Exception as e:
                    print(f"⚠️ Debug output failed: {e}")
            
            return mask
            
        except Exception:
            return None

    @staticmethod
    def crop_square_around_segmentation(
        original_image: np.ndarray,
        margin_ratio: float = 0.5,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        mask = ImageTransforms.segment_image(original_image)
        if mask is None:
            if debug:
                print("Segmentation failed for crop_square_around_segmentation")
            return None
        return ImageTransforms.crop_square_around_segmentation_mask(
            original_image, mask, margin_ratio, debug
        )

    @staticmethod
    def apply_background_replacement_auto(
        original_image: np.ndarray,
        background_source: callable,
        debug: bool = False,
        **generator_kwargs,
    ) -> Optional[np.ndarray]:
        mask = ImageTransforms.segment_image(original_image)
        if mask is None:
            if debug:
                print("Segmentation failed for background replacement")
            return None
        result = ImageTransforms.apply_background_replacement(
            original_image, mask, background_source, **generator_kwargs
        )
        if debug:
            try:
                import cv2
                contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, (0, 0, 255), 3)
            except Exception:
                pass
        return result

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
        debug: bool = False,
        debug_dir: Optional[str] = None,
        base_filename: Optional[str] = None
    ) -> Optional[np.ndarray]:
        try:
            if face_detector_type == "opencv":
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
                # Get model instance using factory pattern
                model = ImageTransforms._get_model_from_config("pose")
                if model is None:
                    raise ValueError(
                        "Pose model not found in config for yolo_pose face detection"
                    )
                try:
                    from config.loader import load_config
                    cfg = load_config()
                    confidence_threshold = cfg.models.pose.confidence_threshold
                except Exception:
                    confidence_threshold = 0.3
                
                result = ImageTransforms._align_face_yolo_pose(
                    original_image,
                    model,
                    target_eye_distance,
                    confidence_threshold,
                    debug,
                )
                
                # Debug output if enabled
                if debug and debug_dir and base_filename:
                    try:
                        import cv2
                        os.makedirs(debug_dir, exist_ok=True)
                        
                        # Save original image (step 0)
                        debug_path = os.path.join(debug_dir, f"{base_filename}_step0_original.jpg")
                        cv2.imwrite(debug_path, original_image)
                        print(f"🔍 Saved debug image: {debug_path}")
                        print(f"🔍 Original image shape: {original_image.shape}")
                        
                        if result is not None:
                            print("🔍 Face alignment successful!")
                            
                            # Save aligned image
                            debug_path = os.path.join(debug_dir, f"{base_filename}_step1_aligned.jpg")
                            cv2.imwrite(debug_path, result)
                            print(f"🔍 Saved debug aligned image: {debug_path}")
                        else:
                            print("⚠️ Face alignment failed, continuing with original image...")
                            
                            # Save original image as step1 result since alignment failed
                            debug_path = os.path.join(debug_dir, f"{base_filename}_step1_alignment_failed_using_original.jpg")
                            cv2.imwrite(debug_path, original_image)
                            print(f"🔍 Saved debug original image (alignment failed): {debug_path}")
                    except Exception as e:
                        print(f"⚠️ Debug output failed: {e}")
                
                return result
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
        yolo_model: Any,
        target_eye_distance: Optional[float] = None,
        confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Align face using trained YOLO pose model for bat face landmarks."""
        try:
            # Model instance is already created by factory

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

            # Prefer keypoints with confidence and choose leftmost/rightmost as eyes
            kpts_with_conf = result.keypoints.data.cpu().numpy()  # [N, K, 3] -> x,y,conf
            if len(kpts_with_conf) == 0:
                if debug:
                    print("No keypoints found in detection")
                return None
            kpts3 = kpts_with_conf[0]  # [K,3]
            valid = kpts3[kpts3[:, 2] >= confidence_threshold]
            if len(valid) < 2:
                if debug:
                    print("Not enough confident keypoints for alignment")
                return None
            # Choose leftmost and rightmost points as eye proxies
            left_idx = int(np.argmin(valid[:, 0]))
            right_idx = int(np.argmax(valid[:, 0]))
            left_eye = valid[left_idx][:2]
            right_eye = valid[right_idx][:2]

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

            # Optionally draw keypoints before rotation so they rotate with the face
            to_rotate = original_image
            if debug:
                try:
                    annotated = original_image.copy()
                    cv2.circle(annotated, left_eye_center, 5, (0, 0, 255), -1)
                    cv2.circle(annotated, right_eye_center, 5, (255, 0, 0), -1)
                    cv2.line(annotated, left_eye_center, right_eye_center, (0, 255, 255), 3)
                    to_rotate = annotated
                except Exception:
                    to_rotate = original_image

            # Apply rotation
            aligned_image = cv2.warpAffine(
                to_rotate,
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
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Center face landmarks and resize to target size.

        This method detects facial landmarks and centers them in the image,
        then resizes to the target face size while maintaining proportions.

        Args:
            original_image: Input image array
            target_face_size: Target size for the face (width and height)
            eye_y_ratio: Ratio of eye Y position from top (0.0-1.0)
            face_width_ratio: Ratio of face width to image width (0.0-1.0)
            face_detector_type: Type of face detector ('opencv', 'mediapipe', 'yolo_pose')
            confidence_threshold: Confidence threshold for YOLO pose detection
            debug: If True, return debug information

        Returns:
            Centered and resized image array, or None if centering fails

        Note:
            For 'yolo_pose' mode, a trained YOLO pose model is required that can detect
            bat face keypoints (left_eye, right_eye, nose).
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
                # Get model instance using factory pattern
                model = ImageTransforms._get_model_from_config("pose")
                if model is None:
                    raise ValueError(
                        "Pose model not found in config for yolo_pose face detection"
                    )
                try:
                    from config.loader import load_config
                    cfg = load_config()
                    confidence_threshold = cfg.models.pose.confidence_threshold
                except Exception:
                    confidence_threshold = 0.3
                
                return ImageTransforms._center_face_yolo_pose(
                    original_image,
                    model,
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
        yolo_model: Any,
        target_face_size: int = 224,
        eye_y_ratio: float = 0.35,
        face_width_ratio: float = 0.8,
        confidence_threshold: float = 0.3,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Center face using YOLO pose model for bat face landmarks."""
        try:
            # Model instance is already created by factory

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

            # Optionally draw eye keypoints
            if debug:
                try:
                    # Draw all confident keypoints
                    kpts3 = result.keypoints.data.cpu().numpy()[0]
                    for pt in kpts3:
                        if pt[2] >= confidence_threshold:
                            cv2.circle(original_image, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                    # Emphasize eyes and connecting line
                    cv2.circle(original_image, (int(left_eye[0]), int(left_eye[1])), 5, (0, 0, 255), -1)
                    cv2.circle(original_image, (int(right_eye[0]), int(right_eye[1])), 5, (255, 0, 0), -1)
                    cv2.line(original_image, (int(left_eye[0]), int(left_eye[1])), (int(right_eye[0]), int(right_eye[1])), (0, 255, 255), 3)
                except Exception:
                    pass

            # Calculate midpoint between eyes
            eye_center_x = int((left_eye[0] + right_eye[0]) / 2)
            eye_center_y = int((left_eye[1] + right_eye[1]) / 2)
            # No debug prints here

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


        except Exception as e:
            if debug:
                print(f"YOLO pose face centering failed: {e}")
            return None

    @staticmethod
    def crop_face_box_with_margin(
        original_image: np.ndarray,
        margin_ratio: float = 0.2,
        make_square: bool = True,
        debug: bool = False,
    ) -> Optional[np.ndarray]:
        """Crop image around detected face with margin.

        This method detects facial landmarks and crops the image around the face
        with an optional margin and square aspect ratio.

        Args:
            original_image: Input image array
            margin_ratio: Ratio of margin to add around face (0.0-1.0)
            make_square: If True, crop to square aspect ratio
            yolo_model_path: Path to YOLO pose model (.pt file)
            confidence_threshold: Confidence threshold for YOLO pose detection
            debug: If True, return debug information

        Returns:
            Cropped image array, or None if cropping fails
        """
        try:
            # Get model instance using factory pattern
            model = ImageTransforms._get_model_from_config("pose")
            if model is None:
                raise ValueError(
                    "Pose model not found in config for face detection"
                )
            try:
                from config.loader import load_config
                cfg = load_config()
                confidence_threshold = cfg.models.pose.confidence_threshold
            except Exception:
                confidence_threshold = 0.3
            
            results = model(original_image, conf=confidence_threshold)
            
            if not results or len(results) == 0:
                if debug:
                    print("No face detected")
                return None
            
            result = results[0]
            
            if result.keypoints is None or len(result.keypoints) == 0:
                if debug:
                    print("No keypoints detected")
                return None
            
            keypoints = result.keypoints.data.cpu().numpy()[0]
            
            valid_keypoints = keypoints[keypoints[:, 2] > confidence_threshold]
            
            if len(valid_keypoints) < 3:
                if debug:
                    print(f"Insufficient keypoints detected: {len(valid_keypoints)}")
                return None
            
            x_coords = valid_keypoints[:, 0]
            y_coords = valid_keypoints[:, 1]
            
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            
            width = x_max - x_min
            height = y_max - y_min
            margin_x = width * margin_ratio
            margin_y = height * margin_ratio
            
            x_min = max(0, x_min - margin_x)
            x_max = min(original_image.shape[1], x_max + margin_x)
            y_min = max(0, y_min - margin_y)
            y_max = min(original_image.shape[0], y_max + margin_y)
            
            cropped = original_image[int(y_min):int(y_max), int(x_min):int(x_max)]
            
            if make_square:
                height, width = cropped.shape[:2]
                size = max(height, width)
                
                square = np.zeros((size, size, 3), dtype=cropped.dtype)
                
                y_offset = (size - height) // 2
                x_offset = (size - width) // 2
                
                square[y_offset:y_offset + height, x_offset:x_offset + width] = cropped
                cropped = square
            
            if debug:
                print(f"Face crop: {cropped.shape}")
            
            return cropped
            
        except Exception as e:
            if debug:
                print(f"Face cropping failed: {e}")
            return None

    @staticmethod
    def crop_square_around_segmentation_mask(
        original_image: np.ndarray,
        segmentation_mask: np.ndarray,
        margin_ratio: float = 0.2,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        base_filename: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Crop and center image using segmentation mask and pose keypoints.

        Creates a dummy image of target size, extracts max possible crop from input,
        then centers the crop using the pose model's keypoint triangle center.

        Args:
            original_image: Input image array
            segmentation_mask: Binary segmentation mask (0s and 1s or 0s and 255s)
            margin_ratio: Ratio of margin to add around the mask bounding box
            debug: If True, print debug information

        Returns:
            Centered square image of target size, or None if processing fails
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
 
            # Calculate mask dimensions and center
            mask_width = max_x - min_x + 1
            mask_height = max_y - min_y + 1
            mask_center_x = (min_x + max_x) / 2
            mask_center_y = (min_y + max_y) / 2
 
            if debug:
                print(f"Mask bounding box: [{min_x}, {min_y}, {max_x}, {max_y}]")
                print(f"Mask dimensions: {mask_width}x{mask_height}")
                print(f"Mask center: ({mask_center_x:.1f}, {mask_center_y:.1f})")
 
            # Calculate target square size with margin
            max_mask_dim = max(mask_width, mask_height)
            square_size = max_mask_dim * (1 + 2 * margin_ratio)
             
            # Use computed size directly
            target_size = int(square_size)
             
            if debug:
                print(f"Target output size: {target_size}x{target_size}")
 
            # Create dummy image with neutral background
            dummy_image = np.full((target_size, target_size, 3), 128, dtype=np.uint8)  # Gray background
 
            # Calculate crop coordinates (max possible within image bounds)
            margin_pixels = int(max_mask_dim * margin_ratio)
            crop_x1 = max(0, int(mask_center_x - max_mask_dim//2 - margin_pixels))
            crop_y1 = max(0, int(mask_center_y - max_mask_dim//2 - margin_pixels))
            crop_x2 = min(original_image.shape[1], crop_x1 + int(square_size))
            crop_y2 = min(original_image.shape[0], crop_y1 + int(square_size))
             
            # Extract the crop from original image
            crop = original_image[crop_y1:crop_y2, crop_x1:crop_x2]
             
            if debug:
                print(f"Crop region: [{crop_x1}, {crop_y1}, {crop_x2}, {crop_y2}]")
                print(f"Crop size: {crop.shape}")
 
            # Get pose keypoints to determine centering using factory pattern
            try:
                from config.loader import load_config
                cfg = load_config()
                confidence_threshold = cfg.models.pose.confidence_threshold
                 
                # Get model instance using factory pattern
                model = ImageTransforms._get_model_from_config("pose")
                if model is not None:
                    results = model.predict(source=crop, conf=confidence_threshold, verbose=False, save=False)
                     
                    if results and len(results) > 0:
                        result = results[0]
                        if hasattr(result, "keypoints") and result.keypoints is not None:
                            kpts3 = result.keypoints.data.cpu().numpy()[0]  # [K,3]
                            valid = kpts3[kpts3[:, 2] >= confidence_threshold]
                             
                            if len(valid) >= 3:  # Need left eye, right eye, nose
                                # Calculate triangle center from 3 keypoints
                                triangle_center_x = np.mean(valid[:3, 0])
                                triangle_center_y = np.mean(valid[:3, 1])
                                 
                                if debug:
                                    print(f"Triangle center: ({triangle_center_x:.1f}, {triangle_center_y:.1f})")
                                 
                                # Calculate offset to center in dummy image
                                crop_height, crop_width = crop.shape[:2]
                                offset_x = (target_size - crop_width) // 2
                                offset_y = (target_size - crop_height) // 2
                                 
                                # Adjust offset based on triangle center relative to crop center
                                crop_center_x = crop_width // 2
                                crop_center_y = crop_height // 2
                                 
                                # Fine-tune positioning based on triangle center
                                triangle_offset_x = int(triangle_center_x - crop_center_x)
                                triangle_offset_y = int(triangle_center_y - crop_center_y)
                                 
                                final_offset_x = offset_x - triangle_offset_x
                                final_offset_y = offset_y - triangle_offset_y
                                 
                                # Ensure offsets keep crop within dummy bounds
                                final_offset_x = max(0, min(final_offset_x, target_size - crop_width))
                                final_offset_y = max(0, min(final_offset_y, target_size - crop_height))
                                 
                                if debug:
                                    print(f"Final offset: ({final_offset_x}, {final_offset_y})")
                                 
                                # Place crop in dummy image
                                dummy_image[final_offset_y:final_offset_y+crop_height, 
                                          final_offset_x:final_offset_x+crop_width] = crop
                                
                                # Debug output if enabled
                                if debug and debug_dir and base_filename:
                                    try:
                                        import cv2
                                        os.makedirs(debug_dir, exist_ok=True)
                                        
                                        debug_path = os.path.join(debug_dir, f"{base_filename}_step3_cropped.jpg")
                                        cv2.imwrite(debug_path, dummy_image)
                                        print(f"🔍 Saved debug cropped image: {debug_path}")
                                        print(f"🔍 Cropped image shape: {dummy_image.shape}")
                                    except Exception as e:
                                        print(f"⚠️ Debug output failed: {e}")
                                
                                return dummy_image
                            else:
                                if debug:
                                    print("Insufficient keypoints for triangle center")
                        else:
                            if debug:
                                print("No keypoints detected")
                    else:
                        if debug:
                            print("Pose model produced no results")
                else:
                    if debug:
                        print("Pose model not available")
            except Exception as e:
                if debug:
                    print(f"Pose processing error: {e}")
             
            # Fallback: center crop in dummy image without pose adjustment
            if debug:
                print("Using fallback centering")
             
            crop_height, crop_width = crop.shape[:2]
            offset_x = (target_size - crop_width) // 2
            offset_y = (target_size - crop_height) // 2
             
            dummy_image[offset_y:offset_y+crop_height, offset_x:offset_x+crop_width] = crop
            
            # Debug output if enabled
            if debug and debug_dir and base_filename:
                try:
                    import cv2
                    os.makedirs(debug_dir, exist_ok=True)
                    
                    debug_path = os.path.join(debug_dir, f"{base_filename}_step3_cropped.jpg")
                    cv2.imwrite(debug_path, dummy_image)
                    print(f"🔍 Saved debug cropped image: {debug_path}")
                    print(f"🔍 Cropped image shape: {dummy_image.shape}")
                except Exception as e:
                    print(f"⚠️ Debug output failed: {e}")
            
            return dummy_image
 
        except Exception as e:
            if debug:
                print(f"Segmentation mask cropping failed: {e}")
            return None

    @staticmethod
    def draw_all_predictions(
        original_image: np.ndarray,
        debug: bool = False
    ) -> Optional[np.ndarray]:
        """Draw segmentation and pose predictions on the image.
        
        This method runs both segmentation and pose models on the input image
        and draws their predictions with different colors and styles:
        - Segmentation masks: Green contours with mask indices
        - Pose bounding boxes: Red rectangles with detection indices  
        - Pose keypoints: Blue circles with keypoint indices
        - Keypoint connections: Yellow triangle connecting first 3 keypoints

        Args:
            original_image: Input image array (can be any size)
            debug: If True, print debug information

        Returns:
            Image with all predictions drawn, or None if processing fails
        """
        try:
            annotated_image = original_image.copy()
            original_height, original_width = original_image.shape[:2]
            
            if debug:
                print(f"🖼️  Processing image size: {original_width}x{original_height}")
            
            # Load config
            try:
                from config.loader import load_config
                cfg = load_config()
            except Exception as e:
                if debug:
                    print(f"⚠️  Config loading failed: {e}")
                return annotated_image
            
            # Step 1: Run segmentation model using factory pattern
            seg_model = ImageTransforms._get_model_from_config("segmentation")
            seg_confidence = cfg.models.segmentation.confidence_threshold
            
            if seg_model is not None:
                try:
                    seg_results = seg_model.predict(
                        source=original_image, 
                        conf=seg_confidence, 
                        verbose=False, 
                        save=False
                        # Don't specify imgsz - let YOLO handle it automatically
                    )
                    
                    if seg_results and len(seg_results) > 0:
                        result = seg_results[0]
                        
                        # Get the original image shape that YOLO processed
                        yolo_orig_shape = result.orig_shape  # (height, width)
                        yolo_img_shape = result.masks.data.shape[-2:] if hasattr(result, "masks") and result.masks is not None else None
                        
                        if debug:
                            print(f"🎯 YOLO original shape: {yolo_orig_shape}")
                            print(f"🎯 Current image shape: ({original_height}, {original_width})")
                            if yolo_img_shape:
                                print(f"🎯 YOLO mask shape: {yolo_img_shape}")
                        
                        if hasattr(result, "masks") and result.masks is not None and len(result.masks) > 0:
                            masks = result.masks.data.cpu().numpy()
                            if debug:
                                print(f"🎯 Segmentation: {len(masks)} masks found")
                            
                            # Draw each mask contour
                            for i, mask in enumerate(masks):
                                # Always resize mask to match original image dimensions
                                # YOLO masks are at model resolution, need to scale to original image
                                mask_resized = cv2.resize(
                                    mask.astype(np.float32), 
                                    (original_width, original_height), 
                                    interpolation=cv2.INTER_NEAREST
                                )
                                mask_uint8 = (mask_resized > 0.5).astype(np.uint8)
                                
                                # Find and draw contours
                                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                if contours:
                                    # Draw contours with adaptive thickness based on image size
                                    thickness = max(1, min(original_width, original_height) // 200)
                                    cv2.drawContours(annotated_image, contours, -1, (0, 255, 0), thickness)  # Green contours
                                    
                                    # Draw mask index at centroid
                                    largest_contour = max(contours, key=cv2.contourArea)
                                    M = cv2.moments(largest_contour)
                                    if M["m00"] != 0:
                                        cx = int(M["m10"] / M["m00"])
                                        cy = int(M["m01"] / M["m00"])
                                        
                                        # Adaptive font size and thickness
                                        font_scale = max(0.4, min(original_width, original_height) / 1000)
                                        font_thickness = max(1, int(font_scale * 2))
                                        
                                        cv2.putText(
                                            annotated_image, f"SEG{i}", 
                                            (cx - 15, cy), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            font_scale, (0, 255, 0), font_thickness
                                        )
                                        
                                        if debug:
                                            print(f"   Mask {i}: center=({cx}, {cy}), area={cv2.contourArea(largest_contour):.0f}")
                        else:
                            if debug:
                                print("🎯 Segmentation: No masks found")
                    else:
                        if debug:
                            print("🎯 Segmentation: No results")
                except Exception as e:
                    if debug:
                        print(f"❌ Segmentation error: {e}")
            else:
                if debug:
                    print("⚠️  Segmentation model not available or path invalid")
            
            # Step 2: Run pose model using factory pattern
            pose_model = ImageTransforms._get_model_from_config("pose")
            pose_confidence = cfg.models.pose.confidence_threshold
            
            if pose_model is not None:
                try:
                    pose_results = pose_model.predict(
                        source=original_image, 
                        conf=pose_confidence, 
                        verbose=False, 
                        save=False
                        # Don't specify imgsz - let YOLO handle it automatically
                    )
                    
                    if pose_results and len(pose_results) > 0:
                        result = pose_results[0]
                        
                        # Get YOLO's processed image dimensions for coordinate scaling
                        yolo_orig_shape = result.orig_shape  # (height, width) - original image size YOLO saw
                        
                        if debug:
                            print(f"📦 YOLO original shape: {yolo_orig_shape}")
                            print(f"📦 Current image shape: ({original_height}, {original_width})")
                        
                        # Modern YOLO versions return coordinates in original image space
                        # Only scale if there's actually a size mismatch
                        need_scaling = (yolo_orig_shape[1] != original_width or yolo_orig_shape[0] != original_height)
                        
                        if need_scaling:
                            scale_x = original_width / yolo_orig_shape[1]
                            scale_y = original_height / yolo_orig_shape[0]
                            if debug:
                                print(f"📦 Scaling needed: x={scale_x:.3f}, y={scale_y:.3f}")
                        else:
                            scale_x = scale_y = 1.0
                            if debug:
                                print(f"📦 No scaling needed - coordinates already in original image space")
                        
                        # Adaptive styling based on image size
                        box_thickness = max(1, min(original_width, original_height) // 300)
                        font_scale = max(0.4, min(original_width, original_height) / 1000)
                        font_thickness = max(1, int(font_scale * 2))
                        keypoint_radius = max(2, min(original_width, original_height) // 200)
                        
                        # Draw bounding boxes if available
                        if hasattr(result, "boxes") and result.boxes is not None and len(result.boxes) > 0:
                            boxes_xyxy = result.boxes.xyxy.cpu().numpy()
                            confidences = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else [1.0] * len(boxes_xyxy)
                            
                            if debug:
                                print(f"📦 Pose: {len(boxes_xyxy)} bounding boxes found")
                            
                            for i, ((x1, y1, x2, y2), conf) in enumerate(zip(boxes_xyxy, confidences)):
                                # Scale coordinates if needed
                                if need_scaling:
                                    x1_final = int(x1 * scale_x)
                                    y1_final = int(y1 * scale_y)
                                    x2_final = int(x2 * scale_x)
                                    y2_final = int(y2 * scale_y)
                                else:
                                    x1_final = int(x1)
                                    y1_final = int(y1)
                                    x2_final = int(x2)
                                    y2_final = int(y2)
                                
                                # Clamp coordinates to image bounds
                                x1_clamped = max(0, min(x1_final, original_width - 1))
                                y1_clamped = max(0, min(y1_final, original_height - 1))
                                x2_clamped = max(x1_clamped + 1, min(x2_final, original_width))
                                y2_clamped = max(y1_clamped + 1, min(y2_final, original_height))
                                
                                # Draw bounding box
                                cv2.rectangle(annotated_image, (x1_clamped, y1_clamped), (x2_clamped, y2_clamped), (0, 0, 255), box_thickness)  # Red boxes
                                
                                # Draw label with confidence
                                label = f"POSE{i} {conf:.2f}"
                                label_y = max(y1_clamped - 5, 15)  # Ensure label is visible
                                cv2.putText(
                                    annotated_image, label, 
                                    (x1_clamped, label_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_scale, (0, 0, 255), font_thickness
                                )
                                
                                if debug:
                                    if need_scaling:
                                        print(f"   Box {i}: orig=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}) -> scaled=({x1_clamped}, {y1_clamped}, {x2_clamped}, {y2_clamped}), conf={conf:.3f}")
                                    else:
                                        print(f"   Box {i}: ({x1_clamped}, {y1_clamped}, {x2_clamped}, {y2_clamped}), conf={conf:.3f}")
                        
                        # Draw keypoints if available
                        if hasattr(result, "keypoints") and result.keypoints is not None:
                            kpts3 = result.keypoints.data.cpu().numpy()  # [N, K, 3] where 3 = (x, y, confidence)
                            
                            if len(kpts3) > 0 and kpts3.shape[1] > 0:
                                if debug:
                                    print(f"🔵 Pose: {len(kpts3)} detections, {kpts3.shape[1]} keypoints each")
                                
                                for det_idx, detection in enumerate(kpts3):
                                    # Get all keypoints with their original indices
                                    all_keypoints = [(i, kpt) for i, kpt in enumerate(detection) if kpt[2] >= pose_confidence]
                                    
                                    if debug and len(all_keypoints) > 0:
                                        print(f"   Detection {det_idx}: {len(all_keypoints)} valid keypoints")
                                    
                                    # Draw each valid keypoint
                                    triangle_points = []
                                    for original_kpt_idx, (x, y, conf) in all_keypoints:
                                        # Validate coordinates
                                        if x < 0 or y < 0 or not np.isfinite(x) or not np.isfinite(y):
                                            if debug:
                                                print(f"     Skipping invalid keypoint {original_kpt_idx}: ({x}, {y})")
                                            continue
                                        
                                        # Scale coordinates if needed
                                        if need_scaling:
                                            x_final = int(x * scale_x)
                                            y_final = int(y * scale_y)
                                        else:
                                            x_final = int(x)
                                            y_final = int(y)
                                        
                                        # Clamp coordinates to image bounds
                                        x_clamped = max(0, min(x_final, original_width - 1))
                                        y_clamped = max(0, min(y_final, original_height - 1))
                                        
                                        # Draw keypoint circle
                                        cv2.circle(annotated_image, (x_clamped, y_clamped), keypoint_radius, (255, 0, 0), -1)  # Blue keypoints
                                        
                                        # Draw keypoint index (use original index, not enumeration index)
                                        cv2.putText(
                                            annotated_image, f"{original_kpt_idx}", 
                                            (x_clamped + keypoint_radius + 2, y_clamped - keypoint_radius - 2), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            font_scale * 0.7, (255, 0, 0), font_thickness
                                        )
                                        
                                        # Collect points for triangle (first 3 keypoints)
                                        if len(triangle_points) < 3:
                                            triangle_points.append([x_clamped, y_clamped])
                                        
                                        if debug:
                                            if need_scaling:
                                                print(f"     Keypoint {original_kpt_idx}: orig=({x:.1f}, {y:.1f}) -> scaled=({x_clamped}, {y_clamped}), conf={conf:.3f}")
                                            else:
                                                print(f"     Keypoint {original_kpt_idx}: ({x_clamped}, {y_clamped}), conf={conf:.3f}")
                                    
                                    # Draw triangle connecting first 3 keypoints if available
                                    if len(triangle_points) >= 3:
                                        triangle_points = np.array(triangle_points[:3], dtype=np.int32)
                                        triangle_thickness = max(1, box_thickness)
                                        cv2.polylines(annotated_image, [triangle_points], True, (255, 255, 0), triangle_thickness)  # Yellow triangle
                                        
                                        # Calculate and draw triangle center
                                        center_x = int(np.mean(triangle_points[:, 0]))
                                        center_y = int(np.mean(triangle_points[:, 1]))
                                        center_radius = max(3, keypoint_radius + 2)
                                        cv2.circle(annotated_image, (center_x, center_y), center_radius, (255, 255, 0), -1)  # Yellow center
                                        cv2.putText(
                                            annotated_image, "C", 
                                            (center_x + center_radius + 2, center_y + center_radius + 2), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            font_scale * 0.8, (255, 255, 0), font_thickness
                                        )
                                        
                                        if debug:
                                            print(f"   Triangle center: ({center_x}, {center_y})")
                            else:
                                if debug:
                                    print("🔵 Pose: No keypoints data available")
                        else:
                            if debug:
                                print("🔵 Pose: No keypoints found")
                    else:
                        if debug:
                            print("📦 Pose: No results")
                except Exception as e:
                    if debug:
                        print(f"❌ Pose error: {e}")
            else:
                if debug:
                    print("⚠️  Pose model not available or path invalid")
            
            if debug:
                print("✅ Prediction drawing completed")
            
            return annotated_image
            
        except Exception as e:
            if debug:
                print(f"❌ draw_all_predictions failed: {e}")
                import traceback
                traceback.print_exc()
            return None


    @staticmethod
    def resize_square_image(
        original_image: np.ndarray,
        target_size: int,
        interpolation: str = "bilinear",
        debug: bool = False,
        debug_dir: Optional[str] = None,
        base_filename: Optional[str] = None
    ) -> np.ndarray:
        """Resize a square image to specified dimension.

        Args:
            original_image: Input image array (should be square)
            target_size: Target size for both width and height
            interpolation: Interpolation method ('bilinear', 'nearest', 'cubic', 'area')

        Returns:
            Resized square image array
        """
        resized_image = ImageTransforms.resize_image(
            original_image,
            target_size=(target_size, target_size),
            interpolation=interpolation,
        )
        
        # Debug output if enabled
        if debug and debug_dir and base_filename:
            try:
                import cv2
                os.makedirs(debug_dir, exist_ok=True)
                
                debug_path = os.path.join(debug_dir, f"{base_filename}_step6_resized.jpg")
                cv2.imwrite(debug_path, resized_image)
                print(f"🔍 Saved debug resized image: {debug_path}")
                print(f"🔍 Resized image shape: {resized_image.shape}")
            except Exception as e:
                print(f"⚠️ Debug output failed: {e}")
        
        return resized_image
