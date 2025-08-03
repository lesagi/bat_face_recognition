import os
import random
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")  # Set the backend to non-interactive
import matplotlib.pyplot as plt
import numpy as np
from utils.image_utils import is_img_file, create_random_image
from image_processor.image_utils import preprocess_siamese_input
from scipy import ndimage


class SiameseModelSaliencyMapCreator:
    def __init__(
        self,
        model,
        input_dir_path,
        output_dir_path=None,
        nesting=None,
        sample_size=50,
        fast_mode=False,
        input_size=224,
        integration_steps=None,
        smoothing_samples=None,
    ):
        self.model = model
        self.sample_size = sample_size
        self.fast_mode = fast_mode
        self.input_size = input_size
        self.integration_steps = integration_steps
        self.smoothing_samples = smoothing_samples
        dir_name = os.path.basename(input_dir_path)
        if not output_dir_path:
            output_dir_path = input_dir_path
        if nesting:
            output_dir_path = os.path.join(output_dir_path, nesting)
        self.output_dir_path = output_dir_path
        self.output_file_path = os.path.join(
            self.output_dir_path, f"{dir_name}_saliency_map.pdf"
        )
        self.input_dir_path = (
            input_dir_path if not nesting else os.path.join(input_dir_path, nesting)
        )

        # Get all subdirectories
        self.subdirs = [
            d
            for d in os.listdir(self.input_dir_path)
            if os.path.isdir(os.path.join(self.input_dir_path, d))
        ]

        # Get images from each subdirectory
        self.images = []
        for subdir in self.subdirs:
            subdir_path = os.path.join(self.input_dir_path, subdir)
            subdir_images = [
                os.path.join(subdir, f)
                for f in os.listdir(subdir_path)
                if is_img_file(f)
            ]
            self.images.extend(subdir_images)

        self.num_images = len(self.images)

        if self.num_images == 0:
            raise ValueError(
                f"No images found in any subdirectory of: {self.input_dir_path}"
            )

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir_path, exist_ok=True)

        # Use the actual sample size for figure creation to avoid blank space
        self.actual_sample_size = min(self.sample_size, self.num_images)

        print(
            f"📏 Using input size: {self.input_size}x{self.input_size} for preprocessing"
        )

        # Show custom parameter settings
        if self.integration_steps is not None:
            print(f"🔧 Custom integration steps: {self.integration_steps}")
        if self.smoothing_samples is not None:
            print(f"🔧 Custom smoothing samples: {self.smoothing_samples}")

    def compute_integrated_gradients(self, image, counterpart, steps=None):
        """
        Compute Integrated Gradients for better attribution.
        This method provides smoother, more interpretable saliency maps.
        """
        if steps is None:
            if self.integration_steps is not None:
                steps = self.integration_steps
            else:
                steps = 10 if self.fast_mode else 20

        print(f"  Using {steps} integration steps")

        # Create a baseline (black image)
        baseline = tf.zeros_like(image)

        # Generate interpolated images between baseline and input
        alphas = tf.linspace(start=0.0, stop=1.0, num=steps + 1)

        # Compute gradients for each interpolated image
        gradients = []
        for i, alpha in enumerate(alphas):
            if i % 5 == 0:  # Progress indicator every 5 steps
                print(f"  Integrated gradients progress: {i+1}/{len(alphas)}")

            interpolated = baseline + alpha * (image - baseline)

            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                output = self.model([interpolated, counterpart])
            grad = tape.gradient(output, interpolated)
            gradients.append(grad)

        gradients = tf.stack(gradients)

        # Integrate gradients
        integrated_gradients = tf.reduce_mean(gradients, axis=0)
        integrated_gradients = integrated_gradients * (image - baseline)

        return integrated_gradients

    def compute_guided_gradients(self, image, counterpart):
        """
        Compute Guided Gradients for better localization.
        This method provides more precise attribution by guiding the gradients.
        """
        with tf.GradientTape() as tape:
            tape.watch(image)
            output = self.model([image, counterpart])

        gradients = tape.gradient(output, image)

        # Apply guided backpropagation: keep only positive gradients
        guided_gradients = tf.where(gradients > 0, gradients, tf.zeros_like(gradients))

        return guided_gradients

    def compute_smoothed_gradients(
        self, image, counterpart, noise_level=0.1, n_samples=None
    ):
        """
        Compute Smoothed Gradients to reduce noise and get cleaner saliency maps.
        """
        if n_samples is None:
            if self.smoothing_samples is not None:
                n_samples = self.smoothing_samples
            else:
                n_samples = 5 if self.fast_mode else 10

        print(f"  Using {n_samples} smoothing samples")

        gradients = []

        for i in range(n_samples):
            if i % 5 == 0:  # Progress indicator every 5 samples
                print(f"  Smoothed gradients progress: {i+1}/{n_samples}")
            # Add noise to the image
            noise = tf.random.normal(
                shape=tf.shape(image), mean=0.0, stddev=noise_level
            )
            noisy_image = image + noise
            # Ensure values are in valid range [0, 1]
            noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)

            with tf.GradientTape() as tape:
                tape.watch(noisy_image)
                output = self.model([noisy_image, counterpart])

            grad = tape.gradient(output, noisy_image)
            gradients.append(grad)

        # Average the gradients
        smoothed_gradients = tf.reduce_mean(tf.stack(gradients), axis=0)

        return smoothed_gradients

    def process_saliency_map(self, gradients, method="magnitude"):
        """
        Process gradients to create saliency maps with different aggregation methods.
        """
        if method == "magnitude":
            # L2 norm across channels (smoother than max)
            saliency = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=-1))
        elif method == "max":
            # Original method - max across channels
            saliency = tf.reduce_max(tf.abs(gradients), axis=-1)
        elif method == "mean":
            # Mean across channels
            saliency = tf.reduce_mean(tf.abs(gradients), axis=-1)
        elif method == "sum":
            # Sum across channels
            saliency = tf.reduce_sum(tf.abs(gradients), axis=-1)
        else:
            raise ValueError(f"Unknown method: {method}")

        return saliency

    def apply_gaussian_smoothing(self, saliency_map, sigma=1.0):
        """
        Apply Gaussian smoothing to the saliency map for better visualization.
        """
        # Convert to numpy for scipy operations
        saliency_np = saliency_map.numpy()

        # Apply Gaussian filter
        smoothed = ndimage.gaussian_filter(saliency_np, sigma=sigma)

        return tf.convert_to_tensor(smoothed)

    def normalize_saliency_map(self, saliency_map):
        """
        Normalize saliency map to [0, 1] range for better visualization.
        """
        min_val = tf.reduce_min(saliency_map)
        max_val = tf.reduce_max(saliency_map)

        if max_val > min_val:
            normalized = (saliency_map - min_val) / (max_val - min_val)
        else:
            normalized = tf.zeros_like(saliency_map)

        return normalized

    def compute_saliency_map(self):
        """
        Original saliency map computation - kept for backward compatibility.
        """
        # choose randomly sample_size images to do the saliency maps on from self.images
        saliency_maps_images = random.sample(self.images, self.actual_sample_size)

        self.fig, self.axes = plt.subplots(
            self.actual_sample_size, 2, figsize=(8.27, self.actual_sample_size * 5)
        )  # A4 size

        # Handle single row case - ensure axes is always 2D
        if self.actual_sample_size == 1:
            self.axes = self.axes.reshape(1, -1)

        plt.subplots_adjust(hspace=0.5)  # adjust space between rows

        for idx, img_path in enumerate(saliency_maps_images):
            full_img_path = os.path.join(self.input_dir_path, img_path)
            print(full_img_path)
            ax1 = self.axes[idx, 0]
            ax2 = self.axes[idx, 1]

            img = preprocess_siamese_input(
                full_img_path, target_size=(self.input_size, self.input_size)
            )
            anchor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)

            random_counterpart = create_random_image(img)
            counterpart = tf.convert_to_tensor(
                np.expand_dims(random_counterpart, axis=0), dtype=tf.float32
            )

            with tf.GradientTape() as tape:
                tape.watch(anchor)
                output = self.model([anchor, counterpart])
            gradients = tape.gradient(output, anchor)

            saliency_map = tf.reduce_max(tf.abs(gradients), axis=-1).numpy()

            img = tf.image.resize(img, (350, 350))
            ax1.imshow(img)
            ax1.axis("off")
            ax1.set_title(img_path)  # Show the relative path as title

            saliency_map_image = tf.image.resize(
                tf.expand_dims(saliency_map[0], axis=-1), (350, 350)
            )
            ax2.imshow(saliency_map_image, cmap="hot")
            ax2.axis("off")

        os.makedirs(self.output_dir_path, exist_ok=True)
        plt.savefig(self.output_file_path, format="pdf")

    def compute_advanced_saliency_maps(
        self, method="integrated_gradients", smoothing=True
    ):
        """
        Compute advanced saliency maps with multiple visualization methods.

        Args:
            method: 'integrated_gradients', 'guided_gradients', 'smoothed_gradients', or 'comparison'
            smoothing: Whether to apply Gaussian smoothing
        """
        # Choose random sample of images
        saliency_maps_images = random.sample(self.images, self.actual_sample_size)

        if method == "comparison":
            # Create comparison with multiple methods
            self.fig, self.axes = plt.subplots(
                self.actual_sample_size, 5, figsize=(20, self.actual_sample_size * 4)
            )
            if self.actual_sample_size == 1:
                self.axes = self.axes.reshape(1, -1)
            plt.subplots_adjust(hspace=0.4, wspace=0.2)

            column_titles = ["Original", "Standard", "Integrated", "Guided", "Smoothed"]

        else:
            # Single method visualization
            self.fig, self.axes = plt.subplots(
                self.actual_sample_size, 2, figsize=(8.27, self.actual_sample_size * 5)
            )
            if self.actual_sample_size == 1:
                self.axes = self.axes.reshape(1, -1)
            plt.subplots_adjust(hspace=0.5)

        print(f"Computing {method} saliency maps...")

        for idx, img_path in enumerate(saliency_maps_images):
            full_img_path = os.path.join(self.input_dir_path, img_path)
            print(f"Processing: {full_img_path}")

            # Load and preprocess image
            img = preprocess_siamese_input(
                full_img_path, target_size=(self.input_size, self.input_size)
            )
            anchor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)

            # Create random counterpart
            random_counterpart = create_random_image(img)
            counterpart = tf.convert_to_tensor(
                np.expand_dims(random_counterpart, axis=0), dtype=tf.float32
            )

            # Resize image for display
            img_display = tf.image.resize(img, (350, 350))

            if method == "comparison":
                # Show original image
                self.axes[idx, 0].imshow(img_display)
                self.axes[idx, 0].axis("off")
                self.axes[idx, 0].set_title(f"Original\n{os.path.basename(img_path)}")

                # Compute different saliency methods
                methods = [
                    (
                        "standard",
                        lambda: self.compute_standard_gradients(anchor, counterpart),
                    ),
                    (
                        "integrated",
                        lambda: self.compute_integrated_gradients(anchor, counterpart),
                    ),
                    (
                        "guided",
                        lambda: self.compute_guided_gradients(anchor, counterpart),
                    ),
                    (
                        "smoothed",
                        lambda: self.compute_smoothed_gradients(anchor, counterpart),
                    ),
                ]

                for col, (method_name, compute_func) in enumerate(methods, 1):
                    try:
                        gradients = compute_func()
                        saliency = self.process_saliency_map(
                            gradients, method="magnitude"
                        )

                        if smoothing:
                            saliency = self.apply_gaussian_smoothing(
                                saliency[0], sigma=1.0
                            )
                        else:
                            saliency = saliency[0]

                        saliency = self.normalize_saliency_map(saliency)

                        # Resize for display
                        saliency_display = tf.image.resize(
                            tf.expand_dims(saliency, axis=-1), (350, 350)
                        )

                        self.axes[idx, col].imshow(
                            saliency_display[:, :, 0], cmap="hot", alpha=0.7
                        )
                        self.axes[idx, col].imshow(
                            img_display, alpha=0.3
                        )  # Overlay original image
                        self.axes[idx, col].axis("off")
                        self.axes[idx, col].set_title(f"{method_name.capitalize()}")

                    except Exception as e:
                        print(f"Error computing {method_name}: {e}")
                        self.axes[idx, col].text(
                            0.5,
                            0.5,
                            f"Error: {method_name}",
                            transform=self.axes[idx, col].transAxes,
                            ha="center",
                            va="center",
                        )
                        self.axes[idx, col].axis("off")

            else:
                # Single method visualization
                self.axes[idx, 0].imshow(img_display)
                self.axes[idx, 0].axis("off")
                self.axes[idx, 0].set_title(f"Original\n{os.path.basename(img_path)}")

                # Compute specified method
                if method == "integrated_gradients":
                    gradients = self.compute_integrated_gradients(anchor, counterpart)
                elif method == "guided_gradients":
                    gradients = self.compute_guided_gradients(anchor, counterpart)
                elif method == "smoothed_gradients":
                    gradients = self.compute_smoothed_gradients(anchor, counterpart)
                else:
                    raise ValueError(f"Unknown method: {method}")

                # Process saliency map
                saliency = self.process_saliency_map(gradients, method="magnitude")

                if smoothing:
                    saliency = self.apply_gaussian_smoothing(saliency[0], sigma=1.0)
                else:
                    saliency = saliency[0]

                saliency = self.normalize_saliency_map(saliency)

                # Resize for display
                saliency_display = tf.image.resize(
                    tf.expand_dims(saliency, axis=-1), (350, 350)
                )

                # Show saliency map overlaid on original image
                self.axes[idx, 1].imshow(
                    saliency_display[:, :, 0], cmap="hot", alpha=0.7
                )
                self.axes[idx, 1].imshow(img_display, alpha=0.3)
                self.axes[idx, 1].axis("off")
                self.axes[idx, 1].set_title(f"{method.replace('_', ' ').title()}")

        # Save results
        os.makedirs(self.output_dir_path, exist_ok=True)

        if method == "comparison":
            output_file = os.path.join(
                self.output_dir_path,
                f"{os.path.basename(self.input_dir_path)}_saliency_comparison.pdf",
            )
        else:
            output_file = os.path.join(
                self.output_dir_path,
                f"{os.path.basename(self.input_dir_path)}_saliency_{method}.pdf",
            )

        plt.savefig(output_file, format="pdf", bbox_inches="tight", dpi=150)
        plt.close()

        print(f"Advanced saliency maps saved to: {output_file}")
        return output_file

    def compute_standard_gradients(self, image, counterpart):
        """
        Compute standard gradients (original method).
        """
        with tf.GradientTape() as tape:
            tape.watch(image)
            output = self.model([image, counterpart])

        gradients = tape.gradient(output, image)
        return gradients


class MeanSaliencyMapCreator:
    """
    Create mean/averaged saliency maps across all images in the dataset.

    This class computes saliency maps for every image in all subdirectories,
    then creates a single averaged saliency map showing the global attention pattern
    that the model consistently uses across all inputs.
    """

    def __init__(
        self,
        model,
        input_dir_path,
        output_dir_path=None,
        nesting=None,
        input_size=224,
        integration_steps=None,
        smoothing_samples=None,
        fast_mode=False,
    ):
        self.model = model
        self.input_dir_path = (
            input_dir_path if not nesting else os.path.join(input_dir_path, nesting)
        )
        self.output_dir_path = output_dir_path or input_dir_path
        self.input_size = input_size
        self.integration_steps = integration_steps
        self.smoothing_samples = smoothing_samples
        self.fast_mode = fast_mode

        # Create output directory
        os.makedirs(self.output_dir_path, exist_ok=True)

        # Get all images from all subdirectories
        self.all_images = self._collect_all_images()
        print(
            f"📊 Found {len(self.all_images)} total images for mean saliency computation"
        )

        if len(self.all_images) == 0:
            raise ValueError(f"No images found in: {self.input_dir_path}")

    def _collect_all_images(self):
        """Collect all image paths from all subdirectories."""
        all_images = []

        # Get all subdirectories
        subdirs = [
            d
            for d in os.listdir(self.input_dir_path)
            if os.path.isdir(os.path.join(self.input_dir_path, d))
        ]

        for subdir in subdirs:
            subdir_path = os.path.join(self.input_dir_path, subdir)
            subdir_images = [
                os.path.join(subdir_path, f)
                for f in os.listdir(subdir_path)
                if is_img_file(f)
            ]
            all_images.extend(subdir_images)

        return all_images

    def _compute_mean_original_images(self, max_images=None):
        """
        Compute mean of all original images to use as background.

        Args:
            max_images: Maximum number of images to process (None = all images)

        Returns:
            numpy.ndarray: Mean image array normalized to [0, 1]
        """
        print(f"🖼️ Computing mean of original images...")

        # Limit images if specified
        images_to_process = (
            self.all_images[:max_images] if max_images else self.all_images
        )
        print(f"📈 Processing {len(images_to_process)} images for mean image...")

        accumulated_image = None
        successful_count = 0

        for i, image_path in enumerate(images_to_process):
            print(
                f"Processing [{i+1}/{len(images_to_process)}]: {os.path.basename(image_path)}"
            )

            try:
                # Load and preprocess image
                img = preprocess_siamese_input(
                    image_path, target_size=(self.input_size, self.input_size)
                )

                if accumulated_image is None:
                    # Initialize with first successful image
                    accumulated_image = img.astype(np.float64)
                else:
                    # Accumulate
                    accumulated_image += img.astype(np.float64)

                successful_count += 1

            except Exception as e:
                print(f"⚠️  Error processing {image_path}: {e}")
                continue

        if accumulated_image is None:
            raise ValueError("No images could be processed successfully")

        # Compute mean
        mean_image = accumulated_image / successful_count
        print(
            f"✅ Successfully computed mean from {successful_count}/{len(images_to_process)} images"
        )

        return mean_image

    def _compute_single_saliency_map(self, image_path, method):
        """Compute saliency map for a single image using the specified method."""
        try:
            # Load and preprocess image
            img = preprocess_siamese_input(
                image_path, target_size=(self.input_size, self.input_size)
            )
            anchor = tf.convert_to_tensor(np.expand_dims(img, axis=0), dtype=tf.float32)

            # Create random counterpart
            random_counterpart = create_random_image(img)
            counterpart = tf.convert_to_tensor(
                np.expand_dims(random_counterpart, axis=0), dtype=tf.float32
            )

            # Compute gradients based on method
            if method == "standard":
                with tf.GradientTape() as tape:
                    tape.watch(anchor)
                    output = self.model([anchor, counterpart])
                gradients = tape.gradient(output, anchor)

            elif method == "integrated_gradients":
                gradients = self._compute_integrated_gradients_direct(
                    anchor, counterpart
                )

            elif method == "guided_gradients":
                with tf.GradientTape() as tape:
                    tape.watch(anchor)
                    output = self.model([anchor, counterpart])
                gradients = tape.gradient(output, anchor)
                # Apply guided backpropagation: keep only positive gradients
                gradients = tf.where(gradients > 0, gradients, tf.zeros_like(gradients))

            elif method == "smoothed_gradients":
                gradients = self._compute_smoothed_gradients_direct(anchor, counterpart)

            else:
                raise ValueError(f"Unknown method: {method}")

            # Process to saliency map
            saliency = tf.sqrt(
                tf.reduce_sum(tf.square(gradients), axis=-1)
            )  # magnitude method
            saliency = self._normalize_saliency_map(saliency[0])

            return saliency.numpy()

        except Exception as e:
            print(f"⚠️  Error processing {image_path}: {e}")
            return None

    def _compute_integrated_gradients_direct(self, image, counterpart):
        """Direct computation of integrated gradients without using temp creator."""
        # Determine steps
        if self.integration_steps is not None:
            steps = self.integration_steps
        else:
            steps = 10 if self.fast_mode else 20

        # Create baseline and compute integration
        baseline = tf.zeros_like(image)
        alphas = tf.linspace(start=0.0, stop=1.0, num=steps + 1)

        gradients = []
        for alpha in alphas:
            interpolated = baseline + alpha * (image - baseline)
            with tf.GradientTape() as tape:
                tape.watch(interpolated)
                output = self.model([interpolated, counterpart])
            grad = tape.gradient(output, interpolated)
            gradients.append(grad)

        gradients = tf.stack(gradients)
        integrated_gradients = tf.reduce_mean(gradients, axis=0)
        integrated_gradients = integrated_gradients * (image - baseline)

        return integrated_gradients

    def _compute_smoothed_gradients_direct(self, image, counterpart):
        """Direct computation of smoothed gradients without using temp creator."""
        # Determine samples
        if self.smoothing_samples is not None:
            n_samples = self.smoothing_samples
        else:
            n_samples = 5 if self.fast_mode else 10

        gradients = []
        for _ in range(n_samples):
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.1)
            noisy_image = image + noise
            noisy_image = tf.clip_by_value(noisy_image, 0.0, 1.0)

            with tf.GradientTape() as tape:
                tape.watch(noisy_image)
                output = self.model([noisy_image, counterpart])
            grad = tape.gradient(output, noisy_image)
            gradients.append(grad)

        smoothed_gradients = tf.reduce_mean(tf.stack(gradients), axis=0)
        return smoothed_gradients

    def _normalize_saliency_map(self, saliency_map):
        """Normalize saliency map to [0, 1] range."""
        min_val = tf.reduce_min(saliency_map)
        max_val = tf.reduce_max(saliency_map)

        if max_val > min_val:
            normalized = (saliency_map - min_val) / (max_val - min_val)
        else:
            normalized = tf.zeros_like(saliency_map)

        return normalized

    def compute_mean_saliency_map(
        self,
        method="guided_gradients",
        max_images=None,
        use_mean_image_background=False,
    ):
        """
        Compute mean saliency map across all images.

        Args:
            method: Saliency method to use ('standard', 'integrated_gradients',
                   'guided_gradients', 'smoothed_gradients')
            max_images: Maximum number of images to process (None = all images)
            use_mean_image_background: Whether to compute and use mean of original images as background

        Returns:
            str: Path to the saved mean saliency map
        """
        print(f"🧮 Computing mean saliency maps using {method} method...")

        # Limit images if specified
        images_to_process = (
            self.all_images[:max_images] if max_images else self.all_images
        )
        print(f"📈 Processing {len(images_to_process)} images...")

        accumulated_saliency = None
        successful_count = 0

        for i, image_path in enumerate(images_to_process):
            print(
                f"Processing [{i+1}/{len(images_to_process)}]: {os.path.basename(image_path)}"
            )

            saliency_map = self._compute_single_saliency_map(image_path, method)

            if saliency_map is not None:
                if accumulated_saliency is None:
                    # Initialize with first successful saliency map
                    accumulated_saliency = saliency_map.astype(np.float64)
                else:
                    # Accumulate
                    accumulated_saliency += saliency_map.astype(np.float64)

                successful_count += 1

        if accumulated_saliency is None:
            raise ValueError("No saliency maps could be computed successfully")

        # Compute mean
        mean_saliency = accumulated_saliency / successful_count
        print(
            f"✅ Successfully computed mean from {successful_count}/{len(images_to_process)} images"
        )

        # Optionally compute mean of original images
        mean_image = None
        if use_mean_image_background:
            mean_image = self._compute_mean_original_images(max_images)

        # Create visualization
        output_path = self._create_visualization(
            mean_saliency, method, successful_count, mean_image
        )

        return output_path

    def _create_visualization(self, mean_saliency, method, num_images, mean_image=None):
        """Create and save visualization of the mean saliency map."""
        if mean_image is not None:
            # When mean image background is requested, show mean image + overlay version
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot 1: Mean original image
            axes[0].imshow(mean_image)
            axes[0].set_title(f"Mean Original Image\n({num_images} images)")
            axes[0].axis("off")

            # Plot 2: Mean saliency overlaid on mean image
            axes[1].imshow(mean_image, alpha=0.95)
            axes[1].imshow(mean_saliency, cmap="hot", alpha=0.8)
            axes[1].set_title(
                f"Mean {method.replace('_', ' ').title()}\nOverlaid on Mean Image"
            )
            axes[1].axis("off")

        else:
            # When no mean image background, show the original 2-plot version
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # Plot 1: Mean saliency map alone
            im1 = axes[0].imshow(mean_saliency, cmap="hot")
            axes[0].set_title(
                f"Mean {method.replace('_', ' ').title()}\n({num_images} images)"
            )
            axes[0].axis("off")
            plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

            # Plot 2: Mean saliency map with different colormap for comparison
            im2 = axes[1].imshow(mean_saliency, cmap="viridis")
            axes[1].set_title(
                f"Mean {method.replace('_', ' ').title()} (Viridis)\n({num_images} images)"
            )
            axes[1].axis("off")
            plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()

        # Save the visualization
        suffix = "_with_mean_bg" if mean_image is not None else ""
        output_filename = f"mean_saliency_{method}_{num_images}images{suffix}.pdf"
        output_path = os.path.join(self.output_dir_path, output_filename)

        plt.savefig(output_path, format="pdf", bbox_inches="tight", dpi=150)
        plt.close()

        # Also save raw data as numpy arrays
        npy_filename = f"mean_saliency_{method}_{num_images}images.npy"
        npy_path = os.path.join(self.output_dir_path, npy_filename)
        np.save(npy_path, mean_saliency)

        if mean_image is not None:
            mean_image_npy_filename = f"mean_original_image_{num_images}images.npy"
            mean_image_npy_path = os.path.join(
                self.output_dir_path, mean_image_npy_filename
            )
            np.save(mean_image_npy_path, mean_image)
            print(f"💾 Mean image data saved to: {mean_image_npy_path}")

        print(f"📊 Mean saliency visualization saved to: {output_path}")
        print(f"💾 Raw data saved to: {npy_path}")

        return output_path
