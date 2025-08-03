"""
Utility functions for YOLO segmentation training.

This module provides utility functions for data preparation and processing
for YOLO segmentation model training.
"""

import os
import cv2
import glob
import shutil
import random
from typing import Dict, List, Tuple
from pathlib import Path


class YoloTrainingUtils:
    """Utility functions for YOLO training data preparation."""
    
    @staticmethod
    def prepare_labels_from_masks(mask_dir: str, 
                                  output_dir: str,
                                  min_contour_area: int = 200) -> int:
        """Convert binary masks to YOLO segmentation labels.
        
        Args:
            mask_dir: Directory containing binary mask images
            output_dir: Directory to save YOLO label files
            min_contour_area: Minimum contour area to include
            
        Returns:
            Number of label files created
            
        Raises:
            FileNotFoundError: If mask_dir doesn't exist
        """
        if not os.path.exists(mask_dir):
            raise FileNotFoundError(f"Mask directory not found: {mask_dir}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        processed_count = 0
        
        for img_file in os.listdir(mask_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(mask_dir, img_file)
            
            # Load the binary mask and get its contours
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"Warning: Could not load mask from {img_path}")
                continue
                
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
            mask_height, mask_width = mask.shape
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert the contours to normalized polygons
            polygons = []
            for contour in contours:
                if cv2.contourArea(contour) > min_contour_area:
                    polygon = []
                    for point in contour:
                        x, y = point[0]
                        # Normalize coordinates
                        polygon.extend([x / mask_width, y / mask_height])
                    polygons.append(polygon)
            
            # Save the label file
            label_filename = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(output_dir, label_filename)
            
            with open(label_path, 'w') as f:
                for polygon in polygons:
                    # Write class 0 followed by normalized coordinates
                    f.write('0 ')
                    for i, coord in enumerate(polygon):
                        if i == len(polygon) - 1:
                            f.write(f'{coord}\n')
                        else:
                            f.write(f'{coord} ')
            
            processed_count += 1
        
        print(f"Processed {processed_count} mask files to YOLO labels")
        return processed_count
    
    @staticmethod
    def cleanup_images_without_labels(image_base_dir: str, 
                                      label_base_dir: str,
                                      image_extensions: List[str] = None,
                                      verbose: bool = True) -> Dict[str, int]:
        """Remove images that don't have corresponding label files.
        
        This method scans directories containing images and their corresponding
        YOLO label files, and removes images that don't have matching labels.
        Useful for cleaning datasets before training.
        
        Args:
            image_base_dir: Base directory containing image subdirectories
            label_base_dir: Base directory containing label annotation subdirectories
            image_extensions: List of image file extensions to consider
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary with statistics: {'total_removed': int, 'by_directory': dict}
            
        Raises:
            FileNotFoundError: If base directories don't exist
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png']
        
        if not os.path.exists(image_base_dir):
            raise FileNotFoundError(f"Image base directory not found: {image_base_dir}")
        
        if not os.path.exists(label_base_dir):
            raise FileNotFoundError(f"Label base directory not found: {label_base_dir}")
        
        # Get all image subdirectories
        image_dirs = [d for d in os.listdir(image_base_dir) 
                      if os.path.isdir(os.path.join(image_base_dir, d))]
        
        if verbose:
            print(f'Found image directories: {image_dirs}')
        
        total_removed = 0
        stats_by_dir = {}
        
        for dir_name in image_dirs:
            if verbose:
                print(f'\nProcessing {dir_name}...')
            
            # Find corresponding label directory
            label_dirs = glob.glob(f'{label_base_dir}/{dir_name}_annotations_*')
            if not label_dirs:
                if verbose:
                    print(f'  No label directory found for {dir_name}')
                stats_by_dir[dir_name] = {'removed': 0, 'reason': 'no_label_dir'}
                continue
            
            label_dir = label_dirs[0]  # Take the first match
            label_files_dir = os.path.join(label_dir, 'labels', 'train')
            
            if not os.path.exists(label_files_dir):
                if verbose:
                    print(f'  No train labels directory found: {label_files_dir}')
                stats_by_dir[dir_name] = {'removed': 0, 'reason': 'no_train_labels'}
                continue
            
            # Get all label files (without extension)
            label_files = set()
            for f in os.listdir(label_files_dir):
                if f.endswith('.txt'):
                    label_files.add(f[:-4])  # Remove .txt extension
            
            # Get all image files
            image_dir = os.path.join(image_base_dir, dir_name)
            image_files = []
            for f in os.listdir(image_dir):
                if any(f.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(f)
            
            if verbose:
                print(f'  Images: {len(image_files)}, Labels: {len(label_files)}')
            
            # Remove images without corresponding labels
            removed_count = 0
            for image_file in image_files:
                image_name = os.path.splitext(image_file)[0]  # Remove extension
                if image_name not in label_files:
                    image_path = os.path.join(image_dir, image_file)
                    if verbose:
                        print(f'    Removing: {image_file}')
                    os.remove(image_path)
                    removed_count += 1
            
            stats_by_dir[dir_name] = {
                'removed': removed_count,
                'original_images': len(image_files),
                'available_labels': len(label_files),
                'remaining_images': len(image_files) - removed_count
            }
            
            if verbose:
                print(f'  Removed {removed_count} images without labels')
            total_removed += removed_count
        
        result = {
            'total_removed': total_removed,
            'by_directory': stats_by_dir
        }
        
        if verbose:
            print(f'\nTotal images removed: {total_removed}')
        
        return result
    
    @staticmethod
    def create_yolo_dataset(image_base_dir: str,
                           label_base_dir: str,
                           output_dir: str = None,
                           train_split: float = 0.8,
                           seed: int = 42,
                           copy_files: bool = True,
                           image_extensions: List[str] = None,
                           verbose: bool = True) -> Dict[str, any]:
        """Create YOLO dataset structure with train/validation splits from existing directories.
        
        This method takes the cleaned image and label directories and creates a proper
        YOLO dataset structure with train/val splits.
        
        Expected input structure:
        image_base_dir/
        ├── subject1/
        │   ├── img1.jpg
        │   └── img2.jpg
        └── subject2/
            └── img3.jpg
            
        label_base_dir/
        ├── subject1_annotations_*/
        │   └── labels/train/
        │       ├── img1.txt
        │       └── img2.txt
        └── subject2_annotations_*/
            └── labels/train/
                └── img3.txt
        
        Created output structure:
        output_dir/
        ├── train/
        │   ├── images/
        │   │   ├── subject1_img1.jpg
        │   │   └── subject2_img3.jpg
        │   └── labels/
        │       ├── subject1_img1.txt
        │       └── subject2_img3.txt
        ├── val/
        │   ├── images/
        │   │   └── subject1_img2.jpg
        │   └── labels/
        │       └── subject1_img2.txt
        └── data.yaml
        
        Args:
            image_base_dir: Base directory containing image subdirectories
            label_base_dir: Base directory containing label annotation subdirectories
            output_dir: Output directory for YOLO dataset structure (optional, auto-detected if None)
            train_split: Fraction of data to use for training (0.0-1.0)
            seed: Random seed for reproducible splits
            copy_files: If True, copy files; if False, move files
            image_extensions: List of image file extensions to consider
            verbose: Whether to print detailed progress information
            
        Returns:
            Dictionary with dataset statistics and file paths
            
        Raises:
            FileNotFoundError: If base directories don't exist
            ValueError: If train_split is not between 0 and 1
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png']
        
        if not os.path.exists(image_base_dir):
            raise FileNotFoundError(f"Image base directory not found: {image_base_dir}")
        
        if not os.path.exists(label_base_dir):
            raise FileNotFoundError(f"Label base directory not found: {label_base_dir}")
        
        if not 0 <= train_split <= 1:
            raise ValueError(f"train_split must be between 0 and 1, got {train_split}")
        
        # Auto-detect output directory if not provided
        if output_dir is None:
            # Find common parent directory of image_base_dir and label_base_dir
            image_parent = str(Path(image_base_dir).parent)
            label_parent = str(Path(label_base_dir).parent)
            
            if image_parent == label_parent:
                # Both directories are in the same parent, use that
                base_dir = image_parent
            else:
                # Use the parent of image_base_dir as fallback
                base_dir = image_parent
                if verbose:
                    print(f"Warning: Image and label directories have different parents.")
                    print(f"Using image directory parent: {base_dir}")
            
            output_dir = os.path.join(base_dir, 'data')
            if verbose:
                print(f"Auto-detected output directory: {output_dir}")
        
        # Set random seed for reproducible splits
        random.seed(seed)
        
        # Create output directory structure
        output_dir = Path(output_dir)
        train_images_dir = output_dir / 'train' / 'images'
        train_labels_dir = output_dir / 'train' / 'labels'
        val_images_dir = output_dir / 'val' / 'images'
        val_labels_dir = output_dir / 'val' / 'labels'
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        if verbose:
            print(f"Created YOLO dataset structure in: {output_dir}")
        
        # Get all image subdirectories
        image_dirs = [d for d in os.listdir(image_base_dir) 
                      if os.path.isdir(os.path.join(image_base_dir, d))]
        
        if verbose:
            print(f'Found image directories: {image_dirs}')
        
        total_files = 0
        train_count = 0
        val_count = 0
        stats_by_subject = {}
        
        # Process each subject directory
        for subject_name in image_dirs:
            if verbose:
                print(f'\nProcessing {subject_name}...')
            
            # Find corresponding label directory
            label_dirs = glob.glob(f'{label_base_dir}/{subject_name}_annotations_*')
            if not label_dirs:
                if verbose:
                    print(f'  No label directory found for {subject_name}')
                stats_by_subject[subject_name] = {'status': 'no_labels', 'train': 0, 'val': 0}
                continue
            
            label_dir = label_dirs[0]
            label_files_dir = os.path.join(label_dir, 'labels', 'train')
            
            if not os.path.exists(label_files_dir):
                if verbose:
                    print(f'  No train labels directory found: {label_files_dir}')
                stats_by_subject[subject_name] = {'status': 'no_train_labels', 'train': 0, 'val': 0}
                continue
            
            # Get matching image-label pairs
            image_dir = os.path.join(image_base_dir, subject_name)
            image_files = []
            for f in os.listdir(image_dir):
                if any(f.lower().endswith(ext) for ext in image_extensions):
                    image_name = os.path.splitext(f)[0]
                    label_file = os.path.join(label_files_dir, f'{image_name}.txt')
                    if os.path.exists(label_file):
                        image_files.append(f)
            
            if not image_files:
                if verbose:
                    print(f'  No matching image-label pairs found for {subject_name}')
                stats_by_subject[subject_name] = {'status': 'no_matches', 'train': 0, 'val': 0}
                continue
            
            # Shuffle and split files
            random.shuffle(image_files)
            split_idx = int(len(image_files) * train_split)
            train_files = image_files[:split_idx]
            val_files = image_files[split_idx:]
            
            if verbose:
                print(f'  Found {len(image_files)} image-label pairs')
                print(f'  Train: {len(train_files)}, Val: {len(val_files)}')
            
            # Copy/move train files
            subject_train_count = 0
            for img_file in train_files:
                img_name = os.path.splitext(img_file)[0]
                
                # Source paths
                src_img = os.path.join(image_dir, img_file)
                src_label = os.path.join(label_files_dir, f'{img_name}.txt')
                
                # Destination paths (prefix with subject name to avoid conflicts)
                dst_img = train_images_dir / f'{subject_name}_{img_file}'
                dst_label = train_labels_dir / f'{subject_name}_{img_name}.txt'
                
                # Copy or move files
                if copy_files:
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_label, dst_label)
                else:
                    shutil.move(src_img, dst_img)
                    shutil.move(src_label, dst_label)
                
                subject_train_count += 1
            
            # Copy/move validation files
            subject_val_count = 0
            for img_file in val_files:
                img_name = os.path.splitext(img_file)[0]
                
                # Source paths
                src_img = os.path.join(image_dir, img_file)
                src_label = os.path.join(label_files_dir, f'{img_name}.txt')
                
                # Destination paths (prefix with subject name to avoid conflicts)
                dst_img = val_images_dir / f'{subject_name}_{img_file}'
                dst_label = val_labels_dir / f'{subject_name}_{img_name}.txt'
                
                # Copy or move files
                if copy_files:
                    shutil.copy2(src_img, dst_img)
                    shutil.copy2(src_label, dst_label)
                else:
                    shutil.move(src_img, dst_img)
                    shutil.move(src_label, dst_label)
                
                subject_val_count += 1
            
            stats_by_subject[subject_name] = {
                'status': 'processed',
                'total': len(image_files),
                'train': subject_train_count,
                'val': subject_val_count
            }
            
            total_files += len(image_files)
            train_count += subject_train_count
            val_count += subject_val_count
        
        # Create data.yaml configuration file
        data_yaml_content = f"""# YOLO Dataset Configuration
# Generated by YoloTrainingUtils.create_yolo_dataset()

# Dataset root path
path: {output_dir.absolute()}

# Training and validation image directories (relative to 'path')
train: train/images
val: val/images

# Class names (0-indexed)
names:
  0: bat_face

# Number of classes
nc: 1

# Dataset statistics
# Total files: {total_files}
# Train files: {train_count}
# Val files: {val_count}
# Train split: {train_split}
"""
        
        data_yaml_path = output_dir / 'data.yaml'
        with open(data_yaml_path, 'w') as f:
            f.write(data_yaml_content)
        
        result = {
            'dataset_path': str(output_dir),
            'config_file': str(data_yaml_path),
            'total_files': total_files,
            'train_count': train_count,
            'val_count': val_count,
            'train_split': train_split,
            'by_subject': stats_by_subject,
            'directories_created': {
                'train_images': str(train_images_dir),
                'train_labels': str(train_labels_dir),
                'val_images': str(val_images_dir),
                'val_labels': str(val_labels_dir)
            }
        }
        
        if verbose:
            print(f'\nDataset creation complete!')
            print(f'Total files processed: {total_files}')
            print(f'Train files: {train_count}')
            print(f'Val files: {val_count}')
            print(f'Configuration saved to: {data_yaml_path}')
        
        return result