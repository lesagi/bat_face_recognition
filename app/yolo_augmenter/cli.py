#!/usr/bin/env python3
"""
CLI interface for YOLO Segmentation Data Augmentation

Usage:
    python -m app.yolo_augmenter --input /path/to/source --output /path/to/target [options]
"""

import click
import sys
from pathlib import Path
from .segmentation_augmenter import YoloSegmentationAugmenter


def check_dependencies():
    """Check if required dependencies are installed."""
    missing_deps = []
    
    try:
        import cv2
        click.echo(f"✅ OpenCV version: {cv2.__version__}")
    except ImportError:
        missing_deps.append("opencv-python")
    
    try:
        import albumentations
        click.echo(f"✅ Albumentations version: {albumentations.__version__}")
    except ImportError:
        missing_deps.append("albumentations")
    
    try:
        import numpy
        click.echo(f"✅ NumPy version: {numpy.__version__}")
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import yaml
        click.echo("✅ PyYAML available")
    except ImportError:
        missing_deps.append("pyyaml")
    
    try:
        import tqdm
        click.echo("✅ tqdm available")
    except ImportError:
        missing_deps.append("tqdm")
    
    try:
        import click as click_lib
        click.echo(f"✅ Click version: {click_lib.__version__}")
    except ImportError:
        missing_deps.append("click")
    
    if missing_deps:
        click.echo(f"\n❌ Missing dependencies: {', '.join(missing_deps)}", err=True)
        click.echo(f"Install with: pip install {' '.join(missing_deps)}")
        return False
    
    return True


def validate_input_structure(input_dir: Path) -> bool:
    """Validate that input directory has correct YOLO dataset structure."""
    required_dirs = [
        input_dir / "train" / "images",
        input_dir / "train" / "labels"
    ]
    
    # Check required directories
    for dir_path in required_dirs:
        if not dir_path.exists():
            click.echo(f"❌ Required directory not found: {dir_path}", err=True)
            return False
    
    # Check if there are images
    train_images = input_dir / "train" / "images"
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(train_images.glob(f"*{ext}")))
        image_files.extend(list(train_images.glob(f"*{ext.upper()}")))
    
    if not image_files:
        click.echo(f"❌ No image files found in {train_images}", err=True)
        click.echo(f"   Supported formats: {', '.join(image_extensions)}")
        return False
    
    click.echo(f"✅ Found {len(image_files)} images in training set")
    
    # Check validation set (optional)
    val_images = input_dir / "val" / "images"
    if val_images.exists():
        val_image_files = []
        for ext in image_extensions:
            val_image_files.extend(list(val_images.glob(f"*{ext}")))
            val_image_files.extend(list(val_images.glob(f"*{ext.upper()}")))
        click.echo(f"✅ Found {len(val_image_files)} images in validation set")
    else:
        click.echo("⚠️ No validation set found (optional)")
    
    return True


@click.command()
@click.option(
    '--input', '-i', 'input_dir',
    required=True,
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    help='Input directory containing YOLO segmentation dataset'
)
@click.option(
    '--output', '-o', 'output_dir',
    required=True,
    type=click.Path(path_type=Path),
    help='Output directory for augmented dataset'
)
@click.option(
    '--count', '-c',
    default=10,
    type=click.IntRange(1, 1000),
    help='Number of augmentations per image (default: 10)'
)
@click.option(
    '--verbose', '-v',
    is_flag=True,
    help='Enable verbose output'
)
@click.option(
    '--check-deps',
    is_flag=True,
    help='Check if required dependencies are installed and exit'
)
@click.option(
    '--force', '-f',
    is_flag=True,
    help='Overwrite output directory if it exists without prompting'
)
@click.help_option('--help', '-h')
def main(input_dir: Path, output_dir: Path, count: int, verbose: bool, check_deps: bool, force: bool):
    """
    🦇 YOLO Segmentation Dataset Augmentation Tool
    
    Augments YOLO segmentation datasets with proper mask and bbox transformations.
    
    \b
    Expected input directory structure:
      input_dir/
      ├── train/
      │   ├── images/
      │   │   ├── image1.jpg
      │   │   └── image2.png
      │   └── labels/
      │       ├── image1.txt
      │       └── image2.txt
      └── val/ (optional)
          ├── images/
          └── labels/
    
    \b
    Output structure:
      output_dir/
      ├── train/
      │   ├── images/
      │   │   ├── image1_a-000.jpg
      │   │   ├── image1_a-001.jpg
      │   │   └── ...
      │   └── labels/
      │       ├── image1_a-000.txt
      │       ├── image1_a-001.txt
      │       └── ...
      ├── val/
      │   ├── images/
      │   └── labels/
      └── dataset.yaml
    
    \b
    Examples:
      # Basic usage
      python -m app.yolo_augmenter -i ./data/original -o ./data/augmented
      
      # With 20 augmentations per image
      python -m app.yolo_augmenter -i ./data/original -o ./data/augmented -c 20
      
      # Force overwrite existing output
      python -m app.yolo_augmenter -i ./data/original -o ./data/augmented --force
    """
    
    # Check dependencies if requested
    if check_deps:
        if check_dependencies():
            click.echo("✅ All dependencies are available!")
        else:
            sys.exit(1)
        return
    
    click.echo("🦇 YOLO Segmentation Dataset Augmentation Tool")
    click.echo("=" * 60)
    
    # Validate input structure
    if not validate_input_structure(input_dir):
        click.echo("\n❌ Input directory structure is invalid", err=True)
        click.echo("Expected structure:")
        click.echo("  input_dir/")
        click.echo("  ├── train/")
        click.echo("  │   ├── images/")
        click.echo("  │   └── labels/")
        click.echo("  └── val/ (optional)")
        click.echo("      ├── images/")
        click.echo("      └── labels/")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    click.echo(f"\n📂 Input: {input_dir.absolute()}")
    click.echo(f"📂 Output: {output_dir.absolute()}")
    click.echo(f"🔢 Augmentations per image: {count}")
    
    # Handle existing output directory
    if output_dir.exists() and not force:
        if not click.confirm(f"\n⚠️ Output directory exists: {output_dir}\nContinue?"):
            click.echo("Aborted.")
            sys.exit(0)
    
    try:
        # Create augmenter
        augmenter = YoloSegmentationAugmenter(
            source_dir=str(input_dir),
            target_dir=str(output_dir),
            augmentations_per_image=count
        )
        
        # Run augmentation
        with click.progressbar(length=1, label='Running augmentation') as bar:
            results = augmenter.augment_dataset()
            bar.update(1)
        
        click.echo("\n🎉 AUGMENTATION COMPLETE!")
        click.echo("=" * 40)
        click.echo("📈 Results:")
        for split, result_count in results.items():
            click.echo(f"   {split}: {result_count} successful augmentations")
        
        click.echo(f"\n📁 Augmented dataset: {output_dir.absolute()}")
        click.echo(f"📝 Dataset config: {output_dir.absolute() / 'dataset.yaml'}")
        
        click.echo(f"\n💡 Next steps:")
        click.echo("1. Review the augmented images and labels")
        click.echo("2. Train your YOLO model with:")
        click.echo(f"   yolo segment train data={output_dir.absolute() / 'dataset.yaml'} model=yolov8n-seg.pt epochs=100")
        
    except KeyboardInterrupt:
        click.echo("\n\n⚠️ Interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Error during augmentation: {e}", err=True)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()