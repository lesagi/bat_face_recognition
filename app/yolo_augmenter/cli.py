#!/usr/bin/env python3
"""
CLI interface for the YOLO Augmenter system using Click.

This module provides a unified command-line interface that can auto-detect
data types and route to appropriate augmenters.
"""

import click
import sys
from pathlib import Path
from .base.config import AugmentationConfig
from .base.pipeline_factory import AugmentationPipelineFactory
from .augmenters.plain_image_augmenter import PlainImageAugmenter


def detect_data_type(input_dir: Path) -> str:
    """Auto-detect the type of data in the input directory."""
    input_dir = Path(input_dir)
    
    # Check for YOLO dataset structure
    train_images = input_dir / "train" / "images"
    train_labels = input_dir / "train" / "labels"
    
    if train_images.exists() and train_labels.exists():
        # Check if labels are segmentation or detection format
        label_files = list(train_labels.glob("*.txt"))
        if label_files:
            # Read first label file to determine format
            try:
                with open(label_files[0], 'r') as f:
                    first_line = f.readline().strip()
                    if first_line:
                        parts = first_line.split()
                        if len(parts) > 5:  # More than class + bbox coords
                            return "segmentation"
                        else:
                            return "detection"
            except Exception:
                pass
        
        # Default to detection if we can't determine
        return "detection"
    
    # Check for plain images
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(input_dir.glob(f"*{ext}")))
        image_files.extend(list(input_dir.glob(f"*{ext.upper()}")))
    
    if image_files:
        return "plain"
    
    return "unknown"


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
        import click as click_lib
        click.echo(f"✅ Click version: {click_lib.__version__}")
    except ImportError:
        missing_deps.append("click")
    
    if missing_deps:
        click.echo(f"\n❌ Missing dependencies: {', '.join(missing_deps)}", err=True)
        click.echo(f"Install with: pip install {' '.join(missing_deps)}")
        return False
    
    return True


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--debug', '-d', is_flag=True, help='Enable debug mode')
@click.pass_context
def cli(ctx, config, debug):
    """🦇 YOLO Augmenter - Modular image augmentation system."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = config
    ctx.obj['debug'] = debug


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Input directory')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory')
@click.option('--preset', '-p', required=True, help='Augmentation preset name (must be defined in config file)')
@click.option('--force', '-f', is_flag=True, help='Overwrite output directory without prompting')
@click.pass_context
def auto(ctx, input, output, preset, force):
    """Auto-detect data type and apply appropriate augmentation (default mode)."""
    input_dir = Path(input)
    output_dir = Path(output)
    
    click.echo("🔍 Auto-detecting data type...")
    data_type = detect_data_type(input_dir)
    
    if data_type == "unknown":
        click.echo("❌ Could not determine data type. Please specify manually.")
        sys.exit(1)
    
    click.echo(f"✅ Detected data type: {data_type}")
    
    # Route to appropriate augmenter
    if data_type == "plain":
        plain(ctx, input, output, preset, force)
    elif data_type == "detection":
        detection(ctx, input, output, preset, force)
    elif data_type == "segmentation":
        segmentation(ctx, input, output, preset, force)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Input directory with images')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory for augmented images')
@click.option('--preset', '-p', required=True, help='Augmentation preset name (must be defined in config file)')
@click.option('--force', '-f', is_flag=True, help='Overwrite output directory without prompting')
@click.pass_context
def plain(ctx, input, output, preset, force):
    """Augment plain images without labels."""
    input_dir = Path(input)
    output_dir = Path(output)
    
    # Handle existing output directory
    if output_dir.exists() and not force:
        if not click.confirm(f"⚠️ Output directory exists: {output_dir}\nContinue?"):
            click.echo("Aborted.")
            return
    
    try:
        # Create augmenter
        augmenter = PlainImageAugmenter(
            source_dir=str(input_dir),
            target_dir=str(output_dir),
            preset=preset,
            config_path=ctx.obj['config'],
            debug=ctx.obj['debug']
        )
        
        # Run augmentation
        results = augmenter.augment_dataset()
        
        if results:
            click.echo(f"\n🎉 Augmentation completed!")
            click.echo(f"📁 Output: {output_dir.absolute()}")
        else:
            click.echo("❌ Augmentation failed")
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        if ctx.obj['debug']:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Input directory with YOLO detection dataset')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory for augmented dataset')
@click.option('--preset', '-p', required=True, help='Augmentation preset name (must be defined in config file)')
@click.option('--force', '-f', is_flag=True, help='Overwrite output directory without prompting')
@click.pass_context
def detection(ctx, input, output, preset, force):
    """Augment YOLO detection dataset (bounding boxes)."""
    click.echo("⚠️ YOLO detection augmentation not yet implemented")
    click.echo("This will be implemented in the next phase")
    sys.exit(1)


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True), help='Input directory with YOLO segmentation dataset')
@click.option('--output', '-o', required=True, type=click.Path(), help='Output directory for augmented dataset')
@click.option('--preset', '-p', required=True, help='Augmentation preset name (must be defined in config file)')
@click.option('--force', '-f', is_flag=True, help='Overwrite output directory without prompting')
@click.pass_context
def segmentation(ctx, input, output, preset, force):
    """Augment YOLO segmentation dataset (masks)."""
    click.echo("⚠️ YOLO segmentation augmentation not yet implemented")
    click.echo("This will be implemented in the next phase")
    sys.exit(1)


@cli.command()
@click.option('--preset', '-p', help='Show info for specific preset')
@click.pass_context
def info(ctx, preset):
    """Show information about available presets and configurations."""
    config = AugmentationConfig(ctx.obj['config'])
    
    if preset:
        # Show specific preset info
        pipeline_factory = AugmentationPipelineFactory(config)
        info = pipeline_factory.get_pipeline_info(preset)
        if info:
            click.echo(f"🔧 Preset: {info['preset_name']}")
            click.echo(f"📊 Total augmentations: {info['total_augmentations']}")
            click.echo(f"⚙️ Parameters:")
            for param_name, param_info in info['parameters'].items():
                steps = param_info['steps']
                custom = "custom" if param_info['custom'] else "default"
                click.echo(f"   {param_name}: {steps} steps ({custom})")
        else:
            click.echo(f"❌ Preset '{preset}' not found")
    else:
        # List all presets
        presets = config.list_presets()
        click.echo("📋 Available presets:")
        for preset_name in presets:
            pipeline_factory = AugmentationPipelineFactory(config)
            info = pipeline_factory.get_pipeline_info(preset_name)
            click.echo(f"   {preset_name}: {info['total_augmentations']} augmentations")


@cli.command()
def check_deps():
    """Check if required dependencies are installed."""
    if check_dependencies():
        click.echo("✅ All dependencies are available!")
    else:
        sys.exit(1)


if __name__ == '__main__':
    cli()