import os
import re
import sys
from pathlib import Path

import click
import cv2

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.config.loader import load_config
from app.models.yolo_factory import create_yolo_model


def parse_video_name(filename: str):
    """Parse `<type>_<bat_name>_VID_<vid_name>.*` into parts.

    Returns (type_str, bat_name, vid_name) or (None, None, None) if pattern doesn't match.
    """
    name = Path(filename).stem
    m = re.match(r"^(?P<type>[^_]+)_(?P<bat>[^_]+)_VID_(?P<vid>.+)$", name)
    if not m:
        return None, None, None
    return m.group("type"), m.group("bat"), m.group("vid")


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def get_video_rotation(video_path: str) -> int:
    """Get video rotation from metadata using ffprobe if available."""
    try:
        import subprocess
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', video_path
        ], capture_output=True, text=True, check=True)
        
        import json
        data = json.loads(result.stdout)
        for stream in data.get('streams', []):
            if stream.get('codec_type') == 'video':
                # Check for rotation in side_data or tags
                side_data = stream.get('side_data_list', [])
                for side in side_data:
                    if side.get('side_data_type') == 'Display Matrix':
                        rotation = side.get('rotation', 0)
                        return int(rotation)
                
                # Check tags for rotation
                tags = stream.get('tags', {})
                rotation = tags.get('rotate', 0)
                if rotation:
                    return int(rotation)
                
                # Check for display matrix in side_data
                for side in side_data:
                    if side.get('side_data_type') == 'Display Matrix':
                        matrix = side.get('display_matrix', '')
                        if matrix:
                            # Parse display matrix to determine rotation
                            # Common matrices: 90°=90, 180°=180, 270°=270
                            if '90' in matrix and '0' in matrix:
                                return 90
                            elif '180' in matrix:
                                return 180
                            elif '270' in matrix:
                                return 270
    except Exception as e:
        print(f"ffprobe error: {e}")
    
    # Fallback: if video is wider than tall, it's likely rotated
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            # If width > height, it's likely a portrait video stored as landscape
            if width > height:
                return 90  # Most common case for portrait videos
    except Exception:
        pass
    
    return 0


def rotate_frame(frame, rotation: int):
    """Rotate frame by the specified degrees (0, 90, 180, 270)."""
    if rotation == 0:
        return frame
    elif rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame


def has_segmentation_detection(model, frame_bgr, conf_threshold: float) -> bool:
    """Run YOLO segmentation and return True if any detection meets threshold."""
    # YOLO expects RGB by default; ultralytics can handle BGR, but convert to be safe
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    results = model.predict(frame_rgb, verbose=False, conf=conf_threshold)
    if not results:
        return False
    r0 = results[0]
    # For segmentation models, boxes may be present; check conf scores
    try:
        if r0.boxes is None or r0.boxes.data is None:
            return False
        # Any box above conf threshold counts as detection
        return len(r0.boxes) > 0
    except Exception:
        return False


@click.command()
@click.option("--input-dir", "input_dir", "-i", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help="Directory containing input videos")
@click.option("--output-dir", "output_dir", "-o", type=click.Path(file_okay=False, path_type=Path), required=True, help="Directory to save extracted frames")
@click.option("--ext", "img_ext", default="jpg", show_default=True, help="Image extension for saved frames")
@click.option("--threshold", "threshold", default=None, type=float, help="Confidence threshold override (defaults to config)")
@click.option("--limit", "limit", default=None, type=int, help="Optional max number of videos to process")
@click.option("--rotation", "manual_rotation", default=None, type=click.Choice(['0', '90', '180', '270']), help="Manual rotation override (0, 90, 180, 270 degrees)")
def main(input_dir: Path, output_dir: Path, img_ext: str, threshold: float | None, limit: int | None, manual_rotation: str | None):
    """Extract full-frame images from videos when rousettus segmentation detects anything."""
    # Load configuration and model
    cfg = load_config()
    seg_cfg = cfg.models.segmentation
    if not seg_cfg:
        click.echo("No segmentation model configured in app/config/config.yml", err=True)
        sys.exit(1)

    conf_threshold = threshold if threshold is not None else seg_cfg.confidence_threshold

    try:
        model = create_yolo_model(seg_cfg)
    except Exception as e:
        click.echo(f"Failed loading segmentation model: {e}", err=True)
        sys.exit(1)

    ensure_dir(output_dir)

    # Get list of video files to process
    video_files = [f for f in sorted(input_dir.iterdir()) 
                   if f.is_file() and f.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v"}]
    
    if not video_files:
        click.echo("No video files found in input directory")
        return
    
    total_videos = len(video_files) if limit is None else min(len(video_files), limit)
    click.echo(f"Found {len(video_files)} video files, processing {total_videos}...")
    click.echo(f"Using confidence threshold: {conf_threshold}")
    click.echo(f"Output directory: {output_dir}")
    click.echo("-" * 60)

    processed = 0
    total_frames_saved = 0
    
    # Iterate over video files
    for entry in video_files:
        if limit is not None and processed >= limit:
            break
            
        type_str, bat_name, vid_name = parse_video_name(entry.name)
        if not all([type_str, bat_name, vid_name]):
            click.echo(f"⚠️  Skipping file with unexpected name format: {entry.name}")
            continue

        click.echo(f"📹 Processing video {processed + 1}/{total_videos}: {entry.name}")
        
        # Detect video rotation
        if manual_rotation is not None:
            rotation = int(manual_rotation)
            click.echo(f"   🔄 Using manual rotation: {rotation}°")
        else:
            rotation = get_video_rotation(str(entry))
            if rotation != 0:
                click.echo(f"   🔄 Detected rotation: {rotation}°")
            else:
                click.echo(f"   📐 No rotation detected (video appears to be in correct orientation)")
        
        cap = cv2.VideoCapture(str(entry))
        if not cap.isOpened():
            click.echo(f"❌ Failed to open video: {entry}", err=True)
            continue

        # Get video info for progress tracking
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        # Get frame dimensions
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        click.echo(f"   📊 Video info: {total_frames} frames, {fps:.1f} FPS, {duration:.1f}s duration")
        click.echo(f"   📐 Frame dimensions: {frame_width}x{frame_height}")

        video_frames_saved = 0
        frame_count = 0
        last_progress_frame = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                frame_count += 1
                
                # Show progress every 10% or every 100 frames, whichever is more frequent
                progress_interval = max(100, total_frames // 10)
                if frame_number - last_progress_frame >= progress_interval:
                    progress_pct = (frame_number / total_frames) * 100 if total_frames > 0 else 0
                    click.echo(f"   🔄 Processing frame {frame_number}/{total_frames} ({progress_pct:.1f}%)")
                    last_progress_frame = frame_number

                # Apply rotation if needed
                corrected_frame = rotate_frame(frame, rotation)
                
                detected = has_segmentation_detection(model, corrected_frame, conf_threshold)
                if not detected:
                    continue

                # Build filename: <type>--<bat_name>--VID_<vid_name>.<frame_number>.<img_extension>
                out_name = f"{type_str}--{bat_name}--VID_{vid_name}.{frame_number}.{img_ext}"
                out_path = output_dir / out_name
                # Save full frame once per detected frame (with rotation correction)
                cv2.imwrite(str(out_path), corrected_frame)
                video_frames_saved += 1
                total_frames_saved += 1
                
        finally:
            cap.release()

        click.echo(f"   ✅ Video complete: {video_frames_saved} frames saved")
        processed += 1

    click.echo("-" * 60)
    click.echo(f"🎉 Done! Processed {processed} videos, saved {total_frames_saved} frames total")


if __name__ == "__main__":
    main()


