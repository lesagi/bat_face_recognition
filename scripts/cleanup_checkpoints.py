#!/usr/bin/env python3
"""
Cleanup periodic checkpoints from model run directories to recover disk space.

Keeps: best_model_f1/, best_model_loss/, final model, evaluation outputs,
       saliency maps, config_snapshot.yml, training.log
Deletes: checkpoints/ directory (periodic checkpoints only useful during active training)

Usage:
    python scripts/cleanup_checkpoints.py --dry-run          # Preview what would be deleted
    python scripts/cleanup_checkpoints.py --confirm           # Actually delete
    python scripts/cleanup_checkpoints.py --model-dir /path   # Specific directory
"""

import argparse
import os
import shutil
import sys


def get_dir_size(path):
    """Get total size of a directory in bytes."""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total


def format_size(bytes_val):
    """Format bytes into human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


def find_checkpoint_dirs(base_dir):
    """Find all checkpoints/ directories within model run directories."""
    results = []
    if not os.path.isdir(base_dir):
        return results

    for run_dir_name in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, run_dir_name)
        if not os.path.isdir(run_dir):
            continue

        checkpoint_dir = os.path.join(run_dir, "checkpoints")
        if os.path.isdir(checkpoint_dir):
            size = get_dir_size(checkpoint_dir)
            results.append({
                "run_dir": run_dir_name,
                "checkpoint_path": checkpoint_dir,
                "size": size,
            })

    return results


def main():
    parser = argparse.ArgumentParser(description="Clean up model checkpoints to recover disk space")
    parser.add_argument("--model-dir", type=str,
                        default="/home/sagilevi1/bat_face_rec_project/models/siamese",
                        help="Base directory containing model run directories")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete checkpoint directories (default is dry-run)")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Show what would be deleted without deleting (default)")
    args = parser.parse_args()

    if args.confirm:
        args.dry_run = False

    # Search all species subdirs
    base = args.model_dir
    all_checkpoints = []

    if os.path.isdir(base):
        for species_dir_name in sorted(os.listdir(base)):
            species_dir = os.path.join(base, species_dir_name)
            if os.path.isdir(species_dir):
                checkpoints = find_checkpoint_dirs(species_dir)
                for cp in checkpoints:
                    cp["species"] = species_dir_name
                all_checkpoints.extend(checkpoints)

    if not all_checkpoints:
        print("No checkpoint directories found.")
        return

    total_size = sum(cp["size"] for cp in all_checkpoints)

    print(f"\n{'='*70}")
    print(f"  Checkpoint Cleanup {'(DRY RUN)' if args.dry_run else '(DELETING)'}")
    print(f"{'='*70}\n")

    for cp in all_checkpoints:
        status = "WOULD DELETE" if args.dry_run else "DELETING"
        print(f"  [{status}] {cp['species']}/{cp['run_dir']}/checkpoints/")
        print(f"           Size: {format_size(cp['size'])}")

    print(f"\n{'─'*70}")
    print(f"  Total: {len(all_checkpoints)} checkpoint directories")
    print(f"  Space to recover: {format_size(total_size)}")
    print(f"{'─'*70}\n")

    if args.dry_run:
        print("  Run with --confirm to actually delete these directories.")
        return

    # Confirm before deleting
    answer = input("  Are you sure you want to delete these directories? [y/N]: ").strip().lower()
    if answer != 'y':
        print("  Aborted.")
        return

    deleted = 0
    freed = 0
    for cp in all_checkpoints:
        try:
            shutil.rmtree(cp["checkpoint_path"])
            deleted += 1
            freed += cp["size"]
            print(f"  Deleted: {cp['species']}/{cp['run_dir']}/checkpoints/")
        except Exception as e:
            print(f"  Error deleting {cp['checkpoint_path']}: {e}")

    print(f"\n  Done. Deleted {deleted} directories, freed {format_size(freed)}.")


if __name__ == "__main__":
    main()
