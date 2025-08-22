#!/usr/bin/env python3
import os
import sys
import argparse
from typing import List


def should_strip(filename: str, dirname: str, seps: List[str]) -> str | None:
    for sep in seps:
        prefix = f"{dirname}{sep}"
        if filename.startswith(prefix):
            return filename[len(prefix) :]
    if filename.startswith(dirname):
        return filename[len(dirname) :]
    return None


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("root", help="Root directory to process")
    parser.add_argument(
        "--seps",
        default=",",
        help="Separators between dirname and filename prefix (comma-separated). Default: '.'",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Only print planned changes without renaming"
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories (process all nested dirs)",
    )
    parser.add_argument(
        "--include-exts",
        default="",
        help="Optional comma-separated list of file extensions to limit processing (e.g. 'png,jpg')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite if target filename already exists",
    )
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"Invalid directory: {root}")
        sys.exit(1)

    seps = [s for s in args.seps.split(",") if s] if args.seps else ["."]
    if not seps:
        seps = ["."]

    allowed_exts = set()
    if args.include_exts:
        allowed_exts = {("." + e.lower().lstrip(".")) for e in args.include_exts.split(",") if e}

    planned = []

    def process_dir(d: str):
        dirname = os.path.basename(d)
        try:
            for entry in os.scandir(d):
                if entry.is_file():
                    if allowed_exts and os.path.splitext(entry.name)[1].lower() not in allowed_exts:
                        continue
                    new_name = should_strip(entry.name, dirname, seps)
                    if new_name and new_name != entry.name:
                        src = entry.path
                        dst = os.path.join(d, new_name)
                        planned.append((src, dst))
                elif entry.is_dir() and args.recursive:
                    process_dir(entry.path)
        except PermissionError:
            print(f"Skipping (permission denied): {d}")

    if args.recursive:
        for dirpath, dirnames, filenames in os.walk(root):
            # Only process files directly in dirpath; recursion handled by os.walk
            dirname = os.path.basename(dirpath)
            for fname in filenames:
                if allowed_exts and os.path.splitext(fname)[1].lower() not in allowed_exts:
                    continue
                new_name = should_strip(fname, dirname, seps)
                if new_name and new_name != fname:
                    src = os.path.join(dirpath, fname)
                    dst = os.path.join(dirpath, new_name)
                    planned.append((src, dst))
    else:
        process_dir(root)

    if not planned:
        print("No files require renaming.")
        return

    print(f"Planned renames: {len(planned)}")
    for src, dst in planned:
        print(f"{src} -> {os.path.basename(dst)}")

    if args.dry_run:
        print("Dry run complete.")
        return

    success = 0
    skipped = 0
    failed = 0

    for src, dst in planned:
        if os.path.exists(dst) and not args.overwrite:
            print(f"Skip (exists): {dst}")
            skipped += 1
            continue
        try:
            os.replace(src, dst) if args.overwrite else os.rename(src, dst)
            success += 1
        except Exception as e:
            print(f"Failed: {src} -> {dst} ({e})")
            failed += 1

    print(f"Done. Renamed: {success}, Skipped: {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
