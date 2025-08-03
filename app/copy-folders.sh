#!/bin/bash

# Check if the correct number of arguments are provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source_directory> <destination_directory>"
  exit 1
fi

# Source and destination directories from arguments
SOURCE_DIR="$1"
DEST_DIR="$2"

# Create destination directory if it does not exist
mkdir -p "$DEST_DIR"

# Loop through each item in the source directory
for dir in "$SOURCE_DIR"/*; do
  # Check if it is a directory
  if [ -d "$dir" ]; then
    # Get the directory name
    dir_name=$(basename "$dir")
    # Create the directory in the destination
    mkdir -p "$DEST_DIR/$dir_name"
  fi
done

echo "Folders copied without their content."

