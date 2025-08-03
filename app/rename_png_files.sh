#!/bin/bash

# Check if the user provided a directory
if [ -z "$1" ]; then
  echo "Usage: $0 directory"
  exit 1
fi

TARGET_DIR=$1

# Loop through all .jpg and .png files in the specified directory and its subdirectories
find "$TARGET_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | while read -r file; do
  # Check if the file name contains .mp4
  if [[ "$file" == *".mp4"* ]]; then
    # Remove .mp4 from the file name
    new_file="${file//.mp4/}"
    # Log the renaming process
    echo "Renaming '$file' to '$new_file'"
    # Rename the file
    mv -v "$file" "$new_file"
  fi
done

echo "Renaming of .jpg and .png files containing .mp4 in their names is complete."

