#!/bin/bash

# Check if the user provided a directory
if [ -z "$1" ]; then
  echo "Usage: $0 directory"
  exit 1
fi

TARGET_DIR=$1

# Find and delete all PDF files in the specified directory and its subdirectories
find "$TARGET_DIR" -type f -name "*.pdf" -exec rm -v {} \;

echo "All PDF files in $TARGET_DIR and its subdirectories have been deleted."

