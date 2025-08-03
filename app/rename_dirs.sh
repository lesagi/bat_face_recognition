#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Renaming directories to replace spaces with underscores...${NC}"

# Function to rename a directory
rename_dir() {
    local old_name="$1"
    local new_name="$2"
    if [ -d "$old_name" ]; then
        echo -e "${YELLOW}Renaming '$old_name' to '$new_name'...${NC}"
        mv "$old_name" "$new_name"
    else
        echo -e "${RED}Directory '$old_name' not found${NC}"
    fi
}

# Get the current directory
CURRENT_DIR=$(pwd)

# Find all directories with spaces in the current directory and its subdirectories
find . -type d -name "* *" | while read -r dir; do
    # Get the directory name and its parent path
    dir_name=$(basename "$dir")
    parent_path=$(dirname "$dir")
    
    # Create new name by replacing spaces with underscores
    new_name="${dir_name// /_}"
    
    # Rename the directory
    echo -e "${YELLOW}Renaming '$dir' to '$parent_path/$new_name'...${NC}"
    mv "$dir" "$parent_path/$new_name"
done

echo -e "${GREEN}Directory renaming completed!${NC}"
echo -e "${YELLOW}Current path: $(pwd)${NC}" 