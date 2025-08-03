#!/bin/bash

# Function to print all folders in a directory
print_folders_in_directory() {
    local directory=$1

    # Check if the directory exists
    if [ ! -d "$directory" ]; then
        echo "The directory '$directory' does not exist."
        return
    fi

    # Iterate over all entries in the directory
    for entry in "$directory"/*; 
    do
        # Check if entry is a directory
        if [ -d "$entry" ]; then
            # Print the directory name (basename removes the path)
            echo "$(basename "$entry")"
        fi
    done
}

# Specify the directory you want to check as the first argument to the script
directory_to_check=$1

# Call the function
print_folders_in_directory "$directory_to_check"

