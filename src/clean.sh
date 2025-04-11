#!/bin/bash

# Path to the target directory
TARGET_DIR="../dataset"

# Check if the directory exists
if [ -d "$TARGET_DIR" ]; then
    rm -v "$TARGET_DIR"/*/*_compress_*.txt
    rm -v "$TARGET_DIR"/*/*_level_*.txt
else
    echo "Directory $TARGET_DIR does not exist."
    exit 1
fi
