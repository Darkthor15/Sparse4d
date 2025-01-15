#!/bin/bash

if [ ! -d "$1" ]; then
    echo "Usage: $0 /path/to/NuScenes/ROS1_BAG_FOLDER/"
    exit 1
fi

DIR="$1"

for file in "$DIR"/*.bag; do
    if [ ! -e "$file" ]; then
        echo "No .bag files found in the directory."
        exit 0
    fi
    filename=$(basename "$file" .bag)

    echo "Converting $filename"

    output_file="data/${filename}"

    rosbags-convert --dst "$output_file" "$file"

    if [ $? -eq 0 ]; then
        echo "Succesfully converted and saved to $output_file"
    else
        echo "Failed to convert $file"
    fi
done

echo "Conversion complete"