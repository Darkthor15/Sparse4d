#!/bin/bash

check_file_exists() {
    if [ ! -d "$1" ]; then
        echo "Error: ROSBAG '$1' does not exist."
        exit 1
    fi
}

echo "Starting Sparse4Dv3 Node"
ros2 launch launch/sparse_node_launch.py &
sleep 20

for bagfile in "$@"; do
    check_file_exists "$bagfile"
    echo "Playing ROSBAG: $bagfile"
    ros2 bag play -r 0.3 "$bagfile"
    sleep 5
done

echo "All ROSBAGS played."