#!/bin/bash

for video in /home/jorge/rumiaifinal/temp/*.mp4; do
    if [ -f "$video" ]; then
        duration=$(ffmpeg -i "$video" 2>&1 | grep "Duration" | cut -d ' ' -f 4 | cut -d ',' -f 1)
        size=$(ls -lh "$video" | awk '{print $5}')
        basename=$(basename "$video")
        echo "$basename | Duration: $duration | Size: $size"
    fi
done