#!/bin/bash

set -euxo pipefail

INPUT_FILE=${1:-data/video/sample.mp4}
OUTPUT_DIR=${2:-output}
FORCE="no"
OFFSET_SECONDS=0
EVERY_N_SECONDS=5

# Get duration in seconds
duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE")

# Convert to integer
duration=${duration%.*}

for ((i=OFFSET_SECONDS; i<=duration; i+=EVERY_N_SECONDS)); do
    time=$(printf "%02d:%02d:%02d" $((i/3600)) $(( (i/60)%60)) $((i%60)) )
    # Replace the colons with hyphens
    time_safe=${time//:/-}
    filename="$OUTPUT_DIR/img-$time_safe.jpg"

    if [ "$FORCE" == "yes" ] || [ ! -f "$filename" ]; then
      ffmpeg -y -ss "$time" -i "$INPUT_FILE" -vframes 1 "$filename"
    fi
done
