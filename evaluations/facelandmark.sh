#!/bin/bash

# Directory containing the videos
VIDEO_DIR="/home/host/evl/ER-NeRF/whisper"
# CONTAINER_ID=$(docker run -d algebr/openface:latest)
CONTAINER_ID=e604790827a8
# Loop through all the videos in the directory
for video in "$VIDEO_DIR"/*.mp4; do
  video_name=$(basename "$video")
  # Copy the video into the Docker container
  docker cp "$video" "$CONTAINER_ID":/home/openface-build/
  
  # Process the video with OpenFace
  docker exec "$CONTAINER_ID" build/bin/FaceLandmarkVidMulti -f "/home/openface-build/$video_name"   
  # Copy the output CSV file back to the local machine
  output_csv="${video_name%.mp4}.csv"
  docker cp "$CONTAINER_ID":/home/openface-build/processed/"$output_csv" "$VIDEO_DIR"/
  
  echo "Processed $video_name and saved output to $VIDEO_DIR/$output_csv"
done

docker stop $CONTAINER_ID
