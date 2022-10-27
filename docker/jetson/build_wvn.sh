#!/bin/bash

# Exit if error occurs
set -e

# Set arguments
TARGET_IMAGE="rslethz/jetpack-5:r34.1.1-wvn"
DOCKERFILE="Dockerfile.wvn"

echo "building ${TARGET_IMAGE} dependencies"
sudo docker build --network=host -t "$TARGET_IMAGE" -f "$DOCKERFILE" .
echo "done"
