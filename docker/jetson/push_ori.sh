#!/bin/bash
# This is an Oxford Robotics Isntitute (ORI)-specific executable
# It makes tags that are compliant with the internal naming for the images
# and pushes the images to our local server

# ==
# ML + pytorch geometric
# ==

ORIGINAL_TAG=rslethz/jetpack-5:r34.1.1-ml-pyg
ORI_TAG=ori-ci-gateway.robots.ox.ac.uk:12002/drs/jetpack-5:r34.1.1-ml-pyg-latest

# Create new tags to comply with ORI naming
echo "Making new tag $ORI_TAG"
docker tag $ORIGINAL_TAG $ORI_TAG

# Push to ORI server
echo "Pushing $ORI_TAG to server..."
docker push $ORI_TAG

# Remove original tag
echo "Removing tag $ORIGINAL_TAGE"
docker image rm $ORIGINAL_TAG

# ==
# Wild Visual Navigation image
# ==

ORIGINAL_TAG=rslethz/jetpack-5:r34.1.1-wvn
ORI_TAG=ori-ci-gateway.robots.ox.ac.uk:12002/drs/jetpack-5:r34.1.1-wvn-latest

# Create new tags to comply with ORI naming
echo "Making new tag $ORI_TAG"
docker tag $ORIGINAL_TAG $ORI_TAG

# Push to ORI server
echo "Pushing $ORI_TAG to server..."
docker push $ORI_TAG

# Remove original tag
echo "Removing tag $ORIGINAL_TAGE"
docker image rm $ORIGINAL_TAG
