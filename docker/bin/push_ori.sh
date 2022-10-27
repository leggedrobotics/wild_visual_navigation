#!/bin/bash
# This is an Oxford Robotics Institute (ORI)-specific executable
# It makes tags that are compliant with the internal naming for the images
# and pushes the images to our local server

set -e

# Source variables
source bin/env_variables.sh

# Default target
TARGET=none

# Read arguments
for i in "$@"
do
case $i in
  -t=*|--target=*)
    TARGET=${i#*=}
    echo "[build.sh]: User-set target type is: '$TARGET'"
    shift
    ;;
esac
done

update_tags_and_push()
{
    local original_tag=$1
    local new_tag=$2

    # Build pytorch geometric docker
    echo "Making new tag $new_tag"
    docker tag $original_tag $new_tag
    echo "Pushing $new_tag to server..."
    docker push $nw_tag
    echo "Removing tag $original_tag"
    docker image rm $original_tag
}

# Handle different target cases
if [[ "$TARGET" == "jetson" ]]; then
    update_tags_and_push $PYG_JETSON_TAG $ORI_PYG_JETSON_TAG
    update_tags_and_push $WVN_JETSON_TAG $ORI_WVN_JETSON_TAG

elif [[ "$TARGET" == "desktop" ]]; then
    update_tags_and_push $WVN_DESKTOP_TAG $ORI_WVN_DESKTOP_TAG

else
    echo "Error: unsupported target [$TARGET]"
fi





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
