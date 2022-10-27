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
    docker push $new_tag
    #echo "Removing tag $original_tag"
    #docker image rm $original_tag
}

# Handle different target cases
if [[ "$TARGET" == "jetson" ]]; then
    update_tags_and_push "$PYG_JETSON_TAG" "$ORI_PYG_JETSON_TAG"
    update_tags_and_push "$WVN_JETSON_TAG" "$ORI_WVN_JETSON_TAG"

elif [[ "$TARGET" == "desktop" ]]; then
    update_tags_and_push $WVN_DESKTOP_TAG $ORI_WVN_DESKTOP_TAG

else
    echo "Error: unsupported target [$TARGET]"
fi
