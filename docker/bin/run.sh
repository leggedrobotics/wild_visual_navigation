#!/bin/bash
# This script launches a WVN container for the target platform

set -e

# Source variables
source bin/env_variables.sh

# Default target
TARGET=none
GIT_WS=$HOME/git
CATKIN_WS=$HOME/catkin_ws

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


# Handle different target cases
if [[ "$TARGET" == "jetson" ]]; then
    docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v ${GIT_WS}:/root/git -v ${CATKIN_WS}:/root/catkin_ws/ $WVN_JETSON_TAG

elif [[ "$TARGET" == "desktop" ]]; then
    docker run -it --rm --net=host --runtime nvidia -e DISPLAY=$DISPLAY -v /tmp/.X11-unix/:/tmp/.X11-unix -v ${GIT_WS}:/root/git -v ${CATKIN_WS}:/root/catkin_ws/ $WVN_DESKTOP_TAG

else
    echo "Error: unsupported target [$TARGET]"
fi


