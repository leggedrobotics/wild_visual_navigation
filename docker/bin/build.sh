#!/bin/bash
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

# Handle different target cases
if [[ "$TARGET" == "jetson" ]]; then
    echo "Building images for target [$TARGET]"
    
    # Build pytorch geometric docker
    echo "Building ${PYG_TAG} from $PYG_DOCKERFILE"
    sudo docker build --build-arg BASE_IMAGE=$ML_JETSON_TAG --network=host -t $PYG_JETSON_TAG -f $PYG_DOCKERFILE .

    # Build wvn docker
    echo "Building ${WVN_TAG} from $WVN_DOCKERFILE"
    sudo docker build --build-arg BASE_IMAGE=$PYG_JETSON_TAG --network=host -t $WVN_JETSON_TAG -f $WVN_DOCKERFILE .

elif [[ "$TARGET" == "desktop" ]]; then
    echo "Building images for target [$TARGET]"

		# Build wvn docker
    echo "Building ${WVN_DESKTOP_TAG} from $WVN_DOCKERFILE"
    sudo docker build --build-arg BASE_IMAGE=$ML_DESKTOP_TAG --network=host -t $WVN_DESKTOP_TAG -f $WVN_DOCKERFILE .

else
    echo "Error: unsupported target [$TARGET]"
fi