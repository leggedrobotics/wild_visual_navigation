#!/bin/bash

# Docker images
PYG_DOCKERFILE=Dockerfile.pyg
WVN_DOCKERFILE=Dockerfile.wvn

# Source ML images (with Pytorch support)
ML_JETSON_TAG="rslethz/jetpack-5:r34.1.1-ml"
ML_DESKTOP_TAG="nvcr.io/nvidia/pytorch:20.06-py3"

# Export tags
PYG_JETSON_TAG="rslethz/jetpack-5:r34.1.1-ml-pyg"
WVN_JETSON_TAG="rslethz/jetpack-5:r34.1.1-wvn"
WVN_DESKTOP_TAG="rslethz/desktop:r34.1.1-wvn"

# ORI-compliant tags for built images
ORI_PYG_ORI_JETSON_TAG="ori-ci-gateway.robots.ox.ac.uk:12002/drs/jetson:r34.1.1-ml-pyg-latest"
ORI_WVN_JETSON_TAG="ori-ci-gateway.robots.ox.ac.uk:12002/drs/jetson:r34.1.1-wvn-latest"
ORI_WVN_DESKTOP_TAG="ori-ci-gateway.robots.ox.ac.uk:12002/drs/desktop:wvn-latest"