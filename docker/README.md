## Docker build instructions
### General overview
This folder stores basic Dockerfiles and scripts to build the dependencies of the wild_visual_navigation package for (1) desktop machines and (2) Jetson platforms.

The main difference between both are the source images. While the one for desktop/laptops rely on NVidia's images with Pytorch support `nvcr.io/nvidia/pytorch:20.06-py3`, the Jetson ones are more involved and rely on internal images prepared by RSL. Additionally, Jetson requires to build pytorch geometric (`pyg`) from scratch, so we also add Dockerfiles to build an intermediate image for this purpose only.

### Usage
While staying in the docker folder, you can build the images for desktop platforms (`rslethz/desktop:r34.1.1-wvn`) by using:

```sh
./bin/build.sh --target=desktop
```

Similarly, for Jetson ones you must use:
```sh
./bin/build.sh --target=jetson
```
The Jetson version will generate the images `rslethz/jetpack-5:r34.1.1-ml-pyg` and `rslethz/jetpack-5:r34.1.1-wvn`, with pytorch geometric and full wild_visual_navigation dependencies respectively.

### Pushing images to ORI server
For internal use at the Oxford Robotics Institute (ORI), we also provide a helper script to upload the images to the docker server. For example, after building the images you can run:

```sh
./bin/push_ori.sh --target=jetson
```

This will create new tags for the previously mentioned images:
```sh
rslethz/jetpack-5:r34.1.1-ml-pyg -> ori-ci-gateway.robots.ox.ac.uk:12002/drs/jetson:r34.1.1-ml-pyg-latest
rslethz/jetpack-5:r34.1.1-wvn -> ori-ci-gateway.robots.ox.ac.uk:12002/drs/jetson:r34.1.1-wvn-latest
```
which comply with the naming used internally at ORI.