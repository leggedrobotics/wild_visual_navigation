#####################
# Base image #
#####################
FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04 as base

# Labels
LABEL maintainer="Matias Mattamala;Jonas Frey"
LABEL contact="matias@robots.ox.ac.uk;jonfrey@ethz.ch"
LABEL description="WVN Docker"

ARG ROS_VERSION="noetic"

# ==
# Disable dialog frontend
# ==
ARG DEBIAN_FRONTEND=noninteractive

# ==
# Install ROS
# ==
RUN apt update \
    && apt install --no-install-recommends -y curl gnupg lsb-release git build-essential nano \
    && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
    && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - \
    && apt update \
    && apt install --no-install-recommends -y ros-${ROS_VERSION}-desktop-full \
    && apt install --no-install-recommends -y \
        python3-pip \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        python3-catkin-tools \
        python3-osrf-pycommon \
    && rm -f "/etc/ros/rosdep/sources.list.d/20-default.list" \
    && rosdep init \
    && rosdep update \
    && echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# ==
# Install python packages
# ==
RUN pip3 install --no-cache-dir \
         torch==2.1.0+cu118 torchvision --extra-index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
         black \
         flake8 \
         jupyter \
         wget \
         numpy \
         tqdm \
         kornia>=0.6.5 \
         torchmetrics \
         pytorch_lightning==1.6.5 \
         pytest \
         scipy \
         scikit-image \
         scikit-learn \
         matplotlib \
         seaborn \
         pandas \
         pytictac \
         torch_geometric \
         omegaconf \
         optuna \
         neptune \
         fast-slic \
         hydra-core \
         prettytable \
         termcolor \
         pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git#egg=pydensecrf \
         liegroups@git+https://github.com/mmattamala/liegroups#egg=liegroups \
         stego@git+https://github.com/leggedrobotics/stego.git#egg=stego==0.0.1 \
         opencv-python>=4.6

# ==
# Remove cache and extra files
# ==
RUN rm -rf /var/lib/apt/lists/* && apt clean


# ==
# Enable dialog again
# ==
ARG DEBIAN_FRONTEND=dialog

# ==
# Run bash
# ==
CMD ["/bin/bash"]
WORKDIR /root


#####################
# Development image #
#####################

FROM base as dev

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y \
        ros-noetic-jackal-simulator \
        ros-noetic-jackal-desktop \
        ros-noetic-teleop-twist-keyboard \
        ros-noetic-rqt-robot-steering \
    && rm -rf /var/lib/apt/lists/*

ENV DEBIAN_FRONTEND=dialog