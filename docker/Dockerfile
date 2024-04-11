#####################
# Base image #
#####################
FROM nvidia/cuda:12.3.2-runtime-ubuntu20.04 as base

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
# Select shell
# ==
SHELL ["/bin/bash", "-c"]

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
        python3-venv \
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
RUN cd /root \
    && python3 -m venv env --system-site-packages \
    && source /root/env/bin/activate \
    && pip install --upgrade pip \
    && pip3 install --no-cache-dir \
         torch==2.1.0 torchvision --extra-index-url https://download.pytorch.org/whl/cu121 \
    && pip3 install --no-cache-dir \
         black \
         flake8 \
         jupyter \
         wget \
         numpy \
         tqdm \
         kornia \
         torchmetrics \
         pytorch_lightning \
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
         pydensecrf@git+https://github.com/lucasb-eyer/pydensecrf.git \
         liegroups@git+https://github.com/mmattamala/liegroups \
         opencv-python

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
WORKDIR /root/catkin_ws


###############################################################
# Development image
# This adds display access and simulation support
###############################################################

FROM base as dev

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /root/catkin_ws/src \
    && source /opt/ros/noetic/setup.bash \
    && source "/root/.bashrc" \
    && cd /root/catkin_ws && catkin build \
    && apt-get update \
    && apt-get install -y \
        ros-noetic-jackal-simulator \
        ros-noetic-jackal-desktop \
        ros-noetic-teleop-twist-keyboard \
        ros-noetic-rqt-robot-steering \
    && rm -rf /var/lib/apt/lists/*

RUN echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc
RUN echo "source /root/env/bin/activate" >> ~/.bashrc

COPY docker/first_run.sh first_run.sh
ENV DEBIAN_FRONTEND=dialog