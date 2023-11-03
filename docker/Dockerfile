FROM nvidia/cuda:11.8.0-runtime-ubuntu20.04

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
         torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117

RUN pip3 install --no-cache-dir \
         jupyter \
         ruamel.yaml \
         black \
         flake8 \
         shapely==1.7.1 \
         pillow==9.2 \
         wget \
         colorama \
         simple-parsing \
         kornia \
         pytest \
         scipy \
         scikit-image \
         scikit-learn \
         seaborn \
         pandas \
         fast_slic \
         neptune-client[optuna] \
         wandb \
         wget \
         hydra-core \
         pytorch_lightning==1.6.5 \
         colorama \
         fast_slic \
         git+https://github.com/lucasb-eyer/pydensecrf.git#egg=pydensecrf \
         git+https://github.com/kornia/kornia#egg=kornia \
         git+https://github.com/mmattamala/liegroups#egg=liegroups \
         git+https://github.com/leggedrobotics/stego.git#egg=stego==0.0.1 \
         pytictac \
         prettytable \
         termcolor

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