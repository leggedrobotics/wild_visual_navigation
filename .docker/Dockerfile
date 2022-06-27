FROM pytorchlightning/pytorch_lightning:base-conda-py3.8-torch1.10

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

LABEL maintainer="Jonas Frey" mail="jonfrey@ethz.ch"

SHELL ["/bin/bash", "--login", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ENV DEBIAN_frontend noninteractive

RUN mkdir /home/git
RUN mkdir /home/results
RUN mkdir /home/code
RUN mkdir /home/dataset
RUN mkdir /home/work
RUN mkdir /home/tmpdir