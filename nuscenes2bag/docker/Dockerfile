FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN export LC_ALL=en_US.UTF-8

# dependent libraries
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    apt-transport-https apt-utils sudo vim wget build-essential \
    git ffmpeg libsm6 libxext6 libgeos-dev\
    python3-pip python3-tf2-ros ros-noetic-foxglove-msgs

# python libraries
RUN pip install numpy==1.21.6 python-dateutil==2.8.2 matplotlib==3.5.2 \
    opencv-python==4.9.0.80 opencv-contrib-python==4.9.0.80 Shapely==1.8.5 \
    nuscenes-devkit==1.1.10 pypcd_imp==0.1.5 seaborn notebook

