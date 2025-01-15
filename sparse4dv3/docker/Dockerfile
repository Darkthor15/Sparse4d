# Install ROS2 and Necessary python libraries for Sparse4Dv3
FROM dustynv/l4t-pytorch:r35.3.1

ENV DEBIAN_FRONTEND=noninteractive
RUN export LC_ALL=en_US.UTF-8

# dependent libraries
RUN apt-get update && apt-get install -y --no-install-recommends apt-transport-https apt-utils \
    sudo vim git wget build-essential curl ffmpeg libsm6 libxext6 locales \
    python3 python3-dev python3-distutils python3-pip software-properties-common

# python libraries
RUN pip install mmcv-full==1.7.1 --no-cache-dir
RUN pip install --upgrade numpy==1.21.6
RUN pip install matplotlib==3.5.2 scipy==1.10.1 mmdet==2.28.2 pyquaternion==0.9.9 \
    tensorboard==2.14.0 motmetrics==1.1.3 pandas==1.1.5 pypcd_imp==0.1.5 \
    kornia==0.6.10 scikit-learn==1.2.1 ordered-set==4.1.0 rosbags \
    descartes==1.1.0 fire==0.6.0 Shapely==1.8.5 tqdm==4.66.5 IPython
RUN pip install nuscenes-devkit==1.1.10 --no-deps

# sklearn and nsight systems path exports
RUN echo 'export LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0' >> ~/.bashrc
RUN echo 'export PATH=/opt/nvidia/nsight-systems/2023.2.4/bin:/opt/nvidia/nsight-systems/2023.2.4/target-linux-tegra-armv8:/usr/src/tensorrt/bin:${PATH}' >> ~/.bashrc

# ROS Galactic install
RUN apt-get update && locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG en_US.UTF-8 && export LANG=en_US.UTF-8
RUN add-apt-repository universe && apt-get update
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null
RUN apt-get update && apt-get install -y ros-galactic-ros-base ros-galactic-foxglove-msgs python3-colcon-common-extensions
RUN apt-get update && apt-get install -y ros-galactic-rviz2

# source ROS Galactic
RUN echo 'source /opt/ros/galactic/setup.bash' >> ~/.bashrc && \
    echo "source /data/sparse4dv3/install/setup.bash" >> ~/.bashrc
