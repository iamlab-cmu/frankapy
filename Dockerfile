FROM osrf/ros:noetic-desktop-full

# Install apt packages
RUN apt update && apt install -y git nano
RUN apt install ros-noetic-tf2-tools -y
RUN apt install python3-pip python3-tk -y

# install python dependencies to run frankapy within the docker container
RUN pip3 install autolab_core 
RUN pip3 install --force-reinstall pillow==9.0.1 && pip3 install --force-reinstall scipy==1.8
RUN pip3 install numpy-quaternion numba && pip3 install --upgrade google-api-python-client 
RUN pip3 install --force-reinstall numpy==1.23.5

# Install moveit and franka_ros
RUN apt install ros-noetic-moveit ros-noetic-franka-ros  -y

# Make src/git_packages and clone panda_moveit_config
RUN mkdir -p /home/ros_ws/src/git_packages && \
    cd /home/ros_ws/src/git_packages && \
    git clone https://github.com/ros-planning/panda_moveit_config.git -b noetic-devel

# Copy the frankapy folder into the container
COPY frankapy /home/ros_ws/src/git_packages/frankapy/frankapy
COPY catkin_ws /home/ros_ws/src/git_packages/frankapy/catkin_ws

# Copy src folder from desktop to /home/ros_ws/src/devel_packages (this is where we will put our custom packages)
COPY moveit_frankapy/src/devel_packages /home/ros_ws/src/devel_packages

# Install dependencies using rosdep
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; cd /home/ros_ws; rosdep install --from-paths src --ignore-src -r -y"

# add ros workspace to bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
RUN echo "source /home/ros_ws/devel/setup.bash" >> ~/.bashrc

# build workspace
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash; cd /home/ros_ws; catkin_make"

# set workdir as home/ros_ws
WORKDIR /home/ros_ws

CMD [ "bash" ]