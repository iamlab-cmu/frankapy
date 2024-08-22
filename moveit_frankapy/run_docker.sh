xhost +local:root 
docker container prune -f 
docker run --privileged --rm -it \
    --name="moveit_frankapy" \
    --env="DISPLAY=$DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --volume="$XAUTH:$XAUTH" \
    --network host \
    -v $(pwd)/src/devel_packages:/home/ros_ws/src/devel_packages \
    moveit_frankapy \
    bash

# NOTE: --network host is used to allow the container to access the host's network
