#!/bin/bash

docker run \
 --net=host \
 --gpus all\
 -v /tmp/.X11-unix:/tmp/.X11-unix \
 -v /etc/localtime:/etc/localtime:ro \
 -v $PWD/../1:/root/1 \
 -v $PWD/../2:/root/2 \
 -v $PWD/../3:/root/3 \
 -e DISPLAY=$DISPLAY \
 -e NVIDIA_VISIBLE_DEVICES=all \
 -e NVIDIA_DRIVER_CAPABILITIES=all \
 -it --name "torch_tutorial_y-i" torch_tutorial_y-i \


