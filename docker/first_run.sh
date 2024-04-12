#!/bin/bash

echo "pip3 install -e /root/catkin_ws/src/self_supervised_segmentation ..."
pip3 install -e /root/catkin_ws/src/self_supervised_segmentation > /dev/null

echo "pip3 install -e /root/catkin_ws/src/wild_visual_navigation ..."
pip3 install -e /root/catkin_ws/src/wild_visual_navigation > /dev/null

echo "downloading STEGO pretrained weights to self_supervised_segmentation/models/"
./src/self_supervised_segmentation/models/download_pretrained.sh

echo "catkin build ..."
catkin build > /dev/null

echo "source devel/setup.bash ..."
source devel/setup.bash > /dev/null

echo "Setup ready!"