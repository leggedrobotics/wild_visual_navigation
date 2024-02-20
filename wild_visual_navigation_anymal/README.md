Setup on ANYmal:

NPC:
```
sudo apt-get install ros-noetic-anymal-msgs-dev
catkin build wild_visual_navigation_anymal

catkin build wild_visual_navigation_anymal --cmake-args -DBUILD_ANYMAL=1
```

Jetson:
```
```

Currently the resizing is not working which is pretty bad for the wide angle camera.