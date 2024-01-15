# Wild Visual Navigation Sim

Simulation environment to test Wild Visual Navigation (WVN). We use a modified Clearpath Jackal (adding a camera).

## Requirements

```sh
$ sudo apt update 
$ sudo apt install -y \
        ros-noetic-jackal-simulator \
        ros-noetic-jackal-desktop \
        ros-noetic-teleop-twist-keyboard \
        ros-noetic-rqt-robot-steering \
```

## Running

```sh
$ roslaunch wild_visual_navigation_sim sim.launch
```
