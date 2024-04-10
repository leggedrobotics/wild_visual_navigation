# Wild Visual Navigation Sim

Simulation environment to test Wild Visual Navigation (WVN). We use a modified Clearpath Jackal (adding a camera).

## Requirements

```sh
wget https://packages.clearpathrobotics.com/public.key -O - | sudo apt-key add -
sudo sh -c 'echo "deb https://packages.clearpathrobotics.com/stable/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/clearpath-latest.list'
sudo apt-get update
```

```sh
sudo apt update 
sudo apt install -y \
        ros-noetic-jackal-simulator \
        ros-noetic-jackal-desktop
```

## Running

```sh
roslaunch wild_visual_navigation_jackal sim.launch
```

```sh
roslaunch wild_visual_navigation_jackal wild_visual_navigation.launch
```

## License
The tree assets is by `samuelabyan`, obtained from [TurboSquid](https://www.turbosquid.com/3d-models/free-3ds-model-tree/905801)