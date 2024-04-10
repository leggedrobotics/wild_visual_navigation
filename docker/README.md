# Jackal Simulation Demo

We provide an example package to demonstrate Wild Visual Navigation on a simulated environment with a Clearpath Jackal robot.

This example should be self-contained and it should run on a Docker container. This was tested on Ubuntu machines, we do not expect the GUI to run on Windows or Mac computers (due to X11 support).


## Build the image

To build the container:

```sh
docker compose -f docker-compose-gui-nvidia.yaml build
```

## Run the simulation environment in the container

Start the container in detached mode:

```sh
docker compose -f docker-compose-gui-nvidia.yaml up -d
```

Launch a first bash terminal in the container to start the simulation environment:

```sh
docker compose -f docker-compose-gui-nvidia.yaml exec wvn_nvidia /bin/bash
```

Once in the container, run the `first_run` script to install the WVN package that is mounted automatically when the container starts:
```sh
source first_run.sh
```

Launch the Gazebo simulation environment and an RViz window as the main interface.

```sh
roslaunch wild_visual_navigation_jackal sim.launch 
```
If this doesn't open any window, pelase check the troubleshooting section below.


Open a new terminal to launch WVN in the container

```sh
docker compose -f docker-compose-gui-nvidia.yaml exec wvn_nvidia /bin/bash
```

And then, once in the container:
```sh
roslaunch wild_visual_navigation_jackal wild_visual_navigation.launch
```

You can drive the Jackal robot by sending 2D Nav goals using RViz. We implemented a simple [carrot follower](../wild_visual_navigation_jackal/scripts/carrot_follower.py) that was tuned for the demo (not actually used in real experiments)


## Stop the example

Kill all the terminal as usual (Ctrl + D). Then, stop the container using:

```sh
docker compose -f docker-compose-gui-nvidia.yaml down
```


## Troubleshooting

If the GUI doesn't work, you'll see an error like:

> No protocol specified
> qt.qpa.xcb: could not connect to display :1

To fix it, you might need to allow the X Server to connect before running the container:

```sh
xhost +Local:*
xhost
```
