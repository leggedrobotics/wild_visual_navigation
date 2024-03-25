# Jackal Simulation Demo

We provide an example package to demonstrate Wild Visual Navigation on a simulated environment with a Clearpath Jackal robot.

This example should be self-contained and it should run on a Docker container. This was tested on Ubuntu machines, we do not expect the GUI to run on Windows or Mac computers (due to X11 support).


## Build the image

To build the container:

```sh
docker compose -f docker-compose-gui-nvidia.yaml build
```

## Run the container

To run the container (terminal-based):

```sh
docker compose -f docker-compose-gui-nvidia.yaml up -d
```

To launch bash on the container:

```sh
docker compose exec wvn bash
```

## Stop the container

To stop the container:

```sh
docker compose -f docker-compose.yaml stop
```

## Running Wild Visual Navigation

You can either run the following commands in 4 terminals that initialize a bash terminal in the container, or you can use VS Code with the Docker extension to instantiate terminal in the container directly.

### Launch Jackal sim

```sh
roslaunch wild_visual_navigation_jackal sim.launch 
```

### Launch WVN

```sh
roslaunch wild_visual_navigation_jackal wild_visual_navigation.launch
```

### Launch Teleop node

```sh
roslaunch wild_visual_navigation_jackal teleop.launch 
```

### Launch RViz window

```sh
roslaunch wild_visual_navigation_jackal view.launch 
```


## Troubleshooting

If the GUI doesn't work, you might need to allow the X Server to connect before running the container:

```sh
xhost +Local:*
xhost
```
