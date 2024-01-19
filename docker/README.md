# Docker files

## Build and run containers

To build the container:

```sh
docker compose -f docker-compose.yaml build
```

To run the container (terminal-based):

```sh
docker compose -f docker-compose.yaml up -d
```

To launch bash on the container:

```sh
docker compose exec wvn bash
```

To stop the container:

```sh
docker compose -f docker-compose.yaml stop
```

To run the GUI-enabled version and check the Gazebo environment:

```sh
docker compose -f docker-compose-gui.yaml up -d
docker compose exec wvn_gui bash
```
