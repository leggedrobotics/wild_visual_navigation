# Docker files

## Build and run containers

To build and run the standard container (terminal-based):

```sh
docker compose -f docker-compose.yaml up -d
docker compose exec wvn bash
```

To run the GUI-enabled version and check the Gazebo environment:

```sh
docker compose -f docker-compose-gui.yaml up -d
docker compose exec wvn_gui bash
```
