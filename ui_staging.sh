#!/bin/bash

# To be run on dev machine

docker rm -f ui
docker run \
    -d \
    --name ui \
    -p 8000:8000 \
    --env-file .env.dev \
    --pull=always \
    docker.io/tyronen24/ui:latest
