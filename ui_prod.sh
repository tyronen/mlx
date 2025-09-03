#!/bin/bash

# To be run on UI server

docker rm -f ui
docker run \
    -d \
    --name ui \
    -p 80:80 \
    --env-file ~/.env.prod \
    --restart unless-stopped \
    --pull=always \
    docker.io/tyronen24/ui:latest
