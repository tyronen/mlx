#!/bin/bash

# To be run on UI server

docker-compose down
docker-compose pull
docker-compose up -d
