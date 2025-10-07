#!/bin/bash

# To be run on UI server

DOCKER="docker compose --env-file .env.prod"

$DOCKER down
$DOCKER pull
$DOCKER up -d
