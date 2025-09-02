#!/bin/zsh

function run_serverless() {
  docker run --rm -it \
    --platform linux/amd64 \
    -v "$(PWD)/inference/test_$1.json:/app/test_input.json:ro" \
    -v "$(PWD)/data:/data" \
    -e SQLITE_PATH=/data/preds.db \
    $DOCKER_USERNAME/inference:v$VERSION
}

run_serverless predict
run_serverless submit
run_serverless list
