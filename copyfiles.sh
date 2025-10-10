#!/bin/bash

REMOTE="mlx"

function send() {
    rsync -vrt "$1" "$REMOTE:/workspace/mlx/$2"
}

send build_env.sh
send api
send common
send models
send training
send ui
send data/*.pth data/
send data/*.pt data/
send data/*.json data/
send data/hn_predict data/
send data/msmarco_search data/
send data/*.rdb data/