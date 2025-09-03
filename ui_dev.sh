#!/bin/zsh

source .env.dev
export PYTHONPATH=$PWD:$PYTHONPATH
streamlit run ui/webserver.py --server.port=8080 --server.address=0.0.0.0
