#!/bin/zsh

source .env.dev
streamlit run ui/webserver.py --server.port=8080 --server.address=0.0.0.0
