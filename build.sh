#!/bin/zsh

if [[ "$1" == "inference" ]]; then
  # exit if no second arg and no VERSION defined
  if [[ -z "$2" && -z "$VERSION" ]]; then
    echo "Error: must supply a version number as arg2 or set \$VERSION"
    exit 1
  fi

  # pick version from arg2 if present, otherwise from $VERSION
  ver="${2:-$VERSION}"

  docker build --platform linux/amd64 \
    -t docker.io/$DOCKER_USERNAME/inference:v$ver \
    -f inference/Dockerfile .
fi

if [[ "$1" == "ui" ]]; then
  docker build --platform linux/amd64,linux/arm64 -t docker.io/$DOCKER_USERNAME/ui:latest -f ui/Dockerfile .
fi
