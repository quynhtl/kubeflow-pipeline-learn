#!/bin/sh

image_name=$DOCKER_REGISTRY/load_data
image_tag=latest

full_image_name=${image_name}:${image_tag}
base_image_tag=3.6.15-slim

# den file sh
cd "$(dirname "$0")"

docker build --build-arg BASE_IMAGE_TAG=${base_image_tag} -t "${full_image_name}" .
docker push "$full_image_name"