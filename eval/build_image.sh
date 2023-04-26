#!/bin/sh

image_name=$DOCKER_REGISTRY/eval
image_tag=dev

full_image_name=${image_name}:${image_tag}
base_image_tag=1.15.2-py3

# den file sh
cd "$(dirname "$0")"

docker build --build-arg BASE_IMAGE_TAG=${base_image_tag} -t "${full_image_name}" .
docker push "$full_image_name"