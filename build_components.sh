#!/bin/sh

export DOCKER_DEFAULT_PLATFORM=linux/amd64

echo "\nBuild and push download data component"
./download_data/build_image.sh

echo "\nBuild and push load data component"
./load_data/build_image.sh

echo "\nBuild and push preprocess data component"
./preprocess_data/build_image.sh

echo "\nBuild and push train component"
./train/build_image.sh

echo "\nBuild and push evaluate component"
./eval/build_image.sh