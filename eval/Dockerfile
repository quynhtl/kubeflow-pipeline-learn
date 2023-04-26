ARG BASE_IMAGE_TAG=1.15.2-py3
FROM tensorflow/tensorflow:$BASE_IMAGE_TAG
RUN python3 -m pip install --no-cache-dir numpy pandas
COPY ./src /pipelines/component/src