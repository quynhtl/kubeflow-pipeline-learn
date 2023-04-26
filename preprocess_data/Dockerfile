ARG BASE_IMAGE_TAG=3.6.15-slim
FROM python:${BASE_IMAGE_TAG}

RUN python3 -m pip install --no-cache-dir pandas joblib scikit-learn

COPY ./src /pipelines/component/src

