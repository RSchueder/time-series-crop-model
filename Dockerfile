FROM python:3.11-slim as base

ARG ARM_TORCH_VERSION="2.5.1"
WORKDIR /code
ENV PYTHONPATH /code

RUN apt-get update && \
    apt-get -y install \
    curl \ 
    gdal-bin \
    git \
    htop \
    libgdal-dev \
    python-is-python3 \
    python3-dev \
    python3-opencv \
    python3-pip \
    python3-shapely \
    wget \
    unzip && \
    apt-get clean

COPY requirements.txt /code
COPY constraints.txt /code
RUN pip install -U pip && pip install -r requirements.txt -c constraints.txt \
    && pip install torch==${ARM_TORCH_VERSION} --no-cache-dir -c constraints.txt \
    && pip install GDAL==3.6.2 --no-cache-dir -c constraints.txt