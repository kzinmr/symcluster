FROM ubuntu:focal-20220105

WORKDIR /app

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

ENV APP_ROOT /app
ENV WORK_DIR /app/workspace
ENV DEBIAN_FRONTEND noninteractive

RUN mkdir -p $APP_ROOT
WORKDIR $APP_ROOT

RUN ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime
RUN apt-get update && apt-get install -y \
    build-essential \
    sudo \
    cmake \
    file \
    unzip \
    gcc \
    g++ \
    python3-pip \
    python3-scipy python3-numpy \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install -U pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py /app/main.py
CMD ["python3", "/app/main.py"]