#Name: Dockerfile
#Version: 1.0
#Summary: Docker recipe file 
#Author: suxing liu
#Author-email: suxingliu@gmail.com
#Created: 2024-10-29

#USAGE:
#docker build -t test_docker -f Dockerfile .
#docker run -v /input_path:/srv/images -it test_docker
#cd /opt/code/
#python3 /opt/code/ai_seg.py -i /input/ -o /output/

FROM ubuntu:22.04

LABEL maintainer='Suxing Liu, Wes Bonelli'

COPY ./ /opt/code


RUN apt-get update && apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" TZ="America/New_York" apt-get install -y \
    build-essential \
    python3-setuptools \
    python3-pip \
    python3 \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    cmake-gui \
    nano \
    libdmtx0b

RUN python3 -m pip install --upgrade pip

RUN pip3 install numpy \
    Pillow \
    scipy \
    scikit-image==0.19.3 \
    scikit-learn\
    matplotlib \
    pandas \
    pytest \
    opencv-python-headless \
    openpyxl \
    imutils \
    numba \
    skan \
    tabulate \
    pylibdmtx \
    psutil \
    natsort \
    pathlib \
    kmeans1d \
    onnxruntime \
    rembg




RUN chmod -R a+rwx /opt/code/

WORKDIR /opt/code/




