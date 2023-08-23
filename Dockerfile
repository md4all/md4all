FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV TZ=Europe
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y ffmpeg=7:4.2.7-0ubuntu0.1 libsm6=2:1.2.3-1 libxext6=2:1.3.4-0ubuntu1 python3-pip=20.0.2-5ubuntu1.7 git=1:2.25.1-1ubuntu3.8

RUN pip3 --no-cache-dir install torch==1.13.1 torchvision==0.14.1 torchaudio==1.13.1 --extra-index-url https://download.pytorch.org/whl/cu113

RUN pip --no-cache-dir install \
pytorch-lightning==1.9.0 \
fvcore==0.1.5.post20221221 \
yacs==0.1.8 \
matplotlib==3.6.3 \
opencv-python==4.7.0.68 \
nuscenes-devkit==1.1.9 \
pandas==1.5.3 \
kornia==0.6.9 \
tensorboard==2.11.2

RUN pip --no-cache-dir install git+https://github.com/morbi25/robotcar-dataset-sdk.git

RUN ln -s /usr/bin/python3 /usr/bin/python

