FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 as base

RUN apt update
RUN apt install python3 python3-dev python3-pip -y
RUN apt install sudo -y
RUN adduser --quiet --disabled-password --shell /bin/bash --home /home/ubuntu --gecos "User" ubuntu
RUN usermod -aG sudo ubuntu
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
USER ubuntu
WORKDIR /home/ubuntu

RUN sudo apt install openssh-server sshpass openssh-client -y

RUN sudo apt install ninja-build -y
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu118
ADD requirements.txt .
RUN pip3 install -r requirements.txt

RUN mkdir app
RUN pip3 install fastapi sse-starlette uvicorn
RUN sudo apt update
RUN sudo apt install git -y
RUN FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE pip3 install flash-attn --no-build-isolation

RUN pip3 install hf-transfer

ENV PYTHONPATH "${PYTHONPATH}:/home/ubuntu/app"
COPY ./app/ /home/ubuntu/app
RUN ls app