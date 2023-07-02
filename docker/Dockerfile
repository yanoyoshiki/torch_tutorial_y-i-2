FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

LABEL authors="Y-Yano"

# Install python3 pip3
RUN apt -y update && apt -y upgrade
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN pip3 install --upgrade pip

# Other install
RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        git \
        vim 



# python libraries
RUN pip install notebook \
		matplotlib \
		numpy \
		scipy \
 		gym \ 
		tensorboard \
		ipdb


RUN mkdir 1 2 3
