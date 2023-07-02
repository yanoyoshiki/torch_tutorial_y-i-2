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
        vim \
	ipdb \



# python libraries
RUN pip install notebook \\
		matplotlib\\
		numpy\\
		scipy\\
 		torch\\ 
		torchvision\\ 
		torchaudio\\	
		gym\\ 
		tensorboard\\

ARG uname
ARG password
ARG uid
ARG gid
RUN groupadd -g $gid $uname
RUN useradd -s /bin/bash -u $uid -g $gid -m $uname && echo "$uname:$password" | chpasswd && adduser $uname sudo
RUN apt-get update -qq && apt-get -y install sudo && rm -rf /var/lib/apt/lists/*

WORKDIR /home/$uname
USER $uname

