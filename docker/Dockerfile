FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

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
RUN pip3 install notebook==6.5.4 \
		matplotlib==3.7.2 \
		numpy==1.22.3 \
		scipy==1.11.1 \
		tensorboard==2.13.0 \
		optuna==3.2.0 \
		ipywidgets==8.0.7 \
		pytorch_lightning==2.0.5 \
		tensorflow==2.13.0
#COPY requirements.txt .
#RUN pip install -r requirements.txt


