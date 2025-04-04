# Use a base image with CUDA 11.8 support (Ubuntu 20.04)
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set non-interactive mode for apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create a conda environment with Python 3.8
RUN conda create -n latentgraph python=3.8 -y

# Install PyTorch with CUDA 11.8 support and DGL
RUN conda install -n latentgraph pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    conda install -n latentgraph -c dglteam/label/cu113 dgl -y

# Use the latentgraph environment for subsequent commands
SHELL ["conda", "run", "-n", "latentgraph", "/bin/bash", "-c"]

# Clone mmdetection and set MMDETECTION environment variable
RUN cd $HOME && git clone https://github.com/open-mmlab/mmdetection.git && \
    export MMDETECTION=$HOME/mmdetection

# Clone SurgLatentGraph repository
RUN cd $HOME && git clone https://github.com/Valientever/SurgLatentGraph.git

# Set working directory to SurgLatentGraph
WORKDIR $HOME/SurgLatentGraph

# Download pretrained weights
RUN cd weights && \
    wget -O coco_init_wts.zip "https://seafile.unistra.fr/f/71eedc8ce9b44708ab01/?dl=1" && \
    unzip coco_init_wts.zip && \
    cd ..

# Add SurgLatentGraph to PYTHONPATH
ENV PYTHONPATH="$PYTHONPATH:$HOME/SurgLatentGraph"

# Install additional Python dependencies
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html && \
    pip install -U openmim && \
    mim install mmdet && \
    mim install mmengine==0.7.4 && \
    pip install torchmetrics scikit-learn prettytable imagesize networkx opencv-python yapf==0.40.1

# Set default command to launch bash when the container starts
CMD ["bash"]
