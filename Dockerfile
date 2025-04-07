FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set non-interactive mode for apt-get and configure timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Kolkata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies including build tools
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda and create the environment
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create a conda environment with Python 3.8 (per guidelines)
RUN conda create -n latentgraph python=3.8 -y

# Install PyTorch 2.1.0 with CUDA 11.8 support
RUN conda install -n latentgraph pytorch==2.1.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Use the latentgraph environment for subsequent commands
SHELL ["conda", "run", "-n", "latentgraph", "/bin/bash", "-c"]

WORKDIR /workspace
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Install additional Python dependencies:
# - torch-scatter (for PyTorch 2.1.0 + CUDA 11.8)
# - openmim, mmdetection (3.2.0), mmengine (0.7.4), and other dependencies.
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch2.1.0+cu118.html && \
    pip install -U openmim && \
    mim install mmdet==3.2.0 && \
    mim install mmengine==0.7.4 && \
    pip install torchmetrics scikit-learn prettytable imagesize networkx opencv-python yapf==0.40.1 && \
    pip install ipdb && \
    pip install dgl -f https://data.dgl.ai/wheels/torch-2.1/cu118/repo.html

# Install mmcv (version 2.1.0 compatible with mmdetection 3.2.0)
RUN pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html

CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate latentgraph && exec bash"]
