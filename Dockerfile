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
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh
ENV PATH=/opt/conda/bin:$PATH

# Create a conda environment with Python 3.8
RUN conda create -n latentgraph python=3.9 -y

# Install PyTorch with CUDA 11.8 support and DGL
RUN conda install -n latentgraph pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y && \
    conda install -n latentgraph -c dglteam/label/cu113 dgl -y

# Use the latentgraph environment for subsequent commands
SHELL ["conda", "run", "-n", "latentgraph", "/bin/bash", "-c"]

# Set a working directory (this will be used as the mount point)
WORKDIR /workspace

# Add the project directory to PYTHONPATH (so your code can be imported)
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Install additional Python dependencies
RUN pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu117.html && \
    pip install -U openmim && \
    mim install mmdet && \
    mim install mmengine==0.7.4 && \
    pip install torchmetrics scikit-learn prettytable imagesize networkx opencv-python yapf==0.40.1 && \
    pip install ipdb

# Set default command to launch bash when the container starts
CMD ["bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate latentgraph && exec bash"]

