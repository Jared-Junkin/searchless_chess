# Use the NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=10000

# Update CA Certificates
COPY ./JHUAPL-MS-Root-CA-05-21-2038-B64-text.crt /usr/local/share/ca-certificates/JHUAPL-MS-Root-CA-05-21-2038-B64-text.crt
RUN update-ca-certificates

# Environment variables for SSL certificates
ENV PIP_CERT=/etc/ssl/certs/ca-certificates.crt
ENV REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

# Remove the NVIDIA repository to avoid the missing Release file error
RUN sed -i '/^deb.*nvidia\.com\/compute\/cuda/d' /etc/apt/sources.list

# Install Python 3.10 and necessary system packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3.10-distutils \
    build-essential \
    libfreetype6-dev \
    libxft-dev \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*
    

# Install pip for Python 3.10 and upgrade it
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3.10 get-pip.py && rm get-pip.py
RUN pip3 install --upgrade pip

# Ensure python3 points to python3.10
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Ensure 'python' points to 'python3'
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install PyTorch with CUDA 11.8 support
RUN pip3 install torch==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118 --default-timeout=100000
# Install the main packages without jax and jaxlib
RUN pip3 install --default-timeout=100000 \
    transformer_lens==1.10.0 \
    tqdm==4.66.1 \
    jaxtyping==0.2.25 \
    beartype==0.14.1 \
    wandb==0.16.0 \
    fancy_einsum==0.0.3 \
    einops==0.7.0 \
    numpy==1.26.0 \
    python-chess==1.999 \
    pandas==2.1.1 \
    plotly==5.18.0 \
    matplotlib==3.8.0 \
    nbformat==5.9.2 \
    pytest==8.1.1 \
    openai==0.28.0 \
    tiktoken==0.4.0 \
    tenacity==8.2.3 \
    absl-py \
    apache-beam \
    chess \
    chex \
    dm-haiku \
    grain-nightly \
    jupyter \
    optax \
    orbax-checkpoint \
    scipy \
    typing-extensions

# Install JAX and jaxlib with CUDA 11.8 and cuDNN 8 support directly from the JAX project
RUN pip3 install --upgrade --default-timeout=100000 \
    jax==0.4.16 \
    jaxlib==0.4.16+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Copy and install Python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install -r requirements.txt

# Set the working directory
WORKDIR /workspace


# Default command
CMD ["/bin/bash"]

# To mount the current directory: docker run -it -v ${PWD}:/workspace cpp_image
