FROM nvidia/cuda:11.1.1-cudnn8-runtime-ubuntu20.04

ENV TZ=Europe/Warsaw
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
    libboost-python-dev \
    libjpeg-dev \
    libjpeg-turbo8-dev \
    libpng-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html --no-cache

# Let's pretend we've installed CARLA via easy_install
# It's client for Python 3.7 and in Ubuntu 20.04 there's Python 3.8 but hopefully this will work
COPY --from=carlasim/carla:0.9.11 /home/carla/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg /usr/local/lib/python3.8/dist-packages/carla-0.9.11-py3.7-linux-x86_64.egg
RUN echo "import sys; sys.__plen = len(sys.path)\n./carla-0.9.11-py3.7-linux-x86_64.egg\nimport sys; new=sys.path[sys.__plen:]; del sys.path[sys.__plen:]; p=getattr(sys,'__egginsert',0); sys.path[p:p]=new; sys.__egginsert = p+len(new)" > /usr/local/lib/python3.8/dist-packages/easy_install.pth

# Add requirements.txt first for caching purposes.
WORKDIR /app
COPY requirements.txt /app
RUN pip install -r requirements.txt

# Run infinite loop to allow easily attach to container
CMD ["/bin/sh", "-c", "while sleep 1000; do :; done"]
