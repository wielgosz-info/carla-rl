ARG PLATFORM=nvidia
FROM wielgoszinfo/carla-common:${PLATFORM}-latest AS base

ENV torch_version=1.9.1
ENV torchvision_version=0.10.1

# ----------------------------------------------------------------------------
# NVIDIA-specific dependencies
# ----------------------------------------------------------------------------
FROM base as torch-nvidia

RUN /venv/bin/python -m pip install --no-cache-dir -f https://download.pytorch.org/whl/torch_stable.html \
    torch==${torch_version}+cu111 \
    torchvision==${torchvision_version}+cu111

# ----------------------------------------------------------------------------
# CPU-specific dependencies
# ----------------------------------------------------------------------------
FROM base as torch-cpu

RUN /venv/bin/python -m pip install --no-cache-dir -f https://download.pytorch.org/whl/cpu/torch_stable.html \
    torch==${torch_version}+cpu \
    torchvision==${torchvision_version}+cpu

# -------------------------------------------------------------------------------------------------
# Common
# -------------------------------------------------------------------------------------------------
FROM torch-${PLATFORM} as torch

# Add requirements.txt first for caching purposes.
COPY requirements.txt /app
RUN /venv/bin/python -m pip install -r /app/requirements.txt  --no-cache-dir
