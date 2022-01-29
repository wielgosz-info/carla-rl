ARG PLATFORM=nvidia
FROM wielgoszinfo/pedestrians-common:${PLATFORM}-latest AS base

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

# Copy the 'agents' package used by scenario runner. It is a part of provided PythonAPI, but not a part of carla package.
COPY --from=carlasim/carla:0.9.13 --chown=${USERNAME}:${USERNAME} /home/carla/PythonAPI/carla/agents /venv/lib/python3.8/site-packages/agents

# Install direct dependencies and scenario_runner ones
RUN /venv/bin/python -m pip install --no-cache-dir \
    carla==0.9.13

# Add requirements.txt first for caching purposes.
COPY requirements.txt /app
RUN /venv/bin/python -m pip install -r /app/requirements.txt  --no-cache-dir

ENTRYPOINT [ "/bin/sh", "-c", "while sleep 1000; do :; done" ]