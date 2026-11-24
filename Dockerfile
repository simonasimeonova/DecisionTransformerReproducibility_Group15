FROM python:3.8.20-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_PY_MUJOCO_PATH=/opt/mujoco/mujoco200_linux \
    MUJOCO_PY_MJKEY_PATH=/opt/mujoco/mujoco200_linux/bin/mjkey.txt \
    MUJOCO_PY_FORCE_CPU=1 \
    LD_LIBRARY_PATH=/opt/mujoco/mujoco200_linux/bin:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    OMP_NUM_THREADS=1 \
    WANDB_MODE=offline

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl ca-certificates unzip patchelf \
      libosmesa6-dev libgl1-mesa-dev libglew-dev libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# MuJoCo must be installed before mujoco-py.
RUN mkdir -p /opt/mujoco \
    && curl -fL --retry 3 https://www.roboti.us/download/mujoco200_linux.zip -o /tmp/mujoco.zip \
    && unzip -q /tmp/mujoco.zip -d /opt/mujoco \
    && curl -fL --retry 3 https://www.roboti.us/file/mjkey.txt -o "$MUJOCO_PY_MJKEY_PATH" \
    && rm /tmp/mujoco.zip

# Install the old build tools first; mujoco-py uses the pinned NumPy/Cython.
RUN python -m pip install --no-cache-dir \
      pip==24.0 setuptools==65.5.0 wheel==0.38.4 \
    && python -m pip install --no-cache-dir \
      numpy==1.20.3 Cython==0.29.36 scipy==1.7.3 \
      cffi==1.17.1 imageio==2.31.6 glfw==2.7.0 fasteners==0.19 \
    && python -m pip install --no-cache-dir --no-build-isolation \
      gym==0.18.3 mujoco-py==2.0.2.13

RUN python -m pip install --no-cache-dir torch==1.8.1+cu111 \
      -f https://download.pytorch.org/whl/torch_stable.html \
    && python -m pip install --no-cache-dir \
      transformers==4.5.1 packaging==21.3 \
      wandb==0.9.1 click==7.1.2 protobuf==3.20.3

ARG SOURCE_COMMIT=unknown
RUN printf '%s\n' "$SOURCE_COMMIT" > /opt/source-commit.txt
COPY gym /opt/decision-transformer/gym
COPY docker/train.sh /usr/local/bin/train
COPY docker/smoke.py /opt/decision-transformer/smoke.py
RUN chmod 755 /usr/local/bin/train \
    && python -m pip check \
    && chmod -R a+rwX /usr/local/lib/python3.8/site-packages/mujoco_py/generated

# Renku jobs run as a regular user and write under /home/renku/work.
RUN groupadd --gid 1000 renku \
    && useradd --uid 1000 --gid 1000 --create-home renku \
    && mkdir -p /home/renku/work \
    && chown -R renku:renku /home/renku
ENV HOME=/home/renku
USER 1000:1000
WORKDIR /home/renku/work
RUN python /opt/decision-transformer/smoke.py
CMD ["/usr/local/bin/train"]
