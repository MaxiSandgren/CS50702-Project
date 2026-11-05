FROM python:3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir vamp-planner