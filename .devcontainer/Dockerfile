FROM python:3.12-bookworm

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential cmake libeigen3-dev git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# VAMP source code. build_fcit.sh compiles it together with our fcit.hh.
RUN git clone --depth 1 --branch v0.6.4 https://github.com/KavrakiLab/vamp.git /opt/vamp-src
