#!/bin/bash
# Compile our fcit.hh into local_fcit.abi3.so. Run again after every change to fcit.hh.
# The build is tuned to this computer's CPU, so everyone builds their own copy.
set -e
cd "$(dirname "$0")"
python -m pip install --no-cache-dir --force-reinstall --no-deps /opt/vamp-src \
    -Ccmake.define.CMAKE_PROJECT_vamp_INCLUDE="$PWD/local_fcit.cmake"
python -c "import local_fcit; print('local_fcit is built and ready')"
