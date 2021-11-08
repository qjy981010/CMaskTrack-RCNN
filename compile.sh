#!/usr/bin/env bash

PYTHON='python'
# export CUDA_HOME='/usr/local/cuda-10.0/'
# export PATH=/usr/local/cuda-10.0/bin:$PATH
# export LD_LIBRARY_PATH="/usr/local/cuda-10.0/lib64":$LD_LIBRARY_PATH

cd mmdet/ops/roi_align

echo "Building roi align op..."
cd ../roi_align
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building roi pool op..."
cd ../roi_pool
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building nms op..."
cd ../nms
make clean
make PYTHON=${PYTHON}

echo "Building dconv op..."
cd ../dcn
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace

echo "Building correlation op..."
cd ../correlation
if [ -d "build" ]; then
    rm -r build
fi
$PYTHON setup.py build_ext --inplace
