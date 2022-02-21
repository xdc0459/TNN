#!/bin/bash

DEBUG="OFF"
PROFILE="OFF"
OPENMP="OFF"

CC=`which clang`
CXX=`which clang++`

if [ $OPENMP == "ON" ]; then
    # export LIBOMP_ROOT_DIR=/opt/homebrew/opt/libomp
    export LIBOMP_ROOT_DIR=/Users/jacinhu/Library/Logs/Homebrew/libomp
fi

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

TORCHVISION_ENABLE="OFF"
PYBIND_ENABLE="OFF"

# export LIBTORCH_ROOT_DIR=/usr/local/lib/python3.7/site-packages/torch
export LIBTORCH_ROOT_DIR=/usr/local/lib/python3.7/site-packages/torch

# export LIBTORCHVISION_ROOT_DIR=`find /usr/local/ -name "libtorchvision*-0.9.1+*"`

BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_tnntorch_x86_macos
TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/tnntorch_x86_macos_release

TNN_VERSION_PATH=$TNN_ROOT_PATH/scripts/version
cd $TNN_VERSION_PATH
source $TNN_VERSION_PATH/version.sh
source $TNN_VERSION_PATH/add_version_attr.sh

# rm -rf ${BUILD_DIR}
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

cmake ${TNN_ROOT_PATH} \
    -DCMAKE_C_COMPILER=$CC \
    -DCMAKE_CXX_COMPILER=$CXX \
    -DDEBUG:BOOL=$DEBUG \
    -DTNN_PROFILER_ENABLE=$PROFILE \
    -DTNN_OPENMP_ENABLE:BOOL=$OPENMP \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_X86_ENABLE=ON \
    -DTNN_TNNTORCH_ENABLE=ON \
    -DTNN_TORCHVISION_ENABLE=${TORCHVISION_ENABLE} \
    -DTNN_PYBIND_ENABLE=${PYBIND_ENABLE} \
    -DTNN_GLIBCXX_USE_CXX11_ABI_ENABLE=OFF \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF \
    -DTNN_TIACC_MODE=ON \
    -DTNN_APPLE_NPU_ENABLE=ON


echo Building TNN ...
make -j12

if [ -d ${TNN_INSTALL_DIR} ]
then 
    rm -rf ${TNN_INSTALL_DIR}
fi

mkdir -p ${TNN_INSTALL_DIR}
mkdir -p ${TNN_INSTALL_DIR}/lib

cp -r ${TNN_ROOT_PATH}/include ${TNN_INSTALL_DIR}/
cp -R libTNN.* ${TNN_INSTALL_DIR}/lib/

if [ "$PYBIND_ENABLE" = "ON" ]; then
    cp -d _pytnn.*.so ${TNN_INSTALL_DIR}/lib/
    cp ${TNN_ROOT_PATH}/source/pytnn/*.py ${TNN_INSTALL_DIR}/lib/
fi

echo Done
