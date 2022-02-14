#!/bin/bash

DEBUG="OFF"
PROFILE="OFF"
TARGET_ARCH=aarch64

CC=`which clang`
CXX=`which clang++`

if [ -z $TNN_ROOT_PATH ]
then
    TNN_ROOT_PATH=$(cd `dirname $0`; pwd)/..
fi

TORCHVISION_ENABLE="OFF"
PYBIND_ENABLE="OFF"

export LIBTORCH_ROOT_DIR=/opt/homebrew/lib/python3.9/site-packages/torch

# export LIBTORCHVISION_ROOT_DIR=`find /usr/local/ -name "libtorchvision*-0.9.1+*"`

BUILD_DIR=${TNN_ROOT_PATH}/scripts/build_tnntorch_aarch64_macos
TNN_INSTALL_DIR=${TNN_ROOT_PATH}/scripts/tnntorch_aarch64_macos_release

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
    -DCMAKE_SYSTEM_PROCESSOR=$TARGET_ARCH \
    -DDEBUG:BOOL=$DEBUG \
    -DTNN_PROFILER_ENABLE=$PROFILE \
    -DTNN_TEST_ENABLE=ON \
    -DTNN_CPU_ENABLE=ON \
    -DTNN_ARM_ENABLE=ON \
    -DTNN_ARM82_ENABLE=ON \
    -DTNN_TNNTORCH_ENABLE=ON \
    -DTNN_TORCHVISION_ENABLE=${TORCHVISION_ENABLE} \
    -DTNN_PYBIND_ENABLE=${PYBIND_ENABLE} \
    -DTNN_GLIBCXX_USE_CXX11_ABI_ENABLE=OFF \
    -DTNN_BENCHMARK_MODE=OFF \
    -DTNN_BUILD_SHARED=ON \
    -DTNN_CONVERTER_ENABLE=OFF \
    -DTNN_TIACC_MODE=ON

echo Building TNN ...
make -j6

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
