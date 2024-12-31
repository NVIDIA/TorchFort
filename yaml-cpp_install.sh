#!/bin/bash

git clone https://github.com/jbeder/yaml-cpp.git

cd yaml-cpp

mkdir build

cd build

cmake -DCMAKE_CXX_FLAGS:="-D_GLIBCXX_USE_CXX11_ABI=0" \
-DBUILD_SHARED_LIBS=OFF \
-DCMAKE_INSTALL_PREFIX=$HOME/yaml-cpp-def \
..

cmake --build . --parallel 4

cmake --install .
