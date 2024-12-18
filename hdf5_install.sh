#! /bin/bash

cmake -DCMAKE_INSTALL_PREFIX=/home/tartarughina/hdf5-def \
-G "Unix Makefiles" \
-DCMAKE_BUILD_TYPE:STRING=Release \
-DBUILD_SHARED_LIBS:BOOL=ON \
-DBUILD_TESTING:BOOL=OFF \
-DHDF5_BUILD_TOOLS:BOOL=OFF \
-DHDF5_BUILD_FORTRAN=ON \
-DHDF5_ENABLE_PARALLEL=ON \
..

cmake --build . --config Release

cpack -C Release CPackConfig.cmake

./HDF5-x.x.x-Linux.sh
