
cmake_minimum_required(VERSION 3.0)

#project(argus CXX CUDA)
project(argus CXX)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core)

#pkg_check_modules(tbb REQUIRED IMPORTED_TARGET tbb)

add_subdirectory(app)
add_subdirectory(src)

