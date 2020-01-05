
cmake_minimum_required(VERSION 3.0)

#project(argus CXX CUDA)
project(argus CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)
find_package(TBB REQUIRED)

add_subdirectory(app)
add_subdirectory(src)

