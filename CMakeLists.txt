cmake_minimum_required(VERSION 3.0)

#project(argus CXX CUDA)
project(argus CXX)

find_package(OpenCV REQUIRED)

add_executable(
    argus
    main.cpp)

target_link_libraries(argus PUBLIC opencv_core opencv_imgcodecs)

