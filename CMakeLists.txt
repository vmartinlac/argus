cmake_minimum_required(VERSION 3.0)

#project(argus CXX CUDA)
project(argus CXX)

find_package(OpenCV REQUIRED)
find_package(Threads REQUIRED)

add_executable(
    argus
    main.cpp
    StereoMatcher.h
    StereoMatcher.cpp
    StereoMatcherCPU.h
    StereoMatcherCPU.cpp
    StereoMatcherGPU.h
    StereoMatcherGPU.cpp)

target_link_libraries(argus PUBLIC opencv_core opencv_imgcodecs opencv_highgui Threads::Threads)

