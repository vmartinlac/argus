
add_library(
    argus_core
    STATIC
    StereoMatcher.h
    StereoMatcher.cpp
    StereoMatcherCPU.h
    StereoMatcherCPU.cpp
    StereoMatcherGPU.h
    StereoMatcherGPU.cpp)

target_link_libraries(argus_core PUBLIC opencv_core tbb)
target_include_directories(argus_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

