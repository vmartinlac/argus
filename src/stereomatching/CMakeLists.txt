
add_library(
    argus_stereomatching
    STATIC
    StereoMatcher.h
    StereoMatcher.cpp
    LoopyBeliefPropagation.h
    StereoMatcherCPU.h
    StereoMatcherCPU.cpp
    StereoMatcherGPU.h
    StereoMatcherGPU.cpp)

target_link_libraries(argus_stereomatching PUBLIC argus_core opencv_core tbb)
target_include_directories(argus_stereomatching INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

