
add_library(argus_core STATIC Common.h Common.cpp)

target_link_libraries(argus_core PUBLIC opencv_core)
target_include_directories(argus_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

