
add_library(
    argus_mumfordshah
    STATIC
    ProximalAlternatingMumfordShah.h
    ProximalAlternatingMumfordShah.cpp
    ColorMumfordShah.h
    ColorMumfordShah.cpp
    GrayscaleMumfordShah.h
    GrayscaleMumfordShah.cpp)

target_link_libraries(argus_mumfordshah PUBLIC argus_factorgraph opencv_core)
target_include_directories(argus_mumfordshah INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

