
add_executable(
    tmp
    LBP.h
    LBP.cpp
    LBPSM.h
    LBPSM.cpp
    Tsukuba.h
    Tsukuba.cpp
    main.cpp)

target_link_libraries(tmp PUBLIC argus_core opencv_imgcodecs opencv_highgui Threads::Threads Qt5::Core PkgConfig::tbb)

