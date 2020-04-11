
add_executable(
    tmp
    LBPSM.cpp
    LBPSM.h
    main.cpp
    Tsukuba.cpp
    Tsukuba.h)

target_link_libraries(tmp PUBLIC argus_factorgraph opencv_imgcodecs opencv_highgui Threads::Threads Qt5::Core)

