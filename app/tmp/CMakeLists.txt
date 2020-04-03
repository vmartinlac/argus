
add_executable(
    tmp
    LBP.h
    LBP.cpp
    LBPSM.h
    LBPSM.cpp
    main.cpp)

target_link_libraries(tmp PUBLIC argus_core opencv_imgcodecs opencv_highgui Threads::Threads)

