
add_executable(
    argus_demo_stereomatching
    main.cpp)

target_link_libraries(argus_demo_stereomatching PUBLIC argus_stereomatching opencv_imgcodecs opencv_highgui Threads::Threads)

