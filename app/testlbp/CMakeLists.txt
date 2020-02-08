
add_executable(
    testlbp
    main.cpp)

target_link_libraries(testlbp PUBLIC argus_core opencv_imgcodecs opencv_highgui Threads::Threads)

