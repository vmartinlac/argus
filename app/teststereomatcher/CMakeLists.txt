
add_executable(
    teststereomatcher
    main.cpp)

target_link_libraries(teststereomatcher PUBLIC argus_core opencv_imgcodecs opencv_highgui Threads::Threads)

