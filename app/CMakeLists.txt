
add_executable(
    argus
    main.cpp)

target_link_libraries(argus PUBLIC argus_core opencv_imgcodecs opencv_highgui Threads::Threads)

