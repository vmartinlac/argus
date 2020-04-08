
add_executable(
    tmp
    LBPSM.cpp
    LBPSM.h
    LoopyBeliefPropagationSolver.cpp
    LoopyBeliefPropagationSolver.h
    StochasticSearchSolver.h
    StochasticSearchSolver.cpp
    main.cpp
    MarkovRandomField.cpp
    MarkovRandomField.h
    Solver.h
    Solver.cpp
    Tsukuba.cpp
    Tsukuba.h)

target_link_libraries(tmp PUBLIC argus_core opencv_imgcodecs opencv_highgui Threads::Threads Qt5::Core)

