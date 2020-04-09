
add_library(
    argus_markovrandomfield
    STATIC
    LoopyBeliefPropagationSolver.cpp
    LoopyBeliefPropagationSolver.h
    MarkovRandomField.cpp
    MarkovRandomField.h
    Solver.cpp
    Solver.h
    StochasticSearchSolver.cpp
    StochasticSearchSolver.h)

target_link_libraries(argus_markovrandomfield PUBLIC argus_core)
target_include_directories(argus_markovrandomfield INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

