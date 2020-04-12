
add_library(
    argus_factorgraph
    STATIC
    ExhaustiveSolver.cpp
    ExhaustiveSolver.h
    BeliefPropagationSolver.cpp
    BeliefPropagationSolver.h
    LoopyBeliefPropagationSolver.cpp
    LoopyBeliefPropagationSolver.h
    FactorGraph.cpp
    FactorGraph.h
    Solver.cpp
    Solver.h
    StochasticSearchSolver.cpp
    StochasticSearchSolver.h)

target_link_libraries(argus_factorgraph PUBLIC argus_core)
target_include_directories(argus_factorgraph INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

