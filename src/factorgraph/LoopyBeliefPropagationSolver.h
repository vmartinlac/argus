
#pragma once

#include <opencv2/core.hpp>
#include "Solver.h"

class LoopyBeliefPropagationSolver : public Solver
{
public:

    LoopyBeliefPropagationSolver();

    void setNumIterations(int value);

    bool solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution) override;

protected:

    int myNumIterations;
};

