
#pragma once

#include <opencv2/core.hpp>
#include "Solver.h"

class BeliefPropagationSolver : public Solver
{
public:

    BeliefPropagationSolver();

    bool solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution) override;

protected:
};

