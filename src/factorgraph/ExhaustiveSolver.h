
#pragma once

#include "Solver.h"

class ExhaustiveSolver : public Solver
{
public:

    ExhaustiveSolver() = default;

    bool solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution) override;
};

