
#pragma once

#include "FactorGraph.h"

class Solver
{
public:

    virtual bool solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution) = 0;
};

