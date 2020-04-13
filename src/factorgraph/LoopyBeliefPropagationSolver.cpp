#include <iostream>
#include "LoopyBeliefPropagationSolver.h"
#include "Common.h"

LoopyBeliefPropagationSolver::LoopyBeliefPropagationSolver()
{
    myNumIterations = 1000;
}

void LoopyBeliefPropagationSolver::setNumIterations(int value)
{
    myNumIterations = value;
}

bool LoopyBeliefPropagationSolver::solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution)
{
    ABORT("Not implemented!");
}

