#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
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
    const int num_variables = graph->getNumVariables();
    const int num_factors = graph->getNumFactors();

    std::vector<int> message_offset(num_variables);
    std::vector<float> factor_messages;
    std::vector<float> variable_messages;

    {
        int offset = 0;

        for(int i=0; i<num_variables; i++)
        {
            message_offset[i] = offset;
            offset += graph->getNumLabels(i);

            if(offset > 2000000) ABORT("Graph size exceeds limits! Lets avoid arithmetic overflow!");
        }

        factor_messages.assign(offset, 0.0f);
        variable_messages.assign(offset, 0.0f);
    }

    for(int iter=0; iter<myNumIterations; iter++)
    {
        for(int i=0; i<num_factors; i++)
        {
            ;
        }
    }
}

