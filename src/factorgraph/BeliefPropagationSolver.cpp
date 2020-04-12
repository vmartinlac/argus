#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
#include "BeliefPropagationSolver.h"
#include "Common.h"

BeliefPropagationSolver::BeliefPropagationSolver()
{
}

bool BeliefPropagationSolver::solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution)
{
    ABORT("Not implemented!"); //FIXME TODO
    const int num_variables = graph->getNumVariables();
    const int num_factors = graph->getNumFactors();

    std::vector<int> variable_edges_offset;
    std::vector<int> factor_edges_offset;
    std::vector<int> variable_edges;
    std::vector<int> factor_edges;
    int sum_message_dimensions;

    std::vector<float> factor_to_variable;
    std::vector<float> variable_to_factor;

    std::vector<int> neighbors0;
    std::vector<int> neighbors1;

    // initialize message offsets for variables.

    {
        sum_message_dimensions = 0;

        variable_edges_offset.resize(num_variables);

        for(int i=0; i<num_variables; i++)
        {
            graph->getFactors(i, neighbors0);
            const int num_local_factors = (int) neighbors0.size();
            const int num_labels = graph->getNumLabels(i);

            variable_edges_offset[i] = (int) variable_edges.size();

            for(int j=0; j<num_local_factors; j++)
            {
                variable_edges.push_back(sum_message_dimensions);
                sum_message_dimensions += num_labels;
            }

            if(sum_message_dimensions > 2000000) ABORT("Graph size exceeds limits! Lets avoid arithmetic overflow!");
        }

        factor_to_variable.assign(sum_message_dimensions, 0.0f);
        variable_to_factor.assign(sum_message_dimensions, 0.0f);
    }

    // initialize message offsets for factors.

    {
        factor_edges_offset.resize(num_factors);

        for(int i=0; i<num_factors; i++)
        {
            graph->getVariables(i, neighbors0);
            const int num_local_variables = (int) neighbors0.size();

            factor_edges_offset[i] = (int) factor_edges.size();

            for(int j=0; j<num_local_variables; j++)
            {
                bool found = false;
                graph->getFactors(neighbors0[j], neighbors1);
                for(int k=0; found == false && k<(int) neighbors1.size(); k++)
                {
                    if(neighbors1[k] == i)
                    {
                        factor_edges.push_back( variable_edges[variable_edges_offset[neighbors0[j]] + k] );
                        found = true;
                    }
                }

                if(found == false)
                {
                    ABORT("Internal error!");
                }
            }
        }
    }

    // min-sum iterations.
}

