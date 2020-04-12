#include "ExhaustiveSolver.h"
#include "Common.h"

bool ExhaustiveSolver::solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution)
{
    const int num_variables = graph->getNumVariables();

    int num_combinations = 1;

    for(int i=0; i<num_variables; i++)
    {
        num_combinations *= graph->getNumLabels(i);

        if( num_combinations > 2000000000 ) ABORT("Factor graph is too big for exhaustive solver!");
    }

    std::vector<int> candidate;
    bool first = true;
    double best_E = 0.0;

    for(int combination=0; combination<num_combinations; combination++)
    {
        candidate.resize(num_variables);

        {
            int x = combination;

            for(int i=0; i<num_variables; i++)
            {
                const int num_labels = graph->getNumLabels(i);
                candidate[i] = x % num_labels;
                x /= num_labels;
            }
        }

        const double this_E = graph->evaluateTotalEnergy(candidate);

        if(first || this_E < best_E)
        {
            candidate.swap(solution);
            best_E = this_E;
            first = false;
        }
    }

    if(first) ABORT("Internal error!");

    return true;
}

