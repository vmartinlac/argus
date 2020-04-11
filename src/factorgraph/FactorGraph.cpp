#include "FactorGraph.h"

double FactorGraph::evaluateTotalEnergy(const std::vector<int>& all_labels) const
{
    std::vector<int> local_variables;
    std::vector<int> local_labels;

    const int num_factors = getNumFactors();

    double ret = 0.0;

    for(int i=0; i<num_factors; i++)
    {
        getVariables(i, local_variables);

        local_labels.resize(local_variables.size());
        for(int j=0; j<(int)local_variables.size(); j++)
        {
            local_labels[j] = all_labels[local_variables[j]];
        }

        ret += evaluateEnergy(i, local_labels);
    }

    return ret;
}

