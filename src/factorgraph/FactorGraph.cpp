#include <iostream>
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

std::string FactorGraph::compileForGraphviz() const
{
    const int num_variables = getNumVariables();
    const int num_factors = getNumFactors();

    const char* indent = "    ";

    std::vector<int> local_factors;

    std::stringstream ss;

    ss << "graph factor_graph {" << std::endl;

    for(int i=0; i<num_variables; i++)
    {
        ss << indent << "v" << i << " [shape=circle]" << std::endl;
    }

    for(int i=0; i<num_factors; i++)
    {
        ss << indent << "f" << i << " [shape=box]" << std::endl;
    }

    for(int i=0; i<num_variables; i++)
    {
        getFactors(i, local_factors);

        for(int j : local_factors)
        {
            ss << indent << "v" << i << " -- f" << j << std::endl;
        }
    }

    ss << "}" << std::endl;

    return ss.str();
}

