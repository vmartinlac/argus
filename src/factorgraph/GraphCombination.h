
#pragma once

#include <vector>
#include <initializer_list>
#include "FactorGraph.h"

class GraphCombination : public FactorGraph
{
public:

    GraphCombination(std::initializer_list<FactorGraph*> graphs);

    int getNumVariables() const final;

    int getNumFactors() const final;

    void getVariables(int factor, std::vector<int>& variables) const final;

    void getFactors(int variable, std::vector<int>& factors) const final;

    int getNumLabels(int variable) const final;

    double evaluateEnergy(int factor, const std::vector<int>& node_labels) const final;

protected:

    std::vector<FactorGraph*> myGraphs;
    std::vector<int> myVariableOffset;
    std::vector<int> myFactorOffset;
};

