
#pragma once

#include <opencv2/core.hpp>

class FactorGraph
{
public:

    virtual int getNumVariables() const = 0;

    virtual int getNumFactors() const = 0;

    virtual void getVariables(int factor, std::vector<int>& variables) const = 0;

    virtual void getFactors(int variable, std::vector<int>& factors) const = 0;

    virtual int getNumLabels(int variable) const = 0;

    virtual double evaluateEnergy(int factor, const std::vector<int>& node_labels) const = 0;

    virtual std::string getNameOfVariable(int variable) const;

    virtual std::string getNameOfFactor(int factor) const;

    double evaluateTotalEnergy(const std::vector<int>& all_labels) const;

    std::string compileForGraphviz() const;
};

