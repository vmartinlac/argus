#include "GraphCombination.h"
#include "Common.h"

GraphCombination::GraphCombination(std::initializer_list<FactorGraph*> graphs) : myGraphs(graphs)
{
    myVariableOffset.resize(myGraphs.size()+1);
    myFactorOffset.resize(myGraphs.size()+1);

    myVariableOffset.front() = 0;
    myFactorOffset.front() = 0;

    for(size_t i=0; i<myGraphs.size(); i++)
    {
        myVariableOffset[i+1] = myVariableOffset[i] + myGraphs[i]->getNumVariables();
        myFactorOffset[i+1] = myFactorOffset[i] + myGraphs[i]->getNumFactors();
    }
}

int GraphCombination::getNumVariables() const
{
    return myVariableOffset.back();
}

int GraphCombination::getNumFactors() const
{
    return myFactorOffset.back();
}

void GraphCombination::getVariables(int factor, std::vector<int>& variables) const
{
    bool go_on = true;

    for(size_t i=0; go_on && i<myGraphs.size(); i++)
    {
        if(factor < myFactorOffset[i+1])
        {
            myGraphs[i]->getVariables(factor - myFactorOffset[i], variables);

            for(int& v : variables)
            {
                v += myVariableOffset[i];
            }

            go_on = false;
        }
    }

    if(go_on) ABORT("Internal error");
}

void GraphCombination::getFactors(int variable, std::vector<int>& factors) const
{
    bool go_on = true;

    for(size_t i=0; go_on && i<myGraphs.size(); i++)
    {
        if( variable < myVariableOffset[i+1] )
        {
            myGraphs[i]->getFactors(variable - myVariableOffset[i], factors);

            for(int& f : factors)
            {
                f += myFactorOffset[i];
            }

            go_on = false;
        }
    }

    if(go_on) ABORT("Internal error");
}

int GraphCombination::getNumLabels(int variable) const
{
    bool go_on = true;
    int ret = 0;

    for(size_t i=0; go_on && i<myGraphs.size(); i++)
    {
        if( variable < myVariableOffset[i+1] )
        {
            ret = myGraphs[i]->getNumLabels(variable - myVariableOffset[i]);
            go_on = false;
        }
    }

    if(go_on) ABORT("Internal error");

    return ret;
}

double GraphCombination::evaluateEnergy(int factor, const std::vector<int>& node_labels) const
{
    bool go_on = true;
    double ret = 0.0;

    for(size_t i=0; go_on && i<myGraphs.size(); i++)
    {
        if(factor < myFactorOffset[i+1])
        {
            ret = myGraphs[i]->evaluateEnergy(factor - myFactorOffset[i], node_labels);
            go_on = false;
        }
    }

    if(go_on) ABORT("Internal error");

    return ret;
}

