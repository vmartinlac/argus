
#pragma once

#include "FactorGraph.h"

class HiddenMarkovModel : public FactorGraph
{
public:

    HiddenMarkovModel(int N, int num_state_labels, int num_observation_labels, const double* prior, const double* transition, const double* observation );

    int getNumVariables() const override;

    int getNumFactors() const override;

    void getVariables(int factor, std::vector<int>& variables) const override;

    void getFactors(int variable, std::vector<int>& factors) const override;

    int getNumLabels(int variable) const override;

    double evaluateEnergy(int factor, const std::vector<int>& node_labels) const override;

    std::string getNameOfVariable(int variable) const override;

    std::string getNameOfFactor(int factor) const override;

    void dump(std::ostream& s);

protected:

    int myN;
    int myNumStateLabels;
    int myNumObservationLabels;
    const double* myPrior;
    const double* myTransition;
    const double* myObservation;
};

