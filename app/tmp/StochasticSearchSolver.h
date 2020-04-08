
#pragma once

#include "MarkovRandomField.h"
#include "Solver.h"

class StochasticSearchSolver : public Solver
{
public:

    StochasticSearchSolver();

    void setInitialTemperature(double value);
    void setLambda(double value);
    void setSeed(int value);
    void setNumIterations(int value);

    bool solve(MarkovRandomField* field, std::vector<int>& solution, bool use_initial_solution) override;

protected:

    int mySeed;
    int myNumIterations;
    double myInitialTemperature;
    double myLambda;
};

