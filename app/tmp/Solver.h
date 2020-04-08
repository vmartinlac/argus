
#pragma once

#include "MarkovRandomField.h"

class Solver
{
public:

    virtual bool solve(MarkovRandomField* field, std::vector<int>& solution, bool use_initial_solution) = 0;
};

