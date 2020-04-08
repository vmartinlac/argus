
#pragma once

#include <opencv2/core.hpp>
#include "MarkovRandomField.h"

class LoopyBeliefPropagationSolver
{
public:

    void solve(MarkovRandomField* problem, int num_iterations, cv::Mat1i& result, bool use_result_as_initial_solution);

protected:

    void updateMessages();
};

