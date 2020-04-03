#pragma once

#include <opencv2/core.hpp>

class LoopyBeliefPropagationSolver
{
public:

    using Label = uint16_t;
    //using Label = uint8_t;

    LoopyBeliefPropagationSolver() = default;

    void run(
        int num_labels,
        const std::vector<cv::Point>& neighbors,
        const cv::Mat1b& connections,
        const cv::Mat1f& data_cost,
        const cv::Mat1f& discontinuity_cost,
        int num_iterations,
        cv::Mat_<Label>& result);
};

