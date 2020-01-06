
#pragma once

#include <array>
#include "StereoMatcher.h"

class StereoMatcherCPU : public StereoMatcher
{
public:

    StereoMatcherCPU();

    ~StereoMatcherCPU() override;

    void compute(
        const cv::Mat1b& left,
        const cv::Mat1b& right,
        cv::Mat1f& disparity) override;

protected:

    void computeDisparity(int i);
    void computeOcclusion(int i);

protected:

    template<typename DataCostFunction, typename DiscontinuityCostFunction>
    void loopyBeliefPropagation(
        int num_labels,
        DataCostFunction data_cost,
        DiscontinuityCostFunction discontinuity_cost,
        cv::Mat1i& result);

protected:

    // parameters.
    std::array<cv::Point,4> myNeighbors;
    int myNumGlobalIterations;
    int myNumBeliefPropagationIterations;
    int myNumDisparities;

    std::array<const cv::Mat1b*,2> myImages;
    std::array<cv::Mat1i,2> myOcclusion;
    std::array<cv::Mat1i,2> myDisparity;
};

