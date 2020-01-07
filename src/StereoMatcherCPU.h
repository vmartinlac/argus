
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

    // parameters.
    int myNumGlobalIterations;
    int myNumBeliefPropagationIterations;
    int myNumDisparities;

    std::array<const cv::Mat1b*,2> myImages;
    std::array<cv::Mat1i,2> myOcclusion;
    std::array<cv::Mat1i,2> myDisparity;
};

