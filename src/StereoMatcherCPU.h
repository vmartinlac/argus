
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

    int myNumGlobalIterations;
    int myNumBeliefPropagationIterations;

    std::vector<int> myDisparityTable[2];
    const cv::Mat1b* myImages[2];
    cv::Mat1i myOcclusion[2];
    cv::Mat1i myDisparity[2];
};

