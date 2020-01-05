
#pragma once

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
};

