
#pragma once

#include "StereoMatcher.h"

class StereoMatcherGPU : public StereoMatcher
{
public:

    StereoMatcherGPU();

    ~StereoMatcherGPU() override;

    void compute(
        const cv::Mat1b& left,
        const cv::Mat1b& right,
        cv::Mat1f& disparity) override;
};

