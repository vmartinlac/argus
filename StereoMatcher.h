
#pragma once

#include <opencv2/core.hpp>

class StereoMatcher
{
public:

    StereoMatcher();

    virtual ~StereoMatcher();

    virtual void compute(
        const cv::Mat1b& left,
        const cv::Mat1b& right,
        cv::Mat1f& disparity) = 0;
};

