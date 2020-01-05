#include <opencv2/core/cuda.hpp>
#include "StereoMatcherGPU.h"

StereoMatcherGPU::StereoMatcherGPU()
{
}

StereoMatcherGPU::~StereoMatcherGPU()
{
}

void StereoMatcherGPU::compute(
    const cv::Mat1b& left,
    const cv::Mat1b& right,
    cv::Mat1f& disparity)
{
    disparity.create(left.size());
}

