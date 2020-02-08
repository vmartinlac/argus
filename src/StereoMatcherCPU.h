
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

    struct Level
    {
        std::array<cv::Mat1b,2> thumbnails;
        std::array<cv::Mat1i,2> occlusion;
        std::array<cv::Mat1i,2> disparity;
    };

protected:

    void updateDisparity(Level& l, int i);
    void updateOcclusion(Level& l, int i);

protected:

    double myScaleFactor;
    int myMinLevelWidth;
    int myNumFixedPointIterations;
    int myNumBeliefPropagationIterations;
    int myNumDisparities;
    /*
    std::vector<int> myDisparityTable[2];
    */
};

