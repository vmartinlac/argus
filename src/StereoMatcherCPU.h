
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
        int level;
        std::array<cv::Mat1b,2> thumbnails;
        std::array<cv::Mat1i,2> occlusion;
        std::array<cv::Mat1i,2> disparity;
    };

protected:

    void updateDisparity(Level& l, int i);
    void updateOcclusion(Level& l, int i);
    float robustNormPixels(float x);
    float robustNormDisparity(float x);

protected:

    double myScaleFactor;
    int myMinLevelWidth;
    int myNumFixedPointIterations;
    int myNumBeliefPropagationIterations;
    int myNumDisparities;
    std::array<int,2> myDirections;

    double myEta0;
    double myBetaW;
    double myBeta0;
    double myED;
    double mySigmaD;
    double myLambda;
    double myTau;
};

