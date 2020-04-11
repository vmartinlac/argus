
#pragma once

#include <opencv2/core.hpp>

class ProximalAlternatingMumfordShah
{
public:

    ProximalAlternatingMumfordShah(double beta, double lambda, const cv::Vec3d& rgb_weights=cv::Vec3d(1.0/3.0, 1.0/3.0, 1.0/3.0));

    void setNumIterations(int value);

    void runColor(const cv::Mat3b& input, cv::Mat3b& result, cv::Mat1b& edges);

    void runGrayscale(const cv::Mat1b& input, cv::Mat1b& result, cv::Mat1b& edges);

protected:

    int myNumIterations;
    double myBeta;
    double myLambda;
    cv::Vec3d myRgbWeights;
};

