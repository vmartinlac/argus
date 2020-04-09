
#pragma once

#include "MarkovRandomField.h"

class ColorMumfordShah : public MarkovRandomField
{
public:

    ColorMumfordShah(cv::Mat3b input_image, double beta, double lambda, const cv::Vec3d& rgb_weights = cv::Vec3d(1.0/3.0, 1.0/3.0, 1.0/3.0));

    int getNumVariables() const final;

    int getNumFactors() const final;

    void getVariables(int factor, std::vector<int>& variables) const final;

    void getFactors(int variable, std::vector<int>& factors) const final;

    int getNumLabels(int variable) const final;

    double evaluateEnergy(int factor, const std::vector<int>& node_labels) const final;

    void getInitialSolution(std::vector<int>& solution);

    void getImage(const std::vector<int>& solution, cv::Mat3b& image);

    void getEdges(const std::vector<int>& solution, cv::Mat1b& edges);

protected:

    cv::Mat3b myImage;
    cv::Size mySize;
    int myOffset[3];
    double myBeta;
    double myLambda;
    cv::Vec3d myRgbWeights;
};

