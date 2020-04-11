
#pragma once

#include "FactorGraph.h"

class GrayscaleMumfordShah : public FactorGraph
{
public:

    GrayscaleMumfordShah(cv::Mat1b input_image, double beta, double lambda);

    int getNumVariables() const final;

    int getNumFactors() const final;

    void getVariables(int factor, std::vector<int>& variables) const final;

    void getFactors(int variable, std::vector<int>& factors) const final;

    int getNumLabels(int variable) const final;

    double evaluateEnergy(int factor, const std::vector<int>& node_labels) const final;

    void getInitialSolution(std::vector<int>& solution);

    void getImage(const std::vector<int>& solution, cv::Mat1b& image);

    void getEdges(const std::vector<int>& solution, cv::Mat1b& edges);

protected:

    cv::Mat1b myImage;
    cv::Size mySize;
    int myOffset[3];
    double myBeta;
    double myLambda;
};

