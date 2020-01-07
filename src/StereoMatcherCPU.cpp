#include <iostream>
#include "StereoMatcherCPU.h"
#include "LoopyBeliefPropagation.h"

StereoMatcherCPU::StereoMatcherCPU()
{
    myNumGlobalIterations = 3;
    myNumBeliefPropagationIterations = 30;
}

StereoMatcherCPU::~StereoMatcherCPU()
{
}

void StereoMatcherCPU::compute(
    const cv::Mat1b& left,
    const cv::Mat1b& right,
    cv::Mat1f& disparity)
{
    if(left.size() != right.size())
    {
        std::cerr << "Size mismatch" << std::endl;
        exit(1);
    }

    myNumDisparities = left.cols*2/10;

    myImages[0] = &left;
    myImages[1] = &right;

    myOcclusion[0].create(left.size());
    myOcclusion[1].create(left.size());
    myDisparity[0].create(left.size());
    myDisparity[1].create(left.size());

    for(int i=0; i<myNumGlobalIterations; i++)
    {
        if(i == 0)
        {
            std::fill(myOcclusion[0].begin(), myOcclusion[0].end(), 0);
            std::fill(myOcclusion[1].begin(), myOcclusion[1].end(), 0);
        }
        else
        {
            computeOcclusion(0);
            computeOcclusion(1);
        }

        computeDisparity(0);
        computeDisparity(1);
    }

    disparity.create(left.size());
    std::transform(myDisparity[0].begin(), myDisparity[0].end(), disparity.begin(), [] (int x) { return static_cast<float>(x); } );

    myImages[0] = nullptr;
    myImages[1] = nullptr;
    myOcclusion[0] = cv::Mat1i();
    myOcclusion[1] = cv::Mat1i();
    myDisparity[0] = cv::Mat1i();
    myDisparity[1] = cv::Mat1i();
}

void StereoMatcherCPU::computeDisparity(int image)
{
    auto data_cost = [this] (const cv::Point& pt, int label) -> float
    {
        return 0.0f;
    };

    auto discontinuity_cost = [this] (int label0, int label1) -> float
    {
        return 0.0;
    };

    LoopyBeliefPropagation::execute(
        myNumDisparities,
        myImages[0]->size(),
        data_cost,
        discontinuity_cost,
        myNumBeliefPropagationIterations,
        myDisparity[image]);
}

void StereoMatcherCPU::computeOcclusion(int image)
{
    auto data_cost = [this] (const cv::Point& pt, int label) -> float
    {
        return 0.0f;
    };

    auto discontinuity_cost = [this] (int label0, int label1) -> float
    {
        return 0.0;
    };

    LoopyBeliefPropagation::execute(
        2,
        myImages[0]->size(),
        data_cost,
        discontinuity_cost,
        myNumBeliefPropagationIterations,
        myOcclusion[image]);
}

