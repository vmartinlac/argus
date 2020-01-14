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

    const int num_disparities = left.cols*2/10;
    myDisparityTable[0].resize(num_disparities);
    myDisparityTable[1].resize(num_disparities);
    for(int i=0; i<num_disparities; i++)
    {
        myDisparityTable[0][i] = -i;
        myDisparityTable[1][i] = i;
    }

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
    std::transform(myDisparity[0].begin(), myDisparity[0].end(), disparity.begin(), [this] (int x) { return static_cast<float>(myDisparityTable[0][x]); } );

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
        myDisparityTable[image].size(),
        myImages[image]->size(),
        data_cost,
        discontinuity_cost,
        myNumBeliefPropagationIterations,
        myDisparity[image]);
}

void StereoMatcherCPU::computeOcclusion(int image)
{
    const int other_image = (image + 1) % 2;

    cv::Mat1b W(myImages[image]->size());
    std::fill(W.begin(), W.end(), 1);

    for(auto it=myDisparity[image].begin(); it!=myDisparity[image].end(); it++)
    {
        const cv::Point other_pt = it.pos() + cv::Point(myDisparityTable[other_image][*it], 0);

        if( 0 <= other_pt.x && other_pt.x < W.cols && 0 <= other_pt.y && other_pt.y < W.rows )
        {
            W(other_pt) = 0;
        }
    }

    auto data_cost = [this] (const cv::Point& pt, int label) -> float
    {
        float ret = 0.0f;

        /*
        if(label)
        {
            ret = myEtaO;
        }
        else
        {
            // TODO
        }

        // TODO
        */

        return 0.0f;
    };

    auto discontinuity_cost = [this] (int label0, int label1) -> float
    {
        float ret = 0.0f;

        /*
        if(label0 != label1)
        {
            ret = static_cast<float>(myBetaO);
        }
        else
        {
            ret = 0.0f;
        }
        */

        return ret;
    };

    LoopyBeliefPropagation::execute(
        2,
        myImages[0]->size(),
        data_cost,
        discontinuity_cost,
        myNumBeliefPropagationIterations,
        myOcclusion[image]);
}

