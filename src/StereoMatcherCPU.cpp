#include <iostream>
#include "StereoMatcherCPU.h"

StereoMatcherCPU::StereoMatcherCPU()
{
    myNumGlobalIterations = 3;
    myNumBeliefPropagationIterations = 30;

    myNeighbors[0].x = 1;
    myNeighbors[0].y = 0;

    myNeighbors[1].x = -1;
    myNeighbors[1].y = 0;

    myNeighbors[2].x = 0;
    myNeighbors[2].y = 1;

    myNeighbors[3].x = 0;
    myNeighbors[3].y = -1;
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
    myOcclusion[0] = cv::Mat1b();
    myOcclusion[1] = cv::Mat1b();
    myDisparity[0] = cv::Mat1i();
    myDisparity[1] = cv::Mat1i();
}

void StereoMatcherCPU::computeDisparity(int i)
{
    cv::Mat1f messages;
    cv::Mat1f new_messages;

    const int dimensions[4] = { myImages[0]->rows, myImages[1]->cols, static_cast<int>(myNeighbors.size()), myNumDisparities };
    messages.create(4, dimensions);
    new_messages.create(4, dimensions);

    for(int i=0; i<myNumBeliefPropagationIterations; i++)
    {
        // TODO

        cv::swap(messages, new_messages);
    }

    // TODO: update disparity according to messages.
}

void StereoMatcherCPU::computeOcclusion(int i)
{
}

