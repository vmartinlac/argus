#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
#include "StereoMatcherCPU.h"

template<typename DataCostFunction, typename DiscontinuityCostFunction>
void StereoMatcherCPU::loopyBeliefPropagation(
    int num_labels,
    DataCostFunction data_cost,
    DiscontinuityCostFunction discontinuity_cost,
    cv::Mat1i& result)
{
    cv::Mat1f messages;
    cv::Mat1f new_messages;

    const cv::Size image_size = myImages[0]->size();

    {
        const int dimensions[4] = { image_size.height, image_size.width, static_cast<int>(myNeighbors.size()), num_labels };
        messages.create(4, dimensions);
        new_messages.create(4, dimensions);
    }

    const cv::Rect ROI(0, 0, image_size.width, image_size.height);

    std::fill(messages.begin(), messages.end(), 0.0f);

    for(int i=0; i<myNumBeliefPropagationIterations; i++)
    {
        tbb::parallel_for(tbb::blocked_range2d<int>(0, image_size.height, 0, image_size.width), [&messages, &new_messages, num_labels, ROI, this] (tbb::blocked_range2d<int> range)
        {
            for(int i=range.rows().begin(); i<range.rows().end(); i++)
            {
                for(int j=range.cols().begin(); j<range.cols().end(); j++)
                {
                    const cv::Point this_point(j,i);

                    for(int k=0; k<myNeighbors.size(); k++)
                    {
                        const cv::Point that_point = this_point + myNeighbors[k];

                        if(ROI.contains(that_point))
                        {
                            for(int m=0; m<num_labels; m++)
                            {
                                bool first = true;
                                float minimal_value = 0.0f;

                                for(int n=0; n<num_labels; n++)
                                {
                                    float value = data_cost(this_point, n) + discontinuity_cost(n,m);

                                    for(cv::Point delta : myNeighbors)
                                    {
                                        if(ROI.contains(this_point+delta) && this_point+delta != that_point)
                                        {
                                            value += messages.at<float>(cv::Vec4i()); // TODO
                                        }
                                    }

                                    if(first || value < minimal_value)
                                    {
                                        minimal_value = value;
                                        first = false;
                                    }
                                }

                                new_messages.at<float>(cv::Vec4i(i,j,k,m)) = minimal_value;
                            }
                        }
                        else
                        {
                            for(int m=0; m<num_labels; m++)
                            {
                                new_messages.at<float>(cv::Vec4i(i,j,k,m)) = 0.0f;
                            }
                        }
                    }
                }
            }
        });

        cv::swap(messages, new_messages);
    }

    result.create(image_size);
    tbb::parallel_for(tbb::blocked_range2d<int>(0, image_size.height, 0, image_size.width), [&messages, num_labels, ROI, this] (tbb::blocked_range2d<int> range)
    {
        for(int i=range.rows().begin(); i<range.rows().end(); i++)
        {
            for(int j=range.cols().begin(); j<range.cols().end(); j++)
            {
                bool first = true;
                float minimal_value = 0.0f;

                for(int k=0; k<num_labels; k++)
                {
                    const value = 0.0f; // TODO: compute this value.

                    if(first || value < minimal_value)
                    {
                        minimal_value = value;
                        result(i,j) = k;
                        first = false;
                    }
                }

                if(first)
                {
                    throw std::runtime_error("internal error");
                }
            }
        }
    }
}

StereoMatcherCPU::StereoMatcherCPU()
{
    myNumGlobalIterations = 3;
    myNumBeliefPropagationIterations = 30;

    myNeighbors[0].x = 1;
    myNeighbors[0].y = 0;


    myNeighbors[1].x = 0;
    myNeighbors[1].y = 1;

    myNeighbors[2].x = -1;
    myNeighbors[2].y = 0;

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

    loopyBeliefPropagation(myNumDisparities, data_cost, discontinuity_cost, myDisparity[image]);
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

    loopyBeliefPropagation(2, data_cost, discontinuity_cost, myOcclusion[image]);
}

