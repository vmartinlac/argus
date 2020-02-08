#pragma once

#include <opencv2/core.hpp>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>

namespace LoopyBeliefPropagation
{
    template<typename DataCostFunction, typename DiscontinuityCostFunction>
    void execute(
        int num_labels,
        const cv::Size& image_size,
        DataCostFunction data_cost,
        DiscontinuityCostFunction discontinuity_cost,
        int num_iterations,
        cv::Mat1i& result)
    {
        std::array<cv::Point,4> neighbors;

        neighbors[0] = cv::Point(1,0);
        neighbors[1] = cv::Point(0,1);
        neighbors[2] = cv::Point(-1,0);
        neighbors[3] = cv::Point(0,-1);

        cv::Mat1f messages;
        cv::Mat1f new_messages;

        const tbb::blocked_range2d<int> whole_range(0, image_size.height, 0, image_size.width);

        {
            const int dimensions[4] = { image_size.height, image_size.width, static_cast<int>(neighbors.size()), num_labels };

            messages.create(4, dimensions);
            new_messages.create(4, dimensions);
        }

        result.create(image_size);

        const cv::Rect ROI(0, 0, image_size.width, image_size.height);

        std::fill(messages.begin(), messages.end(), 0.0f);

        auto update_messages_pred = [&messages, &new_messages, &neighbors, &ROI, num_labels, &data_cost, &discontinuity_cost] (tbb::blocked_range2d<int> range)
        {
            for(int i=range.rows().begin(); i<range.rows().end(); i++)
            {
                for(int j=range.cols().begin(); j<range.cols().end(); j++)
                {
                    const cv::Point this_point(j,i);

                    for(int k=0; k<neighbors.size(); k++)
                    {
                        const cv::Point that_point = this_point + neighbors[k];

                        if(ROI.contains(that_point))
                        {
                            for(int m=0; m<num_labels; m++)
                            {
                                bool first = true;
                                float minimal_value = 0.0f;

                                for(int n=0; n<num_labels; n++)
                                {
                                    float value = data_cost(this_point, n) + discontinuity_cost(n,m);

                                    for(int l=0; l<neighbors.size(); l++)
                                    {
                                        const cv::Point third_point = this_point + neighbors[l];

                                        if(ROI.contains(third_point) && third_point != that_point)
                                        {
                                            value += messages.at<float>(cv::Vec4i(third_point.y, third_point.x, (l+2)%4, n));
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
        };

        auto update_result_pred = [&data_cost, &ROI, &neighbors, &messages, num_labels, &result] (tbb::blocked_range2d<int> range)
        {
            for(int i=range.rows().begin(); i<range.rows().end(); i++)
            {
                for(int j=range.cols().begin(); j<range.cols().end(); j++)
                {
                    const cv::Point this_point(j,i);

                    bool first = true;
                    float minimal_value = 0.0f;

                    for(int k=0; k<num_labels; k++)
                    {
                        float value = data_cost(this_point, k);

                        for(int l=0; l<neighbors.size(); l++)
                        {
                            const cv::Point that_point = this_point + neighbors[l];

                            if(ROI.contains(that_point))
                            {
                                value += messages.at<float>(cv::Vec4i(that_point.y, that_point.x, (l+2)%4, k));
                            }
                        }

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
        };

        for(int i=0; i<num_iterations; i++)
        {
            tbb::parallel_for(whole_range, update_messages_pred);
            cv::swap(messages, new_messages);
        }

        tbb::parallel_for(whole_range, update_result_pred);
    }
}

