#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <LoopyBeliefPropagation.h>

int main(int num_args, char** args)
{
    const int N = 128;

    const int M = 10;

    cv::Mat1f image0(N, 1);
    cv::Mat1f image1(N, 1);
    for(int i=0; i<N; i++)
    {
        image0(i,0) = float(i)*0.1f;
        image1(i,0) = float(i+5)*0.1f;
    }

    auto data_cost = [N, &image0, &image1] (const cv::Point& pt, int label) -> float
    {
        const float val = image0(pt);
        const cv::Point other_point = pt + cv::Point(0, label - M/2);
        float other_val = 0.0f;
        if( 0 <= other_point.y && other_point.y < image0.rows )
        {
            other_val = image1(other_point);
        }

        const float delta = other_val - val;

        return delta*delta;
    };

    auto discontinuity_cost = [] (const cv::Point& pt0, int label0, const cv::Point& pt1, int label1) -> float
    {
        return std::fabs(label1 - label0);
    };

    cv::Mat1i labels(N, 1);

    LoopyBeliefPropagation::execute(
        M,
        image0.size(),
        data_cost,
        discontinuity_cost,
        30,
        labels);

    for(int i=0; i<N; i++)
    {
        std::cout << labels(i,0)-M/2 << std::endl;
    }

    return 0;
}

