#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <LoopyBeliefPropagation.h>

int main(int num_args, char** args)
{
    const int N = 128;
    const int M = 2;

    auto data_cost = [M,N] (const cv::Point& pt, int label) -> float
    {
        if(pt.x == 0 && pt.y == 0) return std::fabs(label-0);
        else return 0.0;

        float ret = 0.0f;
        const int margin = N/4;

        if(pt.x < margin)
        {
            ret = std::fabs(label-0);
        }
        else if(pt.x > N-margin)
        {
            ret = std::fabs(label-(M-1));
        }

        return ret;
    };

    auto discontinuity_cost = [] (int label0, int label1) -> float
    {
        return std::fabs(label1 == label0);
        if( (label0 == 0 && label1 == 2) || (label0 == 2 && label1 == 0) )
        {
            return 1.0f;
        }
        else
        {
            return 0.0;
        }
    };

    cv::Mat1i image(N, N);
    std::fill(image.begin(), image.end(), 0);

    LoopyBeliefPropagation::execute(
        M,
        image.size(),
        data_cost,
        discontinuity_cost,
        100,
        image);
    /*
    template<typename DataCostFunction, typename DiscontinuityCostFunction>
    void execute(
        int num_labels,
        const cv::Size& image_size,
        DataCostFunction data_cost,
        DiscontinuityCostFunction discontinuity_cost,
        int num_iterations,
        cv::Mat1i& result)
    */

    image *= 65535.0/double(M-1);

    cv::imshow("output", image);
    cv::waitKey(0);

    return 0;
}

