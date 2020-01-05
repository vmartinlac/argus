#include <memory>
#include <future>
#include <thread>
#include <chrono>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include "StereoMatcherCPU.h"
#include "StereoMatcherGPU.h"

int main(int num_args, char** args)
{
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

    if(num_args != 3)
    {
        std::cerr << "Bad command line!" << std::endl;
        exit(1);
    }

    cv::Mat1b left = cv::imread(args[1], cv::IMREAD_GRAYSCALE);
    cv::Mat1b right = cv::imread(args[2], cv::IMREAD_GRAYSCALE);

    if( left.data == nullptr || right.data == nullptr || left.size() != right.size() )
    {
        std::cerr << "Bad images!" << std::endl;
        exit(1);
    }

    std::future<cv::Mat1f> disparity_cpu_future = std::async([&left, &right] ()
    {
        StereoMatcherCPU matcher;
        cv::Mat1f disparity;

        matcher.compute(left, right, disparity);

        return disparity;
    });

    std::future<cv::Mat1f> disparity_gpu_future = std::async([&left, &right] ()
    {
        StereoMatcherGPU matcher;
        cv::Mat1f disparity;

        matcher.compute(left, right, disparity);

        return disparity;
    });

    const cv::Mat1f disparity_cpu = disparity_cpu_future.get();
    const cv::Mat1f disparity_gpu = disparity_gpu_future.get();;

    cv::imshow("left", left);
    cv::imshow("right", right);
    //cv::imshow("disparity (CPU)", disparity_cpu);
    //cv::imshow("disparity (GPU)", disparity_gpu);
    cv::waitKey(0);

    return 0;
}

