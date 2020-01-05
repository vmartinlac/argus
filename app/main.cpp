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

using DisparityAndDuration = std::tuple<cv::Mat1f, std::chrono::milliseconds>;

template<typename Matcher>
class MatcherWrapper
{
protected:

    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;

public:

    MatcherWrapper(const cv::Mat1b& left, const cv::Mat1b& right) :
        myLeft(left),
        myRight(right)
    {
    }

    DisparityAndDuration operator()()
    {
        Matcher matcher;
        cv::Mat1f disparity;

        TimePoint t0 = Clock::now();

        matcher.compute(myLeft, myRight, disparity);

        TimePoint t1 = Clock::now();

        return std::make_tuple(disparity, std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0));
    }

protected:

    const cv::Mat1b& myLeft;
    const cv::Mat1b& myRight;
};

int main(int num_args, char** args)
{

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

    std::future<DisparityAndDuration> future_cpu = std::async(MatcherWrapper<StereoMatcherCPU>(left, right));
    std::future<DisparityAndDuration> future_gpu = std::async(MatcherWrapper<StereoMatcherGPU>(left, right));

    const DisparityAndDuration result_cpu = future_cpu.get();
    const DisparityAndDuration result_gpu = future_gpu.get();

    std::cout << "Duration time for CPU stereo matching (ms): " << std::get<1>(result_cpu).count() << std::endl;
    std::cout << "Duration time for GPU stereo matching (ms): " << std::get<1>(result_gpu).count() << std::endl;

    cv::imshow("left", left);
    cv::imshow("right", right);
    cv::imshow("disparity (CPU)", std::get<0>(result_cpu));
    cv::imshow("disparity (GPU)", std::get<0>(result_gpu));
    cv::waitKey(0);

    return 0;
}

