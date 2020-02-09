#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "StereoMatcherCPU.h"

void test0()
{
    const int N = 32;

    cv::Mat1b left_image(N,N);
    cv::Mat1b right_image(N,N);

    std::fill(left_image.begin(), left_image.end(), 0);
    std::fill(right_image.begin(), right_image.end(), 0);

    const int radius = N/4;
    const int delta = 2;
    cv::circle(left_image, cv::Point(N/2, N/2), radius, 255, -1);
    cv::circle(right_image, cv::Point(N/2+delta, N/2), radius, 255, -1);

    /*
    cv::imshow("left_image", left_image);
    cv::imshow("right_image", right_image);
    cv::waitKey(0);
    */

    StereoMatcherCPU m;
    cv::Mat1f disparity;
    m.compute(left_image, right_image, disparity);
}

void test1()
{
    const int N = 128;

    cv::Mat1b image0(1, N);
    cv::Mat1b image1(1, N);
    for(int j=0; j<N; j++)
    {
        image0(0,j) = j;
        image1(0,j) = j+4;
    }

    StereoMatcherCPU m;
    cv::Mat1f disparity;
    m.compute(image0, image1, disparity);
}

int main(int num_args, char** args)
{
    test0();
    return 0;
}

