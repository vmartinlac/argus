#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "Tsukuba.h"

Tsukuba::Tsukuba(const QString& path)
{
    if( myDir.cd(path) == false )
    {
        std::cout << "Could not change directory to specified Tsukuba directory!" << std::endl;
        exit(1);
    }
}


int Tsukuba::getNumImages()
{
    return 1800;
}

cv::Mat1b Tsukuba::loadImage(int index, int left_or_right)
{
    const char* left_pattern = "illumination/daylight/left/tsukuba_daylight_L_%1.png";
    const char* right_pattern = "illumination/daylight/right/tsukuba_daylight_R_%1.png";

    const QString path = myDir.absoluteFilePath( QString((left_or_right == 0) ? left_pattern : right_pattern).arg(index+1, 5, 10, QChar('0')) );

    cv::Mat1b image = cv::imread( path.toUtf8().data(), cv::IMREAD_GRAYSCALE);

    if(image.data == nullptr)
    {
        std::cout << "Could not read image!" << std::endl;
        exit(1);
    }

    return image;
}

cv::Mat1w Tsukuba::loadGroundTruthOcclusion(int index, int left_or_right)
{
    const char* left_pattern = "groundtruth/occlusion_maps/left/tsukuba_occlusion_L_%1.png";
    const char* right_pattern = "groundtruth/occlusion_maps/right/tsukuba_occlusion_R_%1.png";

    const QString path = myDir.absoluteFilePath( QString((left_or_right == 0) ? left_pattern : right_pattern).arg(index+1, 5, 10, QChar('0')) );

    cv::Mat1b image = cv::imread( path.toUtf8().data(), cv::IMREAD_GRAYSCALE);

    if(image.data == nullptr)
    {
        std::cout << "Could not read image!" << std::endl;
        exit(1);
    }

    cv::Mat1w ret;
    image.convertTo(ret, CV_16UC1);

    return ret;
}

cv::Mat1w Tsukuba::loadGroundTruthDisparity(int index, int left_or_right)
{
    const char* left_pattern = "groundtruth/disparity_maps/left/tsukuba_disparity_L_%1.png";
    const char* right_pattern = "groundtruth/disparity_maps/right/tsukuba_disparity_R_%1.png";

    const QString path = myDir.absoluteFilePath( QString((left_or_right == 0) ? left_pattern : right_pattern).arg(index+1, 5, 10, QChar('0')) );

    cv::Mat1b image = cv::imread( path.toUtf8().data(), cv::IMREAD_GRAYSCALE);

    if(image.data == nullptr)
    {
        std::cout << "Could not read image!" << std::endl;
        exit(1);
    }

    cv::Mat1w ret;
    image.convertTo(ret, CV_16UC1);

    return ret;
}

