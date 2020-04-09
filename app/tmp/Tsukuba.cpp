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

cv::Mat3b Tsukuba::loadImage(int index, int left_or_right)
{
    const char* left_pattern = "illumination/daylight/left/tsukuba_daylight_L_%1.png";
    const char* right_pattern = "illumination/daylight/right/tsukuba_daylight_R_%1.png";

    const QString path = myDir.absoluteFilePath( QString((left_or_right == 0) ? left_pattern : right_pattern).arg(index+1, 5, 10, QChar('0')) );

    cv::Mat3b image = cv::imread( path.toUtf8().data(), cv::IMREAD_COLOR);

    if(image.data == nullptr)
    {
        std::cout << path.toStdString() << std::endl;
        std::cout << "Could not read image!" << std::endl;
        exit(1);
    }

    return image;
}

cv::Mat1b Tsukuba::loadGroundTruthOcclusion(int index, int left_or_right)
{
    const char* left_pattern = "groundtruth/occlusion_maps/left/tsukuba_occlusion_L_%1.png";
    const char* right_pattern = "groundtruth/occlusion_maps/right/tsukuba_occlusion_R_%1.png";

    const QString path = myDir.absoluteFilePath( QString((left_or_right == 0) ? left_pattern : right_pattern).arg(index+1, 5, 10, QChar('0')) );

    cv::Mat1b image = cv::imread( path.toUtf8().data(), cv::IMREAD_GRAYSCALE);

    if(image.data == nullptr)
    {
        std::cout << path.toStdString() << std::endl;
        std::cout << "Could not read image!" << std::endl;
        exit(1);
    }

    cv::Mat1b ret(image.size());
    std::transform(
        image.begin(),
        image.end(),
        ret.begin(),
        [] (uint8_t x) -> uint8_t { if(x == 0) return 0; else return 1; });

    return ret;
}

cv::Mat1s Tsukuba::loadGroundTruthDisparity(int index, int left_or_right)
{
    const char* left_pattern = "groundtruth/disparity_maps/left/tsukuba_disparity_L_%1.png";
    const char* right_pattern = "groundtruth/disparity_maps/right/tsukuba_disparity_R_%1.png";

    const QString path = myDir.absoluteFilePath( QString((left_or_right == 0) ? left_pattern : right_pattern).arg(index+1, 5, 10, QChar('0')) );

    cv::Mat1b image = cv::imread( path.toUtf8().data(), cv::IMREAD_GRAYSCALE);

    if(image.data == nullptr)
    {
        std::cout << path.toStdString() << std::endl;
        std::cout << "Could not read image!" << std::endl;
        exit(1);
    }

    cv::Mat1s ret(image.size());

    const int16_t dir = (left_or_right == 0) ? -1 : 1;

    std::transform(
        image.begin(),
        image.end(),
        ret.begin(),
        [dir] (uint8_t x) -> int16_t { return dir*int16_t(x); });

    return ret;
}

