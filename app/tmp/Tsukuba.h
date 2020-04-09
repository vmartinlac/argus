
#pragma once

#include <QDir>
#include <opencv2/core.hpp>

class Tsukuba
{
public:

    Tsukuba(const QString& path);

    int getNumImages();
    cv::Mat3b loadImage(int index, int left_or_right);
    cv::Mat1b loadGroundTruthOcclusion(int index, int left_or_right);
    cv::Mat1s loadGroundTruthDisparity(int index, int left_or_right);

protected:

    QDir myDir;
};

