
#pragma once

#include <QDir>
#include <opencv2/core.hpp>

class Tsukuba
{
public:

    Tsukuba(const QString& path);

    int getNumImages();
    cv::Mat1b loadImage(int index, int left_or_right);
    cv::Mat1w loadGroundTruthOcclusion(int index, int left_or_right);
    cv::Mat1w loadGroundTruthDisparity(int index, int left_or_right);

protected:

    QDir myDir;
};

