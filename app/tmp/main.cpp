#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "LBPSM.h"

struct Input
{
    cv::Mat1b image[2];
    cv::Mat1w disparity[2];
    cv::Mat1w occlusion[2];
};

void load_input(Input& i)
{
    const std::string number = "00020";
    const std::string root_path = "/home/victor/datasets/new_tsukuba//NewTsukubaStereoDataset/";

    const std::string filename_image_left = root_path + "/illumination/daylight/left/tsukuba_daylight_L_" + number + ".png";
    const std::string filename_image_right = root_path + "/illumination//daylight//right/tsukuba_daylight_R_" + number + ".png";
    const std::string filename_disparity_left = root_path + "/groundtruth//disparity_maps//left/tsukuba_disparity_L_" + number + ".png";
    const std::string filename_disparity_right = root_path + "/groundtruth//disparity_maps//right/tsukuba_disparity_R_" + number + ".png";
    const std::string filename_occlusion_left = root_path + "/groundtruth/occlusion_maps/left/tsukuba_occlusion_L_" + number + ".png";
    const std::string filename_occlusion_right = root_path + "/groundtruth/occlusion_maps/right/tsukuba_occlusion_R_" + number + ".png";

    i.image[0] = cv::imread(filename_image_left, cv::IMREAD_GRAYSCALE);
    i.image[1] = cv::imread(filename_image_right, cv::IMREAD_GRAYSCALE);

    if( i.image[0].data == nullptr || i.image[1].data == nullptr )
    {
        std::cout << "Could not read!" << std::endl;
        exit(1);
    }

    cv::Mat1b _disparity_left = cv::imread(filename_disparity_left, cv::IMREAD_GRAYSCALE);
    cv::Mat1b _disparity_right = cv::imread(filename_disparity_right, cv::IMREAD_GRAYSCALE);
    cv::Mat1b _occlusion_left = cv::imread(filename_occlusion_left, cv::IMREAD_GRAYSCALE);
    cv::Mat1b _occlusion_right = cv::imread(filename_occlusion_right, cv::IMREAD_GRAYSCALE);

    if( _disparity_left.data == nullptr || _disparity_right.data == nullptr || _occlusion_left.data == nullptr || _occlusion_right.data == nullptr )
    {
        std::cout << "Could not read!" << std::endl;
        exit(1);
    }

    if(
        _disparity_left.size() != i.image[0].size() ||
        _disparity_right.size() != i.image[1].size() ||
        _occlusion_left.size() != i.image[0].size() ||
        _occlusion_right.size() != i.image[1].size() )
    {
        std::cout << "Incoherent image sizes!" << std::endl;
        exit(1);
    }

    _disparity_left.convertTo(i.disparity[0], CV_16UC1);
    _disparity_right.convertTo(i.disparity[1], CV_16UC1);
    _occlusion_left.convertTo(i.occlusion[0], CV_16UC1);
    _occlusion_right.convertTo(i.occlusion[1], CV_16UC1);
}

int main(int num_args, char** args)
{
    Input input;
    load_input(input);

    LBPSM::Config config;

    std::vector<LBPSM::Level> pyramid;
    LBPSM::build_pyramid(input.image[0], input.image[1], config, pyramid);

    // fill pyramid with grount-truth.
    for(size_t i=0; i<pyramid.size(); i++)
    {
        if(i == 0)
        {
            pyramid.front().gt_disparity[0] = input.disparity[0];
            pyramid.front().gt_disparity[1] = input.disparity[1];
            pyramid.front().gt_occlusion[0] = input.occlusion[0];
            pyramid.front().gt_occlusion[1] = input.occlusion[1];
        }
        else
        {
            cv::resize( input.disparity[0], pyramid[i].gt_disparity[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( input.disparity[1], pyramid[i].gt_disparity[1], pyramid[i].image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( input.occlusion[0], pyramid[i].gt_occlusion[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( input.occlusion[1], pyramid[i].gt_occlusion[1], pyramid[i].image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
        }
    }

#if 0
    for(LBPSM::Level& l : pyramid)
    {
        std::cout << l.level << " " << l.image[0].cols << std::endl;
        cv::imshow("image_left", l.image[0]);
        cv::imshow("image_right", l.image[1]);
        cv::imshow("gt_disparity_left", l.gt_disparity[0]*65535.0/256.0);
        cv::imshow("gt_disparity_right", l.gt_disparity[1]*65535.0/256.0);
        cv::imshow("gt_occlusion_left", l.gt_occlusion[0]*65535);
        cv::imshow("gt_occlusion_right", l.gt_occlusion[1]*65535);
        cv::waitKey(0);
    }
    exit(0);
#endif

    LBPSM::run(pyramid, config);

    return 0;
}

