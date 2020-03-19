#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "LBPSM.h"

int main(int num_args, char** args)
{
    const std::string number = "00020";
    const std::string root_path = "/home/victor/datasets/new_tsukuba//NewTsukubaStereoDataset/";

    const std::string filename_image_left = root_path + "/illumination/daylight/left/tsukuba_daylight_L_" + number + ".png";
    const std::string filename_image_right = root_path + "/illumination//daylight//right/tsukuba_daylight_R_" + number + ".png";
    const std::string filename_disparity_left = root_path + "/groundtruth//disparity_maps//left/tsukuba_disparity_L_" + number + ".png";
    const std::string filename_disparity_right = root_path + "/groundtruth//disparity_maps//right/tsukuba_disparity_R_" + number + ".png";
    const std::string filename_occlusion_left = root_path + "/groundtruth/occlusion_maps/left/tsukuba_occlusion_L_" + number + ".png";
    const std::string filename_occlusion_right = root_path + "/groundtruth/occlusion_maps/right/tsukuba_occlusion_R_" + number + ".png";

    cv::Mat1b left = cv::imread(filename_image_left, cv::IMREAD_GRAYSCALE);
    cv::Mat1b right = cv::imread(filename_image_right, cv::IMREAD_GRAYSCALE);

    std::vector<LBPSM::Level> pyramid;
    LBPSM::Config config;
    LBPSM::build_pyramid(left, right, config, pyramid);

    /*
    cv::Mat1w disparity;
    LBPSM::Config config;
    LBPSM::run(left, right, config, disparity);
    */

    cv::Mat1w disparity_left;
    cv::Mat1w disparity_right;
    cv::Mat1w occlusion_left;
    cv::Mat1w occlusion_right;
    {
        cv::Mat1b _disparity_left = cv::imread(filename_disparity_left, cv::IMREAD_GRAYSCALE);
        cv::Mat1b _disparity_right = cv::imread(filename_disparity_right, cv::IMREAD_GRAYSCALE);
        cv::Mat1b _occlusion_left = cv::imread(filename_occlusion_left, cv::IMREAD_GRAYSCALE);
        cv::Mat1b _occlusion_right = cv::imread(filename_occlusion_right, cv::IMREAD_GRAYSCALE);
        if( _disparity_left.data == nullptr || _disparity_right.data == nullptr || _occlusion_left.data == nullptr || _occlusion_right.data == nullptr )
        {
            std::cout << "Could not read!" << std::endl;
            exit(1);
        }

        _disparity_left.convertTo(disparity_left, CV_16UC1);
        _disparity_right.convertTo(disparity_right, CV_16UC1);
        _occlusion_left.convertTo(occlusion_left, CV_16UC1);
        _occlusion_right.convertTo(occlusion_right, CV_16UC1);
    }

    std::vector<LBPSM::Level> ground_truth(pyramid.size());

    ground_truth.front().disparity[0] = disparity_left;
    ground_truth.front().disparity[1] = disparity_right;
    ground_truth.front().occlusion[0] = occlusion_left;
    ground_truth.front().occlusion[1] = occlusion_right;

    for(size_t i=1; i<pyramid.size(); i++)
    {
        cv::resize(disparity_left, ground_truth[i].disparity[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(disparity_right, ground_truth[i].disparity[1], pyramid[i].image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(occlusion_left, ground_truth[i].occlusion[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(occlusion_right, ground_truth[i].occlusion[1], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
    }

    //for(size_t i=0; i<pyramid.size(); i++)
    {
        const int i = pyramid.size() - 3;
        cv::imshow("left", pyramid[i].image[0]);
        cv::imshow("right", pyramid[i].image[1]);
        cv::imshow("disp_left", ground_truth[i].disparity[0]*655.0);
        cv::imshow("disp_right", ground_truth[i].disparity[1]*655.0);
        cv::imshow("occ_left", ground_truth[i].occlusion[0]*65535.0);
        cv::imshow("occ_right", ground_truth[i].occlusion[1]*65535.0);
        cv::waitKey(0);
    }

    return 0;
}

