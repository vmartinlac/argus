
#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace LBPSM
{
    struct Level
    {
        int level;
        double level0_to_levelk;

        cv::Mat3b image[2];
        cv::Mat1s disparity[2];
        cv::Mat1b occlusion[2];

        cv::Mat1s gt_disparity[2];
        cv::Mat1b gt_occlusion[2];
    };

    struct Config
    {
        Config();

        // options for multiscale.
        bool enable_multiscale;
        double level0_to_level1;
        int min_level_width;

        // options for the algorithm.
        int num_fixed_point_iterations;
        int num_belief_propagation_iterations;
        int num_disparities;
        int margin;

        // directions
        int directions[2];

        // parameters of mathematical model.
        double eta0;
        double betaw;
        double beta0;
        double ed;
        double sigmad;
        double lambda;
        double tau;
    };

    void update_disparity(
        Level& level,
        const Config& config,
        int image);

    void update_occlusion(
        Level& level,
        const Config& config,
        int image);

    void build_pyramid(
        const cv::Mat3b& left,
        const cv::Mat3b& right,
        const Config& config,
        std::vector<Level>& pyramid);

    void init_disparity_and_occlusion(
        std::vector<Level>& pyramid,
        size_t index);

    void run(
        std::vector<Level>& pyramid,
        const Config& config);

    void run(
        const cv::Mat1b& left,
        const cv::Mat1b& right,
        const Config& config,
        cv::Mat1w& disparity);

    float robust_norm_disparity(float x, const Config& config);

    float robust_norm_pixels(float x, const Config& config);
}

