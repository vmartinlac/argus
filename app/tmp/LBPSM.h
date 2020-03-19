
#pragma once

#include <vector>
#include <opencv2/core.hpp>

namespace LBPSM
{

    struct Level
    {
        int level;
        cv::Mat1b image[2];
        cv::Mat1w disparity[2];
        cv::Mat1w occlusion[2];
    };

    struct Config
    {
        Config()
        {
            enable_multiscale = true;
            level0_to_level1 = 0.75;
            min_level_width = 30;

            num_fixed_point_iterations = 3;
            num_belief_propagation_iterations = 150;
            num_disparities = 10;

            directions[0] = -1;
            directions[1] = 1;

            lambda = 1.0; // todo try other values or use the method from the article.
            tau = 2.0;
            eta0 = 2.5;
            sigmad = 4.0;
            ed = 1.0e-2;
            betaw = 4.0;
            beta0 = 1.4;
        }

        // options for multiscale.
        bool enable_multiscale;
        double level0_to_level1;
        int min_level_width;

        // options for the algorithm.
        int num_fixed_point_iterations;
        int num_belief_propagation_iterations;
        int num_disparities;

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
        const cv::Mat1b& left,
        const cv::Mat1b& right,
        const Config& config,
        std::vector<Level>& pyramid);

    void prepare_level(
        Level& level);

    void prepare_level_from(
        Level& level,
        const Level& old_level);

    void run(
        const cv::Mat1b& left,
        const cv::Mat1b& right,
        const Config& config,
        cv::Mat1w& disparity);

    float robust_norm_disparity(float x, const Config& config);

    float robust_norm_pixels(float x, const Config& config);
}

