#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "LBPSM.h"

LBPSM::Config::Config()
{
    enable_multiscale = true;
    level0_to_level1 = 0.65;
    min_level_width = 200;

    num_fixed_point_iterations = 1;
    num_belief_propagation_iterations = 200;
    num_disparities = 30;
    margin = 40;

    directions[0] = -1;
    directions[1] = 1;

    lambda = 0.7;
    tau = 2.5;
    eta0 = 2.5;
    sigmad = 4.0;
    ed = 1.0e-2;
    betaw = 4.0;
    beta0 = 1.4;
}

void LBPSM::build_pyramid(
    const cv::Mat3b& left,
    const cv::Mat3b& right,
    const Config& config,
    std::vector<Level>& pyramid)
{
    pyramid.clear();

    pyramid.emplace_back();
    pyramid.back().level = 0;
    pyramid.back().level0_to_levelk = 1.0;
    pyramid.back().image[0] = left;
    pyramid.back().image[1] = right;

    while( config.enable_multiscale && config.level0_to_level1 * static_cast<double>(pyramid.back().image[0].cols - 2*config.margin) >= static_cast<double>(config.min_level_width) )
    {
        Level new_level;

        new_level.level = pyramid.back().level + 1;

        new_level.level0_to_levelk = std::pow(config.level0_to_level1, new_level.level);

        cv::resize(left, new_level.image[0], cv::Size(), new_level.level0_to_levelk, new_level.level0_to_levelk, cv::INTER_AREA);
        cv::resize(right, new_level.image[1], cv::Size(), new_level.level0_to_levelk, new_level.level0_to_levelk, cv::INTER_AREA);

        pyramid.push_back(std::move(new_level));
    }
}

/*
void LBPSM::init_disparity_and_occlusion(std::vector<Level>& pyramid, size_t level_index)
{
    if( level_index+1 < pyramid.size() )
    {
        Level& level = pyramid[level_index];
        Level& old_level = pyramid[level_index+1];

        cv::resize(old_level.disparity[0], level.disparity[0], level.image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(old_level.disparity[1], level.disparity[1], level.image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(old_level.occlusion[0], level.occlusion[0], level.image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
        cv::resize(old_level.occlusion[1], level.occlusion[1], level.image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);

        const double old_level_to_current_level = pyramid[level_index].level0_to_levelk / pyramid[level_index+1].level0_to_levelk; // should be equal to 1.0/config.level0_to_level1.

        auto proc = [old_level_to_current_level] (uint16_t& x)
        {
            x = static_cast<uint16_t>(old_level_to_current_level * x);
        };

        std::for_each( level.disparity[0].begin(), level.disparity[0].end(), proc );
        std::for_each( level.disparity[1].begin(), level.disparity[1].end(), proc );
    }
    else
    {
        Level& level = pyramid[level_index];

        level.disparity[0].create(level.image[0].size());
        level.occlusion[0].create(level.image[0].size());
        level.disparity[1].create(level.image[1].size());
        level.occlusion[1].create(level.image[1].size());

        level.disparity[0] = 0;
        level.disparity[1] = 0;
        level.occlusion[0] = 0;
        level.occlusion[1] = 0;
    }
}
*/

