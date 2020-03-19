#include <opencv2/imgproc.hpp>
#include "LBPSM.h"
#include "LBP.h"

void LBPSM::build_pyramid(
    const cv::Mat1b& left,
    const cv::Mat1b& right,
    const Config& config,
    std::vector<Level>& pyramid)
{
    pyramid.clear();

    pyramid.emplace_back();
    pyramid.back().level = 0;
    pyramid.back().image[0] = left;
    pyramid.back().image[1] = right;

    while( config.enable_multiscale && config.level0_to_level1 * static_cast<double>(pyramid.back().image[0].cols) >= static_cast<double>(config.min_level_width) )
    {
        Level new_level;

        new_level.level = pyramid.back().level + 1;

        cv::resize(pyramid.back().image[0], new_level.image[0], cv::Size(), config.level0_to_level1, config.level0_to_level1, cv::INTER_AREA);
        cv::resize(pyramid.back().image[1], new_level.image[1], cv::Size(), config.level0_to_level1, config.level0_to_level1, cv::INTER_AREA);

        pyramid.push_back(std::move(new_level));
    }
}

void LBPSM::prepare_level(Level& level)
{
    level.disparity[0].create(level.image[0].size());
    level.occlusion[0].create(level.image[0].size());
    level.disparity[1].create(level.image[1].size());
    level.occlusion[1].create(level.image[1].size());

    std::fill(level.disparity[0].begin(), level.disparity[0].end(), 0);
    std::fill(level.occlusion[0].begin(), level.occlusion[0].end(), 0);
    std::fill(level.disparity[1].begin(), level.disparity[1].end(), 0);
    std::fill(level.occlusion[1].begin(), level.occlusion[1].end(), 0);
}

void LBPSM::prepare_level_from(Level& level, const Level& old_level)
{
    cv::resize(old_level.disparity[0], level.disparity[0], level.image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(old_level.occlusion[0], level.occlusion[0], level.image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(old_level.disparity[1], level.disparity[1], level.image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
    cv::resize(old_level.occlusion[1], level.occlusion[1], level.image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
}

void LBPSM::run(const cv::Mat1b& left, const cv::Mat1b& right, const Config& config, cv::Mat1w& disparity)
{
    std::vector<Level> pyramid;
    build_pyramid(left, right, config, pyramid);

    const size_t num_levels = pyramid.size();

    std::cout << "Number of levels: " << num_levels << std::endl;

    for(size_t i=0; i<num_levels; i++)
    {
        std::cout << "Processing level " << i << "..." << std::endl;
        Level& level = pyramid[num_levels-1-i];

        if(i > 0)
        {
            const Level& old_level = pyramid[num_levels-i];
            prepare_level_from(level, old_level);
        }
        else
        {
            prepare_level(level);
        }

        for(size_t j=0; j<config.num_fixed_point_iterations; j++)
        {
            std::cout << "    Fixed point iteration " << j << "..." << std::endl;

            std::cout << "        Updating left disparity..." << std::endl;
            update_disparity(level, config, 0);
            //cv::imwrite("DisparityLeft.png", level.disparity[0] * 65535.0 / double(myNumDisparities-1));

            std::cout << "        Updating right disparity..." << std::endl;
            update_disparity(level, config, 1);
            //cv::imwrite("DisparityRight.png", level.disparity[1] * 65535.0 / double(myNumDisparities-1));

            std::cout << "        Updating left occlusion..." << std::endl;
            update_occlusion(level, config, 0);
            //cv::imwrite("OcclusionLeft.png", level.occlusion[0] * 65535.0 / 1.0);

            std::cout << "        Updating right occlusion..." << std::endl;
            update_occlusion(level, config, 1);
            //cv::imwrite("OcclusionRight.png", level.occlusion[1] * 65535.0 / 1.0);

            ////////
            /*
            cv::imshow("left_image", level.image[0]);
            //cv::imshow("left_occlusion", level.occlusion[0]*255);
            //cv::imshow("left_disparity", level.disparity[0]);
            cv::imshow("right_image", level.image[1]);
            cv::imshow("right_occlusion", cv::Mat1f(level.occlusion[1]*255));
            cv::imshow("right_disparity", cv::Mat1f(level.disparity[1]*255.0/double(myNumDisparities-1)));
            int mini = level.disparity[0](0,0);
            int maxi = level.disparity[0](0,0);
            for(int i=0; i<level.disparity[1].rows; i++)
            {
                for(int j=0; j<level.disparity[1].cols; j++)
                {
                    maxi = std::max<int>(maxi, level.disparity[0](i,j));
                    mini = std::min<int>(mini, level.disparity[0](i,j));
                }
            }
            std::cout << maxi << " " << mini << std::endl;
            cv::waitKey(0);
            */
        }
    }

    disparity.create(left.size());
    std::transform(
        pyramid.front().disparity[0].begin(),
        pyramid.front().disparity[0].end(),
        disparity.begin(),
        [&config] (int x) { return config.directions[0]*static_cast<float>(x); } );
}

float LBPSM::robust_norm_disparity(float x, const Config& config)
{
    return std::min( config.tau, config.lambda*std::fabs(x) );
}

float LBPSM::robust_norm_pixels(float x, const Config& config)
{
    return -std::log( (1.0-config.ed) * exp( -std::fabs(x) / config.sigmad ) + config.ed );
}

void LBPSM::update_disparity(Level& level, const Config& config, int image)
{
    const int other_image = (image + 1) % 2;

    auto data_cost = [&level, &config, image, other_image] (const cv::Point& pt, int label) -> float
    {
        const cv::Point other_pt = pt + cv::Point(config.directions[image]*label, 0);

        float ret = 0.0f;

        if( (0 <= other_pt.x && other_pt.x < level.image[other_image].cols) == false || level.occlusion[other_image](other_pt) )
        {
            ret += config.betaw;
        }

        if(level.occlusion[image](pt) == 0)
        {
            const float value = level.image[image](pt);
            float other_value = 0.0f;

            if(0 <= other_pt.x && other_pt.x < level.image[other_image].cols)
            {
                other_value = level.image[other_image](other_pt);
            }

            ret += robust_norm_pixels( value - other_value, config );
        }

        return ret;
    };

    auto discontinuity_cost = [&level, &config] (const cv::Point& pt0, int label0, const cv::Point& pt1, int label1) -> float
    {
        float ret = 0.0f;

        if( level.occlusion[0](pt0) == level.occlusion[0](pt1) )
        {
            ret += robust_norm_disparity( label1 - label0, config );
        }

        return ret;
    };

    std::cout << "            LoopyBeliefPropagation starts" << std::endl;

    LBP::execute(
        config.num_disparities,
        level.image[image].size(),
        data_cost,
        discontinuity_cost,
        config.num_belief_propagation_iterations,
        level.disparity[image]);

    std::cout << "            LoopyBeliefPropagation ends" << std::endl;
}

void LBPSM::update_occlusion(Level& level, const Config& config, int image)
{
    const int other_image = (image + 1) % 2;

    cv::Mat1b W( level.image[image].size() );

    std::fill(W.begin(), W.end(), 1);

    for(auto it=level.disparity[other_image].begin(); it!=level.disparity[other_image].end(); it++)
    {
        const cv::Point pt = it.pos() + cv::Point(config.directions[other_image] * *it, 0);

        if( 0 <= pt.x && pt.x < W.cols && 0 <= pt.y && pt.y < W.rows )
        {
            W(pt) = 0;
        }
    }

    auto data_cost = [&config, &W, &level, image, other_image] (const cv::Point& pt, int label) -> float
    {
        double ret = config.betaw * std::fabs( label - W(pt) );

        if(label)
        {
            ret += config.eta0;
        }
        else
        {
            const double value = level.image[image](pt);
            double other_value = 0.0f;

            const cv::Point other_pt = pt + cv::Point(config.directions[image]*level.disparity[image](pt), 0);

            if(0 <= other_pt.x && other_pt.x < level.image[other_image].cols)
            {
                other_value = level.image[other_image](other_pt);
            }

            ret += robust_norm_pixels( value - other_value, config );
        }

        return ret;
    };

    auto discontinuity_cost = [&config, &level, image] (const cv::Point& pt0, int label0, const cv::Point& pt1, int label1) -> float
    {
        double ret = 0.0f;

        if(label0 != label1)
        {
            ret = static_cast<double>(config.beta0);
        }

        return static_cast<double>(ret);
    };

    std::cout << "            LoopyBeliefPropagation starts" << std::endl;

    LBP::execute(
        2,
        level.image[image].size(),
        data_cost, discontinuity_cost,
        config.num_belief_propagation_iterations,
        level.occlusion[image]);

    std::cout << "            LoopyBeliefPropagation ends" << std::endl;
}

