#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "LBPSM.h"

LBPSM::Config::Config()
{
    enable_multiscale = true;
    level0_to_level1 = 0.65;
    min_level_width = 100;

    num_fixed_point_iterations = 1;
    num_belief_propagation_iterations = 200;
    num_disparities = 15;
    margin = 20;

    directions[0] = -1;
    directions[1] = 1;

    lambda = 1.0; // TODO try other values or use the method from the article.
    tau = 2.0;
    eta0 = 2.5;
    sigmad = 4.0;
    ed = 1.0e-2;
    betaw = 4.0;
    beta0 = 1.4;
}

void LBPSM::build_pyramid(
    const cv::Mat1b& left,
    const cv::Mat1b& right,
    const Config& config,
    std::vector<Level>& pyramid)
{
    pyramid.clear();

    pyramid.emplace_back();
    pyramid.back().level = 0;
    pyramid.back().level0_to_levelk = 1.0;
    pyramid.back().image[0] = left;
    pyramid.back().image[1] = right;

    while( config.enable_multiscale && config.level0_to_level1 * static_cast<double>(pyramid.back().image[0].cols) >= static_cast<double>(config.min_level_width) )
    {
        Level new_level;

        new_level.level = pyramid.back().level + 1;

        new_level.level0_to_levelk = std::pow(config.level0_to_level1, new_level.level);

        cv::resize(left, new_level.image[0], cv::Size(), new_level.level0_to_levelk, new_level.level0_to_levelk, cv::INTER_AREA);
        cv::resize(right, new_level.image[1], cv::Size(), new_level.level0_to_levelk, new_level.level0_to_levelk, cv::INTER_AREA);

        pyramid.push_back(std::move(new_level));
    }
}

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

void LBPSM::run(const cv::Mat1b& left, const cv::Mat1b& right, const Config& config, cv::Mat1w& disparity)
{
    std::vector<Level> pyramid;
    build_pyramid(left, right, config, pyramid);

    run(pyramid, config);

    disparity.create(left.size());
    std::transform(
        pyramid.front().disparity[0].begin(),
        pyramid.front().disparity[0].end(),
        disparity.begin(),
        [&config] (int x) { return config.directions[0]*static_cast<float>(x); } );
}

void LBPSM::run(std::vector<Level>& pyramid, const Config& config)
{
    const size_t num_levels = pyramid.size();

    std::cout << "Number of levels: " << num_levels << std::endl;

    for(size_t i=0; i<num_levels; i++)
    {
        const size_t level_index = num_levels-1-i;

        Level& level = pyramid[level_index];

        std::cout << "Processing pyramid level " << level.level << "..." << std::endl;

        init_disparity_and_occlusion(pyramid, level_index);

        for(size_t j=0; j<config.num_fixed_point_iterations; j++)
        {
            std::cout << "    Fixed point iteration " << j << "..." << std::endl;

        /*
        level.disparity[0] = level.gt_disparity[0].clone();
        level.disparity[1] = level.gt_disparity[1].clone();
        */
        level.occlusion[0] = level.gt_occlusion[0].clone();
        level.occlusion[1] = level.gt_occlusion[1].clone();

            //std::cout << "        Updating left disparity..." << std::endl;
            //update_disparity(level, config, 0);

            //std::cout << "        Updating right disparity..." << std::endl;
            //update_disparity(level, config, 1);

            //std::cout << "        Updating left occlusion..." << std::endl;
            //update_occlusion(level, config, 0);

            //std::cout << "        Updating right occlusion..." << std::endl;
            //update_occlusion(level, config, 1);

        cv::imshow("disparity_left", level.disparity[0]*65535.0*20.0/256.0);
        cv::imshow("gt_disparity_left", level.gt_disparity[0]*65535.0*20.0/256.0);
        cv::imshow("disparity_right", level.disparity[1]*65535.0*20.0/256.0);
        cv::imshow("gt_disparity_right", level.gt_disparity[1]*65535.0*20.0/256.0);
        cv::imshow("occlusion_left", level.occlusion[0]*65535.0);
        cv::imshow("gt_occlusion_left", level.gt_occlusion[0]*65535.0);
        cv::imshow("occlusion_right", level.occlusion[1]*65535.0);
        cv::imshow("gt_occlusion_right", level.gt_occlusion[1]*65535.0);
        cv::waitKey(0);
        //exit(0);

        }

        /*
        cv::imshow("disparity_left", level.disparity[0]*65535.0*20.0/256.0);
        cv::imshow("gt_disparity_left", level.gt_disparity[0]*65535.0*20.0/256.0);
        cv::imshow("disparity_right", level.disparity[1]*65535.0*20.0/256.0);
        cv::imshow("gt_disparity_right", level.gt_disparity[1]*65535.0*20.0/256.0);
        cv::imshow("occlusion_left", level.occlusion[0]*65535.0);
        cv::imshow("gt_occlusion_left", level.gt_occlusion[0]*65535.0);
        cv::imshow("occlusion_right", level.occlusion[1]*65535.0);
        cv::imshow("gt_occlusion_right", level.gt_occlusion[1]*65535.0);
        cv::waitKey(0);
        */
    }
}

float LBPSM::robust_norm_disparity(float x, const Config& config)
{
    return std::min( config.tau, config.lambda * std::fabs(x) );
}

float LBPSM::robust_norm_pixels(float x, const Config& config)
{
    return -std::log( (1.0-config.ed) * std::exp( -std::fabs(x) / config.sigmad ) + config.ed );
}

/*
void LBPSM::update_disparity(Level& level, const Config& config, int image)
{
    const cv::Size image_size = level.image[image].size();

    const int other_image = (image+1) % 2;

    std::vector<cv::Point> neighbors;
    cv::Mat1b connections;
    cv::Mat1f data_cost;
    cv::Mat1f discontinuity_cost;

    // initialize neighbors array.

    neighbors.resize(4);
    neighbors[0] = cv::Point(-1, 0);
    neighbors[1] = cv::Point(0, -1);
    neighbors[2] = cv::Point(1, 0);
    neighbors[3] = cv::Point(0, 1);

    // initialize connections.

    {
        const int dims[3] = { image_size.height, image_size.width, 4 };
        connections.create(3, dims);

        for(int i=0; i<image_size.height; i++)
        {
            for(int j=0; j<image_size.width; j++)
            {
                const cv::Point this_point(j,i);

                for(int l=0; l<4; l++)
                {
                    const cv::Point that_point = this_point + neighbors[l];

                    if( 0 <= that_point.x && that_point.x < image_size.width && 0 <= that_point.y && that_point.y < image_size.height )
                    {
                        connections(cv::Vec3i(i,j,l)) = uint8_t( level.occlusion[image](this_point) == level.occlusion[image](that_point) );
                    }
                    else
                    {
                        connections(cv::Vec3i(i,j,l)) = 0;
                    }
                }
            }
        }
    }

    // initialize data cost.

    {
        const int dims[3] = { image_size.height, image_size.width, config.num_disparities };
        data_cost.create(3, dims);

        for(int i=0; i<image_size.height; i++)
        {
            for(int j=0; j<image_size.width; j++)
            {
                const cv::Point this_point(j,i);

                for(int k=0; k<config.num_disparities; k++)
                {
                    float cost = 0.0f;

                    const cv::Point other_pt = this_point + cv::Point(config.directions[image]*k, 0);

                    if( (0 <= other_pt.x && other_pt.x < level.image[other_image].cols) == false || level.occlusion[other_image](other_pt) )
                    {
                        cost += config.betaw;
                    }

                    if( level.occlusion[image](this_point) == 0 && 0 <= other_pt.x && other_pt.x < level.image[other_image].cols)
                    {
                        const float value = level.image[image](this_point);
                        const float other_value = level.image[other_image](other_pt);
                        cost += robust_norm_pixels( value - other_value, config );
                    }

                    //cost = std::fabs( float(level.image[image](this_point)) - 255.0f * k / 11.0f );

                    data_cost( cv::Vec3i(i,j,k) ) = cost;
                }
            }
        }
    }

    // initialize discontinuity cost.

    {
        const int dims[2] = { config.num_disparities, config.num_disparities };
        discontinuity_cost.create(2, dims);

        for(int i=0; i<config.num_disparities; i++)
        {
            for(int j=0; j<config.num_disparities; j++)
            {
                discontinuity_cost( cv::Vec3i(i,j) ) = robust_norm_disparity( i - j, config );
            }
        }
    }

    // run loopy belief propagation.

    LoopyBeliefPropagationSolver solver;
    solver.run(
        config.num_disparities,
        neighbors,
        connections,
        data_cost,
        discontinuity_cost,
        config.num_belief_propagation_iterations,
        level.disparity[image]);
}

void LBPSM::update_occlusion(Level& level, const Config& config, int image)
{
    const cv::Size image_size = level.image[image].size();

    const int other_image = (image+1) % 2;

    std::vector<cv::Point> neighbors(4);
    cv::Mat1b connections;
    cv::Mat1f data_cost;
    cv::Mat1f discontinuity_cost;

    // initialize neighbors array.

    neighbors[0] = cv::Point(-1, 0);
    neighbors[1] = cv::Point(0, -1);
    neighbors[2] = cv::Point(1, 0);
    neighbors[3] = cv::Point(0, 1);

    // initialize connections.

    {
        const int dims[3] = { image_size.height, image_size.width, 4 };
        connections.create(3, dims);

        for(int i=0; i<image_size.height; i++)
        {
            for(int j=0; j<image_size.width; j++)
            {
                const cv::Point this_point(j,i);

                for(int l=0; l<4; l++)
                {
                    const cv::Point that_point = this_point + neighbors[l];

                    if( 0 <= that_point.x && that_point.x < image_size.width && 0 <= that_point.y && that_point.y < image_size.height )
                    {
                        connections(cv::Vec3i(i,j,l)) = 1;
                    }
                    else
                    {
                        connections(cv::Vec3i(i,j,l)) = 0;
                    }
                }
            }
        }
    }

    // initialize data cost.

    {
        cv::Mat1b W(image_size);

        W = 1;

        for(auto it=level.disparity[other_image].begin(); it!=level.disparity[other_image].end(); it++)
        {
            const cv::Point pt = it.pos() + cv::Point(config.directions[other_image] * *it, 0);

            if( 0 <= pt.x && pt.x < W.cols && 0 <= pt.y && pt.y < W.rows )
            {
                W(pt) = 0;
            }
        }

        const int dims[3] = { image_size.height, image_size.width, 2 };
        data_cost.create(3, dims);

        for(int i=0; i<image_size.height; i++)
        {
            for(int j=0; j<image_size.width; j++)
            {
                const cv::Point this_point(j,i);

                for(int k=0; k<2; k++)
                {
                    float cost = config.betaw * std::fabs( k - W(this_point) );

                    if(k)
                    {
                        cost += config.eta0;
                    }
                    else
                    {
                        const float value = level.image[image](this_point);
                        float other_value = 0.0f;

                        const cv::Point other_pt = this_point + cv::Point(config.directions[image] * level.disparity[image](this_point), 0);

                        if(0 <= other_pt.x && other_pt.x < level.image[other_image].cols)
                        {
                            other_value = level.image[other_image](other_pt);
                        }

                        cost += robust_norm_pixels( value - other_value, config );
                    }

                    data_cost( cv::Vec3i(i,j,k) ) = cost;
                }
            }
        }
    }

    // initialize discontinuity cost.

    {
        const int dims[2] = { 2, 2 };
        discontinuity_cost.create(2, dims);

        for(int i=0; i<2; i++)
        {
            for(int j=0; j<2; j++)
            {

                if(i != j)
                {
                    discontinuity_cost( cv::Vec3i(i,j) ) = static_cast<float>(config.beta0);
                }
                else
                {
                    discontinuity_cost( cv::Vec3i(i,j) ) = 0.0f;
                }
            }
        }
    }

    // run loopy belief propagation.

    LoopyBeliefPropagationSolver solver;
    solver.run(
        2,
        neighbors,
        connections,
        data_cost,
        discontinuity_cost,
        config.num_belief_propagation_iterations,
        level.occlusion[image]);
}
*/

