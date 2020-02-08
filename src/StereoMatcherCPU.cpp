#include <iostream>
#include <opencv2/imgproc.hpp>
#include "StereoMatcherCPU.h"
#include "LoopyBeliefPropagation.h"

StereoMatcherCPU::StereoMatcherCPU()
{
    myScaleFactor = 0.7;
    myMinLevelWidth = 80;
    myNumFixedPointIterations = 3;
    myNumBeliefPropagationIterations = 30;
    myNumDisparities = 20;

    /*
    const int num_disparities = 20;
    myDisparities[0].resize(num_disparities);
    myDisparities[1].resize(num_disparities);
    for(int i=0; i<num_disparities; i++)
    {
        myDisparityTable[0][i] = -static_cast<float>(i);
        myDisparityTable[1][i] = static_cast<float>(i);
    }
    */
}

StereoMatcherCPU::~StereoMatcherCPU()
{
}

void StereoMatcherCPU::compute(
    const cv::Mat1b& left,
    const cv::Mat1b& right,
    cv::Mat1f& disparity)
{
    if( left.size() != right.size() )
    {
        std::cerr << "Size mismatch" << std::endl;
        exit(1);
    }

    std::cout << "Building image pyramids..." << std::endl;

    std::vector<Level> levels;

    levels.emplace_back();
    levels.back().thumbnails[0] = left;
    levels.back().thumbnails[1] = right;

    while( myScaleFactor * static_cast<double>(levels.back().thumbnails[0].cols) >= static_cast<double>(myMinLevelWidth) )
    {
        cv::Mat1b new_left;
        cv::Mat1b new_right;

        cv::resize(levels.back().thumbnails[0], new_left, cv::Size(), myScaleFactor, myScaleFactor, cv::INTER_AREA);
        cv::resize(levels.back().thumbnails[1], new_right, cv::Size(), myScaleFactor, myScaleFactor, cv::INTER_AREA);

        levels.emplace_back();
        levels.back().thumbnails[0] = new_left;
        levels.back().thumbnails[1] = new_right;
    }

    const size_t num_levels = levels.size();

    for(size_t i=0; i<num_levels; i++)
    {
        Level& level = levels[num_levels-1-i];

        if(i > 0)
        {
            Level& old_level = levels[num_levels-i];

            cv::resize(old_level.disparity[0], level.disparity[0], level.thumbnails[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize(old_level.occlusion[0], level.occlusion[0], level.thumbnails[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize(old_level.disparity[1], level.disparity[1], level.thumbnails[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize(old_level.occlusion[1], level.occlusion[1], level.thumbnails[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
        }
        else
        {
            level.disparity[0].create(level.thumbnails[0].size());
            level.occlusion[0].create(level.thumbnails[0].size());
            level.disparity[1].create(level.thumbnails[1].size());
            level.occlusion[1].create(level.thumbnails[1].size());

            std::fill(level.disparity[0].begin(), level.disparity[0].end(), 0);
            std::fill(level.occlusion[0].begin(), level.occlusion[0].end(), 0);
            std::fill(level.disparity[1].begin(), level.disparity[1].end(), 0);
            std::fill(level.occlusion[1].begin(), level.occlusion[1].end(), 0);
        }

        for(size_t j=0; j<myNumFixedPointIterations; j++)
        {
            updateDisparity(level, 0);
            updateDisparity(level, 1);
            updateOcclusion(level, 0);
            updateOcclusion(level, 1);
        }
    }

    disparity.create(left.size());
    std::transform(levels.front().disparity[0].begin(), levels.front().disparity[0].end(), disparity.begin(), [this] (int x) { return static_cast<float>(x); } );
}

void StereoMatcherCPU::updateDisparity(Level& level, int image)
{
    /*
    auto data_cost = [this] (const cv::Point& pt, int label) -> float
    {
        return 0.0f;
    };

    auto discontinuity_cost = [this] (int label0, int label1) -> float
    {
        return 0.0;
    };

    LoopyBeliefPropagation::execute(
        myDisparityTable[image].size(),
        myImages[image]->size(),
        data_cost,
        discontinuity_cost,
        myNumBeliefPropagationIterations,
        myDisparity[image]);
    */
}

void StereoMatcherCPU::updateOcclusion(Level& level, int image)
{
    /*
    const int other_image = (image + 1) % 2;

    cv::Mat1b W(myImages[image]->size());
    std::fill(W.begin(), W.end(), 1);

    for(auto it=myDisparity[image].begin(); it!=myDisparity[image].end(); it++)
    {
        const cv::Point other_pt = it.pos() + cv::Point(myDisparityTable[other_image][*it], 0);

        if( 0 <= other_pt.x && other_pt.x < W.cols && 0 <= other_pt.y && other_pt.y < W.rows )
        {
            W(other_pt) = 0;
        }
    }

    auto data_cost = [this] (const cv::Point& pt, int label) -> float
    {
        float ret = 0.0f;
*/

        /*
        if(label)
        {
            ret = myEtaO;
        }
        else
        {
            // TODO
        }

        // TODO
        */
        /*

        return 0.0f;
    };

    auto discontinuity_cost = [this] (int label0, int label1) -> float
    {
        float ret = 0.0f;

        */
        /*
        if(label0 != label1)
        {
            ret = static_cast<float>(myBetaO);
        }
        else
        {
            ret = 0.0f;
        }
        */

        /*
        return ret;
    };

    LoopyBeliefPropagation::execute(
        2,
        myImages[0]->size(),
        data_cost,
        discontinuity_cost,
        myNumBeliefPropagationIterations,
        myOcclusion[image]);
    */
}

