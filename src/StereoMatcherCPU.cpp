#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "StereoMatcherCPU.h"
#include "LoopyBeliefPropagation.h"

StereoMatcherCPU::StereoMatcherCPU()
{
    myScaleFactor = 0.7;
    myMinLevelWidth = 30;
    myNumFixedPointIterations = 3;
    myNumBeliefPropagationIterations = 40;
    myNumDisparities = 20;
    myDirections[0] = -1;
    myDirections[1] = 1;

    myLambda = 1.0; // TODO try other values or use the method from the article.
    myTau = 2.0;
    myEta0 = 2.5;
    mySigmaD = 4.0;
    myED = 1.0e-2;
    myBetaW = 4.0;
    myBeta0 = 1.4;

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
    levels.back().level = 0;
    levels.back().thumbnails[0] = left;
    levels.back().thumbnails[1] = right;

    while( myScaleFactor * static_cast<double>(levels.back().thumbnails[0].cols) >= static_cast<double>(myMinLevelWidth) )
    {
        Level new_level;

        new_level.level = levels.back().level + 1;

        cv::resize(levels.back().thumbnails[0], new_level.thumbnails[0], cv::Size(), myScaleFactor, myScaleFactor, cv::INTER_AREA);
        cv::resize(levels.back().thumbnails[1], new_level.thumbnails[1], cv::Size(), myScaleFactor, myScaleFactor, cv::INTER_AREA);

        levels.push_back(std::move(new_level));
    }

    /*
    for(Level& l : levels)
    {
        cv::imshow("left", l.thumbnails[0]);
        cv::imshow("right", l.thumbnails[1]);
        cv::waitKey(0);
    }
    */

    const size_t num_levels = levels.size();

    std::cout << "Number of levels: " << num_levels << std::endl;

    for(size_t i=0; i<num_levels; i++)
    {
        std::cout << "Processing level " << i << "..." << std::endl;
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
            std::cout << "    Fixed point iteration " << j << "..." << std::endl;

            std::cout << "        Updating left disparity..." << std::endl;
            updateDisparity(level, 0);
            std::cout << std::all_of(level.disparity[0].begin(), level.disparity[0].end(), [] (int i) { return (i==0); } ) << std::endl;
            cv::imwrite("DisparityLeft.png", level.disparity[0] * 65535.0 / double(myNumDisparities-1));

            std::cout << "        Updating right disparity..." << std::endl;
            updateDisparity(level, 1);
            cv::imwrite("DisparityRight.png", level.disparity[1] * 65535.0 / double(myNumDisparities-1));

            std::cout << "        Updating left occlusion..." << std::endl;
            updateOcclusion(level, 0);
            cv::imwrite("OcclusionLeft.png", level.occlusion[0] * 65535.0 / 1.0);

            std::cout << "        Updating right occlusion..." << std::endl;
            updateOcclusion(level, 1);
            cv::imwrite("OcclusionRight.png", level.occlusion[1] * 65535.0 / 1.0);
        }
    }

    disparity.create(left.size());
    std::transform(levels.front().disparity[0].begin(), levels.front().disparity[0].end(), disparity.begin(), [] (int x) { return static_cast<float>(x); } );
}

void StereoMatcherCPU::updateDisparity(Level& level, int image)
{
    const int other_image = (image + 1) % 2;

    auto data_cost = [level, image, other_image, this] (const cv::Point& pt, int label) -> float
    {
        const cv::Point other_pt = pt + cv::Point(myDirections[image]*level.disparity[image](pt), 0);

        double ret = myBetaW * level.occlusion[other_image](other_pt);

        if(level.occlusion[image](pt) == 0)
        {
            const double value = level.thumbnails[image](pt);
            double other_value = 0.0f;

            if(0 <= other_pt.x && other_pt.x < level.thumbnails[other_image].cols)
            {
                other_value = level.thumbnails[other_image](other_pt);
            }

            ret += robustNormPixels( value - other_value );
        }

        return static_cast<float>(ret);
    };

    auto discontinuity_cost = [level, this] (const cv::Point& pt0, int label0, const cv::Point& pt1, int label1) -> float
    {
        float ret = 0.0f;

        if( level.occlusion[0](pt0) == level.occlusion[0](pt1) )
        {
            ret += robustNormDisparity( level.disparity[0](pt0) - level.disparity[0](pt1) );
        }

        return ret;
    };

    LoopyBeliefPropagation::execute(
        myNumDisparities,
        level.thumbnails[image].size(),
        data_cost,
        discontinuity_cost,
        myNumBeliefPropagationIterations,
        level.disparity[image]);
}

void StereoMatcherCPU::updateOcclusion(Level& level, int image)
{
    const int other_image = (image + 1) % 2;

    cv::Mat1b W( level.thumbnails[image].size() );

    std::fill(W.begin(), W.end(), 1);

    for(auto it=level.disparity[other_image].begin(); it!=level.disparity[other_image].end(); it++)
    {
        const cv::Point pt = it.pos() + cv::Point(myDirections[other_image] * *it, 0);

        if( 0 <= pt.x && pt.x < W.cols && 0 <= pt.y && pt.y < W.rows )
        {
            W(pt) = 0;
        }
    }

    auto data_cost = [this, &W, &level, image, other_image] (const cv::Point& pt, int label) -> float
    {
        double ret = myBetaW * std::fabs( label - W(pt) );

        if(label)
        {
            ret += myEta0;
        }
        else
        {
            const double value = level.thumbnails[image](pt);
            double other_value = 0.0f;

            const cv::Point other_pt = pt + cv::Point(myDirections[image]*level.disparity[image](pt), 0);

            if(0 <= other_pt.x && other_pt.x < level.thumbnails[other_image].cols)
            {
                other_value = level.thumbnails[other_image](other_pt);
            }

            ret += robustNormPixels( value - other_value );
        }

        return ret;
    };

    auto discontinuity_cost = [this, &level, image] (const cv::Point& pt0, int label0, const cv::Point& pt1, int label1) -> float
    {
        double ret = 0.0f;

        if(label0 != label1)
        {
            ret = static_cast<double>(myBeta0);
        }

        return static_cast<double>(ret);
    };

    LoopyBeliefPropagation::execute( 2, level.thumbnails[image].size(), data_cost, discontinuity_cost, myNumBeliefPropagationIterations, level.occlusion[image]);
}

float StereoMatcherCPU::robustNormDisparity(float x)
{
    return std::min( myTau, myLambda*std::fabs(x) );
}

float StereoMatcherCPU::robustNormPixels(float x)
{
    return -std::log( (1.0-myED) * exp( -std::fabs(x) / mySigmaD ) + myED );
}

