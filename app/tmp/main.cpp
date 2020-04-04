#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "LBPSM.h"
#include "Tsukuba.h"

int main(int num_args, char** args)
{
    Tsukuba data("/home/victor/datasets/new_tsukuba/NewTsukubaStereoDataset/");
    const int index = 20;

    LBPSM::Config config;

    std::vector<LBPSM::Level> pyramid;
    LBPSM::build_pyramid(data.loadImage(index, 0), data.loadImage(index, 1), config, pyramid);

    // fill pyramid with grount-truth.
    for(size_t i=0; i<pyramid.size(); i++)
    {
        if(i == 0)
        {
            pyramid.front().gt_disparity[0] = data.loadGroundTruthDisparity(index, 0);
            pyramid.front().gt_disparity[1] = data.loadGroundTruthDisparity(index, 1);
            pyramid.front().gt_occlusion[0] = data.loadGroundTruthOcclusion(index, 0);
            pyramid.front().gt_occlusion[1] = data.loadGroundTruthOcclusion(index, 1);

            auto pred_occlusion = [] (uint16_t& x)
            {
                switch(x)
                {
                case 0:
                    x = 1;
                    break;
                case 1:
                case 255:
                    x = 0;
                    break;
                default:
                    std::cout << x << std::endl;
                    std::cout << "Bad tsukuba input data!" << std::endl;
                    exit(1);
                }
            };

            std::for_each( pyramid[i].gt_occlusion[0].begin(), pyramid[i].gt_occlusion[0].end(), pred_occlusion );
            std::for_each( pyramid[i].gt_occlusion[1].begin(), pyramid[i].gt_occlusion[1].end(), pred_occlusion );
        }
        else
        {
            cv::resize( pyramid.front().gt_disparity[0], pyramid[i].gt_disparity[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( pyramid.front().gt_disparity[1], pyramid[i].gt_disparity[1], pyramid[i].image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( pyramid.front().gt_occlusion[0], pyramid[i].gt_occlusion[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( pyramid.front().gt_occlusion[1], pyramid[i].gt_occlusion[1], pyramid[i].image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);

            const double gamma = pyramid[i].level0_to_levelk;

            auto pred_disparity = [gamma] (uint16_t& x)
            {
                x = static_cast<uint16_t>(gamma * x);
            };

            std::for_each( pyramid[i].gt_disparity[0].begin(), pyramid[i].gt_disparity[0].end(), pred_disparity );
            std::for_each( pyramid[i].gt_disparity[1].begin(), pyramid[i].gt_disparity[1].end(), pred_disparity );
        }
    }

#if 0
    for(LBPSM::Level& l : pyramid)
    {
        std::cout << l.level << " " << l.image[0].cols << std::endl;
        cv::imshow("image_left", l.image[0]);
        cv::imshow("image_right", l.image[1]);
        cv::imshow("gt_disparity_left", l.gt_disparity[0]*65535.0*20.0/256.0);
        cv::imshow("gt_disparity_right", l.gt_disparity[1]*65535.0*20.0/256.0);
        cv::imshow("gt_occlusion_left", l.gt_occlusion[0]*65535);
        cv::imshow("gt_occlusion_right", l.gt_occlusion[1]*65535);
        cv::waitKey(0);
    }
    exit(0);
#endif

    LBPSM::run(pyramid, config);

    return 0;
}

