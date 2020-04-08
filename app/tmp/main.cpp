#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "LBPSM.h"
#include "Tsukuba.h"
#include "MarkovRandomField.h"
#include "StochasticSearchSolver.h"
#include "common.h"

namespace LBPSM
{
    class DisparityRandomField : public MarkovRandomField
    {
    public:

        DisparityRandomField(const Config& config, Level& level, int image) :
            myConfig(config),
            myLevel(level),
            myImage(image),
            myOtherImage( (image+1)%2 ),
            mySize(level.image[image].cols - 2*config.margin, level.image[image].rows - 2*config.margin)
        {
            myFactorOffset[0] = mySize.width * mySize.height;
            myFactorOffset[1] = myFactorOffset[0] + (mySize.width-1) * mySize.height;
            myFactorOffset[2] = myFactorOffset[1] + mySize.width * (mySize.height-1);
        }

        int getNumVariables() const override
        {
            return mySize.width * mySize.height;
        }

        int getNumFactors() const override
        {
            return myFactorOffset[2];
        }

        void getVariables(int factor, std::vector<int>& variables) const override
        {
            if( factor < 0 || myFactorOffset[2] <= factor )
            {
                ABORT("Internal error!");
            }

            if(factor < myFactorOffset[0])
            {
                variables.assign({factor});
            }
            else if(factor < myFactorOffset[1])
            {
                const int x = (factor-myFactorOffset[0]) % (mySize.width-1);
                const int y = (factor-myFactorOffset[0]) / (mySize.width-1);
                const int v0 = y*mySize.width + x;
                const int v1 = y*mySize.width + (x+1);
                variables.assign({v0, v1});
            }
            else if(factor < myFactorOffset[2])
            {
                const int x = (factor-myFactorOffset[1]) % mySize.width;
                const int y = (factor-myFactorOffset[1]) / mySize.width;
                const int v0 = y*mySize.width + x;
                const int v1 = (y+1)*mySize.width + x;
                variables.assign({v0, v1});
            }
            else
            {
                ABORT("Internal error!");
            }
        }

        void getFactors(int variable, std::vector<int>& factors) const override
        {
            const cv::Point pt(variable % mySize.width, variable / mySize.width);

            if( pt.x >= mySize.width || pt.y >= mySize.height )
            {
                ABORT("Internal error!");
            }

            factors.assign({variable});

            if(0 <= pt.x-1)
            {
                factors.push_back(myFactorOffset[0] + pt.y*(mySize.width-1) + (pt.x-1));
            }

            if(pt.x+1 < mySize.width)
            {
                factors.push_back(myFactorOffset[0] + pt.y*(mySize.width-1) + pt.x);
            }

            if(0 <= pt.y-1)
            {
                factors.push_back(myFactorOffset[1] + (pt.y-1)*mySize.width + pt.x);
            }

            if(pt.y+1 < mySize.height)
            {
                factors.push_back(myFactorOffset[1] + pt.y*mySize.width + pt.x);
            }
        }

        int getNumLabels(int variable) const override
        {
            return myConfig.num_disparities;
        }

        double evaluateEnergy(int factor, std::vector<int>& node_labels) const override
        {
            double ret = 0.0;

            if( factor < myFactorOffset[0] )
            {
                const int x = factor % mySize.width;
                const int y = factor / mySize.width;
                const int xo = x + myConfig.directions[myOtherImage]*node_labels[0];

                if( xo < 0 || myLevel.image[myOtherImage].cols <= xo )
                {
                    ABORT("Internal error!");
                }

                const double gray = myLevel.image[myImage](y,x);
                const double other_gray = myLevel.image[myOtherImage](y,xo);

                ret = std::fabs(other_gray - gray);

                const double alphax = (x - 0.5*mySize.width) / (0.5*mySize.width);
                const double alphay = (y - 0.5*mySize.height) / (0.5*mySize.height);
                //const double target = 1.0 - std::hypot( alphax, alphay );
                double target = double(y) / double(mySize.height);
                if( y > mySize.height/2)
                {
                    ret = std::fabs(node_labels[0] - 13.0);
                }
                else
                {
                    ret = std::fabs(node_labels[0] - 3.0);
                }
            }
            else if( factor < myFactorOffset[1] )
            {
                const int x = (factor-myFactorOffset[0]) % (mySize.width-1);
                const int y = (factor-myFactorOffset[0]) / (mySize.width-1);
            }
            else if( factor < myFactorOffset[2] )
            {
                const int x = (factor-myFactorOffset[1]) % mySize.width;
                const int y = (factor-myFactorOffset[1]) / mySize.width;
            }
            else
            {
                ABORT("Internal error!");
            }

            return ret;
        }

    protected:

        const Config& myConfig;
        Level& myLevel;
        const int myImage;
        const int myOtherImage;
        const cv::Size mySize;
        int myFactorOffset[3];
    };
};

int main(int num_args, char** args)
{
    //Tsukuba data("/home/victor/datasets/new_tsukuba/NewTsukubaStereoDataset/");
    Tsukuba data("/home/victor/tsukuba/");
    const int index = 999;

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
                    ABORT("Bad tsukuba input data!");
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

    LBPSM::DisparityRandomField field(config, pyramid.back(), 0);

    StochasticSearchSolver solver;
    std::vector<int> result;

    solver.solve(&field, result, false);

    cv::Mat1b nada(pyramid.back().image[0].size());
    nada = 0;
    for(int i=0; i<pyramid.back().image[0].rows - 2*config.margin; i++)
    {
        for(int j=0; j<pyramid.back().image[0].cols - 2*config.margin; j++)
        {
            nada(config.margin+i, config.margin+j) = result[ i*(pyramid.back().image[0].cols-2*config.margin) + j ] * 255.0 / double(config.num_disparities);
        }
    }
    cv::imshow("", nada);
    cv::waitKey(0);

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

    //LBPSM::run(pyramid, config);

    return 0;
}

