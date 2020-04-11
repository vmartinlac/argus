#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "LBPSM.h"
#include "Tsukuba.h"
#include "FactorGraph.h"
#include "StochasticSearchSolver.h"
#include "Common.h"

namespace LBPSM
{
    class DisparityOcclusionBase : public FactorGraph
    {
    public:

        DisparityOcclusionBase(const Config& config, Level& level, int image) :
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

        virtual void update_level(const std::vector<int>& all_labels) = 0;

        int getNumVariables() const final
        {
            return mySize.width * mySize.height;
        }

        int getNumFactors() const final
        {
            return myFactorOffset[2];
        }

        void getVariables(int factor, std::vector<int>& variables) const final
        {
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

        void getFactors(int variable, std::vector<int>& factors) const final
        {
            const int x = variable % mySize.width;
            const int y = variable / mySize.width;

            if( x >= mySize.width || y >= mySize.height )
            {
                ABORT("Internal error!");
            }

            factors.assign({variable});

            if(0 <= x-1)
            {
                factors.push_back(myFactorOffset[0] + y*(mySize.width-1) + (x-1));
            }

            if(x+1 < mySize.width)
            {
                factors.push_back(myFactorOffset[0] + y*(mySize.width-1) + x);
            }

            if(0 <= y-1)
            {
                factors.push_back(myFactorOffset[1] + (y-1)*mySize.width + x);
            }

            if(y+1 < mySize.height)
            {
                factors.push_back(myFactorOffset[1] + y*mySize.width + x);
            }
        }

    protected:

        double robust_norm_pixels(const cv::Vec3b& c0, const cv::Vec3b& c1) const
        {
            return -std::log( (1.0-myConfig.ed) * std::exp( -cv::norm(c0, c1) / myConfig.sigmad ) + myConfig.ed );
        }

        double robust_norm_disparity(double d0, double d1) const
        {
            return std::min( myConfig.tau, myConfig.lambda*std::fabs(d1 - d0) );
        }

    protected:

        const Config& myConfig;
        Level& myLevel;
        const int myImage;
        const int myOtherImage;
        const cv::Size mySize;
        int myFactorOffset[3];
    };

    class OcclusionField : public DisparityOcclusionBase
    {
    public:

        OcclusionField(const Config& config, Level& level, int image) : DisparityOcclusionBase(config, level, image)
        {
            myUntouched.create(mySize);

            myUntouched = 1;

            for(int i=0; i<mySize.height; i++)
            {
                for(int j=0; j<mySize.width; j++)
                {
                    const int jo = j + config.directions[myOtherImage] * myLevel.disparity[myOtherImage](config.margin+i, config.margin+j);

                    if(0 <= jo && jo < mySize.width)
                    {
                        myUntouched(i,jo) = 0;
                    }
                }
            }
        }

        int getNumLabels(int variable) const final
        {
            return 2;
        }

        double evaluateEnergy(int factor, const std::vector<int>& node_labels) const final
        {
            double ret = 0.0;

            if( factor < myFactorOffset[0] )
            {
                if(node_labels.size() != 1) ABORT("Internal error!");

                const int j = factor % mySize.width;
                const int i = factor / mySize.width;
                const int x = myConfig.margin + j;
                const int y = myConfig.margin + i;
                const int xo = x + myConfig.directions[myImage] * myLevel.disparity[myImage](y,x);

                if( xo < 0 || myLevel.image[myOtherImage].cols <= xo )
                {
                    ABORT("Internal error!");
                }

                if(node_labels[0])
                {
                    ret += myConfig.eta0;
                }
                else
                {
                    const cv::Vec3d color = myLevel.image[myImage](y,x);
                    const cv::Vec3d other_color = myLevel.image[myOtherImage](y,xo);
                    ret += robust_norm_pixels(color, other_color);
                }

                if(node_labels[0] != myUntouched(i,j))
                {
                    ret += myConfig.betaw;
                }
            }
            else if( factor < myFactorOffset[1] )
            {
                if(node_labels.size() != 2) ABORT("Internal error!");

                if( node_labels[0] != node_labels[1] )
                {
                    ret += myConfig.beta0;
                }
            }
            else if( factor < myFactorOffset[2] )
            {
                if(node_labels.size() != 2) ABORT("Internal error!");

                if( node_labels[0] != node_labels[1] )
                {
                    ret += myConfig.beta0;
                }
            }
            else
            {
                ABORT("Internal error!");
            }

            return ret;
        }

    protected:

        cv::Mat1b myUntouched;
    };

    class DisparityField : public DisparityOcclusionBase
    {
    public:

        DisparityField(const Config& config, Level& level, int image) : DisparityOcclusionBase(config, level, image)
        {
        }

        void update_level(const std::vector<int>& all_labels) final
        {
            cv::Mat1s& disparity = myLevel.disparity[myImage];

            disparity.create(myLevel.image[myImage].size());

            disparity = 0;

            for(int i=0; i<mySize.height; i++)
            {
                for(int j=0; j<mySize.width; j++)
                {
                    disparity(myConfig.margin+i, myConfig.margin+j) = all_labels[mySize.width*i+j] * myConfig.directions[myImage];
                }
            }
        }

        int getNumLabels(int variable) const final
        {
            return myConfig.num_disparities;
        }

        double evaluateEnergy(int factor, const std::vector<int>& node_labels) const final
        {
            double ret = 0.0;

            if( factor < myFactorOffset[0] )
            {
                if(node_labels.size() != 1) ABORT("Internal error!");

                const int x = myConfig.margin + factor % mySize.width;
                const int y = myConfig.margin + factor / mySize.width;
                const int xo = x + myConfig.directions[myImage] * node_labels[0];

                if( xo < 0 || myLevel.image[myOtherImage].cols <= xo )
                {
                    ABORT("Internal error!");
                }

                if( myLevel.occlusion[myImage](y,x) == 0 )
                {
                    const cv::Vec3b color = myLevel.image[myImage](y,x);
                    const cv::Vec3b other_color = myLevel.image[myOtherImage](y,xo);
                    ret += robust_norm_pixels(color, other_color);
                }

                if( myLevel.occlusion[myOtherImage](y,xo) == 1 )
                {
                    ret += myConfig.betaw;
                }
            }
            else if( factor < myFactorOffset[1] )
            {
                if(node_labels.size() != 2) ABORT("Internal error!");

                const int x = myConfig.margin + (factor-myFactorOffset[0]) % (mySize.width-1);
                const int y = myConfig.margin + (factor-myFactorOffset[0]) / (mySize.width-1);

                if(myLevel.occlusion[myImage](y,x) == myLevel.occlusion[myImage](y,x+1))
                {
                    ret += robust_norm_disparity( node_labels[0], node_labels[1] );
                }
            }
            else if( factor < myFactorOffset[2] )
            {
                if(node_labels.size() != 2) ABORT("Internal error!");

                const int x = myConfig.margin + (factor-myFactorOffset[1]) % mySize.width;
                const int y = myConfig.margin + (factor-myFactorOffset[1]) / mySize.width;

                if(myLevel.occlusion[myImage](y,x) == myLevel.occlusion[myImage](y+1,x))
                {
                    ret += robust_norm_disparity( node_labels[0], node_labels[1] );
                }
            }
            else
            {
                ABORT("Internal error!");
            }

            return ret;
        }
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
        }
        else
        {
            cv::resize( pyramid.front().gt_disparity[0], pyramid[i].gt_disparity[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( pyramid.front().gt_disparity[1], pyramid[i].gt_disparity[1], pyramid[i].image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( pyramid.front().gt_occlusion[0], pyramid[i].gt_occlusion[0], pyramid[i].image[0].size(), 0.0, 0.0, cv::INTER_NEAREST);
            cv::resize( pyramid.front().gt_occlusion[1], pyramid[i].gt_occlusion[1], pyramid[i].image[1].size(), 0.0, 0.0, cv::INTER_NEAREST);

            const double gamma = pyramid[i].level0_to_levelk;

            auto pred_disparity = [gamma] (int16_t& x)
            {
                x = static_cast<int16_t>(gamma * x);
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

    StochasticSearchSolver solver;
    std::vector<int> result;

    pyramid.back().occlusion[0] = pyramid.back().gt_occlusion[0].clone();
    pyramid.back().occlusion[1] = pyramid.back().gt_occlusion[1].clone();
    pyramid.back().disparity[0] = pyramid.back().gt_disparity[0].clone();
    pyramid.back().disparity[1] = pyramid.back().gt_disparity[1].clone();

    LBPSM::DisparityField field(config, pyramid.back(), 0);
    result.resize( (pyramid.back().image[0].rows - 2*config.margin) * (pyramid.back().image[0].cols - 2*config.margin) );
    std::fill(result.begin(), result.end(), 0);
    solver.solve(&field, result, true);

    cv::Mat1b nada( pyramid.back().image[0].rows - 2*config.margin, pyramid.back().image[0].cols - 2*config.margin );
    cv::Mat3b warped = pyramid.back().image[0].clone();
    cv::Mat1b ref(nada.size());
    for(int i=0; i<nada.rows; i++)
    {
        for(int j=0; j<nada.cols; j++)
        {
            nada(i, j) = result[ i*nada.cols + j ] * 255.0 / double(config.num_disparities-1);
            ref(i,j) = pyramid.back().gt_disparity[0](config.margin+i, config.margin+j) * 255.0 / double(config.num_disparities-1);
            warped(config.margin+i, config.margin+j) = pyramid.back().image[1](config.margin+i, config.margin+j+result[ i*nada.cols+j]*config.directions[0]);
        }
    }
    cv::imshow("est", nada);
    cv::imshow("ref", ref);
    cv::imshow("warped", warped);
    while(cv::waitKey(0) != 'a');

    //LBPSM::run(pyramid, config);

    return 0;
}

