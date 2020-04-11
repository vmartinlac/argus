#include <iostream>
#include "ProximalAlternatingMumfordShah.h"

ProximalAlternatingMumfordShah::ProximalAlternatingMumfordShah(double beta, double lambda, const cv::Vec3d& rgb_weights)
{
    myNumIterations = 1000;
    myBeta = beta;
    myLambda = lambda;
    myRgbWeights = rgb_weights;
}

void ProximalAlternatingMumfordShah::setNumIterations(int value)
{
    myNumIterations = value;
}

void ProximalAlternatingMumfordShah::runColor(const cv::Mat3b& input, cv::Mat3b& result, cv::Mat1b& edges)
{
    /*
    const double gamma = 1.1;
    const double delta = 1.1;

    const cv::Size image_size(input.size());

    double cte_c = 0.0;
    double cte_d = 0.0;

    cv::Mat3f reference;
    input.convertTo(reference, CV_32FC3);

    cv::Mat3f X_image = reference.clone();
    cv::Mat1f X_vertical_edges(image_size.height, image_size.width-1);
    cv::Mat1f X_horizontal_edges(image_size.height-1, image_size.width);

    X_horizontal_edges = 0.0f;
    X_vertical_edges = 0.0f;

    cv::Mat3f derivative_wrt_image(image_size);
    cv::Mat1f derivative_wrt_vertical_edges(image_size.height, image_size.width-1);
    cv::Mat1f derivative_wrt_horizontal_edges(image_size.height-1, image_size.width);

    for(int iter=0; iter<myNumIterations; iter++)
    {
        std::cout << "Iteration " << iter << std::endl;

        // compute cte_c.

        {
            const double L = 4.0 * myBeta * std::sqrt(image_size.width*image_size.height);
            cte_c = gamma * L;
        }

        // update the image with gradient descent.

        {
            derivative_wrt_image = 0.0f;

            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    const float one_minus_edge = 1.0f - X_vertical_edges(i,j);
                    const float cte = 2.0 * myBeta * (X_image(i,j) - X_image(i,j+1)) * one_minus_edge * one_minus_edge;
                    derivative_wrt_image(i,j) += cte;
                    derivative_wrt_image(i,j+1) -= cte;
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    const float one_minus_edge = 1.0f - X_horizontal_edges(i,j);
                    const float cte = 2.0 * myBeta * (X_image(i,j) - X_image(i+1,j)) * one_minus_edge * one_minus_edge;
                    derivative_wrt_image(i,j) += cte;
                    derivative_wrt_image(i+1,j) -= cte;
                }
            }

            X_image -= (1.0f/cte_c) * derivative_wrt_image;
        }

        // update the image with proximal operator.

        {
            const double weight = cte_c / (cte_c + 2.0);
            X_image = weight*X_image + (1.0-weight)*reference;
        }

        // compute cte_d.

        {
            double m = 0.0;

            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    const double diff = X_image(i,j+1) - X_image(i,j);
                    m = std::max<double>(m, diff*diff);
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    const double diff = X_image(i+1,j) - X_image(i,j);
                    m = std::max<double>(m, diff*diff);
                }
            }

            const double L = 4.0 * myBeta * m;
            cte_d = delta * L;
        }

        // update the edges with gradient descent.

        {
            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    const float one_minus_edge = 1.0f - X_vertical_edges(i,j);
                    const float diff = X_image(i,j) - X_image(i,j+1);
                    derivative_wrt_vertical_edges(i,j) = -2.0 * myBeta * diff * diff * one_minus_edge;
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    const float one_minus_edge = 1.0f - X_horizontal_edges(i,j);
                    const float diff = X_image(i,j) - X_image(i+1,j);
                    derivative_wrt_horizontal_edges(i,j) = -2.0 * myBeta * diff * diff * one_minus_edge;
                }
            }

            X_vertical_edges -= (1.0f/cte_d) * derivative_wrt_vertical_edges;
            X_horizontal_edges -= (1.0f/cte_d) * derivative_wrt_horizontal_edges;
        }

        // update the edges with proximal operator.

        {
            const double threshold = myLambda / cte_d;

            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    if(X_vertical_edges(i,j) >= threshold)
                    {
                        X_vertical_edges(i,j) -= threshold;
                    }
                    else if(X_vertical_edges(i,j) <= -threshold)
                    {
                        X_vertical_edges(i,j) += threshold; 
                    }
                    else
                    {
                        X_vertical_edges(i,j) = 0.0;
                    }
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    if(X_horizontal_edges(i,j) >= threshold)
                    {
                        X_horizontal_edges(i,j) -= threshold;
                    }
                    else if(X_horizontal_edges(i,j) <= -threshold)
                    {
                        X_horizontal_edges(i,j) += threshold; 
                    }
                    else
                    {
                        X_horizontal_edges(i,j) = 0.0;
                    }
                }
            }
        }
    }

    X_image.convertTo(result, CV_8UC1);

    edges.create(image_size);
    edges = 0;

    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols-1; j++)
        {
            const uint8_t value = cv::saturate_cast<uint8_t>(255.0f * X_vertical_edges(i,j));
            edges(i,j) = std::max<uint8_t>( edges(i,j), value );
        }
    }

    for(int i=0; i<input.rows-1; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            const uint8_t value = cv::saturate_cast<uint8_t>(255.0f * X_horizontal_edges(i,j));
            edges(i,j) = std::max<uint8_t>( edges(i,j), value );
        }
    }
    */
}

void ProximalAlternatingMumfordShah::runGrayscale(const cv::Mat1b& input, cv::Mat1b& result, cv::Mat1b& edges)
{
    const double gamma = 1.1;
    const double delta = 1.1;

    const cv::Size image_size(input.size());

    double cte_c = 0.0;
    double cte_d = 0.0;

    cv::Mat1f reference;
    input.convertTo(reference, CV_32FC1);

    cv::Mat1f X_image = reference.clone();
    cv::Mat1f X_vertical_edges(image_size.height, image_size.width-1);
    cv::Mat1f X_horizontal_edges(image_size.height-1, image_size.width);

    X_horizontal_edges = 0.0f;
    X_vertical_edges = 0.0f;

    cv::Mat1f derivative_wrt_image(image_size);
    cv::Mat1f derivative_wrt_vertical_edges(image_size.height, image_size.width-1);
    cv::Mat1f derivative_wrt_horizontal_edges(image_size.height-1, image_size.width);

    for(int iter=0; iter<myNumIterations; iter++)
    {
        std::cout << "Iteration " << iter << std::endl;

        // compute cte_c.

        {
            const double L = 4.0 * myBeta * std::sqrt(image_size.width*image_size.height);
            cte_c = gamma * L;
        }

        // update the image with gradient descent.

        {
            derivative_wrt_image = 0.0f;

            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    const float one_minus_edge = 1.0f - X_vertical_edges(i,j);
                    const float cte = 2.0 * myBeta * (X_image(i,j) - X_image(i,j+1)) * one_minus_edge * one_minus_edge;
                    derivative_wrt_image(i,j) += cte;
                    derivative_wrt_image(i,j+1) -= cte;
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    const float one_minus_edge = 1.0f - X_horizontal_edges(i,j);
                    const float cte = 2.0 * myBeta * (X_image(i,j) - X_image(i+1,j)) * one_minus_edge * one_minus_edge;
                    derivative_wrt_image(i,j) += cte;
                    derivative_wrt_image(i+1,j) -= cte;
                }
            }

            X_image -= (1.0f/cte_c) * derivative_wrt_image;
        }

        // update the image with proximal operator.

        {
            const double weight = cte_c / (cte_c + 2.0);
            X_image = weight*X_image + (1.0-weight)*reference;
        }

        // compute cte_d.

        {
            double m = 0.0;

            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    const double diff = X_image(i,j+1) - X_image(i,j);
                    m = std::max<double>(m, diff*diff);
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    const double diff = X_image(i+1,j) - X_image(i,j);
                    m = std::max<double>(m, diff*diff);
                }
            }

            const double L = 4.0 * myBeta * m;
            cte_d = delta * L;
        }

        // update the edges with gradient descent.

        {
            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    const float one_minus_edge = 1.0f - X_vertical_edges(i,j);
                    const float diff = X_image(i,j) - X_image(i,j+1);
                    derivative_wrt_vertical_edges(i,j) = -2.0 * myBeta * diff * diff * one_minus_edge;
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    const float one_minus_edge = 1.0f - X_horizontal_edges(i,j);
                    const float diff = X_image(i,j) - X_image(i+1,j);
                    derivative_wrt_horizontal_edges(i,j) = -2.0 * myBeta * diff * diff * one_minus_edge;
                }
            }

            X_vertical_edges -= (1.0f/cte_d) * derivative_wrt_vertical_edges;
            X_horizontal_edges -= (1.0f/cte_d) * derivative_wrt_horizontal_edges;
        }

        // update the edges with proximal operator.

        {
            const double threshold = myLambda / cte_d;

            for(int i=0; i<image_size.height; i++)
            {
                for(int j=0; j<image_size.width-1; j++)
                {
                    if(X_vertical_edges(i,j) >= threshold)
                    {
                        X_vertical_edges(i,j) -= threshold;
                    }
                    else if(X_vertical_edges(i,j) <= -threshold)
                    {
                        X_vertical_edges(i,j) += threshold; 
                    }
                    else
                    {
                        X_vertical_edges(i,j) = 0.0;
                    }
                }
            }

            for(int i=0; i<image_size.height-1; i++)
            {
                for(int j=0; j<image_size.width; j++)
                {
                    if(X_horizontal_edges(i,j) >= threshold)
                    {
                        X_horizontal_edges(i,j) -= threshold;
                    }
                    else if(X_horizontal_edges(i,j) <= -threshold)
                    {
                        X_horizontal_edges(i,j) += threshold; 
                    }
                    else
                    {
                        X_horizontal_edges(i,j) = 0.0;
                    }
                }
            }
        }
    }

    X_image.convertTo(result, CV_8UC1);

    edges.create(image_size);
    edges = 0;

    for(int i=0; i<input.rows; i++)
    {
        for(int j=0; j<input.cols-1; j++)
        {
            const uint8_t value = cv::saturate_cast<uint8_t>(255.0f * X_vertical_edges(i,j));
            edges(i,j) = std::max<uint8_t>( edges(i,j), value );
        }
    }

    for(int i=0; i<input.rows-1; i++)
    {
        for(int j=0; j<input.cols; j++)
        {
            const uint8_t value = cv::saturate_cast<uint8_t>(255.0f * X_horizontal_edges(i,j));
            edges(i,j) = std::max<uint8_t>( edges(i,j), value );
        }
    }
}

