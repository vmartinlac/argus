#include "Common.h"
#include "ColorMumfordShah.h"

ColorMumfordShah::ColorMumfordShah(cv::Mat3b input_image, double beta, double lambda, const cv::Vec3d& rgb_weights)
{
    myRgbWeights = rgb_weights;
    myImage = input_image;
    myBeta = beta;
    myLambda = lambda;
    mySize = input_image.size();
    myOffset[0] = 3*mySize.width*mySize.height;
    myOffset[1] = myOffset[0] + (mySize.width-1)*mySize.height;
    myOffset[2] = myOffset[1] + mySize.width*(mySize.height-1);
}

int ColorMumfordShah::getNumVariables() const
{
    return myOffset[2];
}

int ColorMumfordShah::getNumFactors() const
{
    return myOffset[2];
}

void ColorMumfordShah::getVariables(int factor, std::vector<int>& variables) const
{
    if(factor < myOffset[0])
    {
        variables.assign({factor});
    }
    else if(factor < myOffset[1])
    {
        const int x = (factor-myOffset[0]) % (mySize.width-1);
        const int y = (factor-myOffset[0]) / (mySize.width-1);

        const int v0 = factor;

        const int v1 = 3*(y*mySize.width + x)+0;
        const int v2 = 3*(y*mySize.width + x)+1;
        const int v3 = 3*(y*mySize.width + x)+2;

        const int v4 = 3*(y*mySize.width + (x+1))+0;
        const int v5 = 3*(y*mySize.width + (x+1))+1;
        const int v6 = 3*(y*mySize.width + (x+1))+2;

        variables.assign({v0, v1, v2, v3, v4, v5, v6});
    }
    else if(factor < myOffset[2])
    {
        const int x = (factor-myOffset[1]) % mySize.width;
        const int y = (factor-myOffset[1]) / mySize.width;

        const int v0 = factor;

        const int v1 = 3*(y*mySize.width + x)+0;
        const int v2 = 3*(y*mySize.width + x)+1;
        const int v3 = 3*(y*mySize.width + x)+2;

        const int v4 = 3*((y+1)*mySize.width + x)+0;
        const int v5 = 3*((y+1)*mySize.width + x)+1;
        const int v6 = 3*((y+1)*mySize.width + x)+2;

        variables.assign({v0, v1, v2, v3, v4, v5, v6});
    }
    else
    {
        ABORT("Internal error!");
    }
}

void ColorMumfordShah::getFactors(int variable, std::vector<int>& factors) const
{
    if(variable < myOffset[0])
    {
        const int x = (variable/3) % mySize.width;
        const int y = (variable/3) / mySize.width;

        factors.assign({variable});

        if(0 <= x-1 )
        {
            factors.push_back( myOffset[0] + y*(mySize.width-1)+(x-1) );
        }

        if(x+1 < mySize.width)
        {
            factors.push_back( myOffset[0] + y*(mySize.width-1)+x );
        }

        if(0 <= y-1 )
        {
            factors.push_back( myOffset[1] + (y-1)*mySize.width+x );
        }

        if(y+1 < mySize.height)
        {
            factors.push_back( myOffset[1] + y*mySize.width+x );
        }
    }
    else if(variable < myOffset[2])
    {
        factors.assign({variable});
    }
    else
    {
        ABORT("Internal error!");
    }
}

int ColorMumfordShah::getNumLabels(int variable) const
{
    if(variable < myOffset[0])
    {
        return 255;
    }
    else if(variable < myOffset[2])
    {
        return 2;
    }
    else
    {
        ABORT("Internal error!");
    }
}

double ColorMumfordShah::evaluateEnergy(int factor, const std::vector<int>& node_labels) const
{
    double ret = 0.0;

    if(factor < myOffset[0])
    {
        if( node_labels.size() != 1 ) ABORT("Internal error!");

        const int channel = factor % 3;
        const int x = (factor/3) % mySize.width;
        const int y = (factor/3) / mySize.width;

        const double value0 = node_labels[0];
        const double value1 = myImage(y,x)[channel];
        const double delta = value1 - value0;

        ret = myRgbWeights[channel] * delta * delta;
    }
    else if(factor < myOffset[2])
    {
        if( node_labels.size() != 7 ) ABORT("Internal error!");

        if(node_labels[0])
        {
            ret = myLambda;
        }
        else
        {
            double color0[3];
            double color1[3];
            double delta[3];

            color0[0] = node_labels[1];
            color0[1] = node_labels[2];
            color0[2] = node_labels[3];

            color1[0] = node_labels[4];
            color1[1] = node_labels[5];
            color1[2] = node_labels[6];

            delta[0] = color1[0] - color0[0];
            delta[1] = color1[1] - color0[1];
            delta[2] = color1[2] - color0[2];

            ret =
                myBeta * myRgbWeights[0] * delta[0] * delta[0] +
                myBeta * myRgbWeights[1] * delta[1] * delta[1] +
                myBeta * myRgbWeights[2] * delta[2] * delta[2];
        }
    }
    else
    {
        ABORT("Internal error!");
    }

    return ret;
}

void ColorMumfordShah::getInitialSolution(std::vector<int>& solution)
{
    solution.resize(myOffset[2]);

    for(int i=0; i<mySize.height; i++)
    {
        for(int j=0; j<mySize.width; j++)
        {
            solution[3*(i*mySize.width+j)+0] = myImage(i,j)[0];
            solution[3*(i*mySize.width+j)+1] = myImage(i,j)[1];
            solution[3*(i*mySize.width+j)+2] = myImage(i,j)[2];
        }
    }

    std::fill(solution.begin() + myOffset[0], solution.begin() + myOffset[2], 0);
}

void ColorMumfordShah::getImage(const std::vector<int>& solution, cv::Mat3b& image)
{
    image.create(mySize);

    for(int i=0; i<mySize.height; i++)
    {
        for(int j=0; j<mySize.width; j++)
        {
            image(i,j)[0] = solution[3*(i*mySize.width+j)+0];
            image(i,j)[1] = solution[3*(i*mySize.width+j)+1];
            image(i,j)[2] = solution[3*(i*mySize.width+j)+2];
        }
    }
}

void ColorMumfordShah::getEdges(const std::vector<int>& solution, cv::Mat1b& edges)
{
    edges.create(mySize);

    edges = 0;

    for(int i=0; i<mySize.height; i++)
    {
        for(int j=0; j<mySize.width-1; j++)
        {
            if( solution[myOffset[0] + i*(mySize.width-1)+j] == 1 )
            {
                edges(i,j) = 255;
            }
        }
    }

    for(int i=0; i<mySize.height-1; i++)
    {
        for(int j=0; j<mySize.width; j++)
        {
            if( solution[myOffset[1] + i*mySize.width+j] == 1 )
            {
                edges(i,j) = 255;
            }
        }
    }
}

