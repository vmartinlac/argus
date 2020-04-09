#include <opencv2/imgcodecs.hpp>
#include <random>
#include "Common.h"
#include "MumfordShah.h"
#include "MarkovRandomField.h"
#include "StochasticSearchSolver.h"

MumfordShah::MumfordShah(cv::Mat1b input_image, double beta, double lambda)
{
    myImage = input_image;
    myBeta = beta;
    myLambda = lambda;
    mySize = input_image.size();
    myOffset[0] = mySize.width*mySize.height;
    myOffset[1] = myOffset[0] + (mySize.width-1)*mySize.height;
    myOffset[2] = myOffset[1] + mySize.width*(mySize.height-1);
}

int MumfordShah::getNumVariables() const
{
    return myOffset[2];
}

int MumfordShah::getNumFactors() const
{
    return myOffset[2];
}

void MumfordShah::getVariables(int factor, std::vector<int>& variables) const
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
        const int v1 = y*mySize.width + x;
        const int v2 = y*mySize.width + (x+1);
        variables.assign({v0, v1, v2});
    }
    else if(factor < myOffset[2])
    {
        const int x = (factor-myOffset[1]) % mySize.width;
        const int y = (factor-myOffset[1]) / mySize.width;
        const int v0 = factor;
        const int v1 = y*mySize.width + x;
        const int v2 = (y+1)*mySize.width + x;
        variables.assign({v0, v1, v2});
    }
    else
    {
        ABORT("Internal error!");
    }
}

void MumfordShah::getFactors(int variable, std::vector<int>& factors) const
{
    if(variable < myOffset[0])
    {
        const int x = variable % mySize.width;
        const int y = variable / mySize.width;

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

int MumfordShah::getNumLabels(int variable) const
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

double MumfordShah::evaluateEnergy(int factor, const std::vector<int>& node_labels) const
{
    double ret = 0.0;

    if(factor < myOffset[0])
    {
        if( node_labels.size() != 1 ) ABORT("Internal error!");

        const int x = factor % mySize.width;
        const int y = factor / mySize.width;

        const double gray0 = node_labels[0];
        const double gray1 = myImage(y,x);
        const double delta = gray1 - gray0;

        ret = delta * delta;
    }
    else if(factor < myOffset[2])
    {
        if( node_labels.size() != 3 ) ABORT("Internal error!");

        if(node_labels[0])
        {
            ret = myLambda;
        }
        else
        {
            const double gray0 = node_labels[1];
            const double gray1 = node_labels[2];
            const double delta = gray1 - gray0;

            ret = myBeta * delta * delta;
        }
    }
    else
    {
        ABORT("Internal error!");
    }

    return ret;
}

void MumfordShah::getInitialSolution(std::vector<int>& solution)
{
    solution.resize(myOffset[2]);

    for(int i=0; i<mySize.height; i++)
    {
        for(int j=0; j<mySize.width; j++)
        {
            solution[i*mySize.width+j] = myImage(i,j);
        }
    }

    std::fill(solution.begin() + myOffset[0], solution.begin() + myOffset[2], 0);
}

void MumfordShah::getImage(const std::vector<int>& solution, cv::Mat1b& image)
{
    image.create(mySize);

    for(int i=0; i<mySize.height; i++)
    {
        for(int j=0; j<mySize.width; j++)
        {
            image(i,j) = solution[i*mySize.width+j];
        }
    }
}

void MumfordShah::getEdges(const std::vector<int>& solution, cv::Mat1b& edges)
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

