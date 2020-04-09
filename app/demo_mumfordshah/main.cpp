#include <opencv2/imgcodecs.hpp>
#include <random>
#include "StochasticSearchSolver.h"
#include "GrayscaleMumfordShah.h"
#include "ColorMumfordShah.h"

void demo_color(const std::string& path)
{
    // load image

    cv::Mat3b image = cv::imread(path, cv::IMREAD_COLOR);

    // add noise

    {
        std::default_random_engine w;
        w.seed(1357);

        std::normal_distribution<double> X(0.0, 4.0);
        for(int i=0; i<image.rows; i++)
        {
            for(int j=0; j<image.cols; j++)
            {
                image(i,j)[0] = cv::saturate_cast<uint8_t>(image(i,j)[0] + X(w));
                image(i,j)[1] = cv::saturate_cast<uint8_t>(image(i,j)[1] + X(w));
                image(i,j)[2] = cv::saturate_cast<uint8_t>(image(i,j)[2] + X(w));
            }
        }
    }

    // solve Mumford-Shah model.

    ColorMumfordShah ms(image, 4.0, 500.0);

    std::vector<int> solution;
    ms.getInitialSolution(solution);

    StochasticSearchSolver solver;
    solver.solve(&ms, solution, true);

    // save results.

    cv::Mat3b result;
    cv::Mat1b edges;
    ms.getImage(solution, result);
    ms.getEdges(solution, edges);

    cv::imwrite("A_input.png", image);
    cv::imwrite("B_result.png", result);
    cv::imwrite("C_edges.png", edges);
}

void demo_grayscale(const std::string& path)
{
    // load image

    cv::Mat1b image = cv::imread(path, cv::IMREAD_GRAYSCALE);

    // add noise

    {
        std::default_random_engine w;
        w.seed(1357);

        std::normal_distribution<double> X(0.0, 4.0);
        for(int i=0; i<image.rows; i++)
        {
            for(int j=0; j<image.cols; j++)
            {
                image(i,j) = cv::saturate_cast<uint8_t>(image(i,j) + X(w));
            }
        }
    }

    // solve Mumford-Shah model.

    GrayscaleMumfordShah ms(image, 4.0, 500.0);

    std::vector<int> solution;
    ms.getInitialSolution(solution);

    StochasticSearchSolver solver;
    solver.solve(&ms, solution, true);

    // save results.

    cv::Mat1b result;
    cv::Mat1b edges;
    ms.getImage(solution, result);
    ms.getEdges(solution, edges);

    cv::imwrite("A_input.png", image);
    cv::imwrite("B_result.png", result);
    cv::imwrite("C_edges.png", edges);
}

int main(int num_args, char** args)
{
    const std::string path = "/home/victor/tsukuba/illumination/daylight/left/tsukuba_daylight_L_01000.png";

    demo_color(path);

    return 0;
}

