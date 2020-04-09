#include <opencv2/imgcodecs.hpp>
#include <random>
#include "StochasticSearchSolver.h"
#include "MumfordShah.h"

int main(int num_args, char** args)
{
    const std::string path = "/home/victor/tsukuba/illumination/daylight/left/tsukuba_daylight_L_01000.png";
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

    MumfordShah ms(image, 4.0, 500.0);

    std::vector<int> solution;
    ms.getInitialSolution(solution);

    StochasticSearchSolver solver;
    solver.solve(&ms, solution, true);

    cv::Mat1b result;
    cv::Mat1b edges;
    ms.getImage(solution, result);
    ms.getEdges(solution, edges);

    cv::imwrite("A_input.png", image);
    cv::imwrite("B_result.png", result);
    cv::imwrite("C_edges.png", edges);
}

