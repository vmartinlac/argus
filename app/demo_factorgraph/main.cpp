#include <opencv2/imgcodecs.hpp>
#include <random>
#include "FactorGraph.h"
#include "StochasticSearchSolver.h"
#include "ExhaustiveSolver.h"
#include "BeliefPropagationSolver.h"
#include "HiddenMarkovModel.h"
#include "Common.h"

int main(int num_args, char** args)
{
    const int N = 3;
    const int num_state_labels = 2;
    const int num_observation_labels = 2;

    const double prior[] = { 0.4, 0.6 };

    const double transition[] = { 0.1, 0.9, 0.9, 0.1 };

    const double observation[] = { 0.9, 0.1, 0.1, 0.9 };

    HiddenMarkovModel hmm(N, num_state_labels, num_observation_labels, prior, transition, observation);

    std::vector<int> solution;

    //ExhaustiveSolver solver;
    BeliefPropagationSolver solver;
    solver.solve(&hmm, solution, false);

    hmm.dump(std::cout);

    std::cout << "Most probable explanation:" << std::endl;
    for(int i=0; i<hmm.getNumVariables(); i++)
    {
        std::cout << hmm.getNameOfVariable(i) << " = " << solution[i] << std::endl;
    }

    //std::cout << hmm.compileForGraphviz() << std::endl;

    return 0;
}

