#include <opencv2/imgcodecs.hpp>
#include <random>
#include "FactorGraph.h"
#include "StochasticSearchSolver.h"
#include "ExhaustiveSolver.h"
#include "Common.h"

class HiddenMarkovModelGraph : public FactorGraph
{
public:

    HiddenMarkovModelGraph(int N, int num_state_labels, int num_observation_labels, const double* prior, const double* transition, const double* observation ) :
        myN(N),
        myNumStateLabels(num_state_labels),
        myNumObservationLabels(num_observation_labels),
        myPrior(prior),
        myTransition(transition),
        myObservation(observation)
    {
    }

    int getNumVariables() const
    {
        return myN + (myN-1);
    }

    int getNumFactors() const
    {
        return myN + (myN-1);
    }

    void getVariables(int factor, std::vector<int>& variables) const
    {
        if( factor < 0 || myN+myN-1 <= factor) ABORT("Internal error!");

        if(factor == 0)
        {
            variables.assign({0});
        }
        else if(factor < myN)
        {
            variables.assign({factor, factor-1});
        }
        else
        {
            variables.assign({ factor, factor-myN+1 });
        }
    }

    void getFactors(int variable, std::vector<int>& factors) const
    {
        if( variable < 0 || myN+myN-1 <= variable) ABORT("Internal error!");

        if(variable < myN)
        {
            factors.assign({ variable });

            if(variable+1 < myN)
            {
                factors.push_back(variable+1);
            }

            if(1 <= variable)
            {
                factors.push_back(myN + variable - 1);
            }
        }
        else
        {
            factors.assign({ variable });
        }
    }

    int getNumLabels(int variable) const
    {
        if( variable < 0 || myN+myN-1 <= variable ) ABORT("Internal error!");

        if(variable < myN)
        {
            return myNumStateLabels;
        }
        else
        {
            return myNumObservationLabels;
        }
    }

    double evaluateEnergy(int factor, const std::vector<int>& node_labels) const
    {
        if( factor < 0 || myN+myN-1 <= factor) ABORT("Internal error!");

        double ret = 0.0;

        if(factor == 0)
        {
            if( node_labels.size() != 1 ) ABORT("Internal error!");

            ret = myPrior[node_labels[0]];
        }
        else if(factor < myN)
        {
            if( node_labels.size() != 2 ) ABORT("Internal error!");

            const int child_label = node_labels[0];
            const int parent_label = node_labels[1];

            ret = myTransition[parent_label * myNumStateLabels + child_label];
        }
        else
        {
            if( node_labels.size() != 2 ) ABORT("Internal error!");

            const int child_label = node_labels[0];
            const int parent_label = node_labels[1];

            ret = myObservation[parent_label * myNumObservationLabels + child_label];
        }

        return -std::log(ret);
    }

protected:

    int myN;
    int myNumStateLabels;
    int myNumObservationLabels;
    const double* myPrior;
    const double* myTransition;
    const double* myObservation;
};

int main(int num_args, char** args)
{
    const int N = 3;
    const int num_state_labels = 2;
    const int num_observation_labels = 2;

    const double prior[] = { 0.9, 0.1 };

    const double transition[] = { 0.1, 0.9, 0.9, 0.1 };

    const double observation[] = { 0.9, 0.1, 0.1, 0.9 };

    HiddenMarkovModelGraph hmm(N, num_state_labels, num_observation_labels, prior, transition, observation);

    std::vector<int> solution;

    ExhaustiveSolver solver;
    solver.solve(&hmm, solution, false);

    std::cout << solution[0] << std::endl;
    std::cout << solution[1] << std::endl;
    std::cout << solution[2] << std::endl;
    std::cout << solution[3] << std::endl;
    std::cout << solution[4] << std::endl;

    //std::cout << hmm.compileForGraphviz() << std::endl;

    /*
    std::vector<int> solution({1, 0, 1, 0, 1});
    const double E = hmm.evaluateTotalEnergy(solution);
    const double p = std::exp(-E);
    std::cout << p << std::endl;
    */

    return 0;
}

