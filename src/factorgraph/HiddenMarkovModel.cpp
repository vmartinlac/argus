#include "HiddenMarkovModel.h"
#include "Common.h"

HiddenMarkovModel::HiddenMarkovModel(int N, int num_state_labels, int num_observation_labels, const double* prior, const double* transition, const double* observation ) :
    myN(N),
    myNumStateLabels(num_state_labels),
    myNumObservationLabels(num_observation_labels),
    myPrior(prior),
    myTransition(transition),
    myObservation(observation)
{
}

int HiddenMarkovModel::getNumVariables() const
{
    return myN + (myN-1);
}

int HiddenMarkovModel::getNumFactors() const
{
    return myN + (myN-1);
}

void HiddenMarkovModel::getVariables(int factor, std::vector<int>& variables) const
{
    if( factor < 0 || myN+myN-1 <= factor)
    {
        ABORT("Internal error!");
    }

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

void HiddenMarkovModel::getFactors(int variable, std::vector<int>& factors) const
{
    if( variable < 0 || myN+myN-1 <= variable)
    {
        ABORT("Internal error!");
    }

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

int HiddenMarkovModel::getNumLabels(int variable) const
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

double HiddenMarkovModel::evaluateEnergy(int factor, const std::vector<int>& node_labels) const
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

std::string HiddenMarkovModel::getNameOfVariable(int variable) const
{
    std::stringstream ss;

    if(variable < myN)
    {
        ss << "x" << variable;
    }
    else
    {
        ss << "y" << (variable-myN+1);
    }

    return ss.str();
}

std::string HiddenMarkovModel::getNameOfFactor(int factor) const
{
    std::stringstream ss;

    if(factor == 0)
    {
        ss << "P(x0)";
    }
    else if(factor < myN)
    {
        ss << "P(x" << factor << "|x" << (factor-1) << ")";
    }
    else
    {
        ss << "P(y" << (factor-myN+1) << "|x" << (factor-myN+1) << ")";
    }

    return ss.str();
}

void HiddenMarkovModel::dump(std::ostream& s)
{
    const char* indent = "    ";

    s << "Hidden Markov Model" << std::endl;
    s << std::endl;

    s << "Length: " << myN << std::endl;
    s << "Number of labels for state: " << myNumStateLabels << std::endl;
    s << "Number of labels for observation: " << myNumObservationLabels << std::endl;
    s << std::endl;

    s << "Prior:" << std::endl << std::endl;
    for(int i=0; i<myNumStateLabels; i++)
    {
        s << indent << "P( x0 = " << i << " ) = " << myPrior[i] << std::endl;
    }
    s << std::endl;

    s << "Transition:" << std::endl << std::endl;
    for(int j=0; j<myNumStateLabels; j++)
    {
        for(int i=0; i<myNumStateLabels; i++)
        {
            const double p = myTransition[j * myNumStateLabels + i];
            s << indent << "P( x1 = " << i << " | x0 = " << j << " ) = " << p << std::endl;
        }
    }
    s << std::endl;

    s << "Observation:" << std::endl << std::endl;
    for(int j=0; j<myNumStateLabels; j++)
    {
        for(int i=0; i<myNumObservationLabels; i++)
        {
            const double p = myObservation[j * myNumStateLabels + i];
            s << indent << "P( y1 = " << i << " | x1 = " << j << " ) = " << p << std::endl;
        }
    }
    s << std::endl;
}

