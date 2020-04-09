#include <iostream>
#include <random>
#include "StochasticSearchSolver.h"

StochasticSearchSolver::StochasticSearchSolver()
{
    mySeed = 12345;
    myNumIterations = 1000;
    myLambda = 0.995;
    myInitialTemperature = 2.0;
}

void StochasticSearchSolver::setLambda(double value)
{
    myLambda = value;
}

void StochasticSearchSolver::setInitialTemperature(double value)
{
    myInitialTemperature = value;
}

void StochasticSearchSolver::setSeed(int value)
{
    mySeed = value;
}

void StochasticSearchSolver::setNumIterations(int value)
{
    myNumIterations = value;
}

bool StochasticSearchSolver::solve(MarkovRandomField* field, std::vector<int>& solution, bool use_initial_solution)
{
    std::default_random_engine W;
    W.seed(mySeed);

    std::uniform_real_distribution<double> U(0.0, 1.0);

    const int num_variables = field->getNumVariables();
    const int num_factors = field->getNumFactors();

    double temperature = myInitialTemperature;

    std::vector<double> energies(num_factors);
    std::vector<int> local_variables;
    std::vector<int> local_labels;
    std::vector<int> local_factors;
    std::vector<double> local_energies;

    // initialize solution.

    if(use_initial_solution)
    {
        if( (int) solution.size() != num_variables )
        {
            std::cerr << "Internal error!" << std::endl;
            exit(1);
        }
    }
    else
    {
        solution.resize(num_variables);
        for(int i=0; i<num_variables; i++)
        {
            solution[i] = W() % field->getNumLabels(i);
        }
    }

    // compute energies of all factors.

    for(int i=0; i<num_factors; i++)
    {
        field->getVariables(i, local_variables);

        local_labels.resize(local_variables.size());
        for(int j=0; j<(int) local_variables.size(); j++)
        {
            local_labels[j] = solution[local_variables[j]];
        }

        energies[i] = field->evaluateEnergy(i, local_labels);
    }

    // stochastic optimization.

    for(int iter=0; iter<myNumIterations; iter++)
    {
        for(int v=0; v<num_variables; v++)
        {
            const int new_label = W() % field->getNumLabels(v);

            if(new_label != solution[v])
            {
                double old_local_energy = 0.0;
                double new_local_energy = 0.0;

                field->getFactors(v, local_factors);

                local_energies.resize(local_factors.size());

                for(int f=0; f<(int)local_factors.size(); f++)
                {
                    field->getVariables(local_factors[f], local_variables);
                    local_labels.resize(local_variables.size());

                    for(int vv=0; vv<(int) local_variables.size(); vv++)
                    {
                        if(local_variables[vv] == v)
                        {
                            local_labels[vv] = new_label;
                        }
                        else
                        {
                            local_labels[vv] = solution[local_variables[vv]];
                        }
                    }

                    local_energies[f] = field->evaluateEnergy(local_factors[f], local_labels);
                    new_local_energy += local_energies[f];
                    old_local_energy += energies[local_factors[f]];
                }

                const double delta = new_local_energy - old_local_energy;

                if( delta <= 0.0 || U(W) < std::exp(-delta/temperature) ) // first condition is only for the sake of clarity.
                {
                    solution[v] = new_label;

                    for(int f=0; f<(int)local_factors.size(); f++)
                    {
                        energies[ local_factors[f] ] = local_energies[f];
                    }
                }
            }
        }

        std::cout << "Iteration = " << iter << "\tTemperature = " << temperature << "\tEnergy = " << field->evaluateTotalEnergy(solution) << std::endl;
        temperature *= myLambda;
    }

    return true;
}

