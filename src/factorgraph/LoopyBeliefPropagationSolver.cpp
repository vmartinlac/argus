#include <iostream>
#include <stack>
#include "LoopyBeliefPropagationSolver.h"
#include "Common.h"

LoopyBeliefPropagationSolver::LoopyBeliefPropagationSolver()
{
    myNumIterations = 3;
}

void LoopyBeliefPropagationSolver::setNumIterations(int value)
{
    myNumIterations = value;
}

bool LoopyBeliefPropagationSolver::solve(FactorGraph* graph, std::vector<int>& solution, bool use_initial_solution)
{
    const int num_variables = graph->getNumVariables();
    const int num_factors = graph->getNumFactors();

    std::vector<int> variable_edges_offset; // for each variable, the offset in variable_edges.
    std::vector<int> factor_edges_offset; // for each factor, the offset in factor_edges.
    std::vector<int> variable_edges; // edge indices of each variable.
    std::vector<int> factor_edges; // edge indices of each factor.
    std::vector<int> edge_offset; // for each edge index, an offset into message table.
    int sum_message_dimensions; // dimension of message table.
    int num_edges;

    std::vector<float> factor_to_variable; // messages from factor to variable.
    std::vector<float> variable_to_factor; // messages from variable to factor.
    std::vector<int> argmin_offset; // for each edge, the offset in argmin.
    std::vector<int> argmin; // the argmin.
    int argmin_dim;

    std::vector<int> neighbors0;
    std::vector<int> neighbors1;
    std::vector<int> candidate;

    // initialize message offsets for variables.

    {
        sum_message_dimensions = 0;
        num_edges = 0;

        variable_edges_offset.resize(num_variables);

        for(int i=0; i<num_variables; i++)
        {
            graph->getFactors(i, neighbors0);
            const int num_local_factors = (int) neighbors0.size();
            const int num_labels = graph->getNumLabels(i);

            variable_edges_offset[i] = (int) variable_edges.size();

            for(int j=0; j<num_local_factors; j++)
            {
                variable_edges.push_back(num_edges);
                num_edges++;

                edge_offset.push_back(sum_message_dimensions);
                sum_message_dimensions += num_labels;
            }

            if(sum_message_dimensions > 2000000000) ABORT("Graph size exceeds limits! Lets avoid arithmetic overflow!");
        }

        if( (int) edge_offset.size() != num_edges ) ABORT("Internal error!");

        variable_to_factor.assign(sum_message_dimensions, 0.0f);
        factor_to_variable.assign(sum_message_dimensions, 0.0f);
    }

    // initialize message offsets for factors.

    {
        argmin_dim = 0;

        factor_edges_offset.resize(num_factors);
        argmin_offset.resize(num_edges);

        for(int i=0; i<num_factors; i++)
        {
            graph->getVariables(i, neighbors0);
            const int num_local_variables = (int) neighbors0.size();

            factor_edges_offset[i] = (int) factor_edges.size();

            for(int j=0; j<num_local_variables; j++)
            {
                bool found = false;
                int edge = -1;
                graph->getFactors(neighbors0[j], neighbors1);
                for(int k=0; k<(int) neighbors1.size(); k++)
                {
                    if(neighbors1[k] == i)
                    {
                        if(found)
                        {
                            ABORT("Bad graph");
                        }
                        else
                        {
                            edge = variable_edges[variable_edges_offset[neighbors0[j]] + k];
                            found = true;
                        }
                    }
                }

                if(found == false)
                {
                    ABORT("Internal error!");
                }

                factor_edges.push_back(edge);

                argmin_offset[edge] = argmin_dim;
                argmin_dim += graph->getNumLabels(neighbors0[j]) * ( (int) neighbors0.size()-1 );
                if( argmin_dim >= 2000000000 ) ABORT("Graph is too big");
            }
        }

        if( (int) factor_edges.size() != num_edges ) ABORT("Internal error!");

        argmin.resize(argmin_dim);
    }

    // belief propagation.

    for(int i=0; i<myNumIterations; i++)
    {
        //std::cout << "LoopyBeliefPropagationSolver iteration " << i << std::endl;

        // update factor to variable messages.

        for(int this_factor=0; this_factor<num_factors; this_factor++)
        {
            //std::cout << this_factor << "/" << num_factors << std::endl;
            graph->getVariables(this_factor, neighbors0);

            // send messages.

            for(int i=0; i<(int) neighbors0.size(); i++)
            {
                const int target_variable = neighbors0[i];
                const int edge = factor_edges[factor_edges_offset[this_factor] + i];

                const int num_labels = graph->getNumLabels(target_variable);

                int num_combinations = 1;

                for(int other_variable : neighbors0)
                {
                    if(other_variable != target_variable)
                    {
                        num_combinations *= graph->getNumLabels(other_variable);
                    }
                }

                for(int j=0; j<num_labels; j++)
                {
                    bool first = true;
                    double lowest_energy = 0.0;

                    candidate.resize(neighbors0.size());

                    for(int combination_index=0; combination_index<num_combinations; combination_index++)
                    {
                        double candidate_energy = 0.0;
                        int tmp = combination_index;

                        bool found = false;

                        for(int k=0; k<(int) neighbors0.size(); k++)
                        {
                            const int other_variable = neighbors0[k];
                            const int other_edge = factor_edges[factor_edges_offset[this_factor] + k];
                            const int num_other_labels = graph->getNumLabels(other_variable);

                            if(other_variable == target_variable)
                            {
                                if(found)
                                {
                                    ABORT("Internal error");
                                }
                                else
                                {
                                    candidate[k] = j;
                                    found = true;
                                }
                            }
                            else
                            {
                                candidate[k] = tmp % num_other_labels;
                                tmp /= num_other_labels;

                                candidate_energy += variable_to_factor[edge_offset[other_edge]+candidate[k]];
                            }
                        }

                        if(found == false)
                        {
                            ABORT("Internal error");
                        }

                        candidate_energy += graph->evaluateEnergy(this_factor, candidate);

                        if(first || candidate_energy < lowest_energy)
                        {
                            first = false;
                            lowest_energy = candidate_energy;

                            // copy into argmin.

                            int kk = 0;
                            for(int nn=0; nn<(int) neighbors0.size(); nn++)
                            {
                                if( neighbors0[nn] != target_variable )
                                {
                                    argmin[argmin_offset[edge] + j*( (int) neighbors0.size()-1 ) + kk] = candidate[nn];
                                    kk++;
                                }
                            }
                        }
                    }

                    if(first)
                    {
                        ABORT("Internal error!");
                    }

                    factor_to_variable[edge_offset[edge]+j] = (float) lowest_energy;
                }
            }
        }

        // variable to factor messages.

        for(int this_variable=0; this_variable<num_variables; this_variable++)
        {
            //std::cout << this_variable << "/" << num_variables << std::endl;
            graph->getFactors(this_variable, neighbors0);

            const int num_labels = graph->getNumLabels(this_variable);

            // send messages.

            for(int i=0; i<(int) neighbors0.size(); i++)
            {
                const int target_factor = neighbors0[i];
                const int edge = variable_edges[variable_edges_offset[this_variable] + i];

                for(int l=0; l<num_labels; l++)
                {
                    variable_to_factor[edge_offset[edge]+l] = 0.0f;

                    bool found = false;

                    for(int j=0; j<neighbors0.size(); j++)
                    {
                        const int other_factor = neighbors0[j];
                        const int other_edge = variable_edges[variable_edges_offset[this_variable]+j];

                        if(other_factor != target_factor)
                        {
                            variable_to_factor[edge_offset[edge]+l] += factor_to_variable[edge_offset[other_edge]+l];
                        }
                        else if(found == false)
                        {
                            found = true;
                        }
                        else
                        {
                            ABORT("Logic error");
                        }
                    }

                    if(found == false)
                    {
                        ABORT("Logic error");
                    }
                }
            }
        }

        // 
    }

    // Compute solution.

    {
        std::vector<bool> is_set(num_variables, false);
        solution.resize(num_variables);
        std::stack<int> stack;
        int num_components = 0;

        for(int root_variable=0; root_variable<num_variables; root_variable++)
        {
            if(is_set[root_variable] == false)
            {
                // set the label of root variable.

                {
                    num_components++;

                    is_set[root_variable] = true;
                    const int num_labels = graph->getNumLabels(root_variable);
                    graph->getFactors(root_variable, neighbors0);

                    bool first = true;
                    double lowest_energy = 0.0;

                    for(int l=0; l<num_labels; l++)
                    {
                        double energy = 0.0;

                        for(int j=0; j<(int)neighbors0.size(); j++)
                        {
                            const int edge = variable_edges[variable_edges_offset[root_variable]+j];
                            energy += factor_to_variable[edge_offset[edge]+l];
                        }

                        if(first || energy < lowest_energy)
                        {
                            first = false;
                            solution[root_variable] = l;
                            lowest_energy = energy;
                        }
                    }

                    if(first) ABORT("Internal error");
                }

                // propagate.

                stack.push(root_variable);

                while(stack.empty() == false)
                {
                    const int this_variable = stack.top();
                    stack.pop();

                    if(is_set[this_variable] == false) ABORT("Internal error!");

                    graph->getFactors(this_variable, neighbors0);

                    for(int i=0; i<(int) neighbors0.size(); i++)
                    {
                        const int edge = variable_edges[variable_edges_offset[this_variable]+i];

                        graph->getVariables(neighbors0[i], neighbors1);

                        int index_in_argmin = 0;

                        for(int j=0; j<(int) neighbors1.size(); j++)
                        {
                            const int that_variable = neighbors1[j];

                            if(that_variable != this_variable)
                            {
                                const int that_label = argmin[argmin_offset[edge] + solution[this_variable] * ( (int) neighbors1.size() - 1) + index_in_argmin];
                                index_in_argmin++;

                                if(is_set[that_variable])
                                {
                                    //if( solution[that_variable] != that_label ) ABORT("Strange!");
                                }
                                else
                                {
                                    solution[that_variable] = that_label;
                                    //std::cout << graph->getNameOfVariable(this_variable) << " says that " << graph->getNameOfVariable(that_variable) << " is " << that_label << std::endl;
                                    is_set[that_variable] = true;
                                    stack.push(that_variable);
                                }
                            }
                        }
                    }
                }
            }
        }

        //std::cout << "Num components: " << num_components << std::endl;
    }

    return true;
}

