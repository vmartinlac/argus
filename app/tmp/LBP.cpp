#include <iostream>
#include <tbb/parallel_for.h>
#include <tbb/parallel_for_each.h>
#include <tbb/blocked_range2d.h>
#include "LBP.h"

void LoopyBeliefPropagationSolver::run(
    int num_labels,
    const std::vector<cv::Point>& neighbors,
    const cv::Mat1b& connections,
    const cv::Mat1f& data_cost,
    const cv::Mat1f& discontinuity_cost,
    int num_iterations,
    cv::Mat_<Label>& result)
{
    // define some constants.

    const int num_neighbors = static_cast<int>( neighbors.size() );
    const cv::Size image_size( connections.size[1], connections.size[0] );
    const cv::Rect ROI(0, 0, image_size.width, image_size.height);
    const tbb::blocked_range2d<int> whole_range(0, image_size.height, 0, image_size.width);

    // check that input is coherent.

    assert( neighbors.size() == num_neighbors );
    assert( num_neighbors % 2 == 0 );
    for(int i=0; i<num_neighbors; i++)
    {
        assert( neighbors[ (i+num_neighbors/2)%num_neighbors ] == -neighbors[i] );
    }

    assert( connections.dims == 3 );
    assert( connections.size[0] == image_size.height );
    assert( connections.size[1] == image_size.width );
    assert( connections.size[2] == num_neighbors );
    for( int i=0; i<image_size.height; i++)
    {
        for(int j=0; j<image_size.width; j++)
        {
            const cv::Point this_point(j,i);

            for(int k=0; k<num_neighbors; k++)
            {
                const cv::Point that_point = this_point + neighbors[k];

                if(ROI.contains(that_point))
                {
                    assert( connections.at<uint8_t>( cv::Vec3i(i,j,k) ) == connections.at<uint8_t>( cv::Vec3i(that_point.y, that_point.x, (k+num_neighbors/2)%num_neighbors)) );
                }
                else
                {
                    assert( connections.at<uint8_t>( cv::Vec3i(i,j,k) ) == 0 );
                }
            }
        }
    }

    assert( data_cost.dims == 3 );
    assert( data_cost.size[0] == image_size.height );
    assert( data_cost.size[1] == image_size.width );
    assert( data_cost.size[2] == num_labels );

    assert( discontinuity_cost.dims == 2 );
    assert( discontinuity_cost.size[0] == num_labels );
    assert( discontinuity_cost.size[1] == num_labels );
    for(int i=0; i<num_labels; i++)
    {
        for(int j=0; j<num_labels; j++)
        {
            assert( discontinuity_cost(i,j) == discontinuity_cost(j,i) );
        }
    }

    // initialize message buffers and result array.

    cv::Mat1f messages;
    cv::Mat1f new_messages;

    {
        const int dimensions[4] = { image_size.height, image_size.width, num_neighbors, num_labels };

        messages.create(4, dimensions);
        new_messages.create(4, dimensions);
        messages = 0.0f;
        new_messages = 0.0f;
    }

    result.create(image_size);

    // define procedures.

    auto update_messages_pred = [&connections, &messages, &new_messages, &neighbors, &ROI, num_neighbors, num_labels, &data_cost, &discontinuity_cost] (tbb::blocked_range2d<int> range)
    {
        for(int i=range.rows().begin(); i<range.rows().end(); i++)
        {
            for(int j=range.cols().begin(); j<range.cols().end(); j++)
            {
                const cv::Point this_point(j,i);

                for(int k=0; k<num_neighbors; k++)
                {
                    if(connections(cv::Vec3i(i, j, k)))
                    {
                        const cv::Point that_point = this_point + neighbors[k];

                        if(ROI.contains(that_point) == false)
                        {
                            std::cout << "Internal error!" << std::endl;
                            exit(1);
                        }

                        for(int m=0; m<num_labels; m++)
                        {
                            bool first = true;
                            float minimal_value = 0.0f;

                            for(int n=0; n<num_labels; n++)
                            {
                                float value = data_cost(cv::Vec3i(this_point.y, this_point.x, n)) + discontinuity_cost(n,m);

                                for(int l=0; l<neighbors.size(); l++)
                                {
                                    const cv::Point third_point = this_point + neighbors[l];

                                    if( third_point != that_point && connections(cv::Vec3i(this_point.y, this_point.x, l)) )
                                    {
                                        if(ROI.contains(third_point) == false)
                                        {
                                            std::cout << "Internal error!" << std::endl;
                                            exit(1);
                                        }

                                        value += messages(cv::Vec4i(third_point.y, third_point.x, (l+num_neighbors/2)%num_neighbors, n));

                                        /*
                                        if( std::isnormal(value) == false )
                                        {
                                            static std::mutex m;
                                            m.lock();
                                            std::cout << messages.size[0] << " " << messages.size[1] << " " << messages.size[2] << " " << messages.size[3] << std::endl;
                                            std::cout << messages(cv::Vec4i(third_point.y, third_point.x, (l+num_neighbors/2)%num_neighbors, n)) << std::endl;
                                            std::cout << third_point << std::endl;
                                            std::cout << (l+num_neighbors/2)%num_neighbors << std::endl;
                                            std::cout << n << std::endl;
                                            std::cout << std::endl;
                                            m.unlock();
                                            exit(2);
                                        }
                                        */

                                        /*
                                        if( messages(cv::Vec4i(third_point.y, third_point.x, (l+num_neighbors/2)%num_neighbors, n)) != 0.0f )
                                        {
                                            std::cout << "A" << std::endl;
                                            exit(1);
                                        }
                                        */
                                    }
                                }

                                if(first || value < minimal_value)
                                {
                                    minimal_value = value;
                                    first = false;
                                }
                            }

                            if(first)
                            {
                                std::cout << "Internal error!" << std::endl;
                                exit(1);
                            }

                            new_messages(cv::Vec4i(i,j,k,m)) = minimal_value;
                            //std::cout << messages.at<float>(cv::Vec4i(that_point.y, that_point.x, (l+num_neighbors/2)%num_neighbors, k)) << std::endl;
                            //if(std::isnormal(minimal_value) == false) std::cout << minimal_value << std::endl;
                        }

                        double sum = 0.0;

                        for(int m=0; m<num_labels; m++)
                        {
                            sum += new_messages(cv::Vec4i(i,j,k,m));
                        }

                        if( std::fabs(sum) > 1.0e-4 )
                        {
                            for(int m=0; m<num_labels; m++)
                            {
                                new_messages(cv::Vec4i(i,j,k,m)) /= sum;
                            }
                        }
                    }
                    else
                    {
                        for(int m=0; m<num_labels; m++)
                        {
                            new_messages(cv::Vec4i(i,j,k,m)) = 0.0f;
                        }
                    }
                }
            }
        }
    };

    auto update_result_pred = [&data_cost, &connections, &ROI, &neighbors, &messages, num_neighbors, num_labels, &result] (tbb::blocked_range2d<int> range)
    {
        for(int i=range.rows().begin(); i<range.rows().end(); i++)
        {
            for(int j=range.cols().begin(); j<range.cols().end(); j++)
            {
                const cv::Point this_point(j,i);

                bool first = true;
                float minimal_value = 0.0f;

                for(int k=0; k<num_labels; k++)
                {
                    float value = data_cost( cv::Vec3i(i,j,k) );

                    for(int l=0; l<neighbors.size(); l++)
                    {
                        const cv::Point that_point = this_point + neighbors[l];

                        if( connections( cv::Vec3i(i,j,l) ) )
                        {
                            if(ROI.contains(that_point) == false)
                            {
                                std::cout << "Internal error!" << std::endl;
                                exit(1);
                            }

                            value += messages(cv::Vec4i(that_point.y, that_point.x, (l+num_neighbors/2)%num_neighbors, k));
                        }
                    }

                    if(first || value < minimal_value)
                    {
                        minimal_value = value;
                        result(i,j) = Label(k);
                        first = false;
                    }
                }

                if(first)
                {
                    throw std::runtime_error("internal error");
                }
            }
        }
    };

    // run solver.

    for(int i=0; i<num_iterations; i++)
    {
        std::cout << "LoopyBeliefPropagation iteration " << i << "/" << num_iterations << std::endl;
        tbb::parallel_for(whole_range, update_messages_pred);

        //update_messages_pred(whole_range);
        cv::swap(messages, new_messages);
    }

    tbb::parallel_for(whole_range, update_result_pred);
    //update_result_pred(whole_range);
}

