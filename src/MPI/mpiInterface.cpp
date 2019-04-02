/** @file MPI/mpiInterface.cpp
 *  @brief Class to handle all MPI communication between processes
 *
 *  A class that preforms all MPI communications, and set up for the FDTD grids
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @author Joshua E. Szekely (jeszekely)
 *  @bug No known bugs.
 */

#include "mpiInterface.hpp"
#include <iostream>
#include <iomanip>

mpiInterface::mpiInterface() : boost::mpi::communicator()
{
    npArr_  = { 1, size(), 1 };
    mypArr_ = { 0, rank(), 0 };
}


std::tuple<int,int,int> mpiInterface::getLocxLocyLocz(std::vector<real_grid_ptr> weights) const
{
    std::vector<double> yE_weights( weights[0]->y(), 0 );
    for(auto& grid : weights)
    {
        for(int rr = 0; rr < weights[0]->y(); rr++)
        {
            yE_weights[rr] += std::accumulate(&grid->point(0,rr,0), &grid->point(0,rr,0)+grid->x()*grid->z(), 0);
        }
    }
    double yWeightSum_E = std::accumulate(yE_weights.begin(), yE_weights.end(), 0.0);

    double yWeightAvg_E = static_cast<double>(yWeightSum_E) / static_cast<double>(npArr_[1]);
    double valWeight = 0.0;

    std::vector<int> yStartBound(npArr_[1], 0);
    std::vector<int> yEndBound(npArr_[1], weights[0]->y()-1);
    int curY = 0;
    for(int cc = 0; cc < yEndBound.size()-1; cc++)
    {
        curY = yStartBound[cc];
        while(valWeight + yE_weights[curY] / 2.0 < yWeightAvg_E || curY == yStartBound[cc])
        {
            valWeight += yE_weights[curY];
            curY++;
        }
        valWeight -= yWeightAvg_E;
        yEndBound[cc]     = curY-1;
        yStartBound[cc+1] = curY;
    }
    return std::make_tuple(weights[0]->x()+2, yEndBound[mypArr_[1]] - yStartBound[mypArr_[1]] + 3, weights[0]->z()+2 );
}

std::tuple<int,int,int> mpiInterface::getLocxLocyLocz(const int ndim, const int ncol, const int nslice) const
{
    return std::make_tuple(ndim + 2, ncol/npArr_[1] + ( mypArr_[1] < ncol%npArr_[1] ? 1 : 0 ) , nslice + 2 );
}
