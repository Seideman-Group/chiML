/** @file DTC/parallelStorageDTC.cpp
 *  @brief A class that collects the FDTD field information across all processes to one process and place it in a Grid for outputting
 *
 *  Collects FDTD filed information from all processes and transfers it into one place to be
 *  outputted by a detector
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <DTC/parallelStorageDTC.hpp>

parallelStorageDTCReal::parallelStorageDTCReal(real_pgrid_ptr grid, std::array<int,3> loc, std::array<int,3> sz, std::array<int,3> offSet) :
    parallelStorageDTC(grid, loc, sz, offSet)
{}

void parallelStorageDTCReal::getField()
{
    // If process has a part of the field and is stores the outGrid copy relevant field info directly to the out_grid
    if(toOutGrid_)
    {
        for(int ii = 0; ii < toOutGrid_->gridOutGridInds_.size(); ii+=2)
            dcopy_(toOutGrid_->opSz_[0], &grid_->point(toOutGrid_->gridOutGridInds_[ii]), toOutGrid_->stride_, &outGrid_->point(toOutGrid_->gridOutGridInds_[ii+1]), toOutGrid_->strideOutGrid_);
    }
    // If the process is a slave process not holding the outGrid then copy the field information to a vector and send it to master
    if(slave_)
    {
        for(int ii = 0; ii < slave_->gridOutGridInds_.size(); ii+=2)
           dcopy_(slave_->opSz_[0], &grid_->point( slave_->gridOutGridInds_[ii] ), slave_->stride_, &scratch_[ slave_->gridOutGridInds_[ii+1] ], 1);

        gridComm_->send(slave_->masterProc_, gridComm_->cantorTagGen(gridComm_->rank(), slave_->masterProc_, 1, 0), scratch_);
    }
    // If master then for each slave recv the information and copy it to outGrid
    if(masterBool_)
    {
        for(auto & slave : master_)
        {
            gridComm_->recv(slave->slaveProc_, gridComm_->cantorTagGen(slave->slaveProc_, gridComm_->rank(), 1, 0), scratch_);
            for(int ii = 0; ii < slave->gridOutGridInds_.size(); ii+=2)
                dcopy_(slave->sz_[0], &scratch_[ slave->gridOutGridInds_[ii] ], 1, &outGrid_->point( slave->gridOutGridInds_[ii+1] ), slave->stride_);
        }
    }
    return;
}

parallelStorageDTCCplx::parallelStorageDTCCplx(cplx_pgrid_ptr grid, std::array<int,3> loc, std::array<int,3> sz, std::array<int,3> offSet) :
    parallelStorageDTC(grid, loc, sz, offSet)
{}

void parallelStorageDTCCplx::getField()
{
    // If process has a part of the field and is stores the outGrid copy relevant field info directly to the out_grid
    if(toOutGrid_)
    {
        for(int ii = 0; ii < toOutGrid_->gridOutGridInds_.size(); ii+=2)
            zcopy_(toOutGrid_->opSz_[0], &grid_->point(toOutGrid_->gridOutGridInds_[ii]), toOutGrid_->stride_, &outGrid_->point(toOutGrid_->gridOutGridInds_[ii+1]), toOutGrid_->strideOutGrid_);
    }
    // If the process is a slave process not holding the outGrid then copy the field information to a vector and send it to master
    if(slave_)
    {
        for(int ii = 0; ii < slave_->gridOutGridInds_.size(); ii+=2)
           zcopy_(slave_->opSz_[0], &grid_->point( slave_->gridOutGridInds_[ii] ), slave_->stride_, &scratch_[ slave_->gridOutGridInds_[ii+1] ], 1);
        gridComm_->send(slave_->masterProc_, gridComm_->cantorTagGen(gridComm_->rank(), slave_->masterProc_, 1, 0), scratch_);
    }
    // If master then for each slave recv the information and copy it to outGrid
    if(masterBool_)
    {
        for(auto & slave : master_)
        {
            gridComm_->recv(slave->slaveProc_, gridComm_->cantorTagGen(slave->slaveProc_, gridComm_->rank(), 1, 0), scratch_);
            for(int ii = 0; ii < slave->gridOutGridInds_.size(); ii+=2)
                zcopy_(slave->sz_[0], &scratch_[ slave->gridOutGridInds_[ii] ], 1, &outGrid_->point( slave->gridOutGridInds_[ii+1] ), slave->stride_);
        }
    }
    return;
}