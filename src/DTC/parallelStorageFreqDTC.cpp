/** @file DTC/parallelStorageFreqDTC.cpp
 *  @brief A class that collects the FDTD field information across all processes to one process to Fourier transform and place it in a Grid for outputting
 *
 *  Collects FDTD filed information from all processes, Fourier transforms it and transfers
 *  it into one place to be outputted by a detector
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <DTC/parallelStorageFreqDTC.hpp>
parallelStorageFreqDTCReal::parallelStorageFreqDTCReal(int dtcNum, real_pgrid_ptr grid, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList) :
    parallelStorageFreqDTC(dtcNum, grid, propDir, loc, sz, freqList)
{}

parallelStorageFreqDTCReal::parallelStorageFreqDTCReal(int dtcNum, std::pair< real_pgrid_ptr, std::array<int,3> > gridOff, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList) :
    parallelStorageFreqDTC(dtcNum, gridOff, propDir, loc, sz, freqList)
{}


void parallelStorageFreqDTCReal::fieldIn(cplx* fftFact)
{
    if(!fieldInFreq_)
        return;
    for(int ii = 0; ii < fieldInFreq_->fInGridInds_.size(); ii+=2)
    {
        dger_(nfreq_, fieldInFreq_->sz_[0], 1.0, reinterpret_cast<double*>(fftFact)  , 2, &grid_->point(fieldInFreq_->fInGridInds_[ii]), fieldInFreq_->stride_, &fInReal_[ fieldInFreq_->fInGridInds_[ii+1] ], nfreq_);
        dger_(nfreq_, fieldInFreq_->sz_[0], 1.0, reinterpret_cast<double*>(fftFact)+1, 2, &grid_->point(fieldInFreq_->fInGridInds_[ii]), fieldInFreq_->stride_, &fInCplx_[ fieldInFreq_->fInGridInds_[ii+1] ], nfreq_);
    }
}

parallelStorageFreqDTCCplx::parallelStorageFreqDTCCplx(int dtcNum, cplx_pgrid_ptr grid, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList) :
    parallelStorageFreqDTC(dtcNum, grid, propDir, loc, sz, freqList)
{}

parallelStorageFreqDTCCplx::parallelStorageFreqDTCCplx(int dtcNum, std::pair< cplx_pgrid_ptr, std::array<int,3> > gridOff, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList) :
    parallelStorageFreqDTC(dtcNum, gridOff, propDir, loc, sz, freqList)
{}

void parallelStorageFreqDTCCplx::fieldIn(cplx* fftFact)
{
    if(!fieldInFreq_)
        return;
    // Take an outer product of the prefactor vector and the field vectors to get the discrete Fourier Transform at all points
    for(int ii = 0; ii < fieldInFreq_->fInGridInds_.size(); ii+=2)
        zgerc_(nfreq_, fieldInFreq_->sz_[0], 1.0, fftFact, 1, &grid_->point(fieldInFreq_->fInGridInds_[ii]), fieldInFreq_->stride_, &outGrid_->point(fieldInFreq_->fInGridInds_[ii+1]), nfreq_);
}

void parallelStorageFreqDTCReal::toOutGrid()
{
    if(outGrid_)
    {
        dcopy_(outGrid_->size(), fInReal_.data(), 1, reinterpret_cast<double*>( outGrid_->data() )  , 2);
        dcopy_(outGrid_->size(), fInCplx_.data(), 1, reinterpret_cast<double*>( outGrid_->data() )+1, 2);
    }
    return;
}

void parallelStorageFreqDTCCplx::toOutGrid()
{
    return;
}