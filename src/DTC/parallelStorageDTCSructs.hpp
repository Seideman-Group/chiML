/** @file DTC/parallelStorageDTCStructs.cpp
 *  @brief A set of structs to aid the transfer of data for detectors
 *
 *  A set of structs to aid the transfer of data for detectors
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */
#ifndef FDTD_PARALLELDETECTORSTORAGESTRUCTS
#define FDTD_PARALLELDETECTORSTORAGESTRUCTS

#include <vector>
#include <boost/serialization/vector.hpp>
#include <array>

struct slaveProcDtc
{
    int masterProc_; //!< masterProc_ where to send the data
    int stride_; //!< stride of the copy
    std::array<int,3> sz_; //!< sz_ size of the region in grid points of the detection region inside the slave process
    std::array<int,3> opSz_; //!< size of the field listed in order needed for the copy
    std::vector<int> gridOutGridInds_; //!< A vector storing the indexes of of all blas operation starting points
    std::array<int,3> loc_; //!< loc_ lower left corner of the region inside the slave process to send the data to the master processor
    std::array<int,3> addVec1_; //!< vector to determine what component to iterate over for the first copy loop
    std::array<int,3> addVec2_; //!< vector to determine what component to iterate over for the second copy loop
};

struct slaveProcInfo
{
    int slaveProc_; //!< masterProc_ where to send the data
    int stride_; //!< stride of the copy
    std::array<int,3> sz_; //!< sz_ size of the region in grid points of the detection region inside the slave process
    std::vector<int> gridOutGridInds_; //!< A vector storing the indexes of of all blas operation starting points
    template <typename Archive>
    void serialize(Archive& ar, const unsigned int version)
    {
        ar & slaveProc_;
        ar & stride_;
        ar & sz_;
        ar & gridOutGridInds_;
    }
};

struct copyProcDtc
{
    int stride_; //!< stride of the copy
    int strideOutGrid_; //!< stride inside the output grid
    std::array<int,3> sz_; //!< sz_ size of the region in grid points of the detection region inside the slave process
    std::array<int,3> opSz_; //!< size of the field listed in order needed for the copy
    std::vector<int> gridOutGridInds_; //!< A vector storing the indexes of of all blas operation starting points
};

struct fInParam
{
    int stride_; //!< stride of the copy
    int nCopy_; //!< number of elemnts to include in the copy
    std::array<int,3> sz_; //!< sz_ size of the region in grid points of the detection region inside the slave process
    std::vector<int> fInGridInds_; //!< Vector storing the data indices for each copy in the grid
};

#endif