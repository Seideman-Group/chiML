/** @file DTC/parallelStorageDTC.hpp
 *  @brief A class that collects the FDTD field information across all processes to one process and place it in a Grid for outputting
 *
 *  Collects FDTD filed information from all processes and transfers it into one place to be
 *  outputted by a detector
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_PARALLELDETECTORSTORAGE
#define FDTD_PARALLELDETECTORSTORAGE

#include <DTC/parallelStorageDTCSructs.hpp>
#include <src/UTIL/ml_consts.hpp>
#include <cstdio>
#include <UTIL/typedefs.hpp>

namespace mpi = boost::mpi;

/**
* @brief The data storage class for the detectors
* @details Creates the output grid on a single processor that then can be used to output the full data set into a file in the correct order
*
*/
template <typename T> class parallelStorageDTC
{
protected:
    bool masterBool_;
    std::array<int,3> loc_; //!< Grid point location of the lower left corner of the detector (full grid space)
    std::array<int,3> sz_; //!< number of grid points in each direction the detector is storing
    std::array<int,3> offSet_; //!< offset for the grid to do proper averaging for outputting
    std::vector<std::shared_ptr<slaveProcInfo>> master_; //!< A shared pointer that is used to generate the output data grod. If the process is not master this is set to a nullptr.
    std::shared_ptr<slaveProcDtc> slave_; //!< A shared pointer for the slaveProcDtc struct, set to a nullptr if the detector area does not include the process
    std::shared_ptr<copyProcDtc> toOutGrid_; //!< used to transfer from actual grid to the outGrid
    std::shared_ptr<parallelGrid<T>> grid_; //!< A shared pointer to a grid that the detector will need to generate the correct output
    std::shared_ptr<mpiInterface> gridComm_; //!< The communicator for the grid and detector (must be the same communicator)

    std::vector<T> scratch_; //!< vector for scratch space

    std::shared_ptr<Grid<T>> outGrid_; //!< A shared pointer to the serial output grid (what is used to make the file)
public:

    /**
     * @brief      Constructs a storage detector (field collector for outputting)
     *
     * @param[in]  grid  pointer to output grid
     * @param[in]  loc   location of lower left back corner of the dtc
     * @param[in]  sz    size in grid points for the dtc
     */
    parallelStorageDTC(std::shared_ptr<parallelGrid<T>> grid, std::array<int,3> loc, std::array<int,3> sz, std::array<int,3> offSet) :
        masterBool_(false),
        loc_(loc),
        sz_(sz),
        offSet_(offSet),
        grid_(grid),
        gridComm_(grid_->gridComm()),
        scratch_(std::accumulate(sz_.begin(), sz_.end(), 1, std::multiplies<int>()),0.0)
    {
        for(int oo = 0; oo < 3; ++oo)
        {
            sz_ [oo] += ( 1 + std::abs(offSet_[oo]) ) / 2;
            loc_[oo] -= ( 1 -          offSet_[oo]  ) / 2;
        }
        // Output grid only made in the master process
        if(grid_->getLocsProc(loc_[0] + (1 - offSet_[0])/2, loc_[1] + (1 - offSet_[1])/2, loc_[2] + (1 - offSet_[2])/2) == gridComm_->rank() )
            outGrid_ = std::make_shared<Grid<T>>( sz_, grid_->d()  );
        else
            outGrid_ = nullptr;

        for (int oo = 0; oo < 3; ++oo)
        {
            scratch_.reserve(scratch_.size() + sz_[(oo+1)%3]*sz_[(oo+2)%3] );
            scratch_.insert( scratch_.begin(), sz_[(oo+1)%3]*sz_[(oo+2)%3], 0.0 );
            offSet_[oo] = std::abs(offSet_[oo]);
        }
        // Set up the filed input structures
        genDatStruct();
    }
    /**
     * @brief  Accessor function to loc_
     *
     * @return   loc_
     */
    inline std::array<int,3> loc() {return loc_;}

    /**
     * @brief  Accessor function to sz_
     *
     * @return  sz_
     */
    inline std::array<int,3> sz() {return sz_;}

    /**
     * @brief  Accessor function to loc_[ii]
     *
     * @return   loc_[ii]
     */
    inline int loc(int ii) {return loc_[ii];}
    /**
     * @brief  Accessor function to sz_[ii]
     *
     * @return  sz_[ii]
     */
    inline int sz(int ii) {return sz_[ii];}

    /**
     * @brief  Accessor function to offSet_
     *
     * @return  offSet_
     */
    inline int offSet( int ii ) {return offSet_[ii];}

    /**
     * @brief  Accessor function to outGrid_
     *
     * @return outGrid_
     */
    inline std::shared_ptr<Grid<T>> outGrid() { return outGrid_; }

    /**
     * @brief  Accessor function to grid_
     *
     * returns the grid pointer
     */
    inline std::shared_ptr<parallelGrid<T>> grid() {return grid_; }

    /**
     * @brief      Calculates the location of where to start the detector in this process
     *
     * @param[in]  ind  The index of the array to look at (0, 1, or 2)
     *
     * @return     The location of the detector's start in this process for the index ind
     */
    int getLocalLocEl(int ind)
    {
        if(loc_[ind] >= grid_->procLoc(ind) && loc_[ind] < grid_->procLoc(ind) + grid_->ln_vec(ind) -2) //!< Does this process hold the lower, left or back boundary of the detector
            return loc_[ind] - grid_->procLoc(ind) + 1;
        else if(loc_[ind] == -1 && grid_->procLoc(ind) == 0)
            return 0;
        else if(loc_[ind] < grid_->procLoc(ind) && loc_[ind] + sz_[ind] > grid_->procLoc(ind)) //!< Does this process start inside the detectors region?
            return 1;
        else
            return -1;
    }

    /**
     * @brief      Calculates the size of the detector on this process
     *
     * @param[in]  ind       The index of the array to look at (0, 1, or 2)
     * @param[in]  localLoc  The location of the detector's start in this process for the index ind
     *
     * @return     The local size el.
     */
    int getLocalSzEl(int ind, int localLoc)
    {
        if(sz_[ind] + loc_[ind] > grid_->procLoc(ind) + grid_->ln_vec(ind) - 2) // Does the detector go through the end of the process' grid?
            return grid_->ln_vec(ind) - localLoc - 1;
        else
            return loc_[ind] + sz_[ind] - (grid_->procLoc(ind) + localLoc - 1);
    }
    /**
     * @brief      Generates the relevant structs to store the fields for output (this is based on if the process is the master or a slave)
     */
    void genDatStruct()
    {
        // Set master process to process containing the lower, left, back corner of the region (excluding the offset)
        int masterProc = grid_->getLocsProc(loc_[0], loc_[1], loc_[2]);

        // Initialize the temporary values, will be split into toOutGrid_ or slave_
        int stride = 0; //!< stride of the blas operations for the filed
        int outGridStride = 0; //!< stride for the operations (for the outGrid)
        std::array<int,3> localFiledInLoc = {-1, -1, -1}; //!< location of where the detector starts for the process (set to element set to -1 if detector is not in the process)
        std::array<int,3> localFiledInSz = {0, 0, 0};  //!< size of the detector in the process
        std::array<int,3> opSz = {0,0,0}; //!< the size reordered in a way convenient to preform blas operations on {size blas, size of inner loop, size of outer loop}
        std::array<int,3> addVec1 = {0,0,0}; //!< array describing the unit vec that the inner loop acts on
        std::array<int,3> addVec2 = {0,0,0}; //!< array describing the unit vec that the outer loop acts on

        // find the location of where the detector starts in this process
        for(int ii = 0; ii < 3; ++ii)
            localFiledInLoc[ii] = getLocalLocEl(ii);
        // if 2D set the z local location to 0
        if(grid_->local_z() == 1)
            localFiledInLoc[2] = 0;
        // If the detector is in the process check everything else
        if( std::all_of(localFiledInLoc.begin(), localFiledInLoc.end(), [](int ii){return ii != -1;} ) )
        {
            // Find the size of the detector in this process
            for(int ii = 0; ii < 3; ++ii)
                localFiledInSz[ii] = getLocalSzEl(ii, localFiledInLoc[ii]);
            // If 2D set the size of the detector in this process in the z direction to 1
            if(grid_->local_z() == 1)
                localFiledInSz[2] = 1;
            // Preform the blas operations over the direction with the most elements
            if(isamax_(localFiledInSz.size(), localFiledInSz.data(), 1) - 1 == 0 )
            {
                // stride based on the stride needed to get the next element in the grid in the x direction
                stride =  1;
                // outGridStride based on the stride needed to get the next element in the outGrid in the x direction
                outGridStride = 1;
                // Have the outer loop be smaller than the inner loop ties go to the direction with the smallest stride
                if( localFiledInSz[1] > localFiledInSz[2] )
                {
                    opSz = {{ localFiledInSz[0] , localFiledInSz[1], localFiledInSz [2] }};
                    addVec1 = {{ 0, 1, 0 }};
                    addVec2 = {{ 0, 0, 1 }};
                }
                else
                {
                    opSz = {{ localFiledInSz[0] , localFiledInSz[2], localFiledInSz [1] }};
                    addVec1 = {{ 0, 0, 1 }};
                    addVec2 = {{ 0, 1, 0 }};
                }
            }
            else if(isamax_(localFiledInSz.size(), localFiledInSz.data(), 1) - 1 == 1 )
            {
                // stride based on the stride needed to get the next element in the grid in the x direction
                stride =  grid_->local_x() * grid_->local_z();
                // outGridStride based on the stride needed to get the next element in the outGrid in the y direction
                outGridStride = sz_[0]*sz_[2];
                // Have the outer loop be smaller than the inner loop ties go to the direction with the smallest stride
                if( localFiledInSz[0] >= localFiledInSz[2] )
                {
                    opSz = {{ localFiledInSz[1] , localFiledInSz[0], localFiledInSz [2] }};
                    addVec1 = {{ 1, 0, 0 }};
                    addVec2 = {{ 0, 0, 1 }};
                }
                else
                {
                    opSz = {{ localFiledInSz[1] , localFiledInSz[2], localFiledInSz [0] }};
                    addVec1 = {{ 0, 0, 1 }};
                    addVec2 = {{ 1, 0, 0 }};
                }
            }
            else
            {
                // stride based on the stride needed to get the next element in the grid in the x direction
                stride =  grid_->local_x();
                // outGridStride based on the stride needed to get the next element in the outGrid in the y direction
                outGridStride = sz_[0];
                // Have the outer loop be smaller than the inner loop ties go to the direction with the smallest stride
                if( localFiledInSz[0] >= localFiledInSz[1] )
                {
                    opSz = {{ localFiledInSz[2] , localFiledInSz[0], localFiledInSz [1] }};
                    addVec1 = {{ 1, 0, 0 }};
                    addVec2 = {{ 0, 1, 0 }};
                }
                else
                {
                    opSz = {{ localFiledInSz[2] , localFiledInSz[1], localFiledInSz [0] }};
                    addVec1 = {{ 0, 1, 0 }};
                    addVec2 = {{ 1, 0, 0 }};
                }
            }
            // If the process is the master set toOutGrid parameters, otherwise set the slave process
            if(gridComm_->rank() == masterProc)
            {
                toOutGrid_ = std::make_shared<copyProcDtc>();
                toOutGrid_->stride_ = stride;
                toOutGrid_->strideOutGrid_ = outGridStride;
                toOutGrid_->sz_ = localFiledInSz;
                toOutGrid_->opSz_ = opSz;
                // Location of where to place the processes field info in the outGrid
                std::array<int,3> locOutGrid = {grid_->procLoc(0) + localFiledInLoc[0] - 1 - loc_[0], grid_->procLoc(1) + localFiledInLoc[1] - 1 - loc_[1],( grid_->local_z() == 1 ? 0 : grid_->procLoc(2) + localFiledInLoc[2] - 1 - loc_[2])}; //!< make loc relative to process grid
                for(int kk = 0; kk < opSz[2]; ++kk )
                {
                    for(int jj = 0; jj < opSz[1]; ++jj)
                    {
                        toOutGrid_->gridOutGridInds_.push_back(    grid_->getInd( localFiledInLoc[0]+jj*addVec1[0]+kk*addVec2[0], localFiledInLoc[1]+jj*addVec1[1]+kk*addVec2[1], localFiledInLoc[2]+jj*addVec1[2]+kk*addVec2[2] ) );
                        toOutGrid_->gridOutGridInds_.push_back( outGrid_->getInd( locOutGrid     [0]+jj*addVec1[0]+kk*addVec2[0], locOutGrid     [1]+jj*addVec1[1]+kk*addVec2[1], locOutGrid     [2]+jj*addVec1[2]+kk*addVec2[2] ) );
                    }
                }
                toOutGrid_->gridOutGridInds_.reserve( toOutGrid_->gridOutGridInds_.size() );
            }
            else
            {
                slave_ = std::make_shared<slaveProcDtc>();
                slave_->masterProc_ = masterProc;
                slave_->stride_ = stride;
                slave_->sz_ = localFiledInSz;
                slave_->opSz_ = opSz;
                for(int kk = 0; kk < slave_->opSz_[2]; ++kk )
                {
                    for(int jj = 0; jj < slave_->opSz_[1]; ++jj)
                    {
                        slave_->gridOutGridInds_.push_back( grid_->getInd(localFiledInLoc[0]+jj*addVec1[0]+kk*addVec2[0], localFiledInLoc[1]+jj*addVec1[1]+kk*addVec2[1], localFiledInLoc[2]+jj*addVec1[2]+kk*addVec2[2]) );
                        slave_->gridOutGridInds_.push_back( slave_->opSz_[0]*(jj + kk*slave_->opSz_[1])  );
                    }
                }
                slave_->gridOutGridInds_.reserve( slave_->gridOutGridInds_.size() );
            }
        }
        gridComm_->barrier();
        slaveProcInfo toMaster;
        // If there is a salve process then set the master info to be equivalent to the slave, otherwise say that its process is -1 (check to see if master needs to ad it)
        if(slave_)
        {
            toMaster.slaveProc_ = gridComm_->rank();
            // Where does this process's information get stored in the outGrid
            std::array<int,3> locOutGrid = {grid_->procLoc(0) + localFiledInLoc[0] - 1 - loc_[0], grid_->procLoc(1) + localFiledInLoc[1] - 1 - loc_[1], grid_->procLoc(2) + localFiledInLoc[2] - 1 - loc_[2]}; //!< make loc relative to process grid
            if(grid_->local_z() == 1)
                locOutGrid[2]  = 0;

            toMaster.stride_ = outGridStride;
            toMaster.sz_ = slave_->opSz_;
            for(int kk = 0; kk < slave_->opSz_[2]; ++kk)
            {
                for(int jj = 0; jj < slave_->opSz_[1]; ++jj)
                {
                    toMaster.gridOutGridInds_.push_back( (jj + slave_->opSz_[1] * kk) * slave_->opSz_[0] );
                    toMaster.gridOutGridInds_.push_back( addVec1[0]*jj+addVec2[0]*kk+locOutGrid[0] + sz_[0] * ( (addVec1[1]*jj+addVec2[1]*kk+locOutGrid[1]) * sz_[2] + addVec1[2]*jj+addVec2[2]*kk+locOutGrid[2] ) );
                }
            }
            toMaster.gridOutGridInds_.reserve( toMaster.gridOutGridInds_.size() );
        }
        else
        {
            toMaster.slaveProc_ = -1;
            toMaster.stride_ = 0;
            toMaster.sz_ = {{ -1, -1, -1}};
        }

        gridComm_->barrier();
        if(gridComm_->rank() == masterProc)
        {
            masterBool_ = true;
            std::vector<slaveProcInfo> allProcs;
            // collect all slave processes into allProcs
            mpi::gather(*gridComm_, toMaster, allProcs, masterProc);
            for(auto & proc : allProcs)
            {
                // If the process does not have the detector discard it, otherwise add it to master_
                if(proc.slaveProc_ != -1)
                    master_.push_back(std::make_shared<slaveProcInfo>(proc) );
            }
        }
        else
            mpi::gather(*gridComm_, toMaster,  masterProc);
        return;
    }

    /**
     * @brief      Master collects al the fields from the slave processes and puts it into the outGrid
     */
    virtual void getField() = 0;

    /**
     * @brief      Accessor function for masterBoo_
     *
     * @return     true if the process is the master process
     */
    inline bool master() { return masterBool_; }
};

class parallelStorageDTCReal : public parallelStorageDTC<double>
{
public:
    /**
     * @brief      Constructs a storage detector (field collector for outputting)
     *
     * @param[in]  grid    pointer to output grid
     * @param[in]  loc     location of lower left back corner of the dtc
     * @param[in]  sz      size in grid points for the dtc
     * @param[in]  offSet  Offset of the grid to ensure proper averaging for the output
     */
    parallelStorageDTCReal(real_pgrid_ptr grid, std::array<int,3> loc, std::array<int,3> sz, std::array<int,3> offSet);
    /**
     * @brief      Master collects al the fields from the slave processes and puts it into the outGrid
     */
    void getField();
};

class parallelStorageDTCCplx : public parallelStorageDTC<cplx>
{
public:
    /**
     * @brief      Constructs a storage detector (field collector for outputting)
     *
     * @param[in]  grid    pointer to output grid
     * @param[in]  loc     location of lower left back corner of the dtc
     * @param[in]  sz      size in grid points for the dtc
     * @param[in]  offSet  Offset of the grid to ensure proper averaging for the output
     */
    parallelStorageDTCCplx(cplx_pgrid_ptr grid, std::array<int,3> loc, std::array<int,3> sz, std::array<int,3> offSet);

    /**
     * @brief      Master collects al the fields from the slave processes and puts it into the outGrid
     */
    void getField();
};

#endif