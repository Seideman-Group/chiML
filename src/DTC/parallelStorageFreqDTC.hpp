/** @file DTC/parallelStorageFreqDTC.hpp
 *  @brief A class that collects the FDTD field information across all processes to one process to Fourier transform and place it in a Grid for outputting
 *
 *  Collects FDTD filed information from all processes, Fourier transforms it and transfers
 *  it into one place to be outputted by a detector
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_PARALLELDETECTORSTORAGEFREQ
#define FDTD_PARALLELDETECTORSTORAGEFREQ

#include <DTC/parallelStorageDTCSructs.hpp>
#include <src/UTIL/ml_consts.hpp>
#include <UTIL/typedefs.hpp>
#include <cstdio>

namespace mpi = boost::mpi;

template <typename T> class parallelStorageFreqDTC
{
protected:
    char noTranspose_; //!< char for mkl functions
    char transpose_; //!< char for mkl functions
    std::shared_ptr<mpiInterface> gridComm_; //!< mpi interface for all mpi calls
    int nfreq_; //!<  number of frequencies to detect
    int zgemmK_; //!< k value for mkl functions
    cplx ONE_; //!< 1 for mkl functions
    std::array<int,3> loc_; //!< location  of lower left back corner of detection region
    std::array<int,3> sz_; //!< size of the detection region in grid points
    std::array<int,3> offSet_; //!< array storing the offset needed for field averaging
    std::vector<double> freqList_; //!< list of all frequencies
    std::vector<double> fInReal_; //!< vector storing the field input values
    std::vector<double> fInCplx_; //!< vector storing the field input values
    std::vector<cplx> fIn_; //!< vector storing the field input values
    std::vector<cplx> fftFact_; //!< vector storing the exp(i $\omg$ t) values each time step
    std::shared_ptr<std::vector<slaveProcInfo>> master_; //!< parameters for the master process to take in all the slave processes info and combine it
    std::shared_ptr<slaveProcDtc> slave_; //!< parameters for slave processes to get and send info to master
    std::shared_ptr<copyProcDtc> toOutGrid_; //!< a copy param set if master also needs to get info
    std::shared_ptr<parallelGrid<T>> grid_; //!< grid pointer to the field that is being stored

    cplx_grid_ptr outGrid_; //!< the output grid storage

    std::shared_ptr<fInParam> fieldInFreq_; //!< parameters to take in field
public:
    /**
     * @brief      construct the frequency storage dtc
     *
     * @param[in]  dtcNum    the detector number
     * @param[in]  grid      pointer to output grid
     * @param[in]  propDir   The direction of propagation
     * @param[in]  loc       location of lower left corner of the dtc
     * @param[in]  sz        size in grid points for the dtc
     * @param[in]  freqList  The frequency list
     */
    parallelStorageFreqDTC(int dtcNum, std::shared_ptr<parallelGrid<T>> grid, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList) :
        gridComm_(grid->gridComm()),
        grid_(grid),
        noTranspose_('N'),
        transpose_('T'),
        freqList_(freqList),
        nfreq_(freqList.size()),
        zgemmK_(1),
        ONE_(1.0,0.0),
        loc_(loc),
        sz_(sz),
        fftFact_(nfreq_, 0.0)
    {
        genDatStruct(dtcNum, propDir);
        if(outGrid_)
        {
            int szProd = std::accumulate(fieldInFreq_->sz_.begin(), fieldInFreq_->sz_.end(), 1, std::multiplies<int>() );
            fInReal_ = std::vector<double>(szProd*nfreq_, 0.0);
            fInCplx_ = std::vector<double>(szProd*nfreq_, 0.0);
            fIn_ = std::vector<cplx>(szProd*nfreq_, 0.0);
        }
    }

    /**
     * @brief      construct the frequency storage dtc
     *
     * @param[in]  dtcNum    the detector number
     * @param[in]  gridOff   pair of a grid pointer and
     * @param[in]  propDir   The direction of propagation
     * @param[in]  loc       location of lower left corner of the dtc
     * @param[in]  sz        size in grid points for the dtc
     * @param[in]  freqList  The frequency list
     */
    parallelStorageFreqDTC(int dtcNum, std::pair< std::shared_ptr<parallelGrid<T>>, std::array<int,3> > gridOff, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList) :
        gridComm_(std::get<0>(gridOff)->gridComm()),
        grid_( std::get<0>(gridOff) ),
        noTranspose_('N'),
        transpose_('T'),
        nfreq_(freqList.size()),
        freqList_(freqList),
        zgemmK_(1),
        ONE_(1.0,0.0),
        loc_(loc),
        sz_(sz),
        offSet_( std::get<1>(gridOff) ),
        fftFact_(nfreq_, 0.0)
    {
        genDatStruct(dtcNum, propDir);
        if(outGrid_)
        {
            int szProd = std::accumulate(fieldInFreq_->sz_.begin(), fieldInFreq_->sz_.end(), 1, std::multiplies<int>() );
            fInReal_ = std::vector<double>(szProd*nfreq_, 0.0);
            fInCplx_ = std::vector<double>(szProd*nfreq_, 0.0);
            fIn_ = std::vector<cplx>(szProd*nfreq_, 0.0);
        }
    }
    /**
    * @brief  Accessor function for loc_
    *
     * @return  reference to loc_
     */
    inline std::array<int,3> loc() {return loc_;}

    /**
     * @brief  Accessor function for sz_
     *
     * @return reference to sz_
     */
    inline std::array<int,3> sz() {return sz_;}

    /**
     * @brief  Accessor function for offSet_
     *
     * @return    offSet_
     */
    inline std::array<int, 3> offSet() {return offSet_;}
    /**
     * @brief  Accessor function for offSet_[ii]
     *
     * @return    offSet_[ii]
     */
    inline int offSet(int ii) {return offSet_[ii];}

    /**
     * @brief  Accessor function for loc_[ii]
     *
     * @return  reference to loc_[ii]
     */
    inline int loc(int ii) {return loc_[ii];}

    /**
     * @brief  Accessor function for sz_[ii]
     *
     * @return reference to sz_[ii]
     */
    inline int sz(int ii) {return sz_[ii];}

    /**
     * @brief  Accessor function for outGrid_;
     *
     * @return reference to outGrid_
     */
    inline cplx_grid_ptr outGrid() { return outGrid_; }

    /**
     * @brief      Calculates the location of where to start the detector in this process
     *
     * @param[in]  ind  The index of the array to look at (0, 1, or 2)
     *
     * @return     The location of the detector's start in this process for the index ind
     */
    int getLocalLocEl(int ind)
    {
        if(loc_[ind] >= grid_->procLoc(ind) && loc_[ind] < grid_->procLoc(ind) + grid_->ln_vec(ind) - 2) //!< Does this process hold the lower, left or back boundary of the detector
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
     * @brief      Generates the field input parameters
     *
     * @param[in]  propDir  The direction of propagation through the region that is being looked at
     */
    void genDatStruct(int dtcNum, DIRECTION propDir)
    {
        fInParam fieldInFreq;
        std::array<int,3> loc = {-1, -1, -1};
        std::array<int,3> addVec1 = {0,0,0};
        std::array<int,3> addVec2 = {0,0,0};
        fieldInFreq.sz_ = {0,0,0};
        fieldInFreq.stride_ = 0;

        for(int ii = 0; ii < 3; ++ii)
            loc[ii] = getLocalLocEl(ii);
        if(grid_->local_z() == 1)
            loc[2] = 0;

        // If either loc values is 0 then the detector is not in the process
        if(loc[0] != -1 && loc[1] != -1 && loc[2] != -1)
        {
            if(propDir == DIRECTION::X)
            {
                // set stride to number of elements between consecutive z or y (if 2D) points
                fieldInFreq.stride_ = grid_->local_x();

                // set outer loop to direction of propagation (likely 1)
                fieldInFreq.sz_[2] = getLocalSzEl(0, loc[0] );

                if( grid_->local_z() == 1 )
                {
                    // if 2D set blas operations sizes to 1
                    fieldInFreq.sz_[0] = getLocalSzEl(1, loc[1] );
                    // Z size is always 1
                    fieldInFreq.sz_[1] = 1;

                    // make add arrays represent the sz arrays
                    addVec1 = {0, 0, 1};
                }
                else
                {
                    // set blas operations to the z direction
                    fieldInFreq.sz_[0] = getLocalSzEl(2, loc[2] );
                    // set inner loop to direction to y
                    fieldInFreq.sz_[1] = getLocalSzEl(1, loc[1] );

                    // make add arrays represent the sz arrays
                    addVec1 = {0, 1, 0};
                }
                outGrid_ = std::make_shared<Grid<cplx>>( std::array<int,3>( {{nfreq_, fieldInFreq.sz_[1], fieldInFreq.sz_[0] }}) ,  std::array<double,3>( {{freqList_[1] - freqList_[0], grid_->dz(), grid_->dy() }} ) );
            }
            else if(propDir == DIRECTION::Y)
            {
                // set stride to number of elements between consecutive x points
                fieldInFreq.stride_ = 1;

                // set size of blas operations to x sizw
                fieldInFreq.sz_[0] = getLocalSzEl(0, loc[0]);
                // set the inner loop to the larger non blas operation sizes
                fieldInFreq.sz_[1] = (grid_->local_z() == 1) ? 1 : getLocalSzEl(2, loc[2]);
                // set outer loop to be propagation/smallest size direction
                fieldInFreq.sz_[2] = getLocalSzEl(1, loc[1]);

                // make add arrays represent the sz arrays
                addVec1 = {0, 0, 1};
                outGrid_ = std::make_shared<Grid<cplx>>( std::array<int,3>( {{nfreq_, fieldInFreq.sz_[1], fieldInFreq.sz_[0] }}) ,  std::array<double,3>( {{freqList_[1] - freqList_[0], grid_->dz(), grid_->dx() }} ) );
            }
            else if(propDir == DIRECTION::Z)
            {
                // set stride to number of elements between consecutive x points
                fieldInFreq.stride_ = 1;

                // set size of blas operations to x sizw
                fieldInFreq.sz_[0] = getLocalSzEl(0, loc[0]);
                // set the inner loop to the larger non blas operation sizes
                fieldInFreq.sz_[1] = getLocalSzEl(1, loc[1]);
                // set outer loop to be propagation/smallest size direction
                fieldInFreq.sz_[2] = getLocalSzEl(2, loc[2]);

                // make add arrays represent the sz arrays
                addVec1 = {0, 1, 0};
                outGrid_ = std::make_shared<Grid<cplx>>( std::array<int,3>( {{nfreq_, fieldInFreq.sz_[1], fieldInFreq.sz_[0] }}) ,  std::array<double,3>( {{freqList_[1] - freqList_[0], grid_->dy(), grid_->dx() }} ) );
            }
            else
            {
                // set stride to number of elements between consecutive x points
                fieldInFreq.stride_ = 1;

                // set size of blas operations to x sizw
                fieldInFreq.sz_[0] = getLocalSzEl(0, loc[0]);
                // set the inner loop to the larger non blas operation sizes
                fieldInFreq.sz_[1] = getLocalSzEl(2, loc[2]);
                // set outer loop to be propagation/smallest size direction
                fieldInFreq.sz_[2] = getLocalSzEl(1, loc[1]);

                // make add arrays represent the sz arrays
                addVec1 = {0, 0, 1};
                addVec2 = {0, 1, 0};

                outGrid_ = std::make_shared<Grid<cplx>>( std::array<int,3>( {{nfreq_, fieldInFreq.sz_[1]*fieldInFreq.sz_[2], fieldInFreq.sz_[0] }}) ,  std::array<double,3>( {{freqList_[1] - freqList_[0], grid_->dx(), 1.0 }} ) );
            }
            fieldInFreq.fInGridInds_ = std::vector<int>(2*fieldInFreq.sz_[1]*fieldInFreq.sz_[2],0.0);
            if(propDir == DIRECTION::NONE)
            {
                for(int jj = 0; jj < 2*fieldInFreq.sz_[2]; jj+=2)
                {
                    for(int ii = 0; ii < 2*fieldInFreq.sz_[1]; ii+=2)
                    {
                        fieldInFreq.fInGridInds_[ii+jj*fieldInFreq.sz_[1]  ] =    grid_->getInd( loc[0]+ii/2*addVec1[0]+jj/2*addVec2[0], loc[1]+ii/2*addVec1[1]+jj/2*addVec2[1], loc[2]+ii/2*addVec1[2]+jj/2*addVec2[2] );
                        fieldInFreq.fInGridInds_[ii+jj*fieldInFreq.sz_[1]+1] = outGrid_->getInd( 0, ii/2+jj/2*fieldInFreq.sz_[1], 0);
                    }
                }
            }
            else
            {
                for(int ii = 0; ii < 2*fieldInFreq.sz_[1]; ii+=2)
                {
                    fieldInFreq.fInGridInds_[ii  ] =    grid_->getInd( loc[0]+ii/2*addVec1[0], loc[1]+ii/2*addVec1[1], loc[2]+ii/2*addVec1[2] );
                    fieldInFreq.fInGridInds_[ii+1] = outGrid_->getInd( 0, ii/2, 0);
                }
            }
            fieldInFreq.fInGridInds_.reserve( fieldInFreq.fInGridInds_.size() );
            fieldInFreq_ = std::make_shared<fInParam>(fieldInFreq); //!< only make slave active if necessary
        }
        return;
    }

    /**
     * @brief      take in fields
     *
     * @param      fftFact  The fourier transform factors
     */
    virtual void fieldIn(cplx* fftFact) = 0;

    /**
     * @brief      Moves vector representation of fields from the fIn vector to the outgrids
     */
    virtual void toOutGrid() = 0;

    /**
     * @brief    Accessor Function for master_
     * @return   master_
     */
    inline std::shared_ptr<std::vector<slaveProcInfo>> master() { return master_; }
};


class parallelStorageFreqDTCReal : public parallelStorageFreqDTC<double>
{
public:
    /**
     * @brief      construct the frequency storage dtc
     *
     * @param[in]  dtcNum    the detector number
     * @param[in]  grid      pointer to output grid
     * @param[in]  propDir   The direction of propagation
     * @param[in]  loc       location of lower left corner of the dtc
     * @param[in]  sz        size in grid points for the dtc
     * @param[in]  freqList  The frequency list
     */
    parallelStorageFreqDTCReal(int dtcNum, real_pgrid_ptr grid, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList);

    /**
     * @brief      construct the frequency storage dtc
     *
     * @param[in]  dtcNum    the detector number
     * @param[in]  gridOff   pair of a grid pointer and
     * @param[in]  propDir   The direction of propagation
     * @param[in]  loc       location of lower left corner of the dtc
     * @param[in]  sz        size in grid points for the dtc
     * @param[in]  freqList  The frequency list
     */
    parallelStorageFreqDTCReal(int dtcNum, std::pair< real_pgrid_ptr, std::array<int,3> > gridOff, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList);
    /**
     * @brief      take in fields
     *
     * @param      fftFact  The fourier transform factors
     */
    void fieldIn(cplx* fftFact);

    /**
     * @brief      Moves vector representation of fields from the fIn vector to the outgrids
     */
    void toOutGrid();

};

/**
 * Complex version of FluxDTC see base class for more descriptions
 */
class parallelStorageFreqDTCCplx : public parallelStorageFreqDTC<cplx>
{
public:
    /**
     * @brief      construct the frequency storage dtc
     *
     * @param[in]  dtcNum    the detector number
     * @param[in]  grid      pointer to output grid
     * @param[in]  propDir   The direction of propagation
     * @param[in]  loc       location of lower left corner of the dtc
     * @param[in]  sz        size in grid points for the dtc
     * @param[in]  freqList  The frequency list
     */
    parallelStorageFreqDTCCplx(int dtcNum, cplx_pgrid_ptr grid, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList);

    /**
     * @brief      construct the frequency storage dtc
     *
     * @param[in]  dtcNum    the detector number
     * @param[in]  gridOff   pair of a grid pointer and
     * @param[in]  propDir   The direction of propagation
     * @param[in]  loc       location of lower left corner of the dtc
     * @param[in]  sz        size in grid points for the dtc
     * @param[in]  freqList  The frequency list
     */
    parallelStorageFreqDTCCplx(int dtcNum, std::pair< cplx_pgrid_ptr, std::array<int,3> > gridOff, DIRECTION propDir, std::array<int,3> loc, std::array<int,3> sz, std::vector<double> freqList);

    /**
     * @brief      take in fields
     *
     * @param      fftFact  The fourier transform factors
     */
    void fieldIn(cplx* fftFact);

    /**
     * @brief      Moves vector representation of fields from the fIn vector to the outgrids
     */
    void toOutGrid();

};

#endif