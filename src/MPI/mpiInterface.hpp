/** @file MPI/mpiInterface.hpp
 *  @brief Class to handle all MPI communication between processes
 *
 *  A class that preforms all MPI communications, and set up for the FDTD grids
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @author Joshua E. Szekely (jeszekely)
 *  @bug No known bugs.
 */

#ifndef FDTD_MPI_INTERFACE
#define FDTD_MPI_INTERFACE

#include <boost/mpi.hpp>
#include <GRID/Grid.hpp>
#ifdef MKL
#include <UTIL/utilities_MKL.hpp>
#elif defined ACML
#include <UTIL/utilities_acml.hpp>
#else
#include <UTIL/utilities.hpp>
#endif

// const static int blocksize__ = 1 ;
typedef std::complex<double> cplx;
typedef std::shared_ptr<Grid<double>> real_grid_ptr;

extern boost::mpi::environment env;
namespace mpi = boost::mpi;

// Augment the boost mpi communicator class with some other useful data
/**
 * @brief Augment the boost MPI communicator class with some other useful data
 * @details MPI communicator used to transfer data throughout the cell.
 *
 */
class mpiInterface : public boost::mpi::communicator
{
protected:
    std::array<int,3> npArr_; //!< array of number of processes in each direction
    std::array<int,3> mypArr_; //!< The local process's process in each direction

public:
    /**
     * @brief      Constructor for the mpiInterface
     */
    mpiInterface();

    /**
     * @brief Accessors to npX
     * @return npX
     */
    inline int npX() {return npArr_[0];}

    /**
     * @brief Accessors to npY
     * @return npY
     */
    inline int npY() {return npArr_[1];}

    /**
     * @brief Accessors to npZ
     * @return npZ
     */
    inline int npZ() {return npArr_[2];}

    /**
     * @brief Accessors to mypX
     * @return mypX
     */
    inline int mypX() {return mypArr_[0];}

    /**
     * @brief Accessors to mypY
     * @return mypY
     */
    inline int mypY() {return mypArr_[1];}
    /**
     * @brief Accessors to mypZ
     * @return mypZ
     */
    inline int mypZ() {return mypArr_[2];}

    /**
     * @brief      Accessor to npArr_
     *
     * @return     npArr_
     */
    inline std::array<int,3> npArr() {return npArr_;}

    /**
     * @brief      Accessor to npArr_
     *
     * @return     npArr_
     */
    inline int npArr(int cor) {return npArr_[cor];}

    /**
     * @brief      Acessor to mypArr_
     *
     * @return     mypArr_
     */
    inline std::array<int,3> mypArr() {return mypArr_;}


    /**
     * @brief      runs numroc with some of the local parameters
     *
     * @param[in]  nx    number of grid points in the x direction
     * @param[in]  ny    number of grid points in the y direction
     * @param[in]  nz    number of grid points in the z direction
     *
     * @return     numroc values for each direction
     */
    std::tuple<int,int, int> getLocxLocyLocz(const int nx, const int ny, const int nz) const;


    /**
     * @brief      Finds the lower, left, back corner of the processor (proc loc for Grids)
     *
     * @param[in]  weight maps of the grids
     *
     * @return     The procLoc for the Grids
     */
    std::tuple<int,int, int> getLocxLocyLocz(std::vector<real_grid_ptr> weights) const;

    /**
     * @brief      Unique int tag generator
     *
     * @param[in]  procSend   sending process
     * @param[in]  procRecv   receiving process
     * @param[in]  maxOffest  number of different communication processes possible between two processes within the same operation
     * @param[in]  offest     the assigned offset corresponding to a single communication within the same operation
     *
     * @return     A unique tag to send information between two processes
     */
    int cantorTagGen(unsigned int procSend, unsigned int procRecv, unsigned int maxOffest, unsigned int offest) { return (int((procSend + procRecv) * (procSend + procSend +1) / 2) + procRecv) * maxOffest + offest; }


};

#endif