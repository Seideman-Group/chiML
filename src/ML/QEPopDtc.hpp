/** @file ML/QEPopDtc.cpp
 *  @brief Class used to output the average population in a quantum emitter state
 *
 *  Class that collects the quantum emitter density matrix for all processes, averages them
 *  and outputs it to a text file.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef ML_PARALLEL_POP_DTC
#define ML_PARALLEL_POP_DTC


#include <boost/serialization/complex.hpp>
#include <boost/serialization/vector.hpp>
#include <iomanip>
#include <MPI/mpiInterface.hpp>
/**
 * @brief Detector class for printing out field information to a text file
 * @details Copies the field magnitude to a text file
 */

class QEPopDtc
{
protected:
    std::shared_ptr<mpiInterface> gridComm_; //!< The communicator for the processes that are storing the grid
    int level_; //!< level to output
    int nlevel_; //!< number of levels in the QE object
    int npoints_; //!< number of points in QE object
    int t_step_;  //!< current time step
    int timeInt_; //!< time Interval for inputting the populations

    double tcur_; //!< current time
    double dt_; //!< time step of the simulation

    cplx curPop_; //!< current value of population

    std::vector<double> eWeights_; //!< weights for each level

    std::vector<cplx> allPop_; //!< sum of population of the level at all QE points

    std::string outFile_; //!< output file name


public:
    /**
     * @brief      Constructor for QE population detector
     *
     * @param[in]  gridComm  The MPIInterface for the grids
     * @param[in]  level     The level to be outputted
     * @param[in]  nlevel    The nlevel total number of levels in the system
     * @param[in]  npoints   The npoints total number of points of the QE
     * @param[in]  eWeights  The e weights the weight of the energy level
     * @param[in]  outFile   The out file
     * @param[in]  timeInt   The number of time steps before outputting population data
     * @param[in]  nt        number of time steps to be outputting
     * @param[in]  dt        the size of time step
     */
    QEPopDtc(std::shared_ptr<mpiInterface> gridComm, int level, int nlevel, int npoints, std::vector<double> eWeights, std::string outFile, int timeInt, int nt, double dt);

    /**
     * @brief      input the population of the QE at that level
     *
     * @param      denMat  The density matrix starting value
     */
    inline void inPop(cplx* denMat) { if(t_step_ % timeInt_ == 0) curPop_ +=  *(denMat + level_) ; return;}

    /**
     * @brief      accumlate the population into the storage vector and reset inPop to 0
     */
    void accumPop();

    /**
     * @brief      output accumulated population into a vector
     */
    void toFile();

    /**
     * @brief      accessor function to the level to be outputted
     *
     * @return     level_
     */
    inline int level() { return level_; }

    /**
     * @brief      Accessor to npoints_
     *
     * @return     npoints_
     */
    inline int npoints() { return npoints_; }

    /**
     * @brief      Accessor to output file name
     *
     * @return     outFile_
     */
    inline std::string fname() { return outFile_; }

    inline int timeInt() {return timeInt_;}

};

#endif