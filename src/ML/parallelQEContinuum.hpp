#ifndef ML_PRALLEL_QEContinuum
#define ML_PRALLEL_QEContinuum

#include "Hamiltonian.hpp"
#include "density.hpp"
#include <ML/Continuum.hpp>
// #include <src/UTIL/utilities.hpp>
// #include <src/GRID/parallelGrid.hpp>
#include <unordered_map>
#include <UTIL/FDTD_up_eq.hpp>
// #include <functional>


// Parameters to input E field information from the main grids to a processor
struct RecvEFieldSendPFieldCont
{
    int procMainGrid_; //!< Processor of the E/H field Grid
    int tagSend_; //!< tag for sending data
    int tagRecv_; //!< tag for reciving data

    int nx_; //!< number of grid points in the x direction that this part of the QE covers
    int ny_; //!< number of grid points in the y direction that this part of the QE covers

    int sz_; //!< number of grid points that E fields need to receive from the main grid
    int szP_; //!< number of grid points the P fields need to send back to the main grid

    std::array<int,2> loc_; //!< location of the lower left corner that is starting in the main grid

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & procMainGrid_;
        ar & tagSend_;
        ar & tagRecv_;
        ar & nx_;
        ar & ny_;
        ar & szP_;
        // ar & nxP_;
        // ar & nyP_;

        ar & sz_;
        ar & loc_;
    }
};

/**
 * @brief Quantum Emitter Class
 * @details A class that models the quantum emitters for ML
 *
 * @param Px_
 * @param Py_
 * @param Pz_
 * @param prevEx_
 * @param prevEy_
 * @param prevEz_
 *
 */
template <typename T> class parallelQEContinuum
{
public:
// typedef std::shared_ptr<parallelGrid<T>> std::shared_ptr<parallelGrid<T>>;

    struct SendEFieldRecvPFieldCont
    {
        int procCalcQE_;

        int nx_;
        int ny_;

        int ystart_;

        int tagSend_;
        int tagRecv_;

        std::array<int,2> loc_;
        std::array<int,2> locP_;

        std::shared_ptr<Grid<T>> e_;

        std::shared_ptr<Grid<T>> P_;

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & procCalcQE_;
            ar & nx_;
            ar & ny_;
            ar & ystart_;
            ar & tagSend_;
            ar & tagRecv_;
            ar & loc_;
        }
    };
protected:
    //Hamiltonian and rasing and lowering operators go here too
    std::shared_ptr<mpiInterface> gridComm_; //!< The communicator for the processes that are storing the grid

    int nlevel_; //!< number of levels in the qe's basis set
    int t_step_;

    double dt_; //!< the time step of the qe in atomic units
    double na_; //!< molecular density of the qe
    double hbar_; //!< value of hbar in FDTD units
    double tcur_; //!< the current time of the system

    std::vector<std::array<int,2>> locs_; //!< locations of all grid points inside the quantum emitter

    std::vector<std::shared_ptr<SendEFieldRecvPFieldCont>> realGridInfo_; //!< parameters for the actual grids to send out the E fields and receive the P fields
    std::shared_ptr<SendEFieldRecvPFieldCont> sameProcCalc_; //!< parameters to copy information from the actual grids to the QE grids
    std::vector<std::shared_ptr<RecvEFieldSendPFieldCont>> calcQE_; //!< parameters for QE processes to get the right field info and send out the P fields to the actual grids

    std::vector<mpi::request> reqs_; //!< vector storing all MPI requests for wait_all

    std::function< void (std::shared_ptr<Grid<T>>, real_grid_ptr, int, int, int, int, int) > getE_; //!< function to get the Ez field at the correct Pz field point
    std::function< void (std::shared_ptr<parallelGrid<T>>, std::shared_ptr<Grid<T>>, int, int, int, int, int&) > transferE_; //!< function to transfer the Ex fields from actual grids to the QE processes
    std::function< void (real_grid_ptr, real_grid_ptr, real_grid_ptr)> getE_05_; //!< function to get the Ex fields in between time steps
    std::function< void (std::vector<double>&, int&, int&, std::vector<double>&, std::shared_ptr<Grid<T>>, double, int, std::vector<cplx>&) > upP_; //!< function to update the Px_
    std::function< void (real_grid_ptr, real_grid_ptr) > upPrev_; //!< function to update the upPrevX_
    std::function< void (std::shared_ptr<Grid<T>>, std::shared_ptr<parallelGrid<T>>, double&, int&, int&, int&, int&, int&, int, int) > addP_; //!< function to add the Px to the Ex

    std::function<void(mpiInterface&, std::vector<mpi::request>& reqs, int index, std::shared_ptr<Grid<T>>, std::shared_ptr<SendEFieldRecvPFieldCont>)> recvP_; //!< function to receive the Px field
    std::function<void(mpiInterface&, std::vector<mpi::request>& reqs, int index, std::shared_ptr<Grid<T>>, std::shared_ptr<RecvEFieldSendPFieldCont>)> sendP_; //!< function to send the Px field
    std::function<void(mpiInterface&, std::vector<mpi::request>& reqs, int index, std::shared_ptr<Grid<T>>, std::shared_ptr<SendEFieldRecvPFieldCont>)> sendE_; //!< function to send the Ex field
    std::function<void(mpiInterface&, std::vector<mpi::request>& reqs, int index, std::shared_ptr<Grid<T>>, std::shared_ptr<RecvEFieldSendPFieldCont>)> recvE_; //!< function to receive the Ex field
    std::shared_ptr<Grid<T>> eCollect_; //!< Field used to collect the Ex field at the points that they are calculated in FDTD

    real_grid_ptr e_; //!< the Ex grid all at the actual grid points (part of the QE calculated on this process)
    real_grid_ptr e_05_; //!< the Ex in between the current and previous time step (part of the QE calculated on this process)
    real_grid_ptr preve_; //!< Ex of the previous time step (part of the QE calculated on this process)

    std::vector<Continuum> continuum_;
public:
    std::shared_ptr<parallelGrid<T>> E_; //!< pointer to the Ex grid
    std::shared_ptr<Grid<T>> P_; //!< x directional polarization time derivative

    /**
     * @brief Constructor
     * @details Constructor with gam defined
     *
     * @param gam Matrix that lists all relaxation coupling terms
     * @param locs vector of locations of all points contained within the qe
     * @param a unit length in FDTD units
     * @param I0 unit current in FDTD units
     * @param dx step size in x direction
     * @param dy step size in y direction
     * @param dt time step
     * @param na the molecular density of the qe
     *
     */
    parallelQEContinuum(std::shared_ptr<mpiInterface> gridComm, std::array<Grid<double>,3> weights, int nlevel, std::vector<double> gam, std::vector<double> gamP, std::vector<std::array<int,2>> locs, std::shared_ptr<parallelGrid<T>> E, double omgGap, double dOmg, double mu, double a, double I0, double dx, double dy, double dt, double na) :
        gridComm_(gridComm),
        nlevel_( nlevel),
        t_step_(0),
        dt_(dt),
        na_(na),
        hbar_( HBAR * EPS0 * pow(SPEED_OF_LIGHT, 3) / pow(a*I0,2.0) ),
        tcur_(0.0),
        E_(E)
    {
        generateContLists(locs, omgGap, dOmg, mu, gam, gamP);
        reqs_ = std::vector<mpi::request>((realGridInfo_.size() + calcQE_.size() ) , mpi::request() ) ;

        // Alternate Definition of hbar from the fine structure constant
        // hbar_ = 1.0/(4.0*M_PI)*137.03599907444*pow(ELEMENTARY_CHARGE*SPEED_OF_LIGHT/a/I0,2.0);
    }


    /**
     * @brief Generates the density lists for each process
     * @details Checks to see if the qe location is within the process, if it is then it is added to density_
     *
     * @param[in]  locs list of all the locations contained with the qe
     */
    void generateContLists(std::vector<std::array<int,2>> locs, double omgGap, double dOmg, double mu, std::vector<double> gam, std::vector<double> gamP)
    {
        // Split up the QE points as easily as possible across all grids
        int denSz = std::floor(locs.size() / gridComm_->size());
        if(gridComm_->rank() < locs.size() % gridComm_->size() )
            denSz++;

        // Copy the locations of the QE's this process is responsible for into a separate list
        std::vector<std::array<int,2>> localQEs(denSz, std::array<int,2>() );
        (gridComm_->rank() < locs.size() % gridComm_->size()) ? copy_n(locs.begin()+gridComm_->rank()*denSz, denSz, localQEs.begin() ) : copy_n(locs.begin()+gridComm_->rank()*denSz + locs.size() % gridComm_->size(), denSz, localQEs.begin() );

        // Construct and initialize each densiy
        for(auto& loc : localQEs)
        {
            continuum_.push_back(Continuum( loc[0],loc[1], nlevel_, dt_, omgGap, dOmg, mu, hbar_, gam, gamP) );
            continuum_.back().initializePopulation(na_);
        }

        int xmin = E_->x();;
        int ymin = E_->y();;
        int xmax = 0;
        int ymax = 0;
        double dx = E_->dx();
        double dy = E_->dy();
        // Look through all locations to find the max x/y values
        for(auto& loc : localQEs)
        {
            if(loc[0] < xmin)
                xmin = loc[0];
            if(loc[0] > xmax)
                xmax = loc[0];
            if(loc[1] < ymin)
                ymin = loc[1];
            if(loc[1] > ymax)
                ymax = loc[1];
        }
        // number of x and y points occupied by the fields
        int nx = xmax - xmin + 1;
        int ny = ymax - ymin + 1;

        // Set up all the E and P fields needed for QE updates
        e_     = std::make_shared<Grid<double>>(std::vector<int>( {{nx, ny}}), std::vector<double>( {{dx, dy}} ) ) ;

        e_05_  = std::make_shared<Grid<double>>(std::vector<int>( {{nx, ny}}), std::vector<double>( {{dx, dy}} ) ) ;

        preve_ = std::make_shared<Grid<double>>(std::vector<int>( {{nx, ny}}), std::vector<double>( {{dx, dy}} ) ) ;

        P_ = std::make_shared<Grid<T>>(std::vector<int>({{nx+2, ny+2}}), std::vector<double>({{dx, dy}} ) );

        // Actual grid fields may require extra data points so add one to nx
        nx = xmax - xmin + 2;
        ny = ymax - ymin + 2;

        eCollect_ = std::make_shared<Grid<T>>(std::vector<int>({{nx, ny}}), std::vector<double>({{dx, dy}} ) );

        // Density locatons in the QE grids need to start at 0
        for(auto& cont : continuum_)
        {
            cont.x() -= xmin;
            cont.y() -= ymin;
        }

        // Processors are divided by lammella so find the points where processors split in the QE list this processor is responsible for
        std::vector< std::vector<int> > procDivide_y;
        procDivide_y.push_back( std::vector<int>(1,ymin-1) );
        // Different fields used as a reference
        for(int yy = ymin; yy <= ymax; yy++)
        {
            // If the points are not on the processor add a new list for the new processor.
            if(E_->getLocsProc_no_boundaries(xmin, yy) == E_->getLocsProc_no_boundaries(xmin, procDivide_y.back().back()))
                procDivide_y.back().push_back(yy);
            else
                procDivide_y.push_back(std::vector<int>(1, yy) );
        }

        std::vector<RecvEFieldSendPFieldCont> calcQE_all;
        std::vector<SendEFieldRecvPFieldCont> realGrid_all;
        // For each list in y list make a new SendERecvP and RecvESendP for each processor par
        for(auto& yList : procDivide_y)
        {
            RecvEFieldSendPFieldCont  calcQE;
            SendEFieldRecvPFieldCont realGrid;
            // Real grid needs the actual coordinates, the calcQE portion needs the relative ones from the start of what that processor is responsible for
            realGrid.loc_ = {{ xmin-1,  yList.front() }};
            calcQE.loc_   = {{ 0, yList.front() - (ymin-1) }};

            // Find the processor that the fields are actually stored in
            calcQE.procMainGrid_ = E_->getLocsProc_no_boundaries( realGrid.loc_[0], realGrid.loc_[1] );

            //What is the QE calculator's processor
            realGrid.procCalcQE_ = gridComm_->rank();

            // nx is based on the number of points in the x direction, and y based on the number of y points in that list
            realGrid.nx_ = nx;
            realGrid.ny_ = yList.size();

            calcQE.nx_ = nx;
            calcQE.ny_ = yList.size();
            // sz is the product of the two sizes (total number of points in the grid)
            calcQE.sz_ = nx*yList.size();
            // szP is one point larger in both directions because of the offset in calculating P and E fields.
            calcQE.szP_ = (nx+1) * (yList.size() + 1);

            // Calculate all the tags
            calcQE.tagSend_ = gridComm_->cantorTagGen(gridComm_->rank(), calcQE.procMainGrid_, 3, 0);
            calcQE.tagRecv_ = gridComm_->cantorTagGen(calcQE.procMainGrid_, gridComm_->rank(), 3, 0);

            // The opposing tags are the opposite (recv->send send->recv)
            realGrid.tagRecv_ = calcQE.tagSend_;
            realGrid.tagSend_ = calcQE.tagRecv_;

            // Add it to the all vectors
            calcQE_all.push_back(calcQE);
            realGrid_all.push_back(realGrid);
        }


        std::vector<std::vector<RecvEFieldSendPFieldCont>> tempCalcQE(gridComm_->size());
        std::vector<std::vector<SendEFieldRecvPFieldCont>> tempRealGrid(gridComm_->size());
        // All processes get all information in order to construct the correct items
        mpi::all_gather(*gridComm_, calcQE_all, tempCalcQE);
        mpi::all_gather(*gridComm_, realGrid_all, tempRealGrid);

        // Loop through all processes lists and all items in that list
        for(int ii = 0; ii < tempCalcQE.size(); ii++)
        {
            for(int jj = 0; jj < tempCalcQE[ii].size(); jj++)
            {
                // If the current processes is the one with grid's electric field information then construct the grids to transfer data from the actual grids to the QE processor and add it to the list of objects that processor needs to send
                if(gridComm_->rank() == tempCalcQE[ii][jj].procMainGrid_)
                {
                    tempRealGrid[ii][jj].loc_[0] -= E_->procLoc(0)-1;
                    tempRealGrid[ii][jj].loc_[1] -= E_->procLoc(1)-1;

                    if(tempRealGrid[ii][jj].procCalcQE_ == tempCalcQE[ii][jj].procMainGrid_)
                    {
                        tempRealGrid[ii][jj].ystart_ = tempCalcQE[ii][jj].loc_[1];
                        tempRealGrid[ii][jj].locP_ = {{tempRealGrid[ii][jj].loc_[0]-1, tempRealGrid[ii][jj].loc_[1]-1 }};
                        sameProcCalc_ = std::make_shared<SendEFieldRecvPFieldCont>(tempRealGrid[ii][jj]);
                    }
                    else
                    {
                        tempRealGrid[ii][jj].ystart_ = 0;

                        tempRealGrid[ii][jj].e_     = std::make_shared<Grid<T>>( std::vector<int>({{tempRealGrid[ii][jj].nx_, tempRealGrid[ii][jj].ny_}}), std::vector<double>( {{ dx, dy}} ) );

                        tempRealGrid[ii][jj].locP_ = {{tempRealGrid[ii][jj].loc_[0]-1, tempRealGrid[ii][jj].loc_[1]-1 }};
                        tempRealGrid[ii][jj].P_   = std::make_shared<Grid<T>>( std::vector<int>({{tempRealGrid[ii][jj].nx_+1, tempRealGrid[ii][jj].ny_+1}}), std::vector<double>( {{ dx, dy}} ) );

                        realGridInfo_.push_back(std::make_shared<SendEFieldRecvPFieldCont>(tempRealGrid[ii][jj]));
                    }
                }
            }
        }
        // Add the CalcQE parameters to the list of transfers this process needs.
        for(int ii = 0; ii < calcQE_all.size(); ii++)
        {
            if(gridComm_->rank() != calcQE_all[ii].procMainGrid_)
                calcQE_.push_back(std::make_shared<RecvEFieldSendPFieldCont>(calcQE_all[ii]));
        }
        return;
    }
    /**
     * @brief Returns the density matrix of the kth item
     *
     * @param k index of Density in qeArr_
     * @return Density matrix at index k
     */
    Continuum& continuum(int k){return continuum_[k];}

    /**
     * @brief updates the density matrix at a single point forward in time at one point
     */
    void updateDensity()
    {
        // Update previous E fields First
        upPrev_(e_, preve_);

        int ii = 0;
        // Transfer send fields from actual grid to the transfer grids
        for(auto& toQE: realGridInfo_)
        {
            transferE_(E_, toQE->e_, toQE->loc_[0], toQE->loc_[1], toQE->nx_, toQE->ny_, toQE->ystart_);
            sendE_(gridComm_, reqs_, ii, toQE->e_, toQE);
            ii++;
        }
        // Transfer the grids in the same proc to the collect fields
        if(sameProcCalc_)
        {
            transferE_(E_, eCollect_, sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->nx_, sameProcCalc_->ny_, sameProcCalc_->ystart_);
        }
        // Recieve into the collect fields
        for(auto& calcQE : calcQE_)
        {
            recvE_(gridComm_, reqs_, ii, eCollect_, calcQE);

            ii++;
        }
        // Wait for all communications to complete
        mpi::wait_all(reqs_.data(), reqs_.data()+reqs_.size());

        // Do the electric field averaging to ensure all fields are at the grid point
        getE_(eCollect_, e_, 1, 1, eCollect_->x(), eCollect_->y(), 0);

        // Get the 1/2 time values
        getE_05_(e_05_, e_, preve_);
        for(auto& cont : continuum_)
        {
            // Update density using RK4 methods
            // std::cout << "here" << std::endl;
            cont.RK4(e_->point(cont.x(), cont.y() ), e_05_->point(cont.x(), cont.y() ), preve_->point(cont.x(), cont.y() ) );
            // std::cout << "not here" << std::endl;
            // Update polarization fields
            P_->point( cont.x()+1, cont.y()+1 ) = std::accumulate( cont.popAndP().begin()+2*nlevel_, cont.popAndP().end(), 0.0 ) ;

            // std::cout << nlevel_ <<'\t' << P_->point(cont.x()+1, cont.y()+1) << '\t' << std::accumulate( cont.popAndP().begin()+nlevel_, cont.popAndP().begin()+nlevel_*2, 0.0 ) << '\t' << cont.popAndP()[nlevel_+1]<< std::endl;
        }
    }
    /**
     * @brief Adds the polarization fields to the electric field and updates the density matrix and Polarization fields
     */
    void addQE()
    {
        int ii = 0;
        // Add Px directly from the P fields if the QE calc and actual grids are the same
        if(sameProcCalc_)
        {
            // std::cout << sameProcCalc_->ystart_ <<'\t' << sameProcCalc_->nx_ << '\t' << sameProcCalc_->ny_ << '\t' << sameProcCalc_->loc_[0] << '\t' << sameProcCalc_->loc_[1] << std::endl;
            addP_(P_, E_, dt_, sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->nx_, sameProcCalc_->ny_, sameProcCalc_->ystart_, 0, 0);

        }
        // Send polarization fields first
        for(auto& calcQE : calcQE_)
        {

            sendP_(gridComm_, reqs_, ii, P_, calcQE);
            ii++;
        }
        for(auto& toQE: realGridInfo_)
        {
            // Receive all P fields to the temporary girds
            recvP_(gridComm_, reqs_, ii, toQE->P_, toQE);

            // wait for all communication to end
            mpi::wait_all(reqs_.data()+ii, reqs_.data()+(ii+1));

            // Add the polarization fields
            addP_(toQE->P_, E_, dt_, toQE->loc_[0], toQE->loc_[1], toQE->nx_, toQE->ny_, toQE->ystart_, 0, 0);
            ii++;
        }
        // Update the density matrices
        updateDensity();
    }

    inline std::shared_ptr<parallelGrid<T>> P(){return P_;}

};
class parallelQEContinuumReal : public parallelQEContinuum<double>
{
public:
    parallelQEContinuumReal(std::shared_ptr<mpiInterface> gridComm, std::array<Grid<double>,3> weights, int nlevel, std::vector<double> gam, std::vector<double> gamP, std::vector<std::array<int,2>> locs, real_pgrid_ptr E, double omgGap, double dOmg, double mu, double a, double I0, double dx, double dy, double dt, double na) ;
    parallelQEContinuumReal(const parallelQEContinuumReal& o);
};

class parallelQEContinuumCplx : public parallelQEContinuum<cplx>
{
public:
    parallelQEContinuumCplx(std::shared_ptr<mpiInterface> gridComm, std::array<Grid<double>,3> weights, int nlevel, std::vector<double> gam, std::vector<double> gamP, std::vector<std::array<int,2>> locs, cplx_pgrid_ptr E, double omgGap, double dOmg, double mu, double a, double I0, double dx, double dy, double dt, double na) ;
    parallelQEContinuumCplx(const parallelQEContinuumCplx& o);
};


#endif
