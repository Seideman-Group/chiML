/** @file ML/parallelQE.hpp
 *  @brief Class that stores and updates quantum emitter's density matrix and Polarizations
 *
 *  Stores and transfers relevant electric field information across all processes to update
 *  quantum emitter density matrices, updates the density matrices at all points
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef ML_PRALLEL_QE
#define ML_PRALLEL_QE

#include <ML/Hamiltonian.hpp>
#include <ML/density.hpp>
#include <ML/QEPopDtc.hpp>
#include <unordered_map>
#include <UTIL/FDTD_up_eq.hpp>
#include <tuple>

// Relaxation parameters for input
struct relaxParams
{
    int n0_; //!< initial state
    int nf_; //!< final state
    double rate_; //!< rate of relaxation
    double dephasingRate_; //!< is it a dephasing term?
};

// Parameters to input E field information from the main grids to a processor
struct RecvEFieldSendPField
{
    int procMainGrid_; //!< Processor of the E/H field Grid
    int tagSend_; //!< tag for sending data
    int tagRecv_; //!< tag for reciving data

    int sz_; //!< number of grid points that E fields need to receive from the main grid
    int szP_; //!< number of grid points the P fields need to send back to the main grid

    std::array<int,3> loc_; //!< location of the lower left corner that is starting in the main grid
    std::array<int,3> n_vec_; //!< number of grid points in the all directions that this part of the QE covers

    template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
    {
        ar & procMainGrid_;
        ar & tagSend_;
        ar & tagRecv_;
        ar & szP_;
        ar & sz_;
        ar & loc_;
        ar & n_vec_;
    }
};

/**
 * @brief Quantum Emitter Class
 * @details A class that models the quantum emitters for ML
 */
template <typename T> class parallelQEBase
{
protected:
    typedef std::shared_ptr<Grid<T>> grid_ptr;
public:
    struct SendEFieldRecvPField
    {
        int procCalcQE_; //!< process rank of the process that is updating QE information
        int ystart_; //!< value of the y coordinate needed to start QE
        int tagSend_; //!< tag for sending E data
        int tagRecv_; //!< tag for receiving P data

        std::array<int,3> loc_; //!< location of the lower left corner that is starting in the main grid
        std::array<int,3> n_vec_; //!< number of grid points in the all directions that this part of the QE covers

        std::array<grid_ptr,3> e_; //!< pgrid_ptr to the local ex field
        std::array<grid_ptr,3> P_; //!< pgrid_ptr to the local Px field

        template<class Archive>
        void serialize(Archive & ar, const unsigned int version)
        {
            ar & procCalcQE_;
            ar & ystart_;
            ar & tagSend_;
            ar & tagRecv_;
            ar & loc_;
            ar & n_vec_;
        }
    };
    struct qeLevelSystem
    {
        std::shared_ptr<Hamiltonian> ham_; //!< Hamiltonian of the system
        std::vector<Density> den_; //!< Vector of all density matrices for the system
        qeLevelSystem(std::shared_ptr<Hamiltonian> ham, std::vector<Density> den) : ham_(ham), den_(den) {};
    };
protected:
    typedef std::shared_ptr<parallelGrid<T>> pgrid_ptr;
    typedef std::function< void (grid_ptr, cplx_grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int ,int ) > getEFxn;
    typedef std::function< void (pgrid_ptr, grid_ptr, int, int, int, int, int, int, int) > transferEFxn;
    typedef std::function< void (cplx*, int, int, int, cplx*, grid_ptr, double, int) > upPFxn;
    typedef std::function< void (grid_ptr, pgrid_ptr, real_pgrid_ptr, T*, int, int, int, int, int, int, int, int, int, int) > addPFxn;
    typedef std::function<void(std::shared_ptr<mpiInterface>, std::vector<mpi::request>& reqs, int index, grid_ptr, std::shared_ptr<SendEFieldRecvPField>)> sendErecvPFxn;
    typedef std::function<void(std::shared_ptr<mpiInterface>, std::vector<mpi::request>& reqs, int index, grid_ptr, std::shared_ptr<RecvEFieldSendPField>)> sendPrecvEFxn;
    typedef std::function<void(grid_ptr P, std::vector<T>& totVec)> accumPFxn;
    //Hamiltonian and rasing and lowering operators go here too
    std::shared_ptr<mpiInterface> gridComm_; //!< The communicator for the processes that are storing the grid

    bool pAccuulate_; //!< True if P fields should be accumulated
    char noTranspose_; //!< char for zgemm calls
    int outputPAccuProc_; //!< process rank of process outputting the Accumulated P fields
    int nlevel_; //!< number of levels in the qe's basis set
    int denSz_; //!< size of the density matrix nlevel_^2
    int t_step_; //!< current time step
    int zOff_; //!< 1 if 3D 0 if 2 or 1D

    double dt_; //!< the time step of the qe in atomic units
    double na_; //!< molecular density of the qe
    double hbar_; //!< value of hbar in FDTD units
    double tcur_; //!< the current time of the system

    cplx one_over_hbar_; //!< value of 1/hbar in FDTD units
    cplx neg_one_over_hbar_; //!< value of 1/hbar in FDTD units
    cplx ZERO_; //!< constants for updating qe's
    cplx ONE_; //!< constants for updating qe's

    std::vector<std::pair<std::vector<double>, double>> energyWeights_; //!< Vector containing
    std::vector<qeLevelSystem> levelSys_; //!< Vector of all Hamiltonians and densities associated with that Hamiltonian

    std::shared_ptr<Hamiltonian> ham_; //!< Hamiltonian shared_ptr

    std::vector<cplx> P_vec_; //!< vector storing the polarization Vector to add onto E field
    std::vector<cplx> den_predict_; //!< vector storing the predicted solution
    std::vector<cplx> denDeriv_predict_; //!< vector storing the predicted denDeriv

    std::vector<cplx> temp_data_; //!< vector for a denDeriv temp storage

    std::vector<std::array<int,3>> locs_; //!< locations of all grid points inside the quantum emitter
    std::vector<std::unordered_map<int,double>> gam_; //!< relaxation matrix

    std::vector<std::shared_ptr<SendEFieldRecvPField>> realGridInfo_; //!< parameters for the actual grids to send out the E fields and receive the P fields
    std::shared_ptr<SendEFieldRecvPField> sameProcCalc_; //!< parameters to copy information from the actual grids to the QE grids
    std::vector<std::shared_ptr<RecvEFieldSendPField>> calcQE_; //!< parameters for QE processes to get the right field info and send out the P fields to the actual grids

    std::vector<mpi::request> reqs_; //!< vector storing all MPI requests for wait_all

    std::string outputPfname_; //!< output filename for the polarization
    std::array<std::vector<T>,3> totP_; //!< total {Px, Py, Pz} field for outputting

    std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr_; //!< detector of the total population

    std::array<getEFxn,3> getE_; //!< function to get the {Ex, Ey, Ez} field at the correct polarization field point
    std::array<transferEFxn,3> transferE_; //!< function to transfer the {Ex, Ey, Ez} fields from actual grids to the QE processes
    std::array<upPFxn,3> upP_; //!< function to update the {Px, Py, Pz}
    std::array<addPFxn,3> addP_; //!< function to add the {Px, Py, Pz} to the {Ex, Ey, Ez}

    std::array<sendErecvPFxn, 3> recvP_; //!< function to receive the {Px, Py, Pz} field
    std::array<sendPrecvEFxn, 3> sendP_; //!< function to send the {Px, Py, Pz} field
    std::array<sendErecvPFxn, 3> sendE_; //!< function to send the {Ex, Ey, Ez} field
    std::array<sendPrecvEFxn, 3> recvE_; //!< function to receive the {Ex, Ey, Ez} field

    std::array<std::function<void(grid_ptr P)>,3> zeroP_; //!< zeros the {Px, Py, Pz} grid at every time step
    std::array<std::function<void(grid_ptr P, std::vector<T>& totVec)>,3> accumP_; //!< accumulates the {Px, Py, Pz} field into a vector

    std::array<grid_ptr,3> eCollect_; //!< Field used to collect the {Ex, Ey, Ez} field at the points that they are calculated in FDTD
    std::array<cplx_grid_ptr,3> e_; //!< the local {Ex, Ey, Ez} grid all at the points where the quantum emitters are located

    std::vector<T> scratch_; //!< used for scaling P by 1.0/eps
public:
    std::array<pgrid_ptr,3> E_; //!< pointer to the global {Ex, Ey, Ez} grid
    std::array<grid_ptr,3> P_; //!< The local {Px, Py, Pz} fields
    real_pgrid_ptr eps_; //!< stores the relative permittivity of the quantum dot material
    /**
     * @brief      Constructs the parallelQEBase using a defined relaxation matrix
     *
     * @param[in]  gridComm      The mpi communicator
     * @param[in]  hams          Hamiltonians of the system
     * @param[in]  eWeights      The weights of each Hamiltonian for the system
     * @param[in]  dtcPopArr     Array of population detectors
     * @param[in]  basis         Basis Set of the qe
     * @param[in]  gam           Matrix that lists all relaxation coupling terms
     * @param[in]  locs          vector of locations of all points contained within the qe
     * @param[in]  E             The E field of the FDTD grid
     * @param[in]  accumulateP   The accumulate p
     * @param[in]  outputPFname  The output p filename
     * @param[in]  totTime       The total time
     * @param[in]  a             unit length in FDTD units
     * @param[in]  I0            unit current in FDTD units
     * @param[in]  dt            time step
     * @param[in]  na            the molecular density of the qe
     */
    parallelQEBase(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<std::vector<double>> gam, std::vector<std::array<int,3>> locs, std::array<pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputPFname, int totTime, double a, double I0, double dt, double na) :
        gridComm_(gridComm),
        pAccuulate_(accumulateP),
        noTranspose_('N'),
        outputPAccuProc_(-1),
        nlevel_( basis.nbasis() ),
        denSz_( nlevel_*nlevel_ ) ,
        t_step_(0),
        zOff_((E[0] && E[2]) ? 1 : 0),
        dt_(dt),
        na_(na),
        hbar_( HBAR * EPS0 * pow(SPEED_OF_LIGHT, 3) / pow(a*I0,2.0) ),
        tcur_(-1.0*dt),
        one_over_hbar_(0.0,1.0/hbar_),
        neg_one_over_hbar_(0.0,-1.0/hbar_),
        ZERO_(0.0,0.0),
        ONE_(1.0,0.0),
        energyWeights_(eWeights),
        P_vec_(denSz_,0.0),
        den_predict_(denSz_, 0.0),
        denDeriv_predict_(denSz_, 0.0),
        temp_data_(denSz_, 0.0),
        outputPfname_(outputPFname),
        dtcPopArr_(dtcPopArr),
        scratch_( ( E[0] ? E[0]->x() : E[2]->x() ), 0.0),
        E_(E),
        eps_(eps)
    {
        if(hams.size() != energyWeights_.size() )
            throw std::logic_error("The size of the energy weights vector and Hamiltonian Vector are not the same, please fix this error.");
        for(int ii = 0; ii < hams.size(); ++ii)
        {
            for(int ee = 0; ee < std::get<0>(eWeights[ii]).size(); ++ee)
            {
                if(hams[ii].h0()[ee*nlevel_+ee] != std::get<0>(eWeights[ii])[ee] )
                {
                    std::cout << ee << '\t' << hams[ii].h0()[ee*nlevel_+ee] << '\t' << std::get<0>(eWeights[ii])[ee] << std::endl;
                    throw std::logic_error("The energy for the weight and the Hamiltonian at index " + std::to_string(ii) + " do not match, please fix this error.");
                }
            }
        }
        // Copy Gam
        for(int ii = 0; ii< gam.size(); ii++)
        {
            std::unordered_map<int,double> temp;
            for(int jj = 0; jj < gam[ii].size(); jj++)
            {
                if(gam[ii][jj] != 0.0)
                    temp[jj] = gam[ii][jj];
            }
            gam_.push_back(temp);
        }
        // Set up all the density matrix lists
        generateDenLists(locs, hams);
        // set up the request vectors
        reqs_ = std::vector<mpi::request>(3*(realGridInfo_.size() + calcQE_.size() ) , mpi::request() ) ;

        // Alternate Definition of hbar from the fine structure constant
        // hbar_ = 1.0/(4.0*M_PI)*137.03599907444*pow(ELEMENTARY_CHARGE*SPEED_OF_LIGHT/a/I0,2.0);
    }
    /**
     * @brief      Constructs parallelQE using user defined relaxation operators
     *
     * @param[in]  gridComm                The mpi communicator
     * @param[in]  hams                    Hamiltonians of the system
     * @param[in]  eWeights                The weights of each Hamiltonian for the system
     * @param[in]  dtcPopArr               Array of population detectors
     * @param[in]  basis                   Basis Set of the qe
     * @param[in]  relaxStateTransitions   vector describing all the relaxation processes in the QEs
     * @param[in]  locs                    vector of locations of all points contained within the qe
     * @param[in]  E                       The E field of the FDTD grid
     * @param[in]  accumulateP             The accumulate p
     * @param[in]  outputPFname            The output p filename
     * @param[in]  totTime                 The total time
     * @param[in]  a                       unit length in FDTD units
     * @param[in]  I0                      unit current in FDTD units
     * @param[in]  dt                      time step
     * @param[in]  na                      the molecular density of the qe
     */
    parallelQEBase(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<relaxParams> relaxStateTransitions, std::vector<std::array<int,3>> locs, std::array<pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputPFname, int totTime, double a, double I0, double dt, double na)  :
        gridComm_(gridComm),
        pAccuulate_(accumulateP),
        noTranspose_('N'),
        outputPAccuProc_(-1),
        nlevel_( basis.nbasis() ),
        denSz_( nlevel_*nlevel_ ) ,
        t_step_(0),
        zOff_((E[0] && E[2]) ? 1 : 0),
        dt_(dt),
        na_(na),
        hbar_( HBAR * EPS0 * pow(SPEED_OF_LIGHT, 3) / pow(a*I0,2.0) ),
        tcur_(-1.0*dt),
        one_over_hbar_(0.0,1.0/hbar_),
        neg_one_over_hbar_(0.0,-1.0/hbar_),
        ZERO_(0.0,0.0),
        ONE_(1.0,0.0),
        energyWeights_(eWeights),
        P_vec_(denSz_,0.0),
        den_predict_(denSz_, 0.0),
        denDeriv_predict_(denSz_, 0.0),
        temp_data_(denSz_, 0.0),
        outputPfname_(outputPFname),
        dtcPopArr_(dtcPopArr),
        scratch_( ( E[0] ? E[0]->x() : E[2]->x() ), 0.0),
        E_(E),
        eps_(eps)
    {
        if(hams.size() != energyWeights_.size() )
            throw std::logic_error("The size of the energy weights vector and Hamiltonian Vector are not the same, please fix this error.");
        for(int ii = 0; ii < hams.size(); ++ii)
        {
            for(int ee = 0; ee < std::get<0>(eWeights[ii]).size(); ++ee)
            {
                if(hams[ii].h0()[ee*nlevel_+ee] != std::get<0>(eWeights[ii])[ee] )
                {
                    std::cout << ee << '\t' << hams[ii].h0()[ee*nlevel_+ee] << '\t' << std::get<0>(eWeights[ii])[ee] << std::endl;
                    throw std::logic_error("The energy for the weight and the Hamiltonian at index " + std::to_string(ii) + " do not match, please fix this error.");
                }
            }
        }
        std::vector<std::vector<double>> gam(nlevel_*nlevel_, std::vector<double>(nlevel_*nlevel_, 0.0));
        for(auto& relaxTransition : relaxStateTransitions)
        {
            // Initialze the A and density dummy vectors
            std::vector<double> A(nlevel_*nlevel_, 0.0);
            std::vector<double> den(nlevel_*nlevel_, 0.0);
            // Create rasing/lowering operator
            A[ relaxTransition.n0_ * nlevel_ + relaxTransition.nf_ ] = 1.0;
            // See the effects it has on the ground state
            den[0] = 1.0;
            // Add that term into the gam prtocols
            daxpy_(nlevel_*nlevel_, relaxTransition.rate_, linblad(A, den).data(), 1, gam[0].data(), 1);
            // Repeat for every term in the density matrix to build up the correct operator
            for(int ii = 1; ii  < den.size(); ii++)
            {
                den[ii] = 1.0;
                den[ii-1] = 0;
                daxpy_(nlevel_*nlevel_, relaxTransition.rate_, linblad(A, den).data(), 1, gam[ii].data(), 1);
            }
            // Do the same for the dephasing process
            if(relaxTransition.dephasingRate_ > 0.0)
            {
                A[ relaxTransition.n0_ * nlevel_ + relaxTransition.nf_ ] = 0.0;
                A[ relaxTransition.n0_ * nlevel_ + relaxTransition.n0_ ] = sqrt(2.0);
                // SEe the effects it has on the ground state
                den[den.size()-1] = 0.0;
                den[0] = 1.0;
                // Add that term into the gam prtocols
                daxpy_(nlevel_*nlevel_, relaxTransition.dephasingRate_, linblad(A, den).data(), 1, gam[0].data(), 1);
                // Repeat for every term in the density matrix to build up the correct operator
                for(int ii = 1; ii  < den.size(); ii++)
                {
                    den[ii] = 1.0;
                    den[ii-1] = 0;
                    daxpy_(nlevel_*nlevel_, relaxTransition.dephasingRate_, linblad(A, den).data(), 1, gam[ii].data(), 1);
                }
            }
        }
        for(int ii = 0; ii< gam.size(); ii++)
        {
            std::unordered_map<int,double> temp;
            for(int jj = 0; jj < gam[ii].size(); jj++)
                if(gam[ii][jj] != 0.0)
                    temp[jj] = gam[ii][jj];
            gam_.push_back(temp);
        }
        generateDenLists(locs, hams);
        gridComm_->barrier();
        // Set reqs to be the right size
        reqs_ = std::vector<mpi::request>(3*(realGridInfo_.size() + calcQE_.size() ) , mpi::request() ) ;
    }


    /**
     * @brief      Calculates the lindblad operators $\Gamma = A\phi\conj{A} - \frac{1}{2} A A \phi + \phi A A
     *
     * @param[in]  A     Operator describing the ladder operator used to make the Lind bland operator
     * @param[in]  den   The density matrix
     *
     * @return     the Lindblad operator
     */
    std::vector<double> linblad(std::vector<double> A, std::vector<double> den)
    {
        std::vector<double> toReturn_(nlevel_*nlevel_, 0.0);
        std::vector<double> temp(nlevel_*nlevel_, 0.0);
        dgemm_('C', 'N', nlevel_, nlevel_, nlevel_,  1.0 ,    A.data(), nlevel_, den.data(), nlevel_, 0.0,      temp.data(), nlevel_);
        dgemm_('N', 'N', nlevel_, nlevel_, nlevel_,  1.0 , temp.data(), nlevel_,   A.data(), nlevel_, 0.0, toReturn_.data(), nlevel_);

        dgemm_('C', 'N', nlevel_, nlevel_, nlevel_,  1.0 ,    A.data(), nlevel_,   A.data(), nlevel_, 0.0,      temp.data(), nlevel_);
        dgemm_('N', 'N', nlevel_, nlevel_, nlevel_, -0.5 , temp.data(), nlevel_, den.data(), nlevel_, 1.0, toReturn_.data(), nlevel_);

        dgemm_('N', 'C', nlevel_, nlevel_, nlevel_,  1.0 ,  den.data(), nlevel_,   A.data(), nlevel_, 0.0,      temp.data(), nlevel_);
        dgemm_('N', 'N', nlevel_, nlevel_, nlevel_, -0.5 , temp.data(), nlevel_,   A.data(), nlevel_, 1.0, toReturn_.data(), nlevel_);

        return toReturn_;
    }

    /**
     * @brief Generates the density lists for each process
     * @details Checks to see if the qe location is within the process, if it is then it is added to density_
     *
     * @param[in]  locs list of all the locations contained with the qe
     * @param[in]  hams vector storing all Hamiltonians for the system
     */
    void generateDenLists(std::vector<std::array<int,3>> locs, std::vector<Hamiltonian> hams)
    {
        // Start the divisions at the process with the first loc in it
        int loc_rank = -1;
        if(E_[0] && E_[2])
        {
            outputPAccuProc_ = E_[0]->getLocsProc_no_boundaries(locs[0][0], locs[0][1], locs[0][2]);
            loc_rank = (gridComm_->rank() - E_[0]->getLocsProc_no_boundaries(locs[0][0], locs[0][1], locs[0][2]) + gridComm_->size() ) % gridComm_->size();
        }
        else if(E_[0])
        {
            outputPAccuProc_ = E_[0]->getLocsProc_no_boundaries(locs[0][0], locs[0][1]);
            loc_rank = (gridComm_->rank() - E_[0]->getLocsProc_no_boundaries(locs[0][0], locs[0][1]) + gridComm_->size() ) % gridComm_->size();
        }
        else
        {
            outputPAccuProc_ = E_[2]->getLocsProc_no_boundaries(locs[0][0], locs[0][1]);
            loc_rank = (gridComm_->rank() - E_[2]->getLocsProc_no_boundaries(locs[0][0], locs[0][1]) + gridComm_->size() ) % gridComm_->size();
        }

        // Split up the QE points as easily as possible across all grids
        int denSz = std::floor(locs.size() / gridComm_->size());
        if(loc_rank < locs.size() % gridComm_->size() )
            denSz++;
        std::vector<std::array<int,3>> localQEs(denSz );
        // Copy the locations of the QE's this process is responsible for into a separate list
        (loc_rank < locs.size() % gridComm_->size()) ? std::copy_n(locs.begin()+loc_rank*denSz, denSz, localQEs.begin() ) : std::copy_n(locs.begin()+loc_rank*denSz + locs.size() % gridComm_->size(), denSz, localQEs.begin() );

        // Construct and initialize each densiy
        for(int ii = 0; ii < hams.size(); ++ii)
        {
            std::vector<Density> den;
            for(auto& loc : localQEs)
            {
                den.push_back(Density( std::array<int,3>( {{ loc[0],loc[1],loc[2] }} ), -2, nlevel_ ) );
                den.back().initializeDensity( std::get<1>(energyWeights_[ii] ) );
            }
            levelSys_.push_back( qeLevelSystem( std::make_shared<Hamiltonian>(hams[ii]), den) );
        }
        std::array<int, 3>  max = {{ 0, 0, 0 }};
        std::array<int, 3>  min = {{ -1, -1, -1 }};
        std::array<double, 3> d;
        // set dx and dy
        if(E_[0])
        {
            min = {{ E_[0]->x(), E_[0]->y(), E_[0]->z() }};
            d = E_[0]->d();
        }
        else
        {
            min = {{ E_[2]->x(), E_[2]->y(), E_[2]->z() }};
            d = E_[2]->d();
        }
        // Look through all locations to find the max x/y values
        for(auto& loc : localQEs)
        {
            for(int ii = 0; ii < 3; ++ii)
            {
                if(loc[ii] < min[ii])
                    min[ii] = loc[ii];
                if(loc[ii] > max[ii])
                    max[ii] = loc[ii];
            }
        }
        // number of x and y points occupied by the fields
        std::array<int,3> n_vec;
        for(int ii = 0; ii < 3; ++ii)
            n_vec[ii] = max[ii] - min[ii] +1;

        if( (E_[0] && E_[0]->z() == 1) || (E_[2] && E_[2]->z() == 1) )
            n_vec[2] = 1;

        // Set up all the E and P fields needed for QE updates
        for(int ii = 0; ii < 3; ++ii)
        {
            e_[ii] = std::make_shared<Grid<cplx>>(n_vec, d );
            P_[ii] = std::make_shared<Grid<T>>( std::array<int,3>( {{n_vec[0]+2, n_vec[1]+2, n_vec[2]+1+zOff_ }} ), d );
        }
        // Actual grid fields may require extra data points so add one to n_vec[0]
        for(int ii = 0; ii < 3; ++ii)
            n_vec[ii] += 1;
        if( (E_[0] && E_[0]->z() == 1) || (E_[2] && E_[2]->z() == 1) )
            n_vec[2] = 1;
        for(int ii = 0; ii < 3; ++ii)
            eCollect_[ii] = std::make_shared<Grid<T>>( n_vec, d );

        // Density locatons in the QE grids need to start at 0
        for(auto& hamDenPair : levelSys_)
        {
            for(auto& den : hamDenPair.den_ )
            {
                den.x() -= min[0];
                den.y() -= min[1];
                den.z() -= min[2];
                den.ind() = e_[0]->getInd(den.x(), den.y(), den.z());
            }
        }

        // Processors are divided by lammella so find the points where processors split in the QE list this processor is responsible for
        std::vector< std::vector<int> > procDivide_y;
        procDivide_y.push_back( std::vector<int>(1,min[1]-1) );
        // Different fields used as a reference
        if(E_[0])
        {
            for(int yy = min[1]; yy <= max[1]; yy++)
            {
                // If the points are not on the processor add a new list for the new processor.
                if(E_[0]->getLocsProc_no_boundaries(min[0], yy, min[2]) == E_[0]->getLocsProc_no_boundaries(min[0], procDivide_y.back().back(), min[2]))
                    procDivide_y.back().push_back(yy);
                else
                    procDivide_y.push_back(std::vector<int>(1, yy) );
            }
        }
        else
        {
            for(int yy = min[1]; yy <= max[1]; yy++)
            {
                if(E_[2]->getLocsProc_no_boundaries(min[0], yy, min[2]) == E_[2]->getLocsProc_no_boundaries(min[0], procDivide_y.back().back(), min[2]))
                    procDivide_y.back().push_back(yy);
                else
                    procDivide_y.push_back(std::vector<int>(1, yy) );
            }
        }

        std::vector<RecvEFieldSendPField> calcQE_all;
        std::vector<SendEFieldRecvPField> realGrid_all;
        // For each list in y list make a new SendERecvP and RecvESendP for each processor par
        for(auto& yList : procDivide_y)
        {
            RecvEFieldSendPField  calcQE;
            SendEFieldRecvPField realGrid;
            // Real grid needs the actual coordinates, the calcQE portion needs the relative ones from the start of what that processor is responsible for
            realGrid.loc_ = {{ min[0]-1,  yList.front(), min[2]-1 }};
            calcQE.loc_   = {{ 0, yList.front() - (min[1]-1), 0 }};

            // Find the processor that the fields are actually stored in
            if(E_[0])
                calcQE.procMainGrid_ = E_[0]->getLocsProc_no_boundaries( realGrid.loc_[0], realGrid.loc_[1], realGrid.loc_[2] );
            else
                calcQE.procMainGrid_ = E_[2]->getLocsProc_no_boundaries( realGrid.loc_[0], realGrid.loc_[1], realGrid.loc_[2] );

            //What is the QE calculator's processor
            realGrid.procCalcQE_ = gridComm_->rank();

            // n_vec[0] is based on the number of points in the x direction, and y based on the number of y points in that list
            realGrid.n_vec_ = {{ n_vec[0], static_cast<int>(yList.size() ), n_vec[2] }};

            calcQE.n_vec_ = {{ n_vec[0], static_cast<int>(yList.size() ), n_vec[2] }};
            // sz is the product of the two sizes (total number of points in the grid)
            calcQE.sz_ = n_vec[0]*yList.size()*n_vec[2];
            // szP is one point larger in both directions because of the offset in calculating P and E fields.
            calcQE.szP_ = (n_vec[0]+1) * (yList.size() + 1) * (n_vec[2]+1);

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

        std::vector<std::vector<RecvEFieldSendPField>> tempCalcQE(gridComm_->size());
        std::vector<std::vector<SendEFieldRecvPField>> tempRealGrid(gridComm_->size());
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
                    if(E_[0] )
                    {
                        for(int kk = 0; kk < 3; ++kk)
                            tempRealGrid[ii][jj].loc_[kk] -= E_[0]->procLoc(kk)-1;
                    }
                    else
                    {
                        for(int kk = 0; kk < 3; ++kk)
                            tempRealGrid[ii][jj].loc_[kk] -= E_[2]->procLoc(kk)-1;
                    }
                    if(tempRealGrid[ii][jj].procCalcQE_ == tempCalcQE[ii][jj].procMainGrid_)
                    {
                        tempRealGrid[ii][jj].ystart_ = tempCalcQE[ii][jj].loc_[1];
                        sameProcCalc_ = std::make_shared<SendEFieldRecvPField>(tempRealGrid[ii][jj]);
                    }
                    else
                    {
                        tempRealGrid[ii][jj].ystart_ = 0;
                        for(int kk = 0; kk < 3; ++kk)
                        {
                            tempRealGrid[ii][jj].e_[kk] = std::make_shared<Grid<T>>( tempRealGrid[ii][jj].n_vec_, d );
                            tempRealGrid[ii][jj].P_[kk] = std::make_shared<Grid<T>>( std::array<int,3>( {{tempRealGrid[ii][jj].n_vec_[0]+1, tempRealGrid[ii][jj].n_vec_[1]+1, tempRealGrid[ii][jj].n_vec_[2]+1}} ), d );
                        }
                        realGridInfo_.push_back(std::make_shared<SendEFieldRecvPField>(tempRealGrid[ii][jj]));
                    }
                }
            }
        }
        // Add the CalcQE parameters to the list of transfers this process needs.
        for(int ii = 0; ii < calcQE_all.size(); ii++)
        {
            if(gridComm_->rank() != calcQE_all[ii].procMainGrid_)
                calcQE_.push_back(std::make_shared<RecvEFieldSendPField>(calcQE_all[ii]));
        }
        return;
    }

    /**
     * @brief updates the density matrix at a single point forward in time at one point
     */
    void updateDensity()
    {
        int ii = 0;
        // Transfer send fields from actual grid to the transfer grids
        for(auto& toQE: realGridInfo_)
        {
            transferE_[0](E_[0], toQE->e_[0], toQE->loc_[0], toQE->loc_[1], toQE->loc_[2], toQE->n_vec_[0], toQE->n_vec_[1], toQE->n_vec_[2], toQE->ystart_);
            transferE_[1](E_[1], toQE->e_[1], toQE->loc_[0], toQE->loc_[1], toQE->loc_[2], toQE->n_vec_[0], toQE->n_vec_[1], toQE->n_vec_[2], toQE->ystart_);
            transferE_[2](E_[2], toQE->e_[2], toQE->loc_[0], toQE->loc_[1], toQE->loc_[2], toQE->n_vec_[0], toQE->n_vec_[1], toQE->n_vec_[2], toQE->ystart_);

            sendE_[0](gridComm_, reqs_, ii*3  , toQE->e_[0], toQE);
            sendE_[1](gridComm_, reqs_, ii*3+1, toQE->e_[1], toQE);
            sendE_[2](gridComm_, reqs_, ii*3+2, toQE->e_[2], toQE);
            ii++;
        }
        // Transfer the grids in the same proc to the collect fields
        if(sameProcCalc_)
        {
            transferE_[0](E_[0], eCollect_[0], sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->loc_[2], sameProcCalc_->n_vec_[0], sameProcCalc_->n_vec_[1], sameProcCalc_->n_vec_[2], sameProcCalc_->ystart_);
            transferE_[1](E_[1], eCollect_[1], sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->loc_[2], sameProcCalc_->n_vec_[0], sameProcCalc_->n_vec_[1], sameProcCalc_->n_vec_[2], sameProcCalc_->ystart_);
            transferE_[2](E_[2], eCollect_[2], sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->loc_[2], sameProcCalc_->n_vec_[0], sameProcCalc_->n_vec_[1], sameProcCalc_->n_vec_[2], sameProcCalc_->ystart_);
        }
        // Recieve into the collect fields
        for(auto& calcQE : calcQE_)
        {
            recvE_[0](gridComm_, reqs_, ii*3  , eCollect_[0], calcQE);
            recvE_[1](gridComm_, reqs_, ii*3+1, eCollect_[1], calcQE);
            recvE_[2](gridComm_, reqs_, ii*3+2, eCollect_[2], calcQE);
            ii++;
        }
        // Wait for all communications to complete
        mpi::wait_all(reqs_.data(), reqs_.data()+reqs_.size());

        // Do the electric field averaging to ensure all fields are at the grid point
        getE_[0](eCollect_[0], e_[0], P_[0], 1, 1, zOff_, eCollect_[0]->x(), eCollect_[0]->y(), eCollect_[0]->z()+1-zOff_, 0, -1,  0,  0);
        getE_[1](eCollect_[1], e_[1], P_[1], 1, 1, zOff_, eCollect_[1]->x(), eCollect_[1]->y(), eCollect_[1]->z()+1-zOff_, 0,  0, -1,  0);
        getE_[2](eCollect_[2], e_[2], P_[2], 1, 1, zOff_, eCollect_[2]->x(), eCollect_[2]->y(), eCollect_[2]->z()+1-zOff_, 0,  0,  0, -1*zOff_);

        // Zero out P fields for QE and Rad Pols of all qe systems
        zeroP_[0](P_[0]);
        zeroP_[1](P_[1]);
        zeroP_[2](P_[2]);
        for(int qq = 0; qq < levelSys_.size(); ++qq)
        {
            for(auto& den : levelSys_[qq].den_)
            {
                ham_ = levelSys_[qq].ham_;
                // Update density using a 4th order Predictor corrector method
                PCABAM4(den);

                for(auto& dtc: dtcPopArr_)
                    dtc->inPop(den.density());

                // Update polarization fields
                upP_[0](den.density(), den.x(), den.y(), den.z()-1+zOff_, ham_->x_expectation(), P_[0], na_, den.density_vec().size());
                upP_[1](den.density(), den.x(), den.y(), den.z()-1+zOff_, ham_->y_expectation(), P_[1], na_, den.density_vec().size());
                upP_[2](den.density(), den.x(), den.y(), den.z()-1+zOff_, ham_->z_expectation(), P_[2], na_, den.density_vec().size());
            }
        }
        accumP_[0](P_[0], totP_[0]);
        accumP_[1](P_[1], totP_[1]);
        accumP_[2](P_[2], totP_[2]);
        for(auto& dtc: dtcPopArr_)
            dtc->accumPop();
    }
    /**
     * @brief Adds the polarization fields to the electric field and updates the density matrix and Polarization fields
     */
    void addQE()
    {
        tcur_+=dt_;
        int ii = 0;
        // Add Px directly from the P fields if the QE calc and actual grids are the same
        if(sameProcCalc_)
        {
            addP_[0](P_[0], E_[0], eps_, scratch_.data(), sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->loc_[2], sameProcCalc_->n_vec_[0], sameProcCalc_->n_vec_[1], sameProcCalc_->n_vec_[2], sameProcCalc_->ystart_, 1, 0, 0);
            addP_[1](P_[1], E_[1], eps_, scratch_.data(), sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->loc_[2], sameProcCalc_->n_vec_[0], sameProcCalc_->n_vec_[1], sameProcCalc_->n_vec_[2], sameProcCalc_->ystart_, 0, 1, 0);
            addP_[2](P_[2], E_[2], eps_, scratch_.data(), sameProcCalc_->loc_[0], sameProcCalc_->loc_[1], sameProcCalc_->loc_[2], sameProcCalc_->n_vec_[0], sameProcCalc_->n_vec_[1], sameProcCalc_->n_vec_[2], sameProcCalc_->ystart_, 0, 0, zOff_);
        }
        // Send polarization fields first
        for(auto& calcQE : calcQE_)
        {
            sendP_[0](gridComm_, reqs_, ii*3  , P_[0], calcQE);
            sendP_[1](gridComm_, reqs_, ii*3+1, P_[1], calcQE);
            sendP_[2](gridComm_, reqs_, ii*3+2, P_[2], calcQE);
            ii++;
        }
        for(auto& toQE: realGridInfo_)
        {
            // Receive all P fields to the temporary girds
            recvP_[0](gridComm_, reqs_, ii*3  , toQE->P_[0], toQE);
            recvP_[1](gridComm_, reqs_, ii*3+1, toQE->P_[1], toQE);
            recvP_[2](gridComm_, reqs_, ii*3+2, toQE->P_[2], toQE);

            // wait for all communication to end
            mpi::wait_all(reqs_.data()+ii*3, reqs_.data()+(ii+1)*3);
            // Add the polarization fields
            addP_[0](toQE->P_[0], E_[0], eps_, scratch_.data(), toQE->loc_[0], toQE->loc_[1], toQE->loc_[2], toQE->n_vec_[0], toQE->n_vec_[1], toQE->n_vec_[2], toQE->ystart_, 1, 0, 0);
            addP_[1](toQE->P_[1], E_[1], eps_, scratch_.data(), toQE->loc_[0], toQE->loc_[1], toQE->loc_[2], toQE->n_vec_[0], toQE->n_vec_[1], toQE->n_vec_[2], toQE->ystart_, 0, 1, 0);
            addP_[2](toQE->P_[2], E_[2], eps_, scratch_.data(), toQE->loc_[0], toQE->loc_[1], toQE->loc_[2], toQE->n_vec_[0], toQE->n_vec_[1], toQE->n_vec_[2], toQE->ystart_, 0, 0, zOff_);
            ii++;
        }
        // Update the density matrices
        updateDensity();
    }

    /**
     * @brief      Calculates the time derivative of the density matrix
     *
     * @param      H         Pointer to the start of the Hamiltonian storage vector
     * @param      den       Pointer to the start of the density matrix
     * @param      outDeriv  Pointer to the start of the output density derivative vector
     */
    void denDeriv(cplx* H, cplx* den, cplx* outDeriv)
    {
        // Commutator is reversed because zgemm uses column major and not row major multiplication so the order of multiplication is reversed
        #ifdef MKL
            zgemm_(noTranspose_, noTranspose_, nlevel_, nlevel_, nlevel_,     one_over_hbar_,   H, nlevel_, den, nlevel_, ZERO_, temp_data_.data(), nlevel_);
            // MKL has a helper function that allows you to take a conjugate transpose and add it onto another matrix and output into a third, -i/hbar rho H = Conjugate Transpose[i/hbar H rho]
            mkl_zomatadd_('R', 'N', 'C', nlevel_, nlevel_, ONE_, temp_data_.data(), nlevel_, ONE_, temp_data_.data(), nlevel_, outDeriv, nlevel_);
        #else
            zgemm_(noTranspose_, noTranspose_, nlevel_, nlevel_, nlevel_,     one_over_hbar_,   H, nlevel_, den, nlevel_, ZERO_, outDeriv, nlevel_);
            zgemm_(noTranspose_, noTranspose_, nlevel_, nlevel_, nlevel_, neg_one_over_hbar_, den, nlevel_,   H, nlevel_,  ONE_, outDeriv, nlevel_);
        #endif

        // do relaxation state by state the first is the state which is being relaxed  to/from it->second is the rate of that relaxation
        for(int ii = 0; ii < gam_.size(); ii++)
            for(auto it = gam_[ii].begin(); it != gam_[ii].end(); ++it)
                outDeriv[ii] += *(den+it->first) * it->second;
        return;
    }

     /**
     * @brief Predictor-Corrector model based on Adams–Bashforth predictor, and Adams–Moulton corrector
      *
     * @param[in] den density object
     */
    void PCABAM4(Density& den)
    {
        // use the denDeriv calculation for the P updates at the previous time step
        zcopy_(denSz_, den.density(), 1, den_predict_.data(), 1);
        zaxpy_(denSz_,  55.0*dt_/24.0, den.density_deriv_n_        .data(), 1, den_predict_.data(), 1);
        zaxpy_(denSz_, -59.0*dt_/24.0, den.density_deriv_n_minus_1_.data(), 1, den_predict_.data(), 1);
        zaxpy_(denSz_,  37.0*dt_/24.0, den.density_deriv_n_minus_2_.data(), 1, den_predict_.data(), 1);
        zaxpy_(denSz_,  -9.0*dt_/24.0, den.density_deriv_n_minus_3_.data(), 1, den_predict_.data(), 1);

        denDeriv(ham_->getHam(e_[0]->point(den.ind()), e_[1]->point(den.ind()), e_[2]->point(den.ind()) ), den_predict_.data(), denDeriv_predict_.data() );
        zaxpy_(denSz_,  9.0*dt_/24.0, denDeriv_predict_           .data(), 1, den.density(), 1);
        zaxpy_(denSz_, 19.0*dt_/24.0, den.density_deriv_n_        .data(), 1, den.density(), 1);
        zaxpy_(denSz_, -5.0*dt_/24.0, den.density_deriv_n_minus_1_.data(), 1, den.density(), 1);
        zaxpy_(denSz_,      dt_/24.0, den.density_deriv_n_minus_2_.data(), 1, den.density(), 1);

        // Move the time steps over to the future positions so it can add the denDeriv for the p update to density_deriv_n_
        den.moveDensity();
        // Calculate the density derivative for the next time step
        denDeriv( ham_->getHam( e_[0]->point(den.ind() ), e_[1]->point( den.ind() ), e_[2]->point( den.ind() ) ), den.density(), den.density_deriv_n_.data()  );
    }

    /**
     * @brief      outputs the current polarization fields
     */
    virtual void outputPol() = 0;

    /**
     * @brief      Accessor function to the Px pointer
     *
     * @return     P_[0]
     */
    inline pgrid_ptr Px(){return P_[0];}

    /**
     * @brief      Accessor function to the Py pointer
     *
     * @return     P_[1]
     */
    inline pgrid_ptr Py(){return P_[1];}

    /**
     * @brief      Accessor function to the Pz pointer
     *
     * @return     P_[2]
     */
    inline pgrid_ptr Pz(){return P_[2];}

    /**
     * @brief      Accessor function to pAccuulate_ (true if accumulating P fields)
     *
     * @return     pAccuulate_
     */
    inline bool pAccuulate() {return pAccuulate_;}

    /**
     * @brief      Returns the vector storing all the population detectors
     *
     * @return     dtcPopArr_
     */
    inline std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr() { return dtcPopArr_; }
};
class parallelQEReal : public parallelQEBase<double>
{
public:
    /**
     * @brief      Constructs parallelQE using user defined relaxation operators
     *
     * @param[in]  gridComm                The mpi communicator
     * @param[in]  hams                    Hamiltonians of the system
     * @param[in]  eWeights                The weights of each Hamiltonian for the system
     * @param[in]  dtcPopArr               Array of population detectors
     * @param[in]  basis                   Basis Set of the qe
     * @param[in]  relaxStateTransitions   vector describing all the relaxation processes in the QEs
     * @param[in]  locs                    vector of locations of all points contained within the qe
     * @param[in]  E                       The E field of the FDTD grid
     * @param[in]  accumulateP             The accumulate p
     * @param[in]  outputPFname            The output p filename
     * @param[in]  totTime                 The total time
     * @param[in]  a                       unit length in FDTD units
     * @param[in]  I0                      unit current in FDTD units
     * @param[in]  dx                      step size in x direction
     * @param[in]  dy                      step size in y direction
     * @param[in]  dt                      time step
     * @param[in]  na                      the molecular density of the qe
     */
    parallelQEReal(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<relaxParams> relaxStateTransitions, std::vector<std::array<int,3>> locs, std::array<real_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na);

    /**
     * @brief      Constructs the parallelQEBase using a defined relaxation matrix
     *
     * @param[in]  gridComm      The mpi communicator
     * @param[in]  hams          Hamiltonians of the system
     * @param[in]  eWeights      The weights of each Hamiltonian for the system
     * @param[in]  dtcPopArr     Array of population detectors
     * @param[in]  basis         Basis Set of the qe
     * @param[in]  gam           Matrix that lists all relaxation coupling terms
     * @param[in]  locs          vector of locations of all points contained within the qe
     * @param[in]  E             The E field of the FDTD grid
     * @param[in]  accumulateP   The accumulate p
     * @param[in]  outputPFname  The output p filename
     * @param[in]  totTime       The total time
     * @param[in]  a             unit length in FDTD units
     * @param[in]  I0            unit current in FDTD units
     * @param[in]  dx            step size in x direction
     * @param[in]  dy            step size in y direction
     * @param[in]  dt            time step
     * @param[in]  na            the molecular density of the qe
     */
    parallelQEReal(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<std::vector<double>> gam, std::vector<std::array<int,3>> locs, std::array<real_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na);
    /**
     * @brief      outputs the current polarization fields
     */
    void outputPol();
};

class parallelQECplx : public parallelQEBase<cplx>
{
public:
    /**
     * @brief      Constructs parallelQE using user defined relaxation operators
     *
     * @param[in]  gridComm                The mpi communicator
     * @param[in]  hams                    Hamiltonians of the system
     * @param[in]  eWeights                The weights of each Hamiltonian for the system
     * @param[in]  dtcPopArr               Array of population detectors
     * @param[in]  basis                   Basis Set of the qe
     * @param[in]  relaxStateTransitions   vector describing all the relaxation processes in the QEs
     * @param[in]  locs                    vector of locations of all points contained within the qe
     * @param[in]  E                       The E field of the FDTD grid
     * @param[in]  accumulateP             The accumulate p
     * @param[in]  outputPFname            The output p filename
     * @param[in]  totTime                 The total time
     * @param[in]  a                       unit length in FDTD units
     * @param[in]  I0                      unit current in FDTD units
     * @param[in]  dx                      step size in x direction
     * @param[in]  dy                      step size in y direction
     * @param[in]  dt                      time step
     * @param[in]  na                      the molecular density of the qe
     */
    parallelQECplx(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<relaxParams> relaxStateTransitions, std::vector<std::array<int,3>> locs, std::array<cplx_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na);

    /**
     * @brief      Constructs the parallelQEBase using a defined relaxation matrix
     *
     * @param[in]  gridComm      The mpi communicator
     * @param[in]  hams          Hamiltonians of the system
     * @param[in]  eWeights      The weights of each Hamiltonian for the system
     * @param[in]  dtcPopArr     Array of population detectors
     * @param[in]  basis         Basis Set of the qe
     * @param[in]  gam           Matrix that lists all relaxation coupling terms
     * @param[in]  locs          vector of locations of all points contained within the qe
     * @param[in]  E             The E field of the FDTD grid
     * @param[in]  accumulateP   The accumulate p
     * @param[in]  outputPFname  The output p filename
     * @param[in]  totTime       The total time
     * @param[in]  a             unit length in FDTD units
     * @param[in]  I0            unit current in FDTD units
     * @param[in]  dx            step size in x direction
     * @param[in]  dy            step size in y direction
     * @param[in]  dt            time step
     * @param[in]  na            the molecular density of the qe
     */
    parallelQECplx(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<std::vector<double>> gam, std::vector<std::array<int,3>> locs, std::array<cplx_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na);
    /**
     * @brief      outputs the current polarization fields
     */
    void outputPol();
};



#endif
