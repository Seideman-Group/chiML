/** @file FDTD_MANAGER/parallelFDTDField.hpp
 *  @brief Manager that stores the FDTD grids and updates them in time
 *
 *  Class that stores the FDTD grids and all necessary components to update them forward
 *  in time.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef PARALLEL_FDTD_FDTDFIELD
#define PARALLEL_FDTD_FDTDFIELD

#include <INPUTS/parallelInputs.hpp>
#include <DTC/parallelDTC_TXT.hpp>
#include <DTC/parallelDTC_COUT.hpp>
#include <DTC/parallelDTC_BMP.hpp>
#include <DTC/parallelDTC_BIN.hpp>
#include <DTC/parallelDTC_FREQ.hpp>
#include <DTC/parallelFlux.hpp>
#include <SOURCE/parallelSourceNormal.hpp>
#include <SOURCE/parallelSourceOblique.hpp>
#include <SOURCE/parallelTFSF.hpp>
#include <ML/parallelQE.hpp>
#ifdef _OPENMP
   #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_max_threads() 1
#endif

/**
 * @brief The main FDTD propagator class
 * @details This class generates a propagator class that will update all electromagnetic fields
 */
template <typename T> class parallelFDTDFieldBase
{
protected:
    // PML and Grid pointers
    typedef std::shared_ptr<Grid<T>> grid_ptr;
    typedef std::shared_ptr<parallelGrid<T>> pgrid_ptr;
    typedef std::shared_ptr<parallelCPML<T>> pml_ptr;

    // Update Function typedefs
    typedef std::function< void( const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr )> upFxnU;
    typedef std::function< void( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, T*) > upFxnLorPM;
    typedef std::function< void( const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, T*)> upFxnChiLor;
    typedef std::function< void( const std::array<int,6>&, pgrid_ptr, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, T*, T*, T*, T*, T*)> upFxnLorOrDip;

    typedef std::function< void( const std::array<int,6>&, pgrid_ptr, pgrid_ptr, pgrid_ptr, pgrid_ptr, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<real_pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, T*, T*, T*, T*, T*)> upFxnChiLorOrDip;
    typedef std::function< void( const std::array<int,6>&, T, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&) > D2UFxn;
    typedef std::function< void( const std::array<int,6>&, T, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, T*, T*) > orDipD2UFxn;
    typedef std::function< void( const std::array<int,6>&, T, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, T*, T*) > chiOrDipD2UFxn;
    typedef std::function< void(pgrid_ptr, std::array<double, 3>&, int, int, int, int, int, int, int, double&, double&, double&)> BCFxn;

    std::shared_ptr<mpiInterface> gridComm_; //!< mpi communicator for the propagator

    bool dielectricMatInPML_; //!< True if dielectric (constant or dispersive) material are in the PML's
    bool magMatInPML_; //!< True if magnetic (constant or dispersive) material are in the PML's

    int res_; //!< number of grid points per unit length
    int t_step_; //!< the number of time steps that happened

    std::array<int,3> yEPBC_; //!< point to introduce the PBC in the y direction for the {Ex, Ey, Ez} fields
    std::array<int,3> yHPBC_; //!< point to introduce the PBC in the y direction for the {Ex, Ey, Ez} fields

    int zMinPBC_; //!< min value for z PBC
    int zMaxPBC_; //!< min value for z PBC

    int nSteps_; //!< Total number of time steps in the calculation

    double tcur_; //!< the current time of the simulation

    std::array<double,3> d_; //!< the step size in all direction

    std::array<int,3> pmlThickness_; //!< thickness of the PMLS in all directions
    std::array<int,3> n_vec_; //!< the number of grid points in each direction
    std::array<int,3> ln_vec_; //!< the number of grid points in each direction for this process only
    double dt_; //!< the time step  of the simulation

    std::vector<std::shared_ptr<Obj>> objArr_; //!< vector containing all objects in the cell

    std::vector<std::vector<int>> chiDipPInds_; //!< A vector storing indexes  of the chiral polarizations in the DipM vectors
    std::vector<std::vector<int>> chiDipMInds_; //!< A vector storing indexes  of the chiral polarizations in the DipM vectors

    std::array<double,3> r_; //!< the vector descrbing the current location of the grid point for PBC's
    std::array<double,3> k_point_; //!< k-point vector for periodicity

    std::vector<std::shared_ptr<parallelDetectorBase<T> > > dtcArr_; //!< the vector of detectors in the cell
    std::vector<std::shared_ptr<parallelSourceBase<T> > > srcArr_; //!< the vector of all sources in the cell

    std::array<upLists,3> upH_; //!< the list of parameters used to update the {Hx, Hy, Hz} fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upE_; //!< the list of parameters used to update the {Ex, Ey, Ez} fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upB_; //!< the list of parameters used to update the {Bx, By, Bz} fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upD_; //!< the list of parameters used to update the {Dx, Dy, Dz} fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upLorD_; //!< the list of parameters used to update the {Dx, Dy, Dz} fields and polarization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upLorB_; //!< the list of parameters used to update the {Bx, By, Bz} fields and magnetization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upChiD_; //!< the list of parameters used to update the {Dx, Dy, Dz} fields and polarization fields for chiral material containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upChiB_; //!< the list of parameters used to update the {Bx, By, Bz} fields and magnetization fields for chiral material containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upOrDipD_; //!< the list of parameters used to update the {Dx, Dy, Dz} fields and polarization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upOrDipB_; //!< the list of parameters used to update the {Bx, By, Bz} fields and magnetization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upOrDipChiD_; //!< the list of parameters used to update the {Dx, Dy, Dz} fields and polarization fields for chiral material containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    std::array<upLists,3> upOrDipChiB_; //!< the list of parameters used to update the {Bx, By, Bz} fields and magnetization fields for chiral material containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    upLists upOrDipP_; //!< the list of parameters used to update the oriented dipole polarization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    upLists upOrDipM_; //!< the list of parameters used to update the oriented dipole magnetization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )

    upLists upChiOrDipP_; //!< the list of parameters used to update the chiral oriented dipole polarization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )
    upLists upChiOrDipM_; //!< the list of parameters used to update the chiral oriented dipole magnetization fields containing : std::pair(std::array<int,6>( number of elements for the operation, index for the point to be updated, index for the point with an offset in the direction of filed polarization(i),  index for the point offset in the j direction, index of the point offset in the k direction, object array index ), std::array<double,2> ( scaling factor (assumed to be 1 right now since conductivity = 0.0), spatial derivative prefactor for j spatial derivative) )

    std::vector<std::array<int,4>> copy2PrevFields_; //!< the lists of parameters to copy the U (E or H) fields to prevU (E or H) fields for the buffer region (averaging over corners)

    std::array<upFxnU,3> upHFxn_; //!< function that will update the {E/Dx, E/Dy, E/Dz} field
    std::array<upFxnU,3> upEFxn_; //!< function that will update the {H/Bx, H/By, H/Bz} field
    std::array<std::function<void(pml_ptr)>, 3> updateEPML_; //!< wrapper function to update {Ex, Ey, Ez} PMLs
    std::array<std::function<void(pml_ptr)>, 3> updateHPML_; //!< wrapper function to update {Hx, Hy, Hz} PMLs
    std::array<upFxnLorPM,3> upLorPFxn_; //!< function that will update the {Px, Py, Pz} fields
    std::array<upFxnLorPM,3> upLorMFxn_; //!< function that will update the {Mx, My, Mz} fields
    std::array<upFxnChiLor,3> upChiE_; //!< function that will update the chiral {E/Dx, E/Dy, E/Dz } fields
    std::array<upFxnChiLor,3> upChiH_; //!< function that will update the chiral {H/Bx, H/By, H/Bz } fields

    upFxnLorOrDip upLorOrDipP_; //!< function that will update the oriented dipole P fields
    upFxnLorOrDip upLorOrDipM_; //!< function that will update the oriented dipole M fields

    upFxnChiLorOrDip upChiLorOrDipP_; //!< function that will update the chiral oriented dipole P fields
    upFxnChiLorOrDip upChiLorOrDipM_; //!< function that will update the chiral oriented dipole M fields

    std::array<D2UFxn,3> D2EFxn_; //!< function that will convert the {Dx, Dy, Dz} field to the {Ex, Ey, Ez} fields using the {Px, Py, Pz} fields
    std::array<D2UFxn,3> B2HFxn_; //!< function that will convert the {Bx, By, Bz} field to the {Hx, Hy, Hz} fields using the {Mx, My, Mz} fields
    std::array<D2UFxn,3> chiD2EFxn_; //!< function that will add the chiral {Px, Py, Pz} field to the {Ex, Ey, Ez} fields
    std::array<D2UFxn,3> chiB2HFxn_; //!< function that will add the chiral {Mx, My, Mz} field to the {Hx, Hy, Hz} fields
    std::array<orDipD2UFxn,3> orDipD2EFxn_; //!< function that will convert the {Dx, Dy, Dz} field to the {Ex, Ey, Ez} fields using the oriented {Px, Py, Pz} fields
    std::array<orDipD2UFxn,3> orDipB2HFxn_; //!< function that will convert the {Bx, By, Bz} field to the {Hx, Hy, Hz} fields using the oriented {Mx, My, Mz} fields
    std::array<chiOrDipD2UFxn,3> chiOrDipD2EFxn_; //!< function that will add the oriented chiral {Px, Py, Pz} field to the {Ex, Ey, Ez} fields
    std::array<chiOrDipD2UFxn,3> chiOrDipB2HFxn_; //!< function that will add the oriented chiral {Mx, My, Mz} field to the {Hx, Hy, Hz} fields
    std::array<BCFxn,3> applBCE_; //!< function to apply PBC for the {Ex, Ey, Ez} field
    std::array<BCFxn,3> applBCH_; //!< function to apply PBC for the {Hx, Hy, Hz} field
    BCFxn applBCOrDip_; //!< function to apply PBC for the oriented dipole field

    std::vector<std::shared_ptr<parallelTFSFBase<T>>> tfsfArr_; //!< vector of all the TFSF objects

    std::vector<std::shared_ptr<parallelQEBase<T>>> qeArr_; //!< vector of all the QE objects

    std::array<std::vector<cplx>,3> E_incd_; //!< vector of all the incident {Ex, Ey, Ez} field values
    std::array<std::vector<cplx>,3> H_incd_; //!< vector of all the incident {Hx, Hy, Hz} field values
    std::vector<T> scratchx_; //!< vector for scratch operations
    std::vector<T> scratchy_; //!< vector for scratch operations
    std::vector<T> scratchz_; //!< vector for scratch operations
    std::vector<T> scratchOrDipUDeriv_; //!< vector for scratch operations
    std::vector<T> scratchDipDotU_; //!< vector for scratch operations

    std::array<int,6> axParams_; //!< Temp array to store all the axpy parameters for field updates
    std::array<double,3> prefactors_; //!< Temp array to store all the prefactor parameters for field updates

    std::vector<real_grid_ptr> weights_; //!< a map of the weights for each of the x y and z fields used to determine how to split up the grids for parallelization

    std::vector< std::shared_ptr< parallelDetectorFREQ_Base< T > > > dtcFreqArr_; //!< vector storing all dtcFREQ objects
    std::vector< std::shared_ptr< parallelFluxDTC< T > > > fluxArr_; //!< vector storing all flux objects

    std::array<int_pgrid_ptr,3> physE_; //!< Map of what objects are at each grid point for the {Ex, Ey, Ez} field.
    std::array<int_pgrid_ptr,3> physH_; //!< Map of what objects are at each grid point for the {Hx, Hy, Hz} field.

    std::array<real_pgrid_ptr,3> epsRel_; //!< Map of what objects are at each grid point for the {Ex, Ey, Ez} field.
    std::array<real_pgrid_ptr,3> muRel_; //!< Map of what objects are at each grid point for the {Hx, Hy, Hz} field.

    int_pgrid_ptr physPOrDip_; //!< Map of what objects are at each grid point for the oriented dipole polarization fields.
    int_pgrid_ptr physMOrDip_; //!< Map of what objects are at each grid point for the oriented dipole magnetization fields.

    real_pgrid_ptr epsRelOrDip_; //!< Map of what objects are at each grid point for the {Ex, Ey, Ez} field.
    real_pgrid_ptr muRelOrDip_; //!< Map of what objects are at each grid point for the {Hx, Hy, Hz} field.

public:

    std::array<pgrid_ptr,3> H_; //!< parallel grid corresponding to the {Hx, Hy, Hz} fields
    std::array<pgrid_ptr,3> E_; //!< parallel grid corresponding to the {Ex, Ey, Ez} fields
    std::array<pgrid_ptr,3> B_; //!< parallel grid corresponding to the {Bx, By, Bz} fields
    std::array<pgrid_ptr,3> D_; //!< parallel grid corresponding to the {Dx, Dy, Dz} fields
    std::array<pgrid_ptr,3> prevH_; //!< parallel grid corresponding to the {Hx, Hy, Hz} fields at the previous time step
    std::array<pgrid_ptr,3> prevE_; //!< parallel grid corresponding to the {Ex, Ey, Ez} fields at the previous time step

    std::array<std::vector<pgrid_ptr>,3> lorP_; //!< a vectors storing the grids for all the {Px, Py, Pz}  fields for the at the current time step
    std::array<std::vector<pgrid_ptr>,3> prevLorP_; //!< a vectors storing the grids for all the {Px, Py, Pz}  fields for the at the previous time step
    std::array<std::vector<pgrid_ptr>,3> lorM_; //!< a vectors storing the grids for all the {Mx, My, Mz}  fields for the at the current time step
    std::array<std::vector<pgrid_ptr>,3> prevLorM_; //!< a vectors storing the grids for all the {Mx, My, Mz}  fields for the at the previous time step
    std::array<std::vector<pgrid_ptr>,3> lorChiHP_; //!< a vectors storing the grids for all the chiral {Px, Py, Pz}  fields for the at the current time step
    std::array<std::vector<pgrid_ptr>,3> prevLorChiHP_; //!< a vectors storing the grids for all the chiral {Px, Py, Pz}  fields for the at the previous time step
    std::array<std::vector<pgrid_ptr>,3> lorChiEM_; //!< a vectors storing the grids for all the chiral {Mx, My, Mz}  fields for the at the current time step
    std::array<std::vector<pgrid_ptr>,3> prevLorChiEM_; //!< a vectors storing the grids for all the chiral {Mx, My, Mz}  fields for the at the previous time step

    std::vector<pgrid_ptr> prevDotProd_; //!< a vector of grids storing the dot product between the fields and dipole moments at the previous time step

    std::array<std::vector<pgrid_ptr>,3> orDipLorP_; //!< a vector of grids storing the oriented {Px, Py, Pz} fields at the current time step
    std::array<std::vector<pgrid_ptr>,3> orDipLorM_; //!< a vector of grids storing the oriented {Mx, My, Mz} fields at the current time step
    std::array<std::vector<pgrid_ptr>,3> prevOrDipLorP_; //!< a vector of grids storing the oriented {Px, Py, Pz} fields at the previous time step
    std::array<std::vector<pgrid_ptr>,3> prevOrDipLorM_; //!< a vector of grids storing the oriented {Mx, My, Mz} fields at the previous time step
    std::array<std::vector<pgrid_ptr>,3> chiOrDipLorP_; //!< a vector of grids storing the oriented chiral {Px, Py, Pz} fields at the current time step
    std::array<std::vector<pgrid_ptr>,3> chiOrDipLorM_; //!< a vector of grids storing the oriented chiral {Mx, My, Mz} fields at the current time step
    std::array<std::vector<pgrid_ptr>,3> prevChiOrDipLorP_; //!< a vector of grids storing the oriented chiral {Px, Py, Pz} fields at the previous time step
    std::array<std::vector<pgrid_ptr>,3> prevChiOrDipLorM_; //!< a vector of grids storing the oriented chiral {Mx, My, Mz} fields at the previous time step

    std::array<std::vector<real_pgrid_ptr>,3> dipP_; //!< a vector of grids describing the dipole moments for the {Px, Py, Pz} fields
    std::array<std::vector<real_pgrid_ptr>,3> dipM_; //!< a vector of grids describing the dipole moments for the {Mx, My, Mz} fields
    std::array<std::vector<real_pgrid_ptr>,3> dipChiP_; //!< a vector of grids describing the electric dipole moments at positions of the H fields
    std::array<std::vector<real_pgrid_ptr>,3> dipChiM_; //!< a vector of grids describing the magnetic dipole moments at positions of the E fields

    std::array<pml_ptr,3> EPML_; //!< PML for the {Ex, Ey, Ez} field
    std::array<pml_ptr,3> HPML_; //!< PML for the {Hx, Hy, Hz} field

    /**
     * @brief      Constructs a FDTD Propagator class
     *
     * @param[in]  IP        Input parameter object that read in values from a json input file
     * @param[in]  gridComm  A shared_ptr to the MPI interface for the calculation
     */
    parallelFDTDFieldBase(const parallelProgramInputs &IP, std::shared_ptr<mpiInterface> gridComm) :
        gridComm_(gridComm),
        dielectricMatInPML_(false),
        magMatInPML_(false),
        res_(IP.res_),
        t_step_(0),
        zMinPBC_(0),
        zMaxPBC_(1),
        nSteps_( int( std::ceil( IP.tMax_ / IP.dt_ ) ) ),
        d_(IP.d_),
        pmlThickness_(IP.pmlThickness_),
        n_vec_( toN_vec(IP.size_ ) ),
        dt_(IP.dt_),
        tcur_(0),
        scratchx_( 2*std::accumulate(n_vec_.begin(), n_vec_.end(), 0), 0.0),
        scratchy_( 2*std::accumulate(n_vec_.begin(), n_vec_.end(), 0), 0.0),
        scratchz_( 2*std::accumulate(n_vec_.begin(), n_vec_.end(), 0), 0.0),
        scratchOrDipUDeriv_(2*std::accumulate(n_vec_.begin(), n_vec_.end(), 0), 0.0),
        scratchDipDotU_(2*std::accumulate(n_vec_.begin(), n_vec_.end(), 0), 0.0),
        objArr_(IP.objArr_),
        chiDipPInds_(objArr_.size()),
        chiDipMInds_(objArr_.size()),
        k_point_(IP.k_point_),
        weights_()
    {
        for(auto& obj : objArr_)
            obj->setUpConsts(dt_);

        // Reserve memory for all object vectors
        dtcArr_.reserve( IP.dtcLoc_.size() );
        srcArr_.reserve( IP.srcLoc_.size() );
        tfsfArr_.reserve( IP.tfsfLoc_.size() );
        qeArr_.reserve( IP.qeLoc_.size() );
        dtcFreqArr_.reserve( IP.dtcLoc_.size() );
        fluxArr_.reserve( IP.fluxLoc_.size() );

        E_incd_[0] = std::vector<cplx>( 2*( nSteps_ + 1 ), 0.0 ) ;
        E_incd_[1] = std::vector<cplx>( 2*( nSteps_ + 1 ), 0.0 ) ;
        E_incd_[2] = std::vector<cplx>( 2*( nSteps_ + 1 ), 0.0 ) ;
        H_incd_[0] = std::vector<cplx>( 2*( nSteps_ + 1 ), 0.0 ) ;
        H_incd_[1] = std::vector<cplx>( 2*( nSteps_ + 1 ), 0.0 ) ;
        H_incd_[2] = std::vector<cplx>( 2*( nSteps_ + 1 ), 0.0 ) ;

        // Set up weights to scale where the process boundaries should be located (based on Instruction Calls for various objects)
        setupWeightsGrid(IP);

        // If only a 2D calculation set the number of points in the z direction to 1, otherwise like any other direction
        int nz = IP.size_[2] == 0 ? 1 : n_vec_[2]+2*gridComm_->npZ();

        // Construct and set up all the physical grids
        setupPhysFields(physPOrDip_, epsRelOrDip_, true , IP.periodic_, std::array<double,3>( {{ 0.0, 0.0, 0.0}} ), nz, std::array<int,3>( {{ 0,0,0 }} ) );
        setupPhysFields(physE_[0]  , epsRel_[0]  , true , IP.periodic_, std::array<double,3>( {{ 0.5, 0.0, 0.0}} ), nz, std::array<int,3>( {{ 1,0,0 }} ) );
        setupPhysFields(physE_[1]  , epsRel_[1]  , true , IP.periodic_, std::array<double,3>( {{ 0.0, 0.5, 0.0}} ), nz, std::array<int,3>( {{ 0,1,0 }} ) );
        setupPhysFields(physE_[2]  , epsRel_[2]  , true , IP.periodic_, std::array<double,3>( {{ 0.0, 0.0, 0.5}} ), nz, std::array<int,3>( {{ 0,0,1 }} ) );

        setupPhysFields(physMOrDip_, muRelOrDip_ , false, IP.periodic_, std::array<double,3>( {{ 0.5, 0.5, 0.5}} ), nz, std::array<int,3>( {{ 1,1,1 }} ) );
        setupPhysFields(physH_[0]  , muRel_[0]   , false, IP.periodic_, std::array<double,3>( {{ 0.0, 0.5, 0.5}} ), nz, std::array<int,3>( {{ 0,1,1 }} ) );
        setupPhysFields(physH_[1]  , muRel_[1]   , false, IP.periodic_, std::array<double,3>( {{ 0.5, 0.0, 0.5}} ), nz, std::array<int,3>( {{ 1,0,1 }} ) );
        setupPhysFields(physH_[2]  , muRel_[2]   , false, IP.periodic_, std::array<double,3>( {{ 0.5, 0.5, 0.0}} ), nz, std::array<int,3>( {{ 1,1,0 }} ) );

        // Determine if magnetic or electric dielectric material are in the PMLs
        for(int pp = 0; pp < pmlThickness_.size(); ++pp)
        {
            // Determine what the i, j, k values for the PML are i is in the direction normal to the PML and j and k are inside the plane (i.e. if Left/Right ii = 0 (x) jj = 1(y) kk = 2(z))
            int cor_ii = pp;
            int cor_jj = (pp + 1) % 3;
            int cor_kk = (pp + 2) % 3;
            std::array<int, 3> min(n_vec_);
            std::transform(min.begin(), min.end(), min.begin(), [](double mm){return -1*mm/2; } );
            std::array<int, 3> max(n_vec_);
            std::transform(max.begin(), max.end(), max.begin(), [](double mm){return mm/2; } );
            max[cor_ii] = pmlThickness_[cor_ii] - n_vec_[cor_ii]/2.0;
            for(auto& obj : objArr_)
            {
                if(obj->epsInfty() == 1.0 && obj->muInfty() == 1.0 && obj->tellegen() == 0.0 && obj->gamma().size() < 1 && obj->magGamma().size() < 1 && obj->chiGamma().size() < 1 && !obj->ML() )
                    continue;

                bool dielcMat =  (obj->epsInfty() == 1.0 && obj->tellegen() == 0.0 && obj->   gamma().size() < 1 && obj->chiGamma().size() < 1 ) ? false : true;
                bool magMat   =  (obj->muInfty()  == 1.0 && obj->tellegen() == 0.0 && obj->magGamma().size() < 1 && obj->chiGamma().size() < 1 ) ? false : true;
                for(int jj = min[cor_jj]; jj < max[cor_jj]; ++jj)
                {
                    for(int kk = min[cor_kk]; kk < max[cor_kk]; ++kk)
                    {
                        // Bottom, Left, Back PML
                        for(int ii = min[cor_ii]; ii < max[cor_ii]; ++ii)
                        {
                            std::array<double,3> pt = { 0, 0, 0 };
                            pt[cor_ii] = static_cast<double>(ii)*d_[cor_ii];
                            pt[cor_jj] = static_cast<double>(jj)*d_[cor_jj];
                            pt[cor_kk] = static_cast<double>(kk)*d_[cor_kk];
                            // Check if the Ex point is in the object if yes and dielectricMatInPML is not already true, set it to dielecMat
                            pt[0] += d_[0]/2.0;
                            if( !dielectricMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                dielectricMatInPML_ = dielcMat;
                            // Check if the Hz point is in the object if yes and magMatInPML is not already true, set it to magMat
                            pt[1] += d_[1]/2.0;
                            if( !magMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                magMatInPML_ = magMat;

                            // Check if the Ey point is in the object if yes and dielectricMatInPML is not already true, set it to dielecMat
                            pt[0] -= d_[0]/2.0;
                            if( !dielectricMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                dielectricMatInPML_ = dielcMat;
                            // Check if the Hx point is in the object if yes and magMatInPML is not already true, set it to magMat
                            pt[2] += d_[2]/2.0;
                            if( !magMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                magMatInPML_ = magMat;

                            // Check if the Ez point is in the object if yes and dielectricMatInPML is not already true, set it to dielecMat
                            pt[1] -= d_[0]/2.0;
                            if( !dielectricMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                dielectricMatInPML_ = dielcMat;
                            // Check if the Hy point is in the object if yes and magMatInPML is not already true, set it to magMat
                            pt[0] += d_[2]/2.0;
                            if( !magMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                magMatInPML_ = magMat;
                        }
                        // Top, Right, Front PML
                        min[cor_ii] = n_vec_[cor_ii]/2 - pmlThickness_[cor_ii];
                        max[cor_ii] = n_vec_[cor_ii]/2;
                        for(int ii = min[cor_ii]; ii < max[cor_ii]; ++ii)
                        {
                            std::array<double,3> pt = { 0, 0, 0 };
                            pt[cor_ii] = static_cast<double>(ii)*d_[cor_ii];
                            pt[cor_jj] = static_cast<double>(jj)*d_[cor_jj];
                            pt[cor_kk] = static_cast<double>(kk)*d_[cor_kk];
                            // Check if the Ex point is in the object if yes and dielectricMatInPML is not already true, set it to dielecMat
                            pt[0] += d_[0]/2.0;
                            if( !dielectricMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                dielectricMatInPML_ = dielcMat;
                            // Check if the Hz point is in the object if yes and magMatInPML is not already true, set it to magMat
                            pt[1] += d_[1]/2.0;
                            if( !magMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                magMatInPML_ = magMat;

                            // Check if the Ey point is in the object if yes and dielectricMatInPML is not already true, set it to dielecMat
                            pt[0] -= d_[0]/2.0;
                            if( !dielectricMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                dielectricMatInPML_ = dielcMat;
                            // Check if the Hx point is in the object if yes and magMatInPML is not already true, set it to magMat
                            pt[2] += d_[2]/2.0;
                            if( !magMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                magMatInPML_ = magMat;

                            // Check if the Ez point is in the object if yes and dielectricMatInPML is not already true, set it to dielecMat
                            pt[1] -= d_[0]/2.0;
                            if( !dielectricMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                dielectricMatInPML_ = dielcMat;
                            // Check if the Hy point is in the object if yes and magMatInPML is not already true, set it to magMat
                            pt[0] += d_[2]/2.0;
                            if( !magMatInPML_ && obj->isObj( pt, d_[0], obj->geo() ) )
                                magMatInPML_ = magMat;
                        }
                    }
                }
            }
        }
        // See if any objects would require chiral, electric dispersive, or magnetic dispersive materials
        bool chiral = false;
        bool disp = false;
        bool magnetic = false;
        for(auto& obj : objArr_)
        {
            if(obj->chiGamma().size() > 0)
                chiral = true;
            if(obj->magGamma().size() > 0 || obj->muInfty() > 1.0)
                magnetic = true;
            if(obj->gamma().size() > 0 || obj->ML() || obj->epsInfty() > 1.0)
                disp = true;
        }

        if(dielectricMatInPML_)
            disp = true;
        if(magMatInPML_)
            magnetic = true;

        // Initialize all girds
        if(IP.size_[2] != 0 || IP.pol_ == POLARIZATION::HZ || IP.pol_ == POLARIZATION::EX || IP.pol_ == POLARIZATION::EY)
        {
            // Defined TEz mode
            E_[0] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(), nz }}), d_, false);
            H_[2] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(), nz }}), d_, true);
            E_[1] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(), nz }}), d_, true);

            if(disp || chiral)
            {
                D_[0] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(), nz }}), d_, false);
                D_[1] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(), nz }}), d_, true);
            }
            if(magnetic || chiral)
            {
                B_[2] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(), nz }}), d_, true);
            }
            if(chiral)
            {
                prevH_[2] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, true);
                prevE_[0] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, true);
                prevE_[1] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, false);
            }

            ln_vec_[0] = H_[2]->local_x()-2;
            ln_vec_[1] = H_[2]->local_y()-2;
            ln_vec_[2] = (nz == 1) ? 1 : H_[2]->local_z()-2;
        }
        if(IP.size_[2] != 0 || IP.pol_ == POLARIZATION::EZ || IP.pol_ == POLARIZATION::HX || IP.pol_ == POLARIZATION::HY)
        {
            // Define TMz mode
            H_[0] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }}),d_, true);
            E_[2] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }}),d_,false);
            H_[1] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }}),d_,false);

            if(disp || chiral)
            {
                D_[2] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, false);
            }
            if(magnetic || chiral)
            {
                B_[0] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, true);
                B_[1] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, false);
            }
            if(chiral)
            {
                prevH_[0] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, true);
                prevH_[1] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, false);
                prevE_[2] = std::make_shared<parallelGrid<T>> (gridComm_, IP.periodic_, weights_, std::array<int,3>({{ n_vec_[0]+2*gridComm_->npX(), n_vec_[1]+2*gridComm_->npY(),  nz }}), d_, true);
            }
            ln_vec_[0] = E_[2]->local_x()-2;
            ln_vec_[1] = E_[2]->local_y()-2;
            ln_vec_[2] = (nz == 1) ? 1 : E_[2]->local_z()-2;
        }
        zMinPBC_ = (nz == 1) ? 0 : 1;
        zMaxPBC_ = (nz == 1) ? 1 : ln_vec_[2];
        // Initialize object specific grids
        int oo = 0;
        for(auto & obj :objArr_)
        {
            if(D_[2] || B_[0])
            {
                while( obj->gamma().size() > lorP_[2].size())
                {
                    lorP_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    prevLorP_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                }
                while( obj->magGamma().size() > lorM_[0].size())
                {
                    prevLorM_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                    prevLorM_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    lorM_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    lorM_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                }
            }
            if(B_[2] || D_[0])
            {
                while( obj->magGamma().size() > lorM_[2].size())
                {
                    lorM_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    prevLorM_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                }
                while( obj->gamma().size() > lorP_[0].size())
                {
                    prevLorP_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                    prevLorP_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    lorP_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    lorP_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                }
            }
            while( obj->chiGamma().size() > lorChiHP_[0].size())
            {
                prevLorChiHP_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                prevLorChiHP_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                prevLorChiEM_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );

                lorChiHP_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                lorChiHP_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                lorChiEM_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );

                prevLorChiEM_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                prevLorChiEM_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                prevLorChiHP_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );

                lorChiEM_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                lorChiEM_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, true) );
                lorChiHP_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
            }
            if( obj->useOrdDip() )
            {
                if(D_[2] || B_[0])
                {
                    while( obj->gamma().size() > orDipLorP_[0].size() || obj->chiGamma().size() > orDipLorP_[0].size())
                    {
                            orDipLorP_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                        prevOrDipLorP_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                            orDipLorP_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                        prevOrDipLorP_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    }
                    while( obj->magGamma().size() > orDipLorM_[2].size() || obj->chiGamma().size() > orDipLorM_[2].size())
                    {
                            orDipLorM_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                        prevOrDipLorM_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    }
                    while( obj->gamma().size() > dipP_[2].size() )
                    {
                        dipP_[2].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::ELE, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 2, dipP_[2].size() ) );
                    }
                    while( obj->magGamma().size() > dipM_[0].size() )
                    {
                        dipM_[0].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::MAG, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 0, dipM_[0].size() ) );
                        dipM_[1].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::MAG, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 1, dipM_[1].size() ) );
                    }
                    while( obj->chiGamma().size() > dipM_[0].size() )
                    {
                        dipM_[0].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::CHIM, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 0, dipM_[0].size() ) );
                        dipM_[1].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::CHIM, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 1, dipM_[1].size() ) );
                    }
                    while( obj->chiGamma().size() > dipP_[2].size() )
                        dipP_[2].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::CHIE, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 2, dipP_[2].size() ) );
                }
                if(B_[2] || D_[0])
                {
                    while( obj->magGamma().size() > orDipLorM_[0].size() || obj->chiGamma().size() > orDipLorM_[0].size())
                    {
                            orDipLorM_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                        prevOrDipLorM_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                            orDipLorM_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                        prevOrDipLorM_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    }
                    while( obj->gamma().size() > orDipLorP_[2].size() || obj->chiGamma().size() > orDipLorP_[2].size())
                    {
                            orDipLorP_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                        prevOrDipLorP_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    }

                    while( obj->magGamma().size() > dipM_[2].size() )
                    {
                        dipM_[2].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::MAG, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 2, dipM_[2].size() ) );
                    }
                    while( obj->gamma().size() > dipP_[0].size() )
                    {
                        dipP_[0].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::ELE, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 0, dipP_[0].size() ) );
                        dipP_[1].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::ELE, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 1, dipP_[1].size() ) );
                    }
                    while( obj->chiGamma().size() > dipP_[0].size() )
                    {
                        dipP_[0].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::CHIE, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 0, dipP_[0].size() ) );
                        dipP_[1].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::CHIE, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 1, dipP_[1].size() ) );
                    }
                    while( obj->chiGamma().size() > dipM_[2].size() )
                        dipM_[2].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::CHIM, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 2, dipM_[2].size() ) );
                }

                int cc_p = 0, cc_m = 0;
                for( int pp = 0; pp < obj->pols().size() ; ++pp )
                {
                    if( std::abs(obj->pols()[pp].tau_ ) > 0.0)
                    {
                        chiDipPInds_[oo].push_back(cc_p);
                        chiDipMInds_[oo].push_back(cc_m);
                        ++cc_p; ++cc_m;
                    }
                    else
                    {
                        if(obj->pols()[pp].sigP_ > 0.0)
                            ++cc_p;
                        if(obj->pols()[pp].sigM_ > 0.0)
                            ++cc_m;
                    }
                }

                while( obj->chiGamma().size() > dipChiP_[0].size())
                {
                    chiOrDipLorP_    [0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    chiOrDipLorP_    [1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    chiOrDipLorP_    [2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );

                    prevChiOrDipLorP_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    prevChiOrDipLorP_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    prevChiOrDipLorP_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );

                    chiOrDipLorM_    [0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    chiOrDipLorM_    [1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    chiOrDipLorM_    [2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );

                    prevChiOrDipLorM_[0].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    prevChiOrDipLorM_[1].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );
                    prevChiOrDipLorM_[2].push_back(std::make_shared<parallelGrid<T>>(gridComm_, false, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false) );

                    dipChiM_[0].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::CHIM, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 0, dipChiM_[0].size() ) );
                    dipChiM_[1].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::CHIM, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 1, dipChiM_[1].size() ) );
                    dipChiM_[2].push_back(setupDipMoments(physPOrDip_, LOR_POL_TYPE::CHIM, std::array<double,3>( {{ 0.0, 0.0, 0.0 }} ), 2, dipChiM_[2].size() ) );

                    dipChiP_[0].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::CHIE, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 0, dipChiP_[0].size() ) );
                    dipChiP_[1].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::CHIE, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 1, dipChiP_[1].size() ) );
                    dipChiP_[2].push_back(setupDipMoments(physMOrDip_, LOR_POL_TYPE::CHIE, std::array<double,3>( {{ 0.5, 0.5, 0.5 }} ), 2, dipChiP_[2].size() ) );
                }
            }
            ++oo;
        }
        for(auto& xx : IP.inputMapSlicesX_)
            convertInputs2Map(IP, DIRECTION::X, xx);
        for(auto& yy : IP.inputMapSlicesY_)
            convertInputs2Map(IP, DIRECTION::Y, yy);
        for(auto& zz : IP.inputMapSlicesZ_)
            convertInputs2Map(IP, DIRECTION::Z, zz);
    }
    /**
     * @brief      Populates the update Lists using the blas lists generated from getBlasLists
     *
     * @param[in]  blasList  The blas list
     * @param[in]  derivOff  An array describing the offset for the j spatial derivative
     * @param[in]  physGrid  Map of all objects for the gird
     * @param[in]  d         grid spacing
     * @param[in]  upUList   True if the update list corresponds to the E or H field
     * @param      uList     The update list that needs to be filled
     */
    void populateUpLists(const std::vector<std::array<int,5>>& blasList, const std::array<int,3> derivOff, int_pgrid_ptr physGrid, real_pgrid_ptr epMuRel, double dj, double dk, bool upUList, bool E, upLists& uList, std::vector<std::array<int,5>>& locs, bool orDipField=false, bool storeLocs=false)
    {
        for(auto & up : blasList)
        {
            double ep_mu = 1.0;
            if(upUList)
                ep_mu = epMuRel->point(up[0], up[1], up[2]);
            if(!orDipField)
            {
                if(epMuRel->local_z() == 1)
                    uList.push_back(std::make_pair(std::array<int,6>({up[3]  , epMuRel->getInd(up[0], up[1], up[2]), epMuRel->getInd(up[0]-derivOff[1], up[1]-derivOff[2], up[2]            ), epMuRel->getInd(up[0]+derivOff[0], up[1]+derivOff[1], up[2]            ), epMuRel->getInd(up[0]+derivOff[2], up[1]+derivOff[0], up[2]            ), up[4]}), std::array<double,4>({1.0, -1.0*dt_/(ep_mu*dj), -1.0*dt_/(ep_mu*dk), epMuRel->point(up[0], up[1], up[2])}) ) );
                else
                    uList.push_back(std::make_pair(std::array<int,6>({up[3]  , epMuRel->getInd(up[0], up[1], up[2]), epMuRel->getInd(up[0]-derivOff[1], up[1]-derivOff[2], up[2]-derivOff[0]), epMuRel->getInd(up[0]+derivOff[0], up[1]+derivOff[1], up[2]+derivOff[2]), epMuRel->getInd(up[0]+derivOff[2], up[1]+derivOff[0], up[2]+derivOff[1]), up[4]}), std::array<double,4>({1.0, -1.0*dt_/(ep_mu*dj), -1.0*dt_/(ep_mu*dk), epMuRel->point(up[0], up[1], up[2])}) ) );
            }
            else
            {
                uList.push_back(std::make_pair(std::array<int,6>({up[3]  , epMuRel->getInd(up[0], up[1], up[2]), epMuRel->getInd(up[0]+derivOff[0], up[1]            , up[2])            , epMuRel->getInd(up[0]            , up[1]+derivOff[1], up[2]            ), epMuRel->getInd(up[0]            , up[1]            , up[2]+derivOff[2]), up[4]}), std::array<double,4>({1.0, -1.0*dt_/(ep_mu*dj), -1.0*dt_/(ep_mu*dk), epMuRel->point(up[0], up[1], up[2])}) ) );
            }
            if(storeLocs)
                locs.push_back(std::array<int,5>({  up[3], up[0], up[1], up[2], up[4] }));
        }
    }

    /**
     * @brief      Creates the update lists for a field
     *
     * @param[in]  physGrid     Map of all objects for the gird
     * @param[in]  pml          The pml for the grid
     * @param[in]  E            True if gird is an electric field
     * @param[in]  derivOff     An array describing the offset for the j spatial derivative
     * @param[in]  fieldEnd     0 or 1 if last grid point in the i direction is ni or ni-1
     * @param[in]  d            grid spacing
     * @param      upU          upE/upH update list
     * @param      upD          The upD/upLorB update list
     * @param      upChiD       The upChiD/upChiB update list
     * @param      upOrDipD     The upOrDipD/upOrDipB update list
     * @param      upOrDipChiD  The upOrDipChiD/upOrDipChiB update list
     */
    void initializeList(int_pgrid_ptr physGrid, real_pgrid_ptr epMuRel, std::shared_ptr<parallelCPML<T>> pml, bool E, std::array<int,3> derivOff, std::array<int,3> fieldEnd, double dj, double dk, upLists& upU, upLists& upD, upLists& upLorD, upLists& upChiD, upLists& upOrDipD, upLists& upOrDipChiD, std::vector<std::array<int,5>>& chiLocs, bool orDipField=false)
    {
        // Get the blas parameters for the U, D, lorD, chiD, orDipD, orDipChiD update lists
        std::vector<std::array<int, 5>> tempU, tempD, tempLorD, tempChiD, tempOrDipD, tempOrDipChiD;
        std::array<int,3> min = { 1 ,1 , (physGrid->z() == 1 ? 0 : 1) };
        std::array<int,3> max = { physGrid->ln_vec(0)-1-fieldEnd[0], physGrid->ln_vec(1)-1, (physGrid->z() == 1 ? 1 : physGrid->local_z()-1-fieldEnd[2]) };
        if(gridComm_->rank() == gridComm_->size()-1)
            max[1] -= fieldEnd[1];
        getBlasLists(  E, min, max, physGrid, epMuRel, pml, fieldEnd, tempU, tempD, tempLorD, tempChiD, tempOrDipD, tempOrDipChiD);
        // Fill all the lists with correct parameters
        populateUpLists(tempU        , derivOff, physGrid, epMuRel, dj, dk,  true, E, upU, chiLocs, false);
        populateUpLists(tempD        , derivOff, physGrid, epMuRel, dj, dk, false, E, upD, chiLocs, false);
        populateUpLists(tempLorD     , derivOff, physGrid, epMuRel, dj, dk, false, E, upLorD, chiLocs, false);
        populateUpLists(tempChiD     , derivOff, physGrid, epMuRel, dj, dk, false, E, upChiD, chiLocs, orDipField, true);
        populateUpLists(tempOrDipD   , derivOff, physGrid, epMuRel, dj, dk, false, E, upOrDipD, chiLocs, orDipField, false);
        populateUpLists(tempOrDipChiD, derivOff, physGrid, epMuRel, dj, dk, false, E, upOrDipChiD, chiLocs, orDipField, true);
    }

    /**
     * @brief      finds the area where chiral fields exit
     *
     * @param[in]  chiup     The update list for the chiral field
     * @param[in]  objInd    The object array index
     * @param      currXMin  The curr x minimum
     * @param      currYMin  The curr y minimum
     * @param      currZMin  The curr z minimum
     * @param      currXMax  The curr x maximum
     * @param      currYMax  The curr y maximum
     * @param      currZMax  The curr z maximum
     */
    void findMinMaxChiLocsObj(std::vector<std::array<int,5>> chiLocs, int objInd, int& currXMin, int& currYMin, int& currZMin, int& currXMax, int& currYMax, int& currZMax)
    {
        for(auto& ax : chiLocs)
        {

            // Max X is based off of the loc+sz-1
            if(ax[4] == objInd && ax[1] + ax[0]-1 > currXMax)
                currXMax = ax[1] + ax[0] - 1;
            if(ax[4] == objInd && ax[2] > currYMax)
                currYMax = ax[2];
            if(ax[4] == objInd && ax[3] > currZMax)
                currZMax = ax[3];

            if(ax[4] == objInd && ax[1] < currXMin)
                currXMin = ax[1];
            if(ax[4] == objInd && ax[2] < currYMin)
                currYMin = ax[2];
            if(ax[4] == objInd && ax[3] < currZMin)
                currZMin = ax[3];
        }
        return;
    }

    /**
     * @brief      Constructs a DTC based off of the input parameters and puts it in the proper detector vector
     *
     * @param[in]  c             class type of the dtc (bin, bmp, cout, txt, freq)
     * @param[in]  grid          vector of the fields that need to be outputted
     * @param[in]  SI            true if outputting in SI units
     * @param[in]  loc           The location of the detectors lower left corner in grid points
     * @param[in]  sz            The size of the detector in grid points
     * @param[in]  out_name      The output file name
     * @param[in]  fxn           Function used to modify base field data
     * @param[in]  txtType       if BMP what should be outputted to the text file
     * @param[in]  type          The type of the detector (Ex, Ey, Epow, etc)
     * @param[in]  freqList      The frequency list
     * @param[in]  timeInterval  The number of time steps per field output
     * @param[in]  a             unit length of the calculation
     * @param[in]  I0            unit current of the calculation
     * @param[in]  t_max         The time at the final time step
     */
    virtual void coustructDTC(DTCCLASS c, std::vector< std::pair<pgrid_ptr, std::array<int,3>> > grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double t_start, double t_end, bool outputAvg, std::vector<double> freqList, double timeInterval, double a, double I0, double t_max, bool outputMaps) = 0;

    /**
     * @brief      Fills the blas update lists
     *
     * @param[in]  min          Array storing the minimum grid point values in all directions
     * @param[in]  max          Array storing the maximum grid point values in all directions
     * @param[in]  includeU     True if U lists can be updated
     * @param      upU          The tempUpU update list
     * @param      upD          The tempUpD update list
     * @param      upChiD       The tempUpChiD update list
     * @param      upOrDipD     The tempUpOrDipD update list
     * @param      upOrDipChiD  The tempUpOrDipChiD update list
     */
    void fillBlasLists( std::array<int,3> min, std::array<int,3> max, int_pgrid_ptr physGrid, real_pgrid_ptr epMuRel, bool includeU, bool E, std::vector<std::array<int,5>>& upU, std::vector<std::array<int,5>>& upD, std::vector<std::array<int,5>>& upChiD, std::vector<std::array<int,5>>& upOrDipD, std::vector<std::array<int,5>>& upOrDipChiD)
    {
        std::shared_ptr<Obj> obj;
        for(int jj = min[1]; jj < max[1]; ++jj)
        {
            for(int kk = min[2]; kk < max[2]; ++kk )
            {
                int ii = min[0];
                while(ii < max[0])
                {
                    int iistore = ii;
                    // Check if points are in same object
                    while ( (ii < max[0]-1) && ( physGrid->point(ii,jj,kk) == physGrid->point(ii+1,jj,kk) ) && ( epMuRel->point(ii,jj,kk) == epMuRel->point(ii+1,jj,kk) ) )
                        ++ii;

                    obj = objArr_[ physGrid->point(iistore, jj, kk) ];
                    if( obj->useOrdDip() && ( obj->chiGamma().size() > 0 || ( E && obj->gamma().size() > 0 ) || ( !E && obj->magGamma().size() > 0 ) ) )
                    {
                        if( obj->chiGamma().size() > 0 )
                            upOrDipChiD.push_back( {{ iistore, jj, kk, ii-iistore+1, physGrid->point(iistore, jj, kk) }} );
                        else
                            upOrDipD.push_back( {{ iistore, jj, kk, ii-iistore+1, physGrid->point(iistore, jj, kk ) }} );
                    }
                    else if( obj->chiGamma().size() > 0 )
                        upChiD.push_back( {{ iistore, jj, kk, ii-iistore+1,physGrid->point(iistore, jj, kk) }} );
                    else if( includeU && ( ( !E && obj->magGamma().size() < 1 ) || ( E && obj->gamma().size() < 1 && !obj->ML() ) ) )
                        upU.push_back( {{ iistore, jj, kk, ii-iistore+1, physGrid->point(iistore,jj,kk) }} );
                    else
                        upD.push_back( {{ iistore, jj, kk, ii-iistore+1, physGrid->point(iistore,jj,kk) }} );

                    ++ii;
                }
            }
        }
    }

    /**
     * @brief      Gets the blas lists.
     *
     * @param[in]  E                 True if the field the list is being generated for is an electric field
     * @param[in]  min               An array containing the minimum grid points in all direction
     * @param[in]  max               An array containing the maximum grid points in all directions
     * @param[in]  physGrid          The map of all objects on that grid
     * @param[in]  pml               The cPML associated with the grid
     * @param      upULists          The update U field lists
     * @param      upDLists          The update D field lists
     * @param      upLorDLists       The update Lorentzian d lists
     * @param      upChiDLists       The update chiral d lists
     * @param      upOrDipDLists     The update oriented dipole d lists
     * @param      upOrDipChiDLists  The update chiral oriented dipole d lists
     */
    void getBlasLists(bool E, std::array<int,3> min, std::array<int,3> max, int_pgrid_ptr physGrid, real_pgrid_ptr epMuRel, std::shared_ptr<parallelCPML<T>> pml, std::array<int,3> fieldEnd, std::vector<std::array<int,5>>& upULists, std::vector<std::array<int,5>>& upDLists, std::vector<std::array<int,5>>& upLorDLists, std::vector<std::array<int,5>>& upChiDLists, std::vector<std::array<int,5>>& upOrDipDLists, std::vector<std::array<int,5>>& upOrDipChiDLists)
    {

        std::vector<std::array<int,5>> tempUListNotInc;

        // Include PMLs if material is inside them
        int PML_x_left  = ( (E && dielectricMatInPML_) || (!E && magMatInPML_) ) ? pml->lnx_left() : 0;
        int PML_x_right = ( (E && dielectricMatInPML_) || (!E && magMatInPML_) ) ? pml->lnx_right() : 0;
        if(PML_x_right != 0)
            PML_x_right -= fieldEnd[0];
        int PML_y_bot = ( (E && dielectricMatInPML_) || (!E && magMatInPML_) ) ? pml->lny_bot() : 0;
        int PML_y_top = ( (E && dielectricMatInPML_) || (!E && magMatInPML_) ) ? pml->lny_top() : 0;
        if(PML_y_top != 0 && gridComm_->rank() == gridComm_->size() - 1)
            PML_y_top -= fieldEnd[1];
        int PML_z_back  = (E_[2] && H_[2] && ( (E && dielectricMatInPML_) || (!E && magMatInPML_) ) ) ? pml->lnz_back()  : 0;
        int PML_z_front = (E_[2] && H_[2] && ( (E && dielectricMatInPML_) || (!E && magMatInPML_) ) ) ? pml->lnz_front() : 0;
        if(PML_z_front != 0)
            PML_z_front -= fieldEnd[2];

        // Fill Bottom Y PML Region
        fillBlasLists(std::array<int,3>( {{ min[0], min[1]          , min[2] }} ), std::array<int,3>( {{ max[0], min[1]+PML_y_bot, max[2] }} ), physGrid, epMuRel, !(dielectricMatInPML_ || magMatInPML_), E, tempUListNotInc, upLorDLists, upChiDLists, upOrDipDLists, upOrDipChiDLists);
        // Fill Top Y PML Region
        fillBlasLists(std::array<int,3>( {{ min[0], max[1]-PML_y_top, min[2] }} ), std::array<int,3>( {{ max[0], max[1]          , max[2] }} ), physGrid, epMuRel, !(dielectricMatInPML_ || magMatInPML_), E, tempUListNotInc, upLorDLists, upChiDLists, upOrDipDLists, upOrDipChiDLists);

        // Fill Central Y Region

        // Fill Back Z PML Region
        fillBlasLists(std::array<int,3>( {{ min[0], min[1]+PML_y_bot, min[2]             }} ), std::array<int,3>( {{ max[0], max[1]-PML_y_top, min[2]+PML_z_back }} ), physGrid, epMuRel, !(dielectricMatInPML_ || magMatInPML_), E, tempUListNotInc, upLorDLists, upChiDLists, upOrDipDLists, upOrDipChiDLists);
        // Fill Front Z PML Region
        fillBlasLists(std::array<int,3>( {{ min[0], min[1]+PML_y_bot, max[2]-PML_z_front }} ), std::array<int,3>( {{ max[0], max[1]-PML_y_top, max[2]            }} ), physGrid, epMuRel, !(dielectricMatInPML_ || magMatInPML_), E, tempUListNotInc, upLorDLists, upChiDLists, upOrDipDLists, upOrDipChiDLists);

        // Fill Central Z Region

        // Fill Left X PML Region
        fillBlasLists(std::array<int,3>( {{ min[0]            , min[1]+PML_y_bot, min[2]+PML_z_back }} ), std::array<int,3>( {{ min[0]+PML_x_left , max[1]-PML_y_top, max[2]-PML_z_front }} ), physGrid, epMuRel, !(dielectricMatInPML_ || magMatInPML_), E, tempUListNotInc, upLorDLists, upChiDLists, upOrDipDLists, upOrDipChiDLists);
        // Fill Right X PML Region
        fillBlasLists(std::array<int,3>( {{ max[0]-PML_x_right, min[1]+PML_y_bot, min[2]+PML_z_back }} ), std::array<int,3>( {{ max[0]            , max[1]-PML_y_top, max[2]-PML_z_front }} ), physGrid, epMuRel, !(dielectricMatInPML_ || magMatInPML_), E, tempUListNotInc, upLorDLists, upChiDLists, upOrDipDLists, upOrDipChiDLists);
        // Fill Central Region
        fillBlasLists(std::array<int,3>( {{ min[0]+ PML_x_left, min[1]+PML_y_bot, min[2]+PML_z_back }} ), std::array<int,3>( {{ max[0]-PML_x_right, max[1]-PML_y_top, max[2]-PML_z_front }} ), physGrid, epMuRel, true, E, tempUListNotInc, upLorDLists, upChiDLists, upOrDipDLists, upOrDipChiDLists);

        //Fill upD, upB, upE and up H lists
        min[0] += pml->lnx_left();
        max[0] -= pml->lnx_right();
        if(pml->lnx_right() != 0 )
            max[0] += fieldEnd[0];
        min[1] += pml->lny_bot();
        max[1] -= pml->lny_top();
        if(pml->lny_top() != 0 &&  gridComm_->rank() == gridComm_->size() - 1)
            max[1] += fieldEnd[1];
        min[2] += pml->lnz_back() ;
        max[2] -= pml->lnz_front();
        if(pml->lnz_front() != 0 )
            max[2] += fieldEnd[2];
        fillBlasLists(min, max, physGrid, epMuRel, true, E, upULists, upDLists, upDLists, upDLists, upDLists);
    }

    /**
     * @brief      Sets up the object map grids for each field
     *
     * @param      physGrid  The object map to be made
     * @param[in]  PBC       True if periodic boundary conditions used
     * @param[in]  offPt     values for the offset from the base grid point for each field
     * @param[in]  nz        number of grid points in the z direction
     * @param[in]  endOff    each value i is 1 if the field ends at ni - 1
     */
    void setupPhysFields(int_pgrid_ptr& physGrid, real_pgrid_ptr& epMuRel, bool E, bool PBC, std::array<double,3> offPt, int nz, std::array<int,3> endOff)
    {
        physGrid = std::make_shared<parallelGrid<int> >(gridComm_, PBC, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false);
        epMuRel = std::make_shared<parallelGrid<double> >(gridComm_, PBC, weights_, std::array<int,3>( {{ n_vec_[0]+2*gridComm_->npX(),n_vec_[1]+2*gridComm_->npY(), nz }} ), d_, false);
        std::fill_n( epMuRel->data(), epMuRel->size(), 1.0);
        // Test point used
        std::array<double,3> pt = {{ 0,0,0}};
        // Objects on the same points over write each other (last object made wins)
        int zmin = (nz == 1) ? 0 : 1;
        int zmax = (nz == 1) ? 1 : physGrid->ln_vec(2)-1;
        for(int oo = 1; oo < objArr_.size(); ++oo)
        {
            // look at all local points only
            if( objArr_[oo]->ML() || objArr_[oo]->gamma().size() > 0 || objArr_[oo]->epsInfty() != 1.0 || objArr_[oo]->chiGamma().size() > 0 || objArr_[oo]->tellegen() != 0.0 || objArr_[oo]->magGamma().size() > 0 || objArr_[oo]->muInfty() != 1.0 )
            {
                for(int ii = 1; ii < physGrid->ln_vec(0)-1; ++ii)
                {
                    for(int jj = 1; jj < physGrid->ln_vec(1)-1; ++jj)
                    {
                        for(int kk = zmin; kk < zmax; ++kk)
                        {
                            pt[0] = ( (ii-1) + offPt[0] + physGrid->procLoc(0) - (n_vec_[0]-n_vec_[0] % 2)/2.0 )*d_[0];
                            pt[1] = ( (jj-1) + offPt[1] + physGrid->procLoc(1) - (n_vec_[1]-n_vec_[1] % 2)/2.0 )*d_[1];
                            pt[2] = ( (kk-1) + offPt[2] + physGrid->procLoc(2) - (n_vec_[2]-n_vec_[2] % 2)/2.0 )*d_[2];
                            if(objArr_[oo]->isObj( pt,d_[0],objArr_[oo]->geoML() )==true)
                                physGrid->point(ii,jj, kk) = oo;
                            if(objArr_[oo]->isObj( pt,d_[0],objArr_[oo]->geo() )==true)
                                epMuRel->point(ii,jj, kk) = E ? objArr_[oo]->epsInfty() : objArr_[oo]->muInfty() ;
                        }
                    }
                }
            }
        }
        // All borders of between the processors have a -1 to indicate they are borders
        zmin = 0;
        zmax = (nz == 1) ? 1 : physGrid->ln_vec(2);
        if(nz != 1)
        {
            for(int ii = 0; ii < physGrid->ln_vec(0); ++ii)
            {
                for(int jj = 0; jj < physGrid->ln_vec(1); ++jj)
                {
                    physGrid->point(ii, jj, 0   ) = -1;
                    physGrid->point(ii, jj, zmax-1) = -1;
                    if(endOff[2] == 1)
                    {
                        physGrid->point(ii, jj, zmax-2) = -1;
                        epMuRel->point(ii, jj, zmax-2) = 0.0;
                    }
                }
            }
        }

        for(int ii = 0; ii < physGrid->ln_vec(0); ++ii)
        {
            for(int kk = zmin; kk < zmax; ++kk)
            {
                physGrid->point(ii, 0, kk) = -1;
                physGrid->point(ii, physGrid->ln_vec(1)-1, kk) = -1;
                if(endOff[1] == 1 && gridComm_->rank() == gridComm_->size() - 1 && physGrid->n_vec(1) > 3)
                {
                    physGrid->point(ii, physGrid->ln_vec(1)-2, kk) = -1;
                    epMuRel->point(ii, physGrid->ln_vec(1)-2, kk) = 0.0;
                }
            }
        }
        for(int kk = zmin; kk < zmax; ++kk)
        {
            for(int jj = 0; jj < physGrid->ln_vec(1); ++jj)
            {
                physGrid->point(0, jj, kk) = -1;
                physGrid->point(physGrid->ln_vec(0)-1, jj, kk) = -1;
                if(endOff[0] == 1 && physGrid->n_vec(0) > 3)
                {
                    physGrid->point(physGrid->ln_vec(0)-2, jj, kk) = -1;
                    epMuRel->point(physGrid->ln_vec(0)-2, jj, kk) = 0.0;
                }
            }
        }
        epMuRel->transferDat();
    }

    /**
     * @brief      Sets up the dipole orientation grids (how to scale the dipole moments)
     *
     * @param[in]  physGrid   The physGrid associated with the filed
     * @param[in]  fieldType  Describing what type of field the dipole is acting on ( mostly for unidirectional, also for chiral P and M fields)
     * @param[in]  offPt      values for the offset from the base grid point for each field
     * @param[in]  dipInd     The index describing the direction of the dipole moment (x=0, y=1, z=2)
     * @param[in]  polInd     Index of the Lorenz oscillator in the object parameter list
     *
     * @return     The dipole moment orientation gird
     */
    real_pgrid_ptr setupDipMoments(int_pgrid_ptr physGrid, LOR_POL_TYPE fieldType, std::array<double,3> offPt, int dipInd, int polInd)
    {
        real_pgrid_ptr dipField = std::make_shared<parallelGrid<double> >(gridComm_, physGrid->PBC(), weights_, physGrid->n_vec(), d_, false);
        std::array<double,3> pt = {{ 0,0,0}};
        // Objects on the same points over write each other (last object made wins)
        int zmin = (physGrid->local_z() == 1) ? 0 : 1;
        int zmax = (physGrid->local_z() == 1) ? 1 : physGrid->local_z() - 1 - (offPt[2] > 0 ? 1 : 0);
        // look at all local points only
        for(int jj = 1; jj < physGrid->ln_vec(1)-1-( (offPt[1] > 0 && gridComm_->rank() == gridComm_->size()-1) ? 1 : 0); ++jj)
        {
            for(int kk = zmin; kk < zmax; ++kk)
            {
                for(int ii = 1; ii < physGrid->ln_vec(0)-1-(offPt[0] > 0 ? 1 : 0); ++ii)
                {
                    MAT_DIP_ORIENTAITON dipOr;
                    if(fieldType == LOR_POL_TYPE::ELE)
                        dipOr = objArr_[physGrid->point(ii, jj, kk)]->dipOr(polInd);
                    else if(fieldType == LOR_POL_TYPE::MAG)
                        dipOr = objArr_[physGrid->point(ii, jj, kk)]->magDipOr(polInd);
                    else if(fieldType == LOR_POL_TYPE::CHIE)
                        dipOr = objArr_[physGrid->point(ii, jj, kk)]->chiEDipOr(polInd);
                    else if(fieldType == LOR_POL_TYPE::CHIM)
                        dipOr = objArr_[physGrid->point(ii, jj, kk)]->chiMDipOr(polInd);

                    if(dipOr == MAT_DIP_ORIENTAITON::REL_TO_NORM)
                    {
                        pt[0] = ( (ii-1) + offPt[0] + physGrid->procLoc(0) - (n_vec_[0]-n_vec_[0] % 2)/2.0 )*d_[0];
                        pt[1] = ( (jj-1) + offPt[1] + physGrid->procLoc(1) - (n_vec_[1]-n_vec_[1] % 2)/2.0 )*d_[1];
                        pt[2] = ( (kk-1) + offPt[2] + physGrid->procLoc(2) - (n_vec_[2]-n_vec_[2] % 2)/2.0 )*d_[2];
                        if(fieldType == LOR_POL_TYPE::ELE)
                            dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipNormCompE(polInd) * objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt)[dipInd] + getTangentDip( objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt), objArr_[physGrid->point(ii, jj, kk)]->dipTanLatCompE(polInd), objArr_[physGrid->point(ii, jj, kk)]->dipTanLongCompE(polInd) )[dipInd];
                        else if(fieldType == LOR_POL_TYPE::MAG)
                            dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipNormCompM(polInd) * objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt)[dipInd] + getTangentDip( objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt), objArr_[physGrid->point(ii, jj, kk)]->dipTanLatCompM(polInd), objArr_[physGrid->point(ii, jj, kk)]->dipTanLongCompM(polInd) )[dipInd];
                        else if(fieldType == LOR_POL_TYPE::CHIE)
                            dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipNormCompChiE(polInd) * objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt)[dipInd] + getTangentDip( objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt), objArr_[physGrid->point(ii, jj, kk)]->dipTanLatCompChiE(polInd), objArr_[physGrid->point(ii, jj, kk)]->dipTanLongCompChiE(polInd) )[dipInd];
                        else if(fieldType == LOR_POL_TYPE::CHIM)
                            dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipNormCompChiM(polInd) * objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt)[dipInd] + getTangentDip( objArr_[physGrid->point(ii, jj, kk)]->findGradient(pt), objArr_[physGrid->point(ii, jj, kk)]->dipTanLatCompChiM(polInd), objArr_[physGrid->point(ii, jj, kk)]->dipTanLongCompChiM(polInd) )[dipInd];
                    }
                    else if(dipOr == MAT_DIP_ORIENTAITON::ISOTROPIC)
                    {
                        dipField->point(ii,jj,kk) = 1.0;
                    }
                    else if(dipOr == MAT_DIP_ORIENTAITON::UNIDIRECTIONAL)
                    {
                        if(fieldType == LOR_POL_TYPE::ELE)
                        {
                             dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipE(polInd)[dipInd];
                        }
                        else if(fieldType == LOR_POL_TYPE::MAG)
                        {
                             dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipM(polInd)[dipInd];
                        }
                        else if(fieldType == LOR_POL_TYPE::CHIE)
                        {
                             dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipChiE(polInd)[dipInd];
                        }
                        else if(fieldType == LOR_POL_TYPE::CHIM)
                        {
                             dipField->point(ii,jj,kk) = objArr_[physGrid->point(ii, jj, kk)]->dipChiM(polInd)[dipInd];
                        }
                    }
                }
            }
        }
        std::array<int,3>  modN = {{ static_cast<int>(offPt[0]*2), static_cast<int>(offPt[1]*2), static_cast<int>(offPt[2]*2) }};
        int zMin = dipField->z() == 1 ? 0 : 1;
        int zMax = dipField->z() == 1 ? 1 : ln_vec_[2]-modN[2];
        if(dipField->PBC() && gridComm_->size() > 1)
        {
            if(gridComm_->rank() !=0 || gridComm_->rank() != gridComm_->size() - 1)
                FDTDCompUpdateFxnReal::applyBCProcMid(dipField, k_point_, ln_vec_[0]-modN[0], ln_vec_[1], zMax, ln_vec_[0]-modN[0], ln_vec_[1], zMin, zMax, d_[0], d_[1], d_[2]);
            else
            {
                if(gridComm_->rank() == 0)
                    FDTDCompUpdateFxnReal::applyBCProc0(dipField, k_point_, ln_vec_[0]-modN[0], ln_vec_[1], zMax, ln_vec_[0]-modN[0], ln_vec_[1], zMin, zMax, d_[0], d_[1], d_[2]);
                else
                    FDTDCompUpdateFxnReal::applyBCProcMax(dipField, k_point_, ln_vec_[0]-modN[0], ln_vec_[1]-modN[1], zMax, ln_vec_[0]-modN[0], ln_vec_[1]-modN[1], zMin, zMax, d_[0], d_[1], d_[2]);
            }
        }
        else if(dipField->PBC())
        {
            FDTDCompUpdateFxnReal::applyBC1Proc(dipField, k_point_, ln_vec_[0]-modN[0], ln_vec_[1]-modN[1], zMax, ln_vec_[0]-modN[0], ln_vec_[1]-modN[1], zMin, zMax, d_[0], d_[1], d_[2]);
        }
        else if(gridComm_->size() > 1)
        {
            FDTDCompUpdateFxnReal::applyBCNonPer(dipField, k_point_, ln_vec_[0]-modN[0], ln_vec_[1]-modN[1], zMax, ln_vec_[0]-modN[0], ln_vec_[1]-modN[1], zMin, zMax, d_[0], d_[1], d_[2]);
        }
        return dipField;
    }

    /**
     * @brief      Gets the tangent dipole vectors
     *
     * @param[in]  normVec  The vector normal to the surface (describes the tangent plane)
     *
     * @return     An array describing the sum of the tangent vectors
     */
    std::array<double,3> getTangentDip(std::array<double,3> normVec, double latFact, double longFact)
    {
        std::array<double,3> toRet;
        double mag = 0;
        mag = std::sqrt( std::accumulate(normVec.begin(), normVec.end(), 0.0, vecMagAdd<double>() ) );
        std::array<double,3> normVecSph  = { mag, std::acos( normVec[2] / mag ), std::atan( normVec[1] / normVec[0] ) };
        // If the normal vector is not on the y-axis set the sph to pi/2 not nan->if norm is on the axis(0,0,+/-1) phi will be pi/2
        if(normVec[0] == 0.0)
            normVecSph[2] =  normVec[1] >= 0.0 ? M_PI/2.0 : -1.0*M_PI/2.0;
        // Get the correct orientation for phi
        if(normVec[0] < 0)
            normVecSph[2] += M_PI;
        // If the magnitude of the normal vector is 0 then reset the normVecSph to be 0
        if(mag < 1e-20)
            normVecSph = {0.0, 0.0, 0.0};
        if(H_[2] && E_[2])
        {
            // Rotating polar angle by pi/2.0 will give one of the tangent vectors
            std::array<double,3> tanVecSphLong  = {{ mag, normVecSph[1]+M_PI/2.0, normVecSph[2] }};
            std::array<double,3> tanVecCartLong = {{ mag*std::sin(tanVecSphLong[1])*std::cos(tanVecSphLong[2]), mag*std::sin(tanVecSphLong[1])*std::sin(tanVecSphLong[2]), mag*std::cos(tanVecSphLong[1]) }};
            std::array<double,3> tanVecCartLat;
            // Take the cross Product of the normVec and tanVecCartLong to get tanVecCartLat
            for(int ii = 0; ii < 3; ++ii)
                tanVecCartLat[ii] = normVec[(ii+1)%3]*tanVecCartLong[(ii+2)%3] - normVec[(ii+2)%3]*tanVecCartLong[(ii+1)%3];
            // Sum to get isotropic tangent

            std::transform(tanVecCartLong.begin(), tanVecCartLong.end(), tanVecCartLat.begin(), toRet.begin(), [&](double lng, double lat){return longFact*lng + latFact*lat;} );
        }
        else
        {
            toRet = {{ mag*std::cos(normVecSph[2]+M_PI/2.0), mag*std::sin(normVecSph[2]+M_PI/2.0), 0.0 }};
        }
        return toRet;
    }

    /**
     * @brief      Sets up the weight grid for dividing processes by taking average of H/E fields and for 3D calcs the flux region calcs
     *
     * @param[in]  IP    Input parameter object that is being used to construct the propagator
     */
    void setupWeightsGrid(const parallelProgramInputs &IP)
    {
        // test point will move across all grid points
        // Set weights based off of normal materials, number of axpy calls per time step (6 base + 3*n_lor_pol = 6 + 3*(objArr_[kk]->gamma().size() )/3)*number of E fields
        std::array<double,3> pt = {{0,0,0}};
        // Create and initialize the weights grids with the correct baseline cost per grid point
        weights_.push_back(std::make_shared<Grid<double>>(n_vec_, d_) );
        std::fill_n(weights_.back()->data(), weights_.back()->size(), IP.size_[2] != 1 ? 276.0 : 184.0);
        weights_.push_back(std::make_shared<Grid<double>>(n_vec_, d_) );
        std::fill_n(weights_.back()->data(), weights_.back()->size(), IP.size_[2] != 1 ? 276.0 : 184.0);
        weights_.push_back(std::make_shared<Grid<double>>(n_vec_, d_) );
        std::fill_n(weights_.back()->data(), weights_.back()->size(), IP.size_[2] != 1 ? 276.0 : 184.0);

        for(auto& obj : objArr_)
        {
            double dbAdd    = obj->useOrdDip() ? 189.0 : 169.0;
            double dbChiAdd = obj->useOrdDip() ? 908.0 : 464.0;
            for(int ii = 0; ii < n_vec_[0]; ++ii)
            {
                for(int jj = 0; jj < n_vec_[1]; ++jj)
                {
                    for(int kk = 0; kk < n_vec_[2]; ++kk)
                    {
                        // Ex points located at ii+1/2, jj
                        pt[0] = (ii-(n_vec_[0]-1)/2.0+0.5)*d_[0];
                        pt[1] = (jj-(n_vec_[1]-1)/2.0    )*d_[1];
                        pt[2] = (kk-(n_vec_[2]-1)/2.0    )*d_[2];
                        if(obj->isObj(pt,d_[0], obj->geo() ) )
                            weights_[0]->point(ii,jj,kk) += dbAdd*(obj->gamma().size() + obj->magGamma().size() ) + dbChiAdd*( obj->chiGamma().size() ) + ( obj->pols().size() > 0 ? 30.0 : 0.0);
                        // Ey points located ii, jj+1/2
                        pt[1] += 0.5*d_[1];
                        pt[0] -= 0.5*d_[0];
                        if(obj->isObj(pt,d_[0], obj->geo() ) )
                            weights_[1]->point(ii,jj,kk) += dbAdd*(obj->gamma().size() + obj->magGamma().size() ) + dbChiAdd*( obj->chiGamma().size() ) + ( obj->pols().size() > 0 ? 30.0 : 0.0);
                        // Ez point is at ii, jj
                        pt[1] -= 0.5*d_[1];
                        pt[2] += 0.5*d_[2];
                        if(obj->isObj(pt,d_[0], obj->geo() ) )
                            weights_[2]->point(ii,jj,kk) += dbAdd*(obj->gamma().size() + obj->magGamma().size() ) + dbChiAdd*( obj->chiGamma().size() ) + ( obj->pols().size() > 0 ? 30.0 : 0.0);
                    }
                }
            }
        }

        // Add PMLs
        for(int xx = 0; xx < IP.pmlThickness_[0]; ++xx)
        {
            // Ex field has no PML along the left and right
            for(int yy = 0; yy < weights_[0]->y(); ++ yy)
            {
                daxpy_(weights_[1]->z(), 1.0, std::vector<double>(n_vec_[2], 79.0).data(), 1, &weights_[1]->point(            xx, yy, 0), weights_[1]->x());
                daxpy_(weights_[2]->z(), 1.0, std::vector<double>(n_vec_[2], 79.0).data(), 1, &weights_[2]->point(            xx, yy, 0), weights_[2]->x());

                daxpy_(weights_[1]->z(), 1.0, std::vector<double>(n_vec_[2], 79.0).data(), 1, &weights_[1]->point(n_vec_[0]-1-xx, yy, 0), weights_[1]->x());
                daxpy_(weights_[2]->z(), 1.0, std::vector<double>(n_vec_[2], 79.0).data(), 1, &weights_[2]->point(n_vec_[0]-1-xx, yy, 0), weights_[2]->x());
            }
        }
        std::vector<double> PMLweight(n_vec_[0]*n_vec_[2], 79.0);
        for(int yy = 0; yy < IP.pmlThickness_[1]; ++yy)
        {
            // Ey field has no PML along the top and bottom
            std::transform(PMLweight.begin(), PMLweight.end(), &weights_[0]->point(0,             yy, 0), &weights_[0]->point(0,             yy, 0), std::plus<double>() );
            std::transform(PMLweight.begin(), PMLweight.end(), &weights_[2]->point(0,             yy, 0), &weights_[2]->point(0,             yy, 0), std::plus<double>() );

            std::transform(PMLweight.begin(), PMLweight.end(), &weights_[0]->point(0, n_vec_[1]-1-yy, 0), &weights_[0]->point(0, n_vec_[1]-1-yy, 0), std::plus<double>() );
            std::transform(PMLweight.begin(), PMLweight.end(), &weights_[2]->point(0, n_vec_[1]-1-yy, 0), &weights_[2]->point(0, n_vec_[1]-1-yy, 0), std::plus<double>() );
        }
        if(IP.size_[2] > 0)
        {
            for(int zz = 0; zz < IP.pmlThickness_[2]; ++zz)
            {
                // Ez field has no PML along the front/back
                for(int yy = 0; yy < weights_[0]->y(); ++ yy)
                {
                    std::transform(PMLweight.begin(), PMLweight.begin()+weights_[0]->x(), &weights_[0]->point(0, yy,             zz), &weights_[0]->point(0, yy,             zz), std::plus<double>() );
                    std::transform(PMLweight.begin(), PMLweight.begin()+weights_[1]->x(), &weights_[1]->point(0, yy,             zz), &weights_[1]->point(0, yy,             zz), std::plus<double>() );

                    std::transform(PMLweight.begin(), PMLweight.begin()+weights_[0]->x(), &weights_[0]->point(0, yy, n_vec_[2]-1-zz), &weights_[0]->point(0, yy, n_vec_[2]-1-zz), std::plus<double>() );
                    std::transform(PMLweight.begin(), PMLweight.begin()+weights_[1]->x(), &weights_[1]->point(0, yy, n_vec_[2]-1-zz), &weights_[1]->point(0, yy, n_vec_[2]-1-zz), std::plus<double>() );

                }
            }
        }
        // Computational cost of flux for 2D calcs is small so it is neglected.
        if(IP.size_[2] > 0)
        {
            // Include weights for flux regions
            for(int ff = 0; ff < IP.fluxLoc_.size(); ++ff)
            {
                std::vector<double> fluxWeight( std::max( std::max(n_vec_[0], n_vec_[1]), n_vec_[2] ), IP.fluxFreqList_[ff].size()*3 + 20.0 );
                for(int yy = 0; yy < IP.fluxSz_[ff][1]; ++yy)
                {
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[0]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[1]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[2]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), 1);

                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[0]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]+IP.fluxSz_[ff][2]-1), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[1]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]+IP.fluxSz_[ff][2]-1), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[2]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]+IP.fluxSz_[ff][2]-1), 1);

                    daxpy_(IP.fluxSz_[ff][2], 1.0, fluxWeight.data(), 1, &weights_[0]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), 1);
                    daxpy_(IP.fluxSz_[ff][2], 1.0, fluxWeight.data(), 1, &weights_[1]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), 1);
                    daxpy_(IP.fluxSz_[ff][2], 1.0, fluxWeight.data(), 1, &weights_[2]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), 1);

                    daxpy_(IP.fluxSz_[ff][2], 1.0, fluxWeight.data(), 1, &weights_[0]->point(IP.fluxLoc_[ff][0]+IP.fluxSz_[ff][0]-1, IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), weights_[0]->x());
                    daxpy_(IP.fluxSz_[ff][2], 1.0, fluxWeight.data(), 1, &weights_[1]->point(IP.fluxLoc_[ff][0]+IP.fluxSz_[ff][0]-1, IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), weights_[1]->x());
                    daxpy_(IP.fluxSz_[ff][2], 1.0, fluxWeight.data(), 1, &weights_[2]->point(IP.fluxLoc_[ff][0]+IP.fluxSz_[ff][0]-1, IP.fluxLoc_[ff][1]+yy, IP.fluxLoc_[ff][2]), weights_[2]->x());
                }
                for(int zz = 0; zz < IP.fluxSz_[ff][2]; ++zz)
                {
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[0]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1], IP.fluxLoc_[ff][2]+zz), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[1]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1], IP.fluxLoc_[ff][2]+zz), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[2]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1], IP.fluxLoc_[ff][2]+zz), 1);

                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[0]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+IP.fluxSz_[ff][1]-1, IP.fluxLoc_[ff][2]+zz), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[1]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+IP.fluxSz_[ff][1]-1, IP.fluxLoc_[ff][2]+zz), 1);
                    daxpy_(IP.fluxSz_[ff][0], 1.0, fluxWeight.data(), 1, &weights_[2]->point(IP.fluxLoc_[ff][0], IP.fluxLoc_[ff][1]+IP.fluxSz_[ff][1]-1, IP.fluxLoc_[ff][2]+zz), 1);
                }
            }
        }
    }

    /**
     * @brief return the current time
     * @return tcur_
     */
    inline double getTime(){return tcur_;}

    /**
     * @brief      steps the propagator forward one unit in time
     */
    void step()
    {
        updateMagH();
        updateChiH();

        // Update H/B fields
        updateB();
        updateH();

        // Add the H incident field before stepping tfsf objects (like H updates) and then add the E incd (like normal updates)
        for(auto & tfsf : tfsfArr_)
        {
            H_incd_[0][2*t_step_+0] = tfsf->get_incd_Hx();
            H_incd_[0][2*t_step_+1] = tfsf->get_incd_Hx_off();
            H_incd_[1][2*t_step_+0] = tfsf->get_incd_Hy();
            H_incd_[1][2*t_step_+1] = tfsf->get_incd_Hy_off();
            H_incd_[2][2*t_step_+0] = tfsf->get_incd_Hz();
            H_incd_[2][2*t_step_+1] = tfsf->get_incd_Hz_off();

            tfsf->updateFields();

            E_incd_[0][2*t_step_+0] = tfsf->get_incd_Ex();
            E_incd_[0][2*t_step_+1] = tfsf->get_incd_Ex_off();
            E_incd_[1][2*t_step_+0] = tfsf->get_incd_Ey();
            E_incd_[1][2*t_step_+1] = tfsf->get_incd_Ey_off();
            E_incd_[2][2*t_step_+0] = tfsf->get_incd_Ez();
            E_incd_[2][2*t_step_+1] = tfsf->get_incd_Ez_off();
        }

        // Include PML updates before transferring from B to H
        for(int ii = 0; ii < 3; ++ii)
            updateHPML_[ii](HPML_[ii]);

        for(auto & src :srcArr_)
            src->addPul(tcur_);

        B2H();

        // Transfer PBC and MPI related H field information
        applBCH_[0](H_[0], k_point_, ln_vec_[0]  , yHPBC_[0], zMaxPBC_-1, ln_vec_[0]+1, yHPBC_[0], zMinPBC_, zMaxPBC_  , d_[0], d_[1], d_[2] );
        applBCH_[1](H_[1], k_point_, ln_vec_[0]-1, yHPBC_[1], zMaxPBC_-1, ln_vec_[0]  , yHPBC_[1], zMinPBC_, zMaxPBC_  , d_[0], d_[1], d_[2] );
        applBCH_[2](H_[2], k_point_, ln_vec_[0]-1, yHPBC_[2], zMaxPBC_  , ln_vec_[0]  , yHPBC_[2], zMinPBC_, zMaxPBC_+1, d_[0], d_[1], d_[2] );

        // Update all polarization terms to the H field
        updatePolE();
        updateChiE();
        // Update E.D fields
        updateD();
        updateE();

        // Include PML updates before transferring from D to E
        for(int ii = 0; ii < 3; ++ii)
            updateEPML_[ii](EPML_[ii]);
        D2E();
        for(auto & qe : qeArr_)
            qe->addQE();
        // All E-field Updates should now be completed transfer border values for the E-fields
        applBCE_[0](E_[0], k_point_, ln_vec_[0]-1, yEPBC_[0], zMaxPBC_  , ln_vec_[0]  , yEPBC_[0], zMinPBC_, zMaxPBC_+1, d_[0], d_[1], d_[2]);
        applBCE_[1](E_[1], k_point_, ln_vec_[0]  , yEPBC_[1], zMaxPBC_  , ln_vec_[0]+1, yEPBC_[1], zMinPBC_, zMaxPBC_+1, d_[0], d_[1], d_[2]);
        applBCE_[2](E_[2], k_point_, ln_vec_[0]  , yEPBC_[2], zMaxPBC_-1, ln_vec_[0]+1, yEPBC_[2], zMinPBC_, zMaxPBC_  , d_[0], d_[1], d_[2]);

        // Increment time steps to before output as all fields should be updated to the next time step now
        tcur_ += dt_;
        ++t_step_;

        // Output all detector values
        for(auto & dtc : dtcArr_)
            if(t_step_ % dtc->timeInt() == 0)
                dtc->output(tcur_);
        for(auto & dtc : dtcFreqArr_)
            if(t_step_ % dtc->timeInt() == 0)
                dtc->output(tcur_);
        for(auto & flux : fluxArr_)
            if(t_step_ % flux->timeInt() == 0)
                flux->fieldIn(tcur_);
    }

    /**
     * @brief      Updates the H fields forward in time
     */
    void updateH()
    {
        for(int ii = 0; ii < 3; ++ii)
            for(auto& list : upH_[ii])
                upHFxn_[ii](std::get<0>(list), std::get<1>(list), H_[ii], E_[(ii+1)%3], E_[(ii+2)%3]);
    }

    /**
     * @brief      Updates the E fields forward in time
     */
    void updateE()
    {
        for(int ii = 0; ii < 3; ++ii)
            for(auto& list : upE_[ii])
                upEFxn_[ii](std::get<0>(list), std::get<1>(list), E_[ii], H_[(ii+1)%3], H_[(ii+2)%3]);
    }

    /**
     * @brief      Updates the B fields forward in time
     */
    void updateB()
    {
        for(int ii = 0; ii < 3; ++ii)
            for(auto& list : upB_[ii])
                upHFxn_[ii](std::get<0>(list), std::get<1>(list), B_[ii], E_[(ii+1)%3], E_[(ii+2)%3]);
    }

    /**
     * @brief      Updates the D fields forward in time
     */
    void updateD()
    {
        for(int ii = 0; ii < 3; ++ii)
            for(auto& list : upD_[ii])
                upEFxn_[ii](std::get<0>(list), std::get<1>(list), D_[ii], H_[(ii+1)%3], H_[(ii+2)%3]);
    }

    /**
     * @brief      Updates the polarization fields and adds them to the D field to get the E field for electrically dispersive materials
     */
    void updatePolE()
    {
        for(auto& list : upOrDipP_)
        {
            int objInd = std::get<0>(list)[5];
            upLorOrDipP_(std::get<0>(list), E_[0], E_[1], E_[2], orDipLorP_[0], orDipLorP_[1], orDipLorP_[2], prevOrDipLorP_[0], prevOrDipLorP_[1], prevOrDipLorP_[2], dipP_[0], dipP_[1], dipP_[2], objArr_[ objInd ]->alpha(), objArr_[ objInd ]->xi(),  objArr_[ objInd ]->gamma(), scratchx_.data(), scratchy_.data(), scratchz_.data(), scratchOrDipUDeriv_.data(), scratchDipDotU_.data());
        }
        for(int ii = 0; ii < 3; ++ii)
        {
            for(auto& list : upLorD_[ii])
            {
                int objInd = std::get<0>(list)[5];
                upLorPFxn_[ii]( std::get<0>(list), E_[ii], lorP_[ii], prevLorP_[ii], objArr_[ objInd ]->alpha(), objArr_[ objInd ]->xi(), objArr_[ objInd ]->gamma(), scratchx_.data());
            }
            for( auto& grid : orDipLorP_[ii] )
                applBCOrDip_(grid, k_point_, ln_vec_[0], ln_vec_[1], zMaxPBC_, ln_vec_[0]+1, ln_vec_[1]+1, zMinPBC_, zMaxPBC_+1, d_[0], d_[1], d_[2] );
        }
    }

    /**
     * @brief      Updates the magnetization fields and adds them to the B field to get the H field for magnetically dispersive materials
     */
    void updateMagH()
    {
        for(auto& list : upOrDipM_)
        {
            int objInd = std::get<0>(list)[5];
            upLorOrDipM_(std::get<0>(list), H_[0], H_[1], H_[2], orDipLorM_[0], orDipLorM_[1], orDipLorM_[2], prevOrDipLorM_[0], prevOrDipLorM_[1], prevOrDipLorM_[2], dipM_[0], dipM_[1], dipM_[2], objArr_[ objInd ]->magAlpha(), objArr_[ objInd ]->magXi(),  objArr_[ objInd ]->magGamma(), scratchx_.data(), scratchy_.data(), scratchz_.data(), scratchOrDipUDeriv_.data(), scratchDipDotU_.data());
        }
        for(int ii = 0; ii < 3; ++ii)
        {
            for(auto& list : upLorB_[ii])
            {
                int objInd = std::get<0>(list)[5];
                upLorMFxn_[ii]( std::get<0>(list), H_[ii], lorM_[ii], prevLorM_[ii], objArr_[ objInd ]->magAlpha(), objArr_[ objInd ]->magXi(), objArr_[ objInd ]->magGamma(), scratchx_.data());
            }
            for(auto& grid : orDipLorM_[ii])
                applBCOrDip_(grid, k_point_, ln_vec_[0]-1, ln_vec_[1]-1, zMaxPBC_-1, ln_vec_[0], ln_vec_[1], zMinPBC_, zMaxPBC_  , d_[0], d_[1], d_[2] );
        }
    }

    /**
     * @brief Updates the electric chiral and achiral polarization and adds them to the D field to get the E field
     */
    void updateChiE()
    {
        for(auto& list : upChiOrDipP_)
        {
            int objInd = std::get<0>(list)[5];
            upLorOrDipP_(std::get<0>(list), E_[0], E_[1], E_[2], orDipLorP_[0], orDipLorP_[1], orDipLorP_[2], prevOrDipLorP_[0], prevOrDipLorP_[1], prevOrDipLorP_[2], dipP_[0], dipP_[1], dipP_[2], objArr_[ objInd ]->alpha(), objArr_[ objInd ]->xi(),  objArr_[ objInd ]->gamma(), scratchx_.data(), scratchy_.data(), scratchz_.data(), scratchOrDipUDeriv_.data(), scratchDipDotU_.data());
            upChiLorOrDipP_( std::get<0>(list), H_[0], prevH_[0], H_[1], prevH_[1], H_[2], prevH_[2], chiOrDipLorP_[0], chiOrDipLorP_[1], chiOrDipLorP_[2], prevChiOrDipLorP_[0], prevChiOrDipLorP_[1], prevChiOrDipLorP_[2], dipChiM_[0], dipChiM_[1], dipChiM_[2], dipP_[0], dipP_[1], dipP_[2], objArr_[ objInd ]->chiAlpha(), objArr_[ objInd ]->chiXi(), objArr_[ objInd ]->chiGamma(), objArr_[ objInd ]->chiGammaPrev(), scratchx_.data(), scratchy_.data(), scratchz_.data(), scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );
        }
        for(int ii = 0; ii < 3; ++ii)
        {
            for(auto& list : upChiD_[ii])
            {
                int objInd = std::get<0>(list)[5];
                upLorPFxn_[ii]( std::get<0>(list), E_[ii], lorP_[ii], prevLorP_[ii], objArr_[ objInd ]->alpha(), objArr_[ objInd ]->xi(), objArr_[ objInd ]->gamma(), scratchz_.data());
                upChiE_[ii]( std::get<0>(list), H_[ii], prevH_[ii], lorChiHP_[ii], prevLorChiHP_[ii], objArr_[ objInd ]->chiAlpha(), objArr_[ objInd ]->chiXi(), objArr_[ objInd ]->chiGamma(), objArr_[ objInd ]->chiGammaPrev(), scratchz_.data() );
            }
            for( auto& grid : chiOrDipLorP_[ii] )
                applBCOrDip_(grid, k_point_, ln_vec_[0], ln_vec_[1], zMaxPBC_, ln_vec_[0]+1, ln_vec_[1]+1, zMinPBC_, zMaxPBC_+1, d_[0], d_[1], d_[2] );
        }
        for(auto& list : copy2PrevFields_)
        {
            // Copy the H fields to the prevE fields for the next time step
            for(int ii = 0; ii < 3; ++ii)
                std::copy_n(&H_[ii]->point(list[1], list[2], list[3]), list[0], &prevH_[ii]->point(list[1], list[2], list[3]) );
        }
    }

    /**
     * @brief Updates the electric chiral and achiral magnetization and adds them to the B field to get the H field
     */
    void updateChiH()
    {
        for(auto& list : upChiOrDipM_)
        {
            int objInd = std::get<0>(list)[5];
            upLorOrDipM_(std::get<0>(list), H_[0], H_[1], H_[2], orDipLorM_[0], orDipLorM_[1], orDipLorM_[2], prevOrDipLorM_[0], prevOrDipLorM_[1], prevOrDipLorM_[2], dipM_[0], dipM_[1], dipM_[2], objArr_[ objInd ]->magAlpha(), objArr_[ objInd ]->magXi(),  objArr_[ objInd ]->magGamma(), scratchx_.data(), scratchy_.data(), scratchz_.data(), scratchOrDipUDeriv_.data(), scratchDipDotU_.data());
            upChiLorOrDipM_( std::get<0>(list), E_[0], prevE_[0], E_[1], prevE_[1], E_[2], prevE_[2], chiOrDipLorM_[0], chiOrDipLorM_[1], chiOrDipLorM_[2], prevChiOrDipLorM_[0], prevChiOrDipLorM_[1], prevChiOrDipLorM_[2], dipChiP_[0], dipChiP_[1], dipChiP_[2], dipM_[0], dipM_[1], dipM_[2], objArr_[ objInd ]->chiAlpha(), objArr_[ objInd ]->chiXi(), objArr_[ objInd ]->chiGamma(), objArr_[ objInd ]->chiGammaPrev(), scratchx_.data(), scratchy_.data(), scratchz_.data(), scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );
        }
        for(int ii = 0; ii < 3; ++ii)
        {
            for(auto& list : upChiB_[ii])
            {
                int objInd = std::get<0>(list)[5];
                upLorMFxn_[ii]( std::get<0>(list), H_[ii], lorM_[ii], prevLorM_[ii], objArr_[ objInd ]->magAlpha(), objArr_[ objInd ]->magXi(), objArr_[ objInd ]->magGamma(), scratchz_.data());
                upChiH_[ii](std::get<0>(list), E_[ii], prevE_[ii], lorChiEM_[ii], prevLorChiEM_[ii], objArr_[ objInd ]->chiAlpha(), objArr_[ objInd ]->chiXi(), objArr_[ objInd ]->chiGamma(), objArr_[ objInd ]->chiGammaPrev(), scratchz_.data() );
            }
            for(auto& grid : chiOrDipLorM_[ii])
                applBCOrDip_(grid, k_point_, ln_vec_[0]-1, ln_vec_[1]-1, zMaxPBC_-1, ln_vec_[0], ln_vec_[1], zMinPBC_, zMaxPBC_  , d_[0], d_[1], d_[2] );
        }
        for(auto& list : copy2PrevFields_)
        {
            // Copy the E fields to the prevE fields for the next time step
            for(int ii = 0; ii < 3; ++ii)
                std::copy_n(&E_[ii]->point(list[1], list[2], list[3]), list[0], &prevE_[ii]->point(list[1], list[2], list[3]) );
        }
    }

    /**
     * @brief      Updates the E filed given the D field values and current Polarizations
     */
    void D2E()
    {
        for(int ii = 0; ii < 3; ++ii)
        {
            for(auto& list : upLorD_[ii])
            {
                D2EFxn_[ii]( std::get<0>(list), std::get<1>(list)[3], D_[ii], E_[ii], lorP_[ii]);
            }
            for(auto& list : upChiD_[ii])
            {
                   D2EFxn_[ii](std::get<0>(list),      std::get<1>(list)[3], D_[ii], E_[ii], lorP_[ii]);
                chiD2EFxn_[ii](std::get<0>(list), -1.0*std::get<1>(list)[3], D_[ii], E_[ii], lorChiHP_[ii]);
            }
            for(auto& list : upOrDipD_[ii])
                orDipD2EFxn_[ii]( std::get<0>(list), std::get<1>(list)[3], D_[ii], E_[ii], orDipLorP_[ii].size(), orDipLorP_[ii], scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );
            for(auto& list : upOrDipChiD_[ii])
            {
                   orDipD2EFxn_[ii](std::get<0>(list),      std::get<1>(list)[3], D_[ii], E_[ii],    orDipLorP_[ii].size(),    orDipLorP_[ii], scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );
                chiOrDipD2EFxn_[ii](std::get<0>(list), -1.0*std::get<1>(list)[3], D_[ii], E_[ii], chiOrDipLorP_[ii].size(), chiOrDipLorP_[ii], scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );
            }
        }
    }

    /**
     * @brief      Updates the H filed given the B field values and Magnetizations
     */
    void B2H()
    {
        for(int ii = 0; ii < 3; ++ii)
        {
            for(auto& list : upLorB_[ii])
                B2HFxn_[ii]( std::get<0>(list), std::get<1>(list)[3], B_[ii], H_[ii], lorM_[ii]);

            for(auto& list : upChiB_[ii])
            {
                   B2HFxn_[ii](std::get<0>(list), std::get<1>(list)[3], B_[ii], H_[ii], lorM_[ii]);
                chiB2HFxn_[ii](std::get<0>(list), std::get<1>(list)[3], B_[ii], H_[ii], lorChiEM_[ii]);
            }

            for(auto& list : upOrDipB_[ii])
                orDipB2HFxn_[ii]( std::get<0>(list), std::get<1>(list)[3], B_[ii], H_[ii], orDipLorM_[ii].size(), orDipLorM_[ii], scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );

            for(auto& list : upOrDipChiB_[ii])
            {
                   orDipB2HFxn_[ii](std::get<0>(list), std::get<1>(list)[3], B_[ii], H_[ii],    orDipLorM_[ii].size(), orDipLorM_[ii], scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );
                chiOrDipB2HFxn_[ii](std::get<0>(list), std::get<1>(list)[3], B_[ii], H_[ii], chiOrDipLorM_[ii].size(), chiOrDipLorM_[ii], scratchOrDipUDeriv_.data(), scratchDipDotU_.data() );
            }
        }
    }

    /**
     * @brief      Creates a BMP image for a slice of the input maps
     *
     * @param[in]  IP          Input parameters object used to create the FDTD propagator
     * @param[in]  sliceDir    Direction of the normal vector of the plane a slice is being taken of
     * @param[in]  sliceCoord  The value of the slice in real space along the normal direction to output the slice
     */
    void convertInputs2Map(const parallelProgramInputs &IP, DIRECTION sliceDir, double sliceCoord)
    {
        // Only do the output if on the first process
        if(gridComm_->rank() != 0)
            return;
        // iterator is what counts up to make each object have its value
        double iterator = 1;
        // will be set based on the direction if x normal ii=x, jj=y, kk=z
        int cor_ii = -1, cor_jj = -1, cor_kk = -1;
        std::string fname;
        // set the coordinate indexes needed to complete the operation
        if(sliceDir == DIRECTION::X)
        {
            cor_ii = 0;
            cor_jj = 1;
            cor_kk = 2;
            if(!H_[2] || !E_[2])
                throw std::logic_error("Slice in an YZ plane is not possible for a 2D calculation.");
            fname = "InputMap_YZ_plane_" + std::to_string(sliceCoord) + ".bmp";
        }
        else if(sliceDir == DIRECTION::Y)
        {
            cor_ii = 1;
            cor_jj = 0; // Easier to keep x as jj even though it should be kk for x
            cor_kk = 2;
            if(!H_[2] || !E_[2])
                throw std::logic_error("Slice in an XZ plane is not possible for a 2D calculation.");
            fname = "InputMap_XZ_plane_" + std::to_string(sliceCoord) + ".bmp";
        }
        else if(sliceDir == DIRECTION::Z)
        {
            cor_ii = 2;
            cor_jj = 0;
            cor_kk = 1;
            if( (!H_[2] || !E_[2]) && sliceCoord != 0.0)
            {
                sliceCoord = 0.0;
            }
            fname = "InputMap_XY_plane_" + std::to_string(sliceCoord) + ".bmp";
        }
        else
            throw std::logic_error("Slice Direction must be X, Y, or Z");
        int sliceNum = static_cast<int>( (sliceCoord + IP.size_[cor_ii]/2.0 + 0.5*d_[cor_ii] ) / d_[cor_ii] ); // Do things in terms of grid points not actual values
        int map_nx = n_vec_[cor_jj];
        int map_ny = n_vec_[cor_kk];
        //construct map
        real_grid_ptr map = std::make_shared<Grid<double>>( std::array<int,3>({{map_nx, map_ny, 1}}) , std::array<double,3>({{ d_[cor_jj], d_[cor_kk], 1.0}}) );
        std::vector<double> ones(std::max(n_vec_[cor_jj], n_vec_[cor_kk]), 1);
        std::vector<double> includePML(ones.size(), 1.0);

        // PMLs regions initially set to 1
        for(int xx = 0; xx < IP.pmlThickness_[cor_jj]; ++xx)
        {
            dcopy_(map_ny, includePML.data(), 1, &map->point(xx             , 0 ), map->x() );
            dcopy_(map_ny, includePML.data(), 1, &map->point(map->x()-(1+xx), 0 ), map->x() );
        }
        for(int yy = 0; yy < IP.pmlThickness_[cor_kk]; ++yy)
        {
            dcopy_(map_nx, includePML.data(), 1, &map->point(0, yy              ), 1 );
            dcopy_(map_nx, includePML.data(), 1, &map->point(0, map->y()-(1+yy) ), 1 );
        }

        ++iterator;
        for(int dd = 0; dd < IP.dtcLoc_.size(); ++dd)
        {
            std::fill_n(ones.begin(), ones.size(), static_cast<double>(iterator) );
            if( sliceNum >= IP.dtcLoc_[dd][cor_ii] && sliceNum < IP.dtcSz_[dd][cor_ii] + IP.dtcLoc_[dd][cor_ii] )
            {
                if(IP.dtcSz_[dd][cor_jj] > 1)
                {
                    dcopy_(IP.dtcSz_[dd][cor_jj], ones.data(), 1, &map->point( IP.dtcLoc_[dd][cor_jj], IP.dtcLoc_[dd][cor_kk] ), 1);
                    if(IP.dtcSz_[dd][cor_kk] > 1)
                    {
                        dcopy_(IP.dtcSz_[dd][cor_jj]  , ones.data(), 1, &map->point( IP.dtcLoc_[dd][cor_jj]                          , IP.dtcLoc_[dd][cor_kk] + IP.dtcSz_[dd][cor_kk] - 1 ), 1);
                        dcopy_(IP.dtcSz_[dd][cor_kk]-2, ones.data(), 1, &map->point( IP.dtcLoc_[dd][cor_jj]                          , IP.dtcLoc_[dd][cor_kk] + 1)                         , map->x() );
                        dcopy_(IP.dtcSz_[dd][cor_kk]-2, ones.data(), 1, &map->point( IP.dtcLoc_[dd][cor_jj] + IP.dtcSz_[dd][cor_jj]-1, IP.dtcLoc_[dd][cor_kk] + 1)                         , map->x() );
                    }
                }
                else
                {
                    dcopy_(IP.dtcSz_[dd][cor_kk]  , ones.data(), 1, &map->point( IP.dtcLoc_[dd][cor_jj], IP.dtcLoc_[dd][cor_kk] ), map->x() );
                }
            }
            ++iterator;
        }
        // disregard the first object
        for(int tt = 0; tt < IP.tfsfLoc_.size(); ++tt)
        {
            std::fill_n(ones.begin(), ones.size(), static_cast<double>(iterator) );
            if( sliceNum >= IP.tfsfLoc_[tt][cor_ii] && sliceNum < IP.tfsfSize_[tt][cor_ii] + IP.tfsfLoc_[tt][cor_ii] )
            {
                if(IP.tfsfSize_[tt][cor_jj] > 1)
                {
                    dcopy_(IP.tfsfSize_[tt][cor_jj], ones.data(), 1, &map->point( IP.tfsfLoc_[tt][cor_jj], IP.tfsfLoc_[tt][cor_kk] ), 1);
                    if(IP.tfsfSize_[tt][cor_kk] > 1)
                    {
                        dcopy_(IP.tfsfSize_[tt][cor_jj]  , ones.data(), 1, &map->point( IP.tfsfLoc_[tt][cor_jj]                             , IP.tfsfLoc_[tt][cor_kk] + IP.tfsfSize_[tt][cor_kk] - 1 ), 1);
                        dcopy_(IP.tfsfSize_[tt][cor_kk]-2, ones.data(), 1, &map->point( IP.tfsfLoc_[tt][cor_jj]                             , IP.tfsfLoc_[tt][cor_kk] + 1)                            , map->x() );
                        dcopy_(IP.tfsfSize_[tt][cor_kk]-2, ones.data(), 1, &map->point( IP.tfsfLoc_[tt][cor_jj] + IP.tfsfSize_[tt][cor_jj]-1, IP.tfsfLoc_[tt][cor_kk] + 1)                            , map->x() );
                    }
                }
                else
                {
                    dcopy_(IP.tfsfSize_[tt][cor_kk]  , ones.data(), 1, &map->point( IP.tfsfLoc_[tt][cor_jj], IP.tfsfLoc_[tt][cor_kk] ), map->x() );
                }
            }
            ++iterator;
        }

        for(int ff = 0; ff < IP.fluxLoc_.size(); ++ff)
        {
            std::fill_n(ones.begin(), ones.size(), static_cast<double>(iterator) );
            if( sliceNum >= IP.fluxLoc_[ff][cor_ii] && sliceNum < IP.fluxSz_[ff][cor_ii] + IP.fluxLoc_[ff][cor_ii] )
            {
                if(IP.fluxSz_[ff][cor_jj] > 1)
                {
                    dcopy_(IP.fluxSz_[ff][cor_jj], ones.data(), 1, &map->point( IP.fluxLoc_[ff][cor_jj], IP.fluxLoc_[ff][cor_kk] ), 1);
                    if(IP.fluxSz_[ff][cor_kk] > 1)
                    {
                        dcopy_(IP.fluxSz_[ff][cor_jj]  , ones.data(), 1, &map->point( IP.fluxLoc_[ff][cor_jj]                          , IP.fluxLoc_[ff][cor_kk] + IP.fluxSz_[ff][cor_kk] - 1 ), 1);
                        dcopy_(IP.fluxSz_[ff][cor_kk]-2, ones.data(), 1, &map->point( IP.fluxLoc_[ff][cor_jj]                          , IP.fluxLoc_[ff][cor_kk] + 1)                         , map->x() );
                        dcopy_(IP.fluxSz_[ff][cor_kk]-2, ones.data(), 1, &map->point( IP.fluxLoc_[ff][cor_jj] + IP.fluxSz_[ff][cor_jj]-1, IP.fluxLoc_[ff][cor_kk] + 1)                         , map->x() );
                    }
                }
                else
                {
                    dcopy_(IP.fluxSz_[ff][cor_kk]  , ones.data(), 1, &map->point( IP.fluxLoc_[ff][cor_jj], IP.fluxLoc_[ff][cor_kk] ), map->x() );
                }
            }
            ++iterator;
        }

        for(int oo = 1; oo < objArr_.size(); ++oo)
        {
            // look at all local points only
            if( !objArr_[oo]->isVac() )
            {
                std::array<double,3>pt ={sliceCoord, sliceCoord, sliceCoord};
                for(int ii = 0; ii < map->x(); ++ii)
                {
                    for(int jj = 0; jj < map->y(); ++jj)
                    {
                        // split it up by component so you can see what goes where
                        pt[cor_jj] = ((ii)-(n_vec_[cor_jj]-1)/2.0)*d_[cor_jj];
                        pt[cor_kk] = ((jj)-(n_vec_[cor_kk]-1)/2.0)*d_[cor_kk];
                        if(objArr_[oo]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() )==true)
                            map->point(ii,jj) += static_cast<double>(iterator+oo)/3.0;

                        pt[cor_jj] = ((ii)+0.5-(n_vec_[cor_jj]-1)/2.0)*d_[cor_jj];
                        pt[cor_kk] = ((jj)-(n_vec_[cor_kk]-1)/2.0)*d_[cor_kk];
                        if(objArr_[oo]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() )==true)
                            map->point(ii,jj) += static_cast<double>(iterator+oo)/3.0;

                        pt[cor_jj] = ((ii)-(n_vec_[cor_jj]-1)/2.0)*d_[cor_jj];
                        pt[cor_kk] = ((jj)+0.5-(n_vec_[cor_kk]-1)/2.0)*d_[cor_kk];
                        if(objArr_[oo]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() )==true)
                            map->point(ii,jj) += static_cast<double>(iterator+oo)/3.0;
                    }
                }
            }
            else
            {
                for(int o1 = 1; o1 < oo; ++o1)
                {
                    if( objArr_[o1]->isVac() )
                        continue;
                    // If object is vacuum return to initial background values
                    std::array<double,3>pt ={sliceCoord, sliceCoord, sliceCoord};
                    for(int ii = 0; ii < map->x(); ++ii)
                    {
                        for(int jj = 0; jj < map->y(); ++jj)
                        {
                            pt[cor_jj] = ((ii)-(n_vec_[cor_jj]-1)/2.0)*d_[cor_jj];
                            pt[cor_kk] = ((jj)-(n_vec_[cor_kk]-1)/2.0)*d_[cor_kk];
                            if( objArr_[oo]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() ) && objArr_[o1]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() ))
                                map->point(ii,jj) -= static_cast<double>(iterator+o1)/3.0;

                            pt[cor_jj] = ((ii)+0.5-(n_vec_[cor_jj]-1)/2.0)*d_[cor_jj];
                            pt[cor_kk] = ((jj)-(n_vec_[cor_kk]-1)/2.0)*d_[cor_kk];
                            if( objArr_[oo]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() ) && objArr_[o1]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() ))
                                map->point(ii,jj) -= static_cast<double>(iterator+o1)/3.0;

                            pt[cor_jj] = ((ii)-(n_vec_[cor_jj]-1)/2.0)*d_[cor_jj];
                            pt[cor_kk] = ((jj)+0.5-(n_vec_[cor_kk]-1)/2.0)*d_[cor_kk];
                            if( objArr_[oo]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() ) && objArr_[o1]->isObj(pt,d_[cor_jj], objArr_[oo]->geo() ))
                                map->point(ii,jj) -= static_cast<double>(iterator+o1)/3.0;
                        }
                    }
                }
            }
        }
        // OUtput to bmp
        std::function<double(double)> outOpp = [](double a){return std::abs(a);};
        std::function<bool(double, double)> funcComp = [](double a, double b){ return std::abs(a) <= std::abs(b); };
        GridToBitMap(map, fname, outOpp, funcComp );
        return;
    }

    /**
     * @brief      Converts the double size vector into integers (number of grid points)
     *
     * @param[in]  size  The size of the field in real units
     *
     * @return     The size of the grid in number of grid points
     */
    inline std::array<int,3> toN_vec(std::array<double,3> size){ std::array<int,3> toRet; for(int ii = 0; ii < 3; ++ii) toRet[ii] = static_cast<int>(floor(size[ii]/d_[ii] + 0.5) ) + 1; return toRet; }

    /**
     * @brief      Accessor function for dt_
     *
     * @return     dt_
     */
    inline double dt(){return dt_;}

    /**
     * @brief      Accessor function to incident Hx field
     *
     * @return      H_incd_[0]
     */
    inline std::vector<cplx>& HxIncd(){return H_incd_[0];}
    /**
     * @brief      Accessor function to incident Hy field
     *
     * @return      H_incd_[1]
     */
    inline std::vector<cplx>& HyIncd(){return H_incd_[1];}
    /**
     * @brief      Accessor function to incident Hz field
     *
     * @return      H_incd_[2]
     */
    inline std::vector<cplx>& HzIncd(){return H_incd_[2];}

    /**
     * @brief      Accessor function to the incident Ex field
     *
     * @return     E_incd_[0]
     */
    inline std::vector<cplx>& ExIncd(){return E_incd_[0];}
    /**
     * @brief      Accessor function to the incident Ey field
     *
     * @return     E_incd_[1]
     */
    inline std::vector<cplx>& EyIncd(){return E_incd_[1];}
    /**
     * @brief      Accessor function to the incident Ez field
     *
     * @return     E_incd_[2]
     */
    inline std::vector<cplx>& EzIncd(){return E_incd_[2];}

    /**
     * @brief      Accessor function for fluxArr_
     *
     * @return     fluxArr_
     */
    inline std::vector<std::shared_ptr<parallelFluxDTC<T>>>& fluxArr(){return fluxArr_;}

    /**
     * @brief      Accessor function for dtcFreqArr_
     *
     * @return     dtcFreqArr_
     */
    inline std::vector<std::shared_ptr<parallelDetectorFREQ_Base<T>>>& dtcFreqArr() {return dtcFreqArr_;}

    inline std::vector<std::shared_ptr<parallelDetectorBase<T>>>& dtcArr() {return dtcArr_;}
    /**
     * @brief      Accessor function for qeArr_
     *
     * @return     qeArr_
     */
    inline std::vector<std::shared_ptr<parallelQEBase<T>>>& qeArr() {return qeArr_;}
};

class parallelFDTDFieldReal : public parallelFDTDFieldBase<double>
{
public:
    /**
     * @brief      Constructs a FDTD Propagator class
     *
     * @param[in]  IP        Input parameter object that read in values from a json input file
     * @param[in]  gridComm  A shared_ptr to the MPI interface for the calculation
     */
    parallelFDTDFieldReal(parallelProgramInputs &IP, std::shared_ptr<mpiInterface> gridComm);


    /**
     * @brief      Constructs a DTC based off of the input parameters and puts it in the proper detector vector
     *
     * @param[in]  c             class type of the dtc (bin, bmp, cout, txt, freq)
     * @param[in]  grid          vector of the fields that need to be outputted
     * @param[in]  SI            true if outputting in SI units
     * @param[in]  loc           The location of the detectors lower left corner in grid points
     * @param[in]  sz            The size of the detector in grid points
     * @param[in]  out_name      The output file name
     * @param[in]  fxn           Function used to modify base field data
     * @param[in]  txtType       if BMP what should be outputted to the text file
     * @param[in]  type          The type of the detector (Ex, Ey, Epow, etc)
     * @param[in]  freqList      The frequency list
     * @param[in]  timeInterval  The number of time steps per field output
     * @param[in]  a             unit length of the calculation
     * @param[in]  I0            unit current of the calculation
     * @param[in]  t_max         The time at the final time step
     */
    void coustructDTC(DTCCLASS c, std::vector<std::pair<real_pgrid_ptr, std::array<int,3> > > grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double t_start, double t_end, bool otputAvg, std::vector<double> freqList, double timeInterval, double a, double I0, double t_max, bool outputMaps);

};

class parallelFDTDFieldCplx : public parallelFDTDFieldBase<cplx>
{
public:
    /**
     * @brief      Constructs a FDTD Propagator class
     *
     * @param[in]  IP        Input parameter object that read in values from a json input file
     * @param[in]  gridComm  A shared_ptr to the MPI interface for the calculation
     */
    parallelFDTDFieldCplx(parallelProgramInputs &IP, std::shared_ptr<mpiInterface> gridComm);


    /**
     * @brief      Constructs a DTC based off of the input parameters and puts it in the proper detector vector
     *
     * @param[in]  c             class type of the dtc (bin, bmp, cout, txt, freq)
     * @param[in]  grid          vector of the fields that need to be outputted
     * @param[in]  SI            true if outputting in SI units
     * @param[in]  loc           The location of the detectors lower left corner in grid points
     * @param[in]  sz            The size of the detector in grid points
     * @param[in]  out_name      The output file name
     * @param[in]  fxn           Function used to modify base field data
     * @param[in]  txtType       if BMP what should be outputted to the text file
     * @param[in]  type          The type of the detector (Ex, Ey, Epow, etc)
     * @param[in]  freqList      The frequency list
     * @param[in]  timeInterval  The number of time steps per field output
     * @param[in]  a             unit length of the calculation
     * @param[in]  I0            unit current of the calculation
     * @param[in]  t_max         The time at the final time step
     */
    void coustructDTC(DTCCLASS c, std::vector<std::pair<cplx_pgrid_ptr, std::array<int,3> > > grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double t_start, double t_end, bool otputAvg, std::vector<double> freqList, double timeInterval, double a, double I0, double t_max, bool outputMaps);
};

/**
 * @brief      For making the QE it constructs all level combinations and stores them in a vector recursively
 *
 * @param[in]  AllELevs  input describing all of the energy level combinations for inhomo broadening
 * @param      eLevs     output vector to store all combinations
 * @param[in]  depth     how far into the level structure are we
 * @param[in]  current   The current vector being made (a single level set)
 * @param[in]  weight    The weight of the current level set
 */
void GenerateAllELevCombos(std::vector<EnergyLevelDiscriptor> AllELevs, std::vector<std::pair<std::vector<double>, double>>& eLevs, int depth, std::vector<double> current, double weight);

#endif
