/** @file SOURCE/parallelTFSF.hpp
 *  @brief Class creates a TFSF surface to introduce an incident pulse
 *
 *  A class used to introduce a plane wave from a TFSF surface centered at loc in a box the size of sz
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#ifndef FDTD_TFSF
#define FDTD_TFSF

#include <SOURCE/Pulse.hpp>
#include <PML/PML.hpp>
#include <unordered_map>
struct paramUpIncdField
{
    int indI_; //!< Index corresponding to the first point in the incd ifeld
    int indPosJ_; //!< Index corresponding to the first point in the incdJ field (positive contribution in the derivative)
    int indNegJ_; //!< Index corresponding to the first point in the incdJ field (negative contribution in the derivative)
    int indPosK_; //!< Index corresponding to the first point in the incdK field (positive contribution in the derivative)
    int indNegK_; //!< Index corresponding to the first point in the incdK field (negative contribution in the derivative)

    int nSz_; //!< Size of the transform/mkl operations
    double prefactor_j_; //!< prefactor for the incd J derivatives
    double prefactor_k_; //!< prefactor for the incd K derivatives

    std::array<int,2> indChi_;
    // std::array<int,8> indChi_;

    std::vector<double> aChiAlpha_;
    std::vector<double> aChiXi_;
    std::vector<double> aChiGamma_;

    std::vector<double> chiAlpha_;
    std::vector<double> chiXi_;
    std::vector<double> chiGamma_;
    std::vector<double> chiPrevGamma_;
};

/**


* @brief      Updates the incident field with only incd j field
 *
 * @param[in]  pulseVec     Vector to store the pulse values from 0 to mMax
 * @param[in]  pulAddStart  Index of the I field corresponding to where the pulse should be first added
 * @param[in]  prefact      The prefactor to add to the base pulse value
 * @param[in]  upList       The list of update parameters
 * @param[in]  incd_i       The incident field polarized along the i direction
 * @param[in]  incd_j       The incident field polarized along the j direction
 * @param[in]  incd_k       The incident field polarized along the k direction
 * @param[in]  pml_incd     The incident PML
 */
void updateIncdFieldJ (const std::vector<cplx>& pulseVec, int pulAddStart, cplx prefact, const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incd_i, cplx_grid_ptr incd_j, cplx_grid_ptr incd_k, std::shared_ptr<IncdCPMLCplx> pml_incd );

/**
 * @brief      Updates the incident field with only incd j field
 *
 * @param[in]  pulseVec     Vector to store the pulse values from 0 to mMax
 * @param[in]  pulAddStart  Index of the I field corresponding to where the pulse should be first added
 * @param[in]  prefact      The prefactor to add to the base pulse value
 * @param[in]  upList       The list of update parameters
 * @param[in]  incd_i       The incident field polarized along the i direction
 * @param[in]  incd_j       The incident field polarized along the j direction
 * @param[in]  incd_k       The incident field polarized along the k direction
 * @param[in]  pml_incd     The incident PML
 */
void updateIncdFieldK (const std::vector<cplx>& pulseVec, int pulAddStart, cplx prefact, const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incd_i, cplx_grid_ptr incd_j, cplx_grid_ptr incd_k, std::shared_ptr<IncdCPMLCplx> pml_incd );

/**
 * @brief      Updates the incident field with only incd j field
 *
 * @param[in]  pulseVec     Vector to store the pulse values from 0 to mMax
 * @param[in]  pulAddStart  Index of the I field corresponding to where the pulse should be first added
 * @param[in]  prefact      The prefactor to add to the base pulse value
 * @param[in]  upList       The list of update parameters
 * @param[in]  incd_i       The incident field polarized along the i direction
 * @param[in]  incd_j       The incident field polarized along the j direction
 * @param[in]  incd_k       The incident field polarized along the k direction
 * @param[in]  pml_incd     The incident PML
 */
void updateIncdFieldJK(const std::vector<cplx>& pulseVec, int pulAddStart, cplx prefact, const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incd_i, cplx_grid_ptr incd_j, cplx_grid_ptr incd_k, std::shared_ptr<IncdCPMLCplx> pml_incd );

/**
 * @brief      Function used to calculate the total pulse
 *
 * @param[in]  distVec   Vector storing the total distance from the initial source
 * @param[in]  pulses    Vector storing shared_ptrs to the pulses
 * @param[in]  tt        current time
 * @param      pulseVec  Vector storing all the pulse values
 */
void fillPulseVec(const std::vector<double>& distVec, std::vector<std::shared_ptr<Pulse>> pulses, double tt, std::vector<cplx>& pulseVec);

void updateIncdPols(const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incdU , std::vector<cplx_grid_ptr>& incdP, std::vector<cplx_grid_ptr>& incdPrevP, cplx* scratch );

void updateIncdChiPols(const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incdU, cplx_grid_ptr incdPrevU, std::vector<cplx_grid_ptr>& incdChiP, std::vector<cplx_grid_ptr>& incdPrevChiP, cplx* scratchP, cplx* scratchU );

void incdD2U( double chiFact, cplx_grid_ptr incdU, cplx_grid_ptr incdD, real_grid_ptr ep_mu, const std::vector<cplx_grid_ptr>& incdP, cplx* scratch );

struct paramStoreTFSF
{
    cplx_grid_ptr incdField_; //!< incident field to be added
    int incdStart_; //!< incident start for the j grid
    int strideIncd_; //!< stride for the incident grids
    int addIncdProp_; //!< value added to incd start as looping over transCor2
    int strideMain_; //!< stride for the real grids
    double prefactor_; //!< prefactor for the j grid
    std::array<int,2> szTrans_; //!< size of the transverse directions the j grid
    std::vector<int> indsU_; //!< vector storing the indexes of the starting point for the main grid values (U fields)
    std::vector<int> indsD_; //!< vector storing the indexes of the starting point for the main grid values (D fields)
};

/**
 * @brief TFSF region for introducing plane waves into the system
 */
template <typename T> class parallelTFSFBase
{
    typedef std::shared_ptr<parallelGrid<T>> pgrid_ptr;
    typedef std::function< void( const std::vector<cplx>&, int, cplx, const std::vector<paramUpIncdField>&, cplx_grid_ptr, cplx_grid_ptr, cplx_grid_ptr, std::shared_ptr<IncdCPMLCplx>) > updateIncdFieldFunction;
    typedef std::function< void( const std::vector<double>&, std::vector<std::shared_ptr<Pulse>>, double, std::vector<cplx>& ) > getPulVecFunction;
    typedef std::vector<std::shared_ptr<paramStoreTFSF>> tfsfSurfaceAddParamVec;
    typedef std::function< void( pgrid_ptr, pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, T* ) > tfsfSurfaceAddFxn;
    typedef std::function< void( pgrid_ptr, std::shared_ptr<paramStoreTFSF>, double) > chiCorrectFxn;

    typedef std::function< void(double, cplx_grid_ptr, cplx_grid_ptr, real_grid_ptr, const std::vector<cplx_grid_ptr>&, cplx*) > incdD2UFunction;
    typedef std::function< void(const std::vector<paramUpIncdField>&, cplx_grid_ptr, std::vector<cplx_grid_ptr>&, std::vector<cplx_grid_ptr>&, cplx* ) > updateIncdPFunction;
    typedef std::function< void(const std::vector<paramUpIncdField>&, cplx_grid_ptr, cplx_grid_ptr, std::vector<cplx_grid_ptr>&, std::vector<cplx_grid_ptr>&, cplx*, cplx* ) > updateIncdChiPFunction;
protected:
    std::shared_ptr<mpiInterface> gridComm_; //!< Communicator for MPI calls in the grid
    int mTot_; //!< sum of mx, my, mz
    int mMax_; //!< Maximum m value
    int gridLen_; //!< Length of the 1D auxiliary grid for the incident fields
    int originQuadrent_; //!< which quadrant the origin is in 1 is bot left increases in a counter clockwise direction
    int pmlThick_;  //!< incident PML thickness

    double phi_; //!< angle of plane wave's k-vector in the xy plane
    double theta_; //!< polar angle of the plane wave's k-vector
    double psi_; //!< angle describing the polarization of the plane wave from the vector \vec{k}\times e_{z} (if k is along z psi_ = phi_)
    double alpha_; //!< phase difference for circularly polarized light
    double phiPrefactCalc_; //!< phi used in the calculation of the prefactors (may change if 3D calculation)
    double psiPrefactCalc_; //!< psi used in the calculation of the prefactors (may change if chiral light)
    double dt_; //!< time step for the main Grids
    double dr_; //!< step size of 1D auxillary grids
    double t_step_; //!< current integer time step

    std::array<cplx,3> prefactH_; //!< prefactor for the {Hx, Hy, Hz} field
    std::array<cplx,3> prefactE_; //!< prefactor for the {Ex, Ey, Ez} field

    std::array<int, 3> sz_; //!< size of the total field region
    std::array<int, 3> loc_; //!< location of the tfsf region's lower, left corner
    std::array<int, 3> originLoc_; //!< location of the tfsf origin (a corner) based on the angle of propagation
    std::array<int, 3> m_; //!< array storing values for mx, my, mz
    std::array<double, 3> d_; //!< size of the total field region

    std::vector<cplx> pulseVec_; //!< tempoarily stores the pulse for source calculations

    std::array<std::vector<double>,3> distVecE_; //!< Stores the distance offSet for the {Ex, Ey, Ez} field
    std::array<std::vector<double>,3> distVecH_; //!< Stores the distance offSet for the {Hx, Hy, Hz} field

    std::vector<std::shared_ptr<Pulse>> pul_; //!< Pulse of incident wave

    std::array<pgrid_ptr,3> E_; //!< pgrid_ptr to the { Ex, Ey, Ez } field
    std::array<pgrid_ptr,3> H_; //!< pgrid_ptr to the { Hx, Hy, Hz } field

    std::array<pgrid_ptr,3> D_; //!< pgrid_ptr to the { Dx, Dy, Dz } field
    std::array<pgrid_ptr,3> B_; //!< pgrid_ptr to the { Bx, By, Bz } field

    std::array<cplx_grid_ptr,3> E_incd_; //!< TFSF incident { Ex, Ey, Ez } field
    std::array<cplx_grid_ptr,3> H_incd_; //!< TFSF incident { Hx, Hy, Hz } field

    std::array<cplx_grid_ptr,3> D_incd_; //!< TFSF incident { Dx, Dy, Dz } field
    std::array<cplx_grid_ptr,3> B_incd_; //!< TFSF incident { Bx, By, Bz } field

    std::array<std::vector<cplx_grid_ptr>,3> P_incd_; //!< TFSF incident { Px, Py, Pz } field
    std::array<std::vector<cplx_grid_ptr>,3> M_incd_; //!< TFSF incident { Mx, My, Mz } field

    std::array<std::vector<cplx_grid_ptr>,3> prevP_incd_; //!< TFSF incident { prevPx, prevPy, prevPz } field
    std::array<std::vector<cplx_grid_ptr>,3> prevM_incd_; //!< TFSF incident { prevMx, prevMy, prevMz } field

    std::array<cplx_grid_ptr,3> E_pulse_; //!< TFSF incident { Ex, Ey, Ez } field used to calculate their Fourier transform field
    std::array<cplx_grid_ptr,3> H_pulse_; //!< TFSF incident { Hx, Hy, Hz } field used to calculate their Fourier transform field

    std::array<real_grid_ptr,3> eps_; //!< the value of the dielectric constant for the incident { Ex, Ey, Ez } field at each grid point
    std::array<real_grid_ptr,3> mu_; //!< the value of the premeabilty constant for the incident { Hx, Hy, Hz } field at each grid point

    std::array<int_grid_ptr,3> physE_; //!< the value of the physical grid mapping constant for the incident { Ex, Ey, Ez } field at each grid point
    std::array<int_grid_ptr,3> physH_; //!< the value of the physical grid mapping constant for the incident { Hx, Hy, Hz } field at each grid point

    std::array<std::shared_ptr<IncdCPMLCplx>,3> pmlE_incd_; //!< TFSF incident PMLs for the {Ex, Ey, Ez}
    std::array<std::shared_ptr<IncdCPMLCplx>,3> pmlH_incd_; //!< TFSF incident PMLs for the {Hx, Hy, Hz}
    std::array<std::shared_ptr<IncdCPMLCplx>,3> pmlE_pulse_; //!< TFSF incident PMLs for the {Ex, Ey, Ez} fields used  to calculate the Fourier transformed the field
    std::array<std::shared_ptr<IncdCPMLCplx>,3> pmlH_pulse_; //!< TFSF incident PMLs for the {Hx, Hy, Hz} fields used  to calculate the Fourier transformed the field

    std::array<tfsfSurfaceAddParamVec, 3> eSurfaces_; //!< update parameters for the TFSF surfaces with an {Ex, Ey, Ez} component update
    std::array<tfsfSurfaceAddParamVec, 3> hSurfaces_; //!< update parameters for the TFSF surfaces with an {Hx, Hy, Hz} component update

    std::array<tfsfSurfaceAddParamVec, 3> chiECorrectSurfaces_; //!< Used to temporarily add/subtract the incident fields for correct chiral calculations
    std::array<tfsfSurfaceAddParamVec, 3> chiHCorrectSurfaces_; //!< Used to temporarily add/subtract the incident fields for correct chiral calculations

    std::array<tfsfSurfaceAddFxn, 3> addEIncd_; //!< Function to add the {Ex, Ey, Ez} incident fields
    std::array<tfsfSurfaceAddFxn, 3> addHIncd_; //!< Function to add the {Hx, Hy, Hz} incident fields
    std::array<updateIncdFieldFunction, 3> incdUpE_; //!< Function to update the {Ex, Ey, Ez} incident fields
    std::array<updateIncdFieldFunction, 3> incdUpH_; //!< Function to update the {Hx, Hy, Hz} incident fields
    std::array<getPulVecFunction, 3> pulAddE_; //!< Function to add the pulse values to the {Ex, Ey, Ez} incident fields
    std::array<getPulVecFunction, 3> pulAddH_; //!< Function to add the pulse values to the {Hx, Hy, Hz} incident fields
    std::array<updateIncdPFunction, 3> incdUpM_;
    std::array<updateIncdPFunction, 3> incdUpP_;
    std::array<incdD2UFunction, 3> B2H_;
    std::array<incdD2UFunction, 3> D2E_;

    std::array<std::vector<paramUpIncdField>, 3> upIncdParamE_; //!< Parameters used to update the incident {Ex, Ey, Ez} field
    std::array<std::vector<paramUpIncdField>, 3> upIncdParamH_; //!< Parameters used to update the incident {Hx, Hy, Hz} field
    std::array<std::vector<paramUpIncdField>, 3> upPulseParamE_; //!< Parameters used to update the incident {Ex, Ey, Ez} field used to calculate the Fourier transformed fields
    std::array<std::vector<paramUpIncdField>, 3> upPulseParamH_; //!< Parameters used to update the incident {Hx, Hy, Hz} field used to calculate the Fourier transformed fields

    std::vector<T> scratchEpMu_; //!< Temporary scratch space for ep_mu Calcs
    std::vector<cplx> scratchPIncd_; //!< Temporary scratch space for Polariztion and magnetizations
    std::vector<cplx> scratchUIncd_; //!< Temporary scratch space for Polariztion and magnetizations
public:
    std::function<const cplx()> get_incd_Hx; //!< function used to get the incident Hx field
    std::function<const cplx()> get_incd_Hy; //!< function used to get the incident Hy field
    std::function<const cplx()> get_incd_Hz; //!< function used to get the incident Hz field

    std::function<const cplx()> get_incd_Ex; //!< function used to get the incident Ex field
    std::function<const cplx()> get_incd_Ey; //!< function used to get the incident Ey field
    std::function<const cplx()> get_incd_Ez; //!< function used to get the incident Ez field

    std::function<const cplx()> get_incd_Hx_off; //!< function used to get the incident Hx field at an offset for averaging
    std::function<const cplx()> get_incd_Hy_off; //!< function used to get the incident Hy field at an offset for averaging
    std::function<const cplx()> get_incd_Hz_off; //!< function used to get the incident Hz field at an offset for averaging

    std::function<const cplx()> get_incd_Ex_off; //!< function used to get the incident Ex field at an offset for averaging
    std::function<const cplx()> get_incd_Ey_off; //!< function used to get the incident Ey field at an offset for averaging
    std::function<const cplx()> get_incd_Ez_off; //!< function used to get the incident Ez field at an offset for averaging

    /**
     * @brief      Construct a TFSF surface
     *
     * @param[in]  gridComm     The mpiInterface of the grids
     * @param[in]  loc          location of TFSF origin
     * @param[in]  sz           size of the total field region
     * @param[in]  theta        polar angle of the plane wave's k-vector
     * @param[in]  phi          angle of plane wave's k-vector in the xy plane
     * @param[in]  psi          angle describing the polarization of the plane wave from the vector \vec{k}\times e_{z} (if k is along z psi_ = phi_)
     * @param[in]  circPol      POLARIZATION::R if R polarized, L if L polarized, linear if anything else
     * @param[in]  kLenRelJ     ratio between the size of the axis oriented along psi to that perpendicular to it for elliptically polarized light
     * @param[in]  d            Array storing the grid spacing in all directions
     * @param[in]  m            Array storing the values of m for the direction of the tfsf propagation
     * @param[in]  dt           time step of the tfsf surface
     * @param[in]  pul          The pulse list used for the surface
     * @param[in]  E            array of pgrid_ptrs corresponding to the E field
     * @param[in]  H            array of pgrid_ptrs corresponding to the H field
     * @param[in]  physE        array of pgrid_ptrs to the map of materials for the E field
     * @param[in]  physH        array of pgrid_ptrs to the map of materials for the H field
     * @param[in]  objArr       A vector storing all the objects in the field
     * @param[in]  nomPMLThick  The nominal thickness for the incd pmls will be affected by sum of the m vector
     * @param[in]  pmlM         The incd pml scaling value for sigma and kapa
     * @param[in]  pmlMA        The incd pml scaling value for a
     * @param[in]  pmlAMax      The incd pml's maximum a value
     */
    parallelTFSFBase(std::shared_ptr<mpiInterface> gridComm, std::array<int,3> loc, std::array<int,3> sz, double theta, double phi, double psi, POLARIZATION circPol, double kLenRelJ, std::array<double,3> d, std::array<int,3> m, double dt, std::vector<std::shared_ptr<Pulse>> pul, std::array<pgrid_ptr,3> E, std::array<pgrid_ptr,3> H, std::array<pgrid_ptr,3> D, std::array<pgrid_ptr,3> B, std::array<int_pgrid_ptr,3> physE, std::array<int_pgrid_ptr,3> physH, std::vector<std::shared_ptr<Obj>> objArr, int nomPMLThick=20, double pmlM=3.0, double pmlMA=1.0, double pmlAMax=0.25) :
        gridComm_(gridComm),
        mTot_( std::accumulate(m.begin(), m.end(), 0, [](double tot, double m){return tot+std::abs(m); }) ),
        mMax_( std::max(*std::max_element(m.begin(), m.end() ), std::abs(*std::min_element(m.begin(), m.end() ) ) ) ),
        gridLen_( 2*mMax_ + mTot_ * std::accumulate(sz.begin(), sz.end(), 10+2*nomPMLThick) ),
        originQuadrent_(0),
        pmlThick_(nomPMLThick*mTot_),
        phi_(phi),
        theta_(theta),
        psi_(psi),
        alpha_(0.0),
        phiPrefactCalc_(phi),
        psiPrefactCalc_(psi),
        dt_(dt),
        dr_( -1 ),
        t_step_(0),
        prefactH_( std::array<cplx,3>( {1.0, 1.0, 1.0} ) ),
        prefactE_( std::array<cplx,3>( {1.0, 1.0, 1.0} ) ),
        sz_(sz),
        loc_(loc),
        originLoc_(loc),
        m_(m),
        d_(d),
        pulseVec_ (mTot_),
        distVecH_( std::array<std::vector<double>,3>( {std::vector<double>(mTot_), std::vector<double>(mTot_), std::vector<double>(mTot_) } ) ),
        distVecE_( std::array<std::vector<double>,3>( {std::vector<double>(mTot_), std::vector<double>(mTot_), std::vector<double>(mTot_) } ) ),
        pul_(pul),
        scratchEpMu_(2*gridLen_, 0.0),
        scratchPIncd_(2*gridLen_, 0.0),
        scratchUIncd_(2*gridLen_, 0.0),
        E_(E),
        H_(H),
        D_(D),
        B_(B)
    {
        int mInd = -1;
        if( ( m_[0] == 0 && m_[1] == 0 ) || ( m_[2] == 0 && m_[1] == 0 ) || ( m_[0] ==0 && m_[2] == 0 ) )
        {
            if(m_[0] != 0)
                mInd = 0;
            if(m_[1] != 0)
                mInd = 1;
            if(m_[2] != 0)
                mInd = 2;
        }
        // Check to make sure that]+ TFSF is an actual surface
        bool twoDim = (E_[2] && H_[2]) ? false : true;
        pgrid_ptr zField = E_[2] ? E_[2] : H_[2];
        // If TFSF is only a point in 2D and the calc is not a 1D replacement or a line in 3D throw an error since TFSF acts on a plane
        if( ( !twoDim && ( (sz_[0] <= 1 && sz_[1] <= 1) || (sz_[0] <= 1 && sz_[2] <= 1) || (sz_[1] <= 1 && sz_[2] <= 1) ) ) || ( twoDim && sz_[0] <=1 && sz_[1] <=1 && zField->local_x() != 3 && zField->local_y() != 3) )
        {
            throw std::logic_error("TFSF sources are not point/line sources sources");
        }
        // m vector is not {0,0,0}
        if(mTot_ == 0)
            throw std::logic_error("The m array must have at least one non-zero number");
        // Get dr_ and confirm that it is consistent for all directions
        std::array<double,3> drTemp = { d_[0]*std::cos(phi_)*sin(theta_)/static_cast<double>(m_[0]), d_[1]*std::sin(phi_)*std::sin(theta_)/static_cast<double>(m_[1]), d_[2]*std::cos(theta_)/static_cast<double>(m_[2]) };
        for(int ii = 0; ii < 3; ++ii)
        {
            if(m_[ii] != 0  )
                dr_ = drTemp[ii];
        }
        for(int ii = 0; ii < 3; ++ii)
        {
            if( m_[ii] != 0 && std::abs( drTemp[ii] - dr_ ) > 1e-13)
            {
                throw std::logic_error("p[ii]*d[ii]/m[ii] must equal dr for all ii in x, y, z");
            }
        }
        // Determine the origin quadrant, and correct the originLoc based on the size
        if(phi_ <= M_PI/2.0 && phi_ > 0)
        {
            originQuadrent_ = 1;
        }
        else if(phi_ <= M_PI && phi_ > M_PI/2.0)
        {
            originQuadrent_ = 2;
            originLoc_[0] += sz_[0]-1;
        }
        else if(phi_ <= 3*M_PI/2.0 && phi_ > M_PI)
        {
            originQuadrent_ = 3;
            originLoc_[1] += sz_[1]-1;
            originLoc_[0] += sz_[0]-1;
        }
        else if((phi_ <= 2*M_PI && phi_ > 3.0*M_PI/2.0) || phi_ == 0)
        {
            originQuadrent_ = 4;
            originLoc_[1] += sz_[1]-1;
        }
        if(theta > M_PI/2.0 || theta_ < 0.0)
        {
            originQuadrent_ *= -1;
            originLoc_[2] += sz_[2]-1;
        }

        // Check to see if angle of propgation has the correct sz dimenionality
        if(sz_[1] == 1 && m_[0] != 0 && m_[2] !=0 )
            throw std::logic_error("Angle of propgation has a component along the TFSF surface without any other surfaces to cancel it.");
        else if(sz_[0] == 1 && m_[2] != 0 && m_[1] !=0 )
            throw std::logic_error("Angle of propgation has a component along the TFSF surface without any other surfaces to cancel it.");
        else if(sz_[2] == 1 && !twoDim && m_[0] != 0 && m_[1] !=0 )
            throw std::logic_error("Angle of propgation has a component along the TFSF surface without any other surfaces to cancel it.");

        // Determine the prefactors
        if(theta_ == 0 || theta_ == M_PI)
            phiPrefactCalc_ = M_PI/2.0; // Phi does not matter when along z-axis set it to make it easier to calculate the field
        // Determine the psi_/phi_ and alpha_ to match the elliptical shape of the light
        if(circPol == POLARIZATION::R || circPol == POLARIZATION::L )
        {
            // Get alpha parameters
            double c = pow(kLenRelJ, 2.0);
            // phi/psi control the light polarization angle
            psiPrefactCalc_ = 0.5 * asin( sqrt( ( pow(cos(2*psi_),2)*4*c + pow( (1+c)*sin(2*psi_), 2.0) ) / pow(1.0+c, 2.0) ) );
            alpha_ = acos( ( (c - 1.0)*sin(2*psi_) ) / sqrt( pow(cos(2.0*psi_),2.0)*4.0*c + pow( (1.0+c)*sin(2.0*psi_), 2.0) ) );
            if(std::abs( std::tan(psi_) ) > 1)
                psiPrefactCalc_ = M_PI/2.0 - psiPrefactCalc_;
            if(circPol == POLARIZATION::R)
                alpha_ *= -1.0;
        }

        // Construct the incident fields and functions and distance vectors
        std::array<int,3> incdNVec = {{ gridLen_, 1, 1 }};
        std::array<double,3> incdD = {{ dr_, 1, 1 }};

        initializeIncdFields(H_[0], E_[1], E_[2], false, m_[1]+m_[2], mInd, 0, 1, 2, incdNVec, incdD, distVecH_[0], physH[0], objArr, H_incd_[0], B_incd_[0], M_incd_[0], prevM_incd_[0], H_pulse_[0],  mu_[0], physH_[0], upIncdParamH_[0], upPulseParamH_[0], incdUpH_[0], incdUpM_[0], B2H_[0], pulAddH_[0], get_incd_Hx, get_incd_Hx_off);
        initializeIncdFields(H_[1], E_[2], E_[0], false, m_[0]+m_[2], mInd, 1, 2, 0, incdNVec, incdD, distVecH_[1], physH[1], objArr, H_incd_[1], B_incd_[1], M_incd_[1], prevM_incd_[1], H_pulse_[1],  mu_[1], physH_[1], upIncdParamH_[1], upPulseParamH_[1], incdUpH_[1], incdUpM_[1], B2H_[1], pulAddH_[1], get_incd_Hy, get_incd_Hy_off);
        initializeIncdFields(H_[2], E_[0], E_[1], false, m_[0]+m_[1], mInd, 2, 0, 1, incdNVec, incdD, distVecH_[2], physH[2], objArr, H_incd_[2], B_incd_[2], M_incd_[2], prevM_incd_[2], H_pulse_[2],  mu_[2], physH_[2], upIncdParamH_[2], upPulseParamH_[2], incdUpH_[2], incdUpM_[2], B2H_[2], pulAddH_[2], get_incd_Hz, get_incd_Hz_off);

        initializeIncdFields(E_[0], H_[1], H_[2], true, m_[0]       , mInd, 0, 1, 2, incdNVec, incdD, distVecE_[0], physE[0], objArr, E_incd_[0], D_incd_[0], P_incd_[0], prevP_incd_[0], E_pulse_[0], eps_[0], physE_[0], upIncdParamE_[0], upPulseParamE_[0], incdUpE_[0], incdUpP_[0], D2E_[0], pulAddE_[0], get_incd_Ex, get_incd_Ex_off);
        initializeIncdFields(E_[1], H_[2], H_[0], true, m_[1]       , mInd, 1, 2, 0, incdNVec, incdD, distVecE_[1], physE[1], objArr, E_incd_[1], D_incd_[1], P_incd_[1], prevP_incd_[1], E_pulse_[1], eps_[1], physE_[1], upIncdParamE_[1], upPulseParamE_[1], incdUpE_[1], incdUpP_[1], D2E_[1], pulAddE_[1], get_incd_Ey, get_incd_Ey_off);
        initializeIncdFields(E_[2], H_[0], H_[1], true, m_[2]       , mInd, 2, 0, 1, incdNVec, incdD, distVecE_[2], physE[2], objArr, E_incd_[2], D_incd_[2], P_incd_[2], prevP_incd_[2], E_pulse_[2], eps_[2], physE_[2], upIncdParamE_[2], upPulseParamE_[2], incdUpE_[2], incdUpP_[2], D2E_[2], pulAddE_[2], get_incd_Ez, get_incd_Ez_off);
        if( eps_[0] && mu_[0] )
        {
            for(int ii = 0; ii < 3; ++ii)
            {
                scaleDistVecSpeed(eps_[ii], mu_[ii], distVecE_[ii]);
                scaleDistVecSpeed(eps_[ii], mu_[ii], distVecH_[ii]);
            }
        }
        else if(eps_[0])
        {
            scaleDistVecSpeed(eps_[0], mu_[2], distVecE_[0]);
            scaleDistVecSpeed(eps_[1], mu_[2], distVecE_[1]);
            scaleDistVecSpeed(eps_[0], mu_[2], distVecH_[2]);
        }
        else
        {
            scaleDistVecSpeed(eps_[2], mu_[0], distVecH_[0]);
            scaleDistVecSpeed(eps_[2], mu_[1], distVecH_[1]);
            scaleDistVecSpeed(eps_[2], mu_[0], distVecE_[2]);
        }
        // The polarization angle and circular prefactor for the j and k fields
        cplx psiJ = std::cos(psiPrefactCalc_);
        cplx psiK = std::sin(psiPrefactCalc_)*std::exp( cplx(0.0,alpha_) );
        prefactH_[0] = (      psiK * std::sin(phiPrefactCalc_) + psiJ*std::cos(theta_)*std::cos(phiPrefactCalc_) ) * dt_ / (mTot_ * dr_) / ( mu_[0] ? mu_[0]->point(pmlThick_ + mMax_ + 5*mTot_) : mu_[2]->point(pmlThick_ + mMax_ + 5*mTot_) );
        prefactH_[1] = ( -1.0*psiK * std::cos(phiPrefactCalc_) + psiJ*std::cos(theta_)*std::sin(phiPrefactCalc_) ) * dt_ / (mTot_ * dr_) / ( mu_[1] ? mu_[1]->point(pmlThick_ + mMax_ + 5*mTot_) : mu_[2]->point(pmlThick_ + mMax_ + 5*mTot_) );
        prefactH_[2] = ( -1.0*psiJ * std::sin(theta_) ) * dt_ / (mTot_ * dr_) / ( mu_[2] ? mu_[2]->point(pmlThick_ + mMax_ + 5*mTot_) :  mu_[0]->point(pmlThick_ + mMax_ + 5*mTot_) );

        prefactE_[0] = (      psiJ * std::sin(phiPrefactCalc_) - psiK*std::cos(theta_)*std::cos(phiPrefactCalc_) ) * dt_ / (mTot_ * dr_) / ( eps_[0] ? eps_[0]->point(pmlThick_ + mMax_ + 5*mTot_) : eps_[2]->point(pmlThick_ + mMax_ + 5*mTot_) );
        prefactE_[1] = ( -1.0*psiJ * std::cos(phiPrefactCalc_) - psiK*std::cos(theta_)*std::sin(phiPrefactCalc_) ) * dt_ / (mTot_ * dr_) / ( eps_[1] ? eps_[1]->point(pmlThick_ + mMax_ + 5*mTot_) : eps_[2]->point(pmlThick_ + mMax_ + 5*mTot_) );
        prefactE_[2] = (      psiK * std::sin(theta_) ) * dt_ / (mTot_ * dr_) / ( eps_[2] ? eps_[2]->point(pmlThick_ + mMax_ + 5*mTot_) : eps_[0]->point(pmlThick_ + mMax_ + 5*mTot_) );

        // Construct the surface update paramter structures and get the needed functions default pl value is determined by the propagation direction
        bool pl = ( theta_ == 0 || ( theta_ == M_PI/2.0 && std::abs(originQuadrent_) == 4 || std::abs(originQuadrent_) == 1  ) ) ? false :true;
        initializeIncdAddStructs(H_[0], E_incd_[1], E_incd_[2], physH_[0], physH[0],  mu_[0], objArr, DIRECTION::Y, DIRECTION::Z, 0, 1, 2, POLARIZATION::HX, POLARIZATION::EY, POLARIZATION::EZ, false, pl, mInd, hSurfaces_[0] );
        initializeIncdAddStructs(H_[1], E_incd_[2], E_incd_[0], physH_[1], physH[1],  mu_[1], objArr, DIRECTION::Z, DIRECTION::X, 1, 2, 0, POLARIZATION::HY, POLARIZATION::EZ, POLARIZATION::EX, false, pl, mInd, hSurfaces_[1] );
        initializeIncdAddStructs(H_[2], E_incd_[0], E_incd_[1], physH_[2], physH[2],  mu_[2], objArr, DIRECTION::X, DIRECTION::Y, 2, 0, 1, POLARIZATION::HZ, POLARIZATION::EX, POLARIZATION::EY, false, pl, mInd, hSurfaces_[2] );

        initializeIncdAddStructs(E_[0], H_incd_[1], H_incd_[2], physE_[0], physE[0], eps_[0], objArr, DIRECTION::Y, DIRECTION::Z, 0, 1, 2, POLARIZATION::EX, POLARIZATION::HY, POLARIZATION::HZ,  true, pl, mInd, eSurfaces_[0] );
        initializeIncdAddStructs(E_[1], H_incd_[2], H_incd_[0], physE_[1], physE[1], eps_[1], objArr, DIRECTION::Z, DIRECTION::X, 1, 2, 0, POLARIZATION::EY, POLARIZATION::HZ, POLARIZATION::HX,  true, pl, mInd, eSurfaces_[1] );
        initializeIncdAddStructs(E_[2], H_incd_[0], H_incd_[1], physE_[2], physE[2], eps_[2], objArr, DIRECTION::X, DIRECTION::Y, 2, 0, 1, POLARIZATION::EZ, POLARIZATION::HX, POLARIZATION::HY,  true, pl, mInd, eSurfaces_[2] );
        // Construct the incdPMLs
        int pmlThick = nomPMLThick*mTot_; // PML nominal thickness * mTot
        if(H_incd_[0])
        {
            pmlH_incd_ [0] = std::make_shared<IncdCPMLCplx>( B_incd_ [0], E_incd_ [1], E_incd_ [2], POLARIZATION::HX, pmlThick, m_, H_[0]->n_vec(), true, 1.0                  , 1.0                 , 1.0                  , 1.0                 , pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
            pmlH_pulse_[0] = std::make_shared<IncdCPMLCplx>( H_pulse_[0], E_pulse_[1], E_pulse_[2], POLARIZATION::HX, pmlThick, m_, H_[0]->n_vec(), true, eps_[2]->point(mMax_), mu_[0]->point(mMax_), eps_[2]->point(mMax_), mu_[0]->point(mMax_), pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
        }
        if(H_incd_[1])
        {
            pmlH_incd_ [1] = std::make_shared<IncdCPMLCplx>( B_incd_ [1], E_incd_ [2], E_incd_ [0], POLARIZATION::HY, pmlThick, m_, H_[1]->n_vec(), true, 1.0                  , 1.0                 , 1.0                  , 1.0                 , pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
            pmlH_pulse_[1] = std::make_shared<IncdCPMLCplx>( H_pulse_[1], E_pulse_[2], E_pulse_[0], POLARIZATION::HY, pmlThick, m_, H_[1]->n_vec(), true, eps_[2]->point(mMax_), mu_[1]->point(mMax_), eps_[2]->point(mMax_), mu_[1]->point(mMax_), pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
        }
        if(H_incd_[2])
        {
            pmlH_incd_ [2] = std::make_shared<IncdCPMLCplx>( B_incd_ [2], E_incd_ [0], E_incd_ [1], POLARIZATION::HZ, pmlThick, m_, H_[2]->n_vec(), true, 1.0                  , 1.0                 , 1.0                  , 1.0                 , pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
            pmlH_pulse_[2] = std::make_shared<IncdCPMLCplx>( H_pulse_[2], E_pulse_[0], E_pulse_[1], POLARIZATION::HZ, pmlThick, m_, H_[2]->n_vec(), true, eps_[0]->point(mMax_), mu_[2]->point(mMax_), eps_[0]->point(mMax_), mu_[2]->point(mMax_), pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
        }

        if(E_incd_[0])
        {
            pmlE_incd_ [0] = std::make_shared<IncdCPMLCplx>( D_incd_ [0], H_incd_ [1], H_incd_ [2], POLARIZATION::EX, pmlThick, m_, E_[0]->n_vec(), true, 1.0                  , 1.0                 , 1.0                  , 1.0                 , pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
            pmlE_pulse_[0] = std::make_shared<IncdCPMLCplx>( E_pulse_[0], H_pulse_[1], H_pulse_[2], POLARIZATION::EX, pmlThick, m_, E_[0]->n_vec(), true, eps_[0]->point(mMax_), mu_[2]->point(mMax_), eps_[0]->point(mMax_), mu_[2]->point(mMax_), pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
        }
        if(E_incd_[1])
        {
            pmlE_incd_ [1] = std::make_shared<IncdCPMLCplx>( D_incd_ [1], H_incd_ [2], H_incd_ [0], POLARIZATION::EY, pmlThick, m_, E_[1]->n_vec(), true, 1.0                  , 1.0                 , 1.0                  , 1.0                 , pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
            pmlE_pulse_[1] = std::make_shared<IncdCPMLCplx>( E_pulse_[1], H_pulse_[2], H_pulse_[0], POLARIZATION::EY, pmlThick, m_, E_[1]->n_vec(), true, eps_[1]->point(mMax_), mu_[2]->point(mMax_), eps_[1]->point(mMax_), mu_[2]->point(mMax_), pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
        }
        if(E_incd_[2])
        {
            pmlE_incd_ [2] = std::make_shared<IncdCPMLCplx>( D_incd_ [2], H_incd_ [0], H_incd_ [1], POLARIZATION::EZ, pmlThick, m_, E_[2]->n_vec(), true, 1.0                  , 1.0                 , 1.0                  , 1.0                 , pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
            pmlE_pulse_[2] = std::make_shared<IncdCPMLCplx>( E_pulse_[2], H_pulse_[0], H_pulse_[1], POLARIZATION::EZ, pmlThick, m_, E_[2]->n_vec(), true, eps_[2]->point(mMax_), mu_[0]->point(mMax_), eps_[2]->point(mMax_), mu_[0]->point(mMax_), pmlM, pmlMA, 1.0, pmlAMax, d_, dr_, dt_);
        }
    }

    /**
     * @brief      Initializes the incident fields and sets up all necessary functions and containers
     *
     * @param[in]  grid_i       shared_ptr to the main grid that the incident field is acting on
     * @param[in]  grid_j       shared_ptr to the main grid that the incident field is using to update in the k direction acting on
     * @param[in]  grid_k       shared_ptr to the main grid that the incident field is using to update in the j direction acting on
     * @param[in]  E            True if a field is an electric field
     * @param[in]  distBase     The base distance from the source terms for the field
     * @param[in]  mInd         Index corresponding to the axial direction of propagation if materials cross TFSF surfaces(-1 means non normal incidence and will error out)
     * @param[in]  mOffPosJ     Offset from incd field point to get the positive j field contribution for update derivatives
     * @param[in]  mOffNegJ     Offset from incd field point to get the negative j field contribution for update derivatives
     * @param[in]  mOffPosK     Offset from incd field point to get the positive k field contribution for update derivatives
     * @param[in]  mOffNegK     Offset from incd field point to get the negative k field contribution for update derivatives
     * @param[in]  dj           grid spacing in the j direction if i is the field polarization direction
     * @param[in]  dk           grid spacing in the k direction if i is the field polarization direction
     * @param[in]  incdNVec     The incd field n vector (number of grid points in all directions)
     * @param[in]  incdD        The grid spacing for the incident field
     * @param      distVec      Vector storing the distance for each point from the main source term
     * @param[in]  phys         Grid mapping the objects to the grid points
     * @param[in]  objArr       A vector storing all objects (inds corresponding to the maps)
     * @param      incdField    shared_ptr to the incident field
     * @param      pulseField   shared_ptr to the incident field used to calculate pule Fourier transforms
     * @param      ep_mu        shared_ptr to the gird storing the constant permivitty/permeability at each incd field grid point
     * @param      upList       The vector storing all the update parameters for the incident fields
     * @param      upPulseList  The vector storing all the update parameters for the incident fields used to calculate pulse Fourier transform
     * @param      upIncd       function pointer to the function used for incident field updating
     * @param      getPul       Function to get the pulse vector for the incident field
     * @param      getIncd      Function pointer used to get the incident field values
     * @param      getIncdOff   Function pointer used to get the incident field values at an offset for averaging
     */
    void initializeIncdFields(pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k, bool E, int distBase, int mInd, int i, int j, int k, std::array<int,3> incdNVec, std::array<double,3> incdD, std::vector<double>& distVec, int_pgrid_ptr phys, std::vector<std::shared_ptr<Obj>> objArr, cplx_grid_ptr& incdField, cplx_grid_ptr& incdDField, std::vector<cplx_grid_ptr>& incdPField, std::vector<cplx_grid_ptr>& incdPrevPField, cplx_grid_ptr& pulseField, real_grid_ptr& ep_mu, int_grid_ptr& incdPhys, std::vector<paramUpIncdField>& upList, std::vector<paramUpIncdField>& upPulseList, updateIncdFieldFunction& upIncd, updateIncdPFunction& upPIncd, incdD2UFunction& D2U, getPulVecFunction& getPul, std::function<const cplx()>& getIncd, std::function<const cplx()>& getIncdOff)
    {
        if(grid_i)
        {
            std::array<int, 3> fieldEnd = {0,0,0};
            if(E)
            {
                if(sz_[i] == grid_i->n_vec(i) - 2*gridComm_->npArr(i) )
                {
                    fieldEnd[i] = 1;
                }
            }
            else
            {
                if(sz_[j] == grid_i->n_vec(j) - 2*gridComm_->npArr(j) )
                {
                    fieldEnd[j] = 1;
                }
                if(sz_[k] == grid_i->n_vec(k) - 2*gridComm_->npArr(k) )
                {
                    fieldEnd[k] = 1;
                }
            }
            for(int rr = 0; rr < distVec.size(); ++rr)
                distVec[rr] = static_cast<double>( distBase + 2*rr )/2.0*dr_;
            incdField = std::make_shared<Grid<cplx>>(incdNVec, incdD);
            incdDField = std::make_shared<Grid<cplx>>(incdNVec, incdD);
            ep_mu = std::make_shared<Grid<double>>(incdNVec, incdD);
            incdPhys = std::make_shared<Grid<int>>(incdNVec, incdD);
            genEpMu(phys, E, fieldEnd, mInd, objArr, ep_mu, incdPhys);
            // std::fill_n(ep_mu->data(), ep_mu->size(), 1.0);
            genFieldUpParam(E, ep_mu, incdPhys, upList, objArr, i, j, k, mInd);
            int nPols = 0;
            for(auto& param : upList)
            {
                if(param.aChiGamma_.size() > nPols )
                    nPols = param.aChiGamma_.size();
            }
            incdPField = std::vector<cplx_grid_ptr>(nPols);
            incdPrevPField = std::vector<cplx_grid_ptr>(nPols);
            for(int nn = 0; nn < nPols; ++nn)
            {
                incdPField[nn] = std::make_shared<Grid<cplx>>(incdNVec, incdD);
                incdPrevPField[nn] = std::make_shared<Grid<cplx>>(incdNVec, incdD);
            }
            incdNVec[0] = 2*pmlThick_+20*mTot_;
            pulseField = std::make_shared<Grid<cplx>>(incdNVec, incdD);
            real_grid_ptr ep_mu_pul = std::make_shared<Grid<double>>(incdNVec, incdD);
            int_grid_ptr incdPhys_pul = std::make_shared<Grid<int>>(incdNVec, incdD);
            std::fill_n(ep_mu_pul->data(), ep_mu_pul->size(), ep_mu->point(pmlThick_ + mMax_ + 5*mTot_) );
            std::fill_n(incdPhys_pul->data(), incdPhys_pul->size(), incdPhys->point(pmlThick_ + mMax_ + 5*mTot_) );
            genFieldUpParam(E, ep_mu_pul, incdPhys_pul, upPulseList, objArr, i, j, k, mInd);

            if(grid_j && grid_k)
            {
                upIncd = updateIncdFieldJK;
            }
            else if(grid_j)
            {
                upIncd = updateIncdFieldJ;
            }
            else if(grid_k)
            {
                upIncd = updateIncdFieldK;
            }
            upPIncd = updateIncdPols;
            D2U = incdD2U;
            getPul = fillPulseVec;
            getIncd = [=](){return pulseField->point(pmlThick_ + mMax_ + 5*mTot_);};
            getIncdOff = [=](){return pulseField->point(pmlThick_ + mMax_ + 5*mTot_ - distBase%2);};
        }
        else
        {
            upIncd = []( const std::vector<cplx>&, int, cplx, const std::vector<paramUpIncdField>&, cplx_grid_ptr, cplx_grid_ptr, cplx_grid_ptr, std::shared_ptr<IncdCPMLCplx> ){return;};
            upPIncd = [](const std::vector<paramUpIncdField>&, cplx_grid_ptr, std::vector<cplx_grid_ptr>&, std::vector<cplx_grid_ptr>&, cplx* ) {return;} ;
            D2U = []( double, cplx_grid_ptr, cplx_grid_ptr, real_grid_ptr, const std::vector<cplx_grid_ptr>&, cplx* ) {return;} ;
            getPul = [](const std::vector<double>&, std::vector<std::shared_ptr<Pulse>>, double, std::vector<cplx>&){return;};
            getIncd = [](){return 0.0;};
            getIncdOff = [](){return 0.0;};
        }
    }

    /**
     * @brief      Function used to generate the permivitty and permeability grids for incident fields
     *
     * @param[in]  phys    Grid mapping out the objects to grid points
     * @param[in]  E       True if field is an electric field
     * @param[in]  indM    The index of propagation (if -1 it will error out as the filed is not normal incidence)
     * @param[in]  objArr  Vector storing all objects in the material
     * @param      ep_mu   The permivitty and permeability grids for incident fields
     */
    void genEpMu(int_pgrid_ptr phys, bool E, std::array<int,3> fieldEnd, int indM, std::vector<std::shared_ptr<Obj>> objArr, real_grid_ptr& ep_mu, int_grid_ptr incdPhys)
    {
        int off = E ? 0 : -1;
        int fieldOff = pmlThick_ + mMax_ + 5*mTot_;
        std::array<int,6> surInds = {2, 2, 1, 1, 0, 0};
        std::array<int,6> locOff  = {off, sz_[2], off, sz_[1], off, sz_[0]};
        int indStart = (E_[2] && H_[2]) ? 0 : 2;
        for(int ii = indStart; ii < 6; ++ii )
        {
            std::array<int,2> planeLoc = { ( (surInds[ii] != 2) ? loc_[(surInds[ii]+2)%3] :loc_[0] ), ( (surInds[ii] != 2) ? loc_[(surInds[ii]+1)%3] :loc_[1] ) };
            std::array<int,2> planeSz  = { ( (surInds[ii] != 2) ?  sz_[(surInds[ii]+2)%3] : sz_[0] ), ( (surInds[ii] != 2) ?  sz_[(surInds[ii]+1)%3] : sz_[1] ) };
            std::array<int,2> fieldSz  = { ( (surInds[ii] != 2) ?  phys->n_vec((surInds[ii]+2)%3) : phys->n_vec(0) ), ( (surInds[ii] != 2) ?  phys->n_vec((surInds[ii]+1)%3) : phys->n_vec(1) ) };
            if( ( planeSz[0] <= 1  && ( (E_[2] && H_[2]) || ( planeSz[1] <= 1 && (fieldSz[0] > 3 || fieldSz[1] > 3)) ) ) || ( planeSz[1] <= 1 && ( (E_[2] && H_[2]) || ( planeSz[0] <= 1 && (fieldSz[0] > 3 || fieldSz[1] > 3)) ) ) )
                continue;
            else if( (std::any_of(m_.begin(), m_.end(), [](int m){return m < 0;}) && (ii%2 == 1) ) || (ii%2 == 0) )
                continue;
            std::vector<int> physVals;
            if(surInds[ii] == 0)
                physVals = phys->getPlaneYZ(loc_[0]+locOff[ii], planeLoc, planeSz);
            else if(surInds[ii] == 1)
                physVals = phys->getPlaneXZ(loc_[1]+locOff[ii], planeLoc, planeSz);
            else if(surInds[ii] == 2)
                physVals = phys->getPlaneXY(loc_[2]+locOff[ii], planeLoc, planeSz);
            std::vector<double> epMuVals(physVals.size(), 0.0);
            if(E)
                std::transform( physVals.begin(), physVals.end(), epMuVals.begin(), [&](int pp){return (pp > 0 ? objArr[pp]->epsInfty() : 1.0 ) ;} );
            else
                std::transform( physVals.begin(), physVals.end(), epMuVals.begin(), [&](int pp){return (pp > 0 ? objArr[pp]-> muInfty() : 1.0 ) ;} );
            bool epMuChange = false;
            if( std::any_of( physVals.begin(), physVals.end(), [&](double a){return a != physVals[0] && a > -1; } ) )
                epMuChange = true;
            if(epMuChange)
            {
                if(indM == -1)
                    throw std::logic_error("Material changes crossing TFSF boundaries only defined for normal incidence.");
                if(indM == surInds[ii])
                    throw std::logic_error("Material changes can't occur along surface that is propagating the pulse.");
                else
                {
                    int epMuInd = -1; int epMuStride = 0; int epMuSz = -1;
                    if( indM == 0 || (indM == 2 && surInds[ii] == 0) )
                    {
                        epMuSz = planeSz[1];
                        epMuStride = planeSz[0];
                    }
                    else
                    {
                        epMuSz = planeSz[0];
                        epMuStride = 1;
                    }
                    for(int jj = 0; jj < sz_[indM]; ++jj)
                    {
                        if( indM == 0 || (indM == 2 && surInds[ii] == 0) )
                            epMuInd = (jj < sz_[indM] - fieldEnd[indM]) ? jj : (jj-1);
                        else
                            epMuInd = (jj < sz_[indM] - fieldEnd[indM]) ? planeSz[0]*jj : planeSz[0]*(jj-1);
                        if( std::abs(dasum_(epMuSz, &epMuVals[epMuInd], epMuStride) - epMuVals[epMuInd] * epMuSz) > 1e-10 )
                            throw std::logic_error("1) The material is not consistent across all TFSF surfaces for each slice perpendicular to the propagating direction");

                        int ep_mu_ind = fieldOff + ( m_[indM] >=0 ? jj : sz_[indM] - jj - 1 );
                        if( ep_mu->point( ep_mu_ind ) == 0)
                        {
                            ep_mu->point( ep_mu_ind ) = epMuVals[epMuInd];
                            incdPhys->point( ep_mu_ind ) = physVals[epMuInd] > -1 ? physVals[epMuInd] : 0;
                        }
                        else if(ep_mu->point( ep_mu_ind) != epMuVals[epMuInd] )
                        {
                            throw std::logic_error("2) The material is not consistent across all TFSF surfaces for each slice perpendicular to the propagating direction");
                        }
                    }
                    std::fill_n( ep_mu->data(), fieldOff, ep_mu->point(fieldOff ) );
                    std::fill_n(&ep_mu->point(fieldOff +sz_[indM]), ep_mu->size() - ( sz_[indM] + fieldOff ), ep_mu->point(fieldOff + sz_[indM]-1) );

                    std::fill_n( incdPhys->data(), fieldOff, incdPhys->point(fieldOff ) );
                    std::fill_n(&incdPhys->point(fieldOff +sz_[indM]), incdPhys->size() - ( sz_[indM] + fieldOff ), incdPhys->point(fieldOff + sz_[indM]-1) );
                }
            }
            else if(surInds[ii] != indM)
            {
                if( std::any_of( ep_mu->data(), ep_mu->data()+ep_mu->size(), [&](double a){return (a != epMuVals[0]) && (a != 0.0) ; } ) )
                    throw std::logic_error("3) The material is not consistent across all TFSF surfaces for each slice perpendicular to the propagating direction.");
                std::fill_n(ep_mu->data(), ep_mu->size(), epMuVals[0] );
                std::fill_n(incdPhys->data(), incdPhys->size(), *std::max_element( physVals.begin(), physVals.end() ) );
            }
            else if(std::any_of(sz_.begin(), sz_.end(), [](int s){return s == 1;} ) )
            {
                if( std::any_of( ep_mu->data(), ep_mu->data()+ep_mu->size(), [&](double a){return (a != epMuVals[0]) && (a != 0.0) ; } ) )
                    throw std::logic_error("3) The material is not consistent across all TFSF surfaces for each slice perpendicular to the propagating direction.");
                std::fill_n(ep_mu->data(), ep_mu->size(), epMuVals[0] );
                std::fill_n(incdPhys->data(), incdPhys->size(), *std::max_element( physVals.begin(), physVals.end() ) );
            }
        }
    }

    /**
     * @brief      Creates the update parameters vector
     *
     * @param[in]  ep_mu     The permivitty and permeability grids for incident fields
     * @param      upList    The update parameters list
     * @param[in]  dj           grid spacing in the j direction if i is the field polarization direction
     * @param[in]  dk           grid spacing in the k direction if i is the field polarization direction
     * @param[in]  mOffPosJ     Offset from incd field point to get the positive j field contribution for update derivatives
     * @param[in]  mOffNegJ     Offset from incd field point to get the negative j field contribution for update derivatives
     * @param[in]  mOffPosK     Offset from incd field point to get the positive k field contribution for update derivatives
     * @param[in]  mOffNegK     Offset from incd field point to get the negative k field contribution for update derivatives
     */
    void genFieldUpParam(bool E, real_grid_ptr ep_mu, int_grid_ptr incdPhys, std::vector<paramUpIncdField>& upList, std::vector<std::shared_ptr<Obj>> objArr, int i, int j, int k, int mInd )
    {
        int ii = mTot_;
        while(ii < incdPhys->size() - mTot_)
        {
            paramUpIncdField param;
            param.prefactor_j_ = dt_/d_[k];
            param.prefactor_k_ = dt_/d_[j];
            param.indI_ = ii;
            param.indPosJ_ = ii + (E ?     m_[j]           /2 : -1* m_[j]           /2 );
            param.indNegJ_ = ii + (E ? -1*(m_[j] + m_[j]%2)/2 :    (m_[j] + m_[j]%2)/2 );
            param.indPosK_ = ii + (E ? -1*(m_[k] + m_[k]%2)/2 :    (m_[k] + m_[k]%2)/2 );
            param.indNegK_ = ii + (E ?     m_[k]           /2 : -1* m_[k]           /2 );

            if( objArr[incdPhys->point(ii)]->chiAlpha().size() > 0 )
                throw std::logic_error("Chiral materials can't be intersect the TFSF surfaces");
            param.aChiAlpha_ = !E ? objArr[incdPhys->point(ii)]->magAlpha() : objArr[incdPhys->point(ii)]->alpha();
            param.aChiXi_    = !E ? objArr[incdPhys->point(ii)]->magXi() : objArr[incdPhys->point(ii)]->xi()   ;
            param.aChiGamma_ = !E ? objArr[incdPhys->point(ii)]->magGamma() : objArr[incdPhys->point(ii)]->gamma();
            ++ii;
            while(ii < incdPhys->size() -mTot_ && ( incdPhys->point(ii)  == incdPhys->point(ii-1) ) )
                ++ii;
            param.nSz_ = ii - param.indI_;
            upList.push_back(param);
        }
    }

    inline int getMOffSet(int mI, int mJ, int mK)
    {
        return static_cast<int>( floor(static_cast<double>(mI+mJ+mK)/2.0 - static_cast<double>(mI%2+mJ%2+mK%2)/2.0 ) );
    }

    /**
     * @brief      Scales the distance vector by the speed of light inside that material
     *
     * @param[in]  ep       grid pointer to the permiviity grid
     * @param[in]  mu       grid pointer to the permeability gird
     * @param      distVec  The distance vector
     */
    void scaleDistVecSpeed(real_grid_ptr ep, real_grid_ptr mu, std::vector<double>& distVec)
    {
        std::vector<double> epMu(distVec.size());
        std::transform( ep->data(), ep->data()+epMu.size(), mu->data(), epMu.data(), [](double e, double m){ return std::sqrt(e*m); } );
        std::transform( epMu.begin(), epMu.end(), distVec.begin(), distVec.begin(), std::multiplies<double>() );
    }

    /**
     * @brief      Initialize the update parameter structs for the TFSF surface for field grid_i
     *
     * @param[in]  grid_i       Main grid field the struct is being made for
     * @param[in]  incd_j       Incident J field used to update grid_i
     * @param[in]  incd_k       Incident K field used to update grid_i
     * @param[in]  j            Direction of j
     * @param[in]  k            Direction of j
     * @param[in]  pol_i        Polarization of grid_i
     * @param[in]  pol_j        Polarization of incd_j
     * @param[in]  pol_k        Polarization of incd_k
     * @param[in]  sz_i         The size of the surface in the i dierction
     * @param[in]  sz_j         The size of the surface in the j dierction
     * @param[in]  sz_k         The size of the surface in the k dierction
     * @param[in]  E            True if the polarization is for an electric field
     * @param[in]  pl           True if the surface is for either the top, right, or front surface
     * @param      paramStruct  Cector storing the parameter update struct to be made
     */
    void initializeIncdAddStructs(pgrid_ptr grid_i, cplx_grid_ptr incd_j, cplx_grid_ptr incd_k, int_grid_ptr physField, int_pgrid_ptr mainPhysGrid, real_grid_ptr ep_mu, std::vector<std::shared_ptr<Obj>> objArr, DIRECTION j, DIRECTION k, int cor_ii, int cor_jj, int cor_kk, POLARIZATION pol_i, POLARIZATION pol_j, POLARIZATION pol_k, bool E, bool pl, int mInd, tfsfSurfaceAddParamVec& paramStruct)
    {
        if(!grid_i)
            return;
        std::shared_ptr<paramStoreTFSF> sur;
        // If sz_i == 1 then there is no surfaces are to be made
        if( sz_[cor_ii] > 1 || grid_i->n_vec(cor_ii) <= 3 )
        {
            // If incd_j is a nullptr or sz_j ==1 no surface in k direction
            if(sz_[cor_jj] > 1 || grid_i->n_vec(2) == 1)
            {
                if(incd_j)
                {
                    sur = genSurface(grid_i, incd_j, physField, mainPhysGrid, ep_mu, objArr, pol_i, pol_j, k, pl, E, mInd);
                    // If sur is nullptr then the surface does not exist on this process
                    if(sur)
                        paramStruct.push_back( sur );
                    if( sz_[cor_kk] > 1 )
                    {
                        sur = genSurface(grid_i, incd_j, physField, mainPhysGrid, ep_mu, objArr, pol_i, pol_j, k, !pl, E, mInd);
                        // If sur is nullptr then the surface does not exist on this process
                        if(sur)
                            paramStruct.push_back( sur );
                    }
                }
            }
            // If incd_k is a nullptr or sz_k ==1 no surface in j direction
            if( sz_[cor_kk] > 1 || grid_i->n_vec(2) == 1)
            {
                if(incd_k)
                {
                    sur = genSurface(grid_i, incd_k, physField, mainPhysGrid, ep_mu, objArr, pol_i, pol_k, j, pl, E, mInd);
                    // If sur is nullptr then the surface does not exist on this process
                    if(sur)
                        paramStruct.push_back( sur );
                    // If sz_j > 1 then both the plus and minus surface are present so construct both
                    if( sz_[cor_jj] > 1 )
                    {
                        sur = genSurface(grid_i, incd_k, physField, mainPhysGrid, ep_mu, objArr, pol_i, pol_k, j, !pl, E, mInd);
                        // If sur is nullptr then the surface does not exist on this process
                        if(sur)
                            paramStruct.push_back( sur );
                    }
                }
            }
        }
        return;
    }

    /**
     * @brief     generates a tfsf surface data structures
     *
     * @param[in]  mainGrid   Main grid field correpsoinding to the incdient field structure to be constrcuted
     * @param[in]  incdField  The incd field used to get the current terms for the main grid
     * @param[in]  pol        The polarization of the main Grid Field
     * @param[in]  polIncd    The polarization of the incd field
     * @param[in]  dir        The of the surface's normal
     * @param[in]  pl         True if top, right or front surface
     * @param[in]  E          true if E field
     *
     * @return     a TFSF Surface data structure
     */
    std::shared_ptr<paramStoreTFSF> genSurface(pgrid_ptr mainGrid, cplx_grid_ptr incdField, int_grid_ptr physField, int_pgrid_ptr physMain, real_grid_ptr ep_mu, const std::vector<std::shared_ptr<Obj>>& objArr, POLARIZATION pol, POLARIZATION polIncd, DIRECTION dir, bool pl, bool E, int mInd)
    {
        std::array<int, 3> fieldEnd_i = {0, 0, 0}; // 0 if field value is not cutoff in Yee cell, 1 if it is
        // Needed to determine where to get the correct incident field values
        int mOff = 0;
        double mOffSet = 0; // half integer value for mOff
        double mOffHalf = 0; // Used to sum up the half integer offsets ( used because offset by 1/2 is treated with assumptions in the update)
        // Used to determine the sizes of the surfaces
        if(pol == POLARIZATION::EX || pol == POLARIZATION::HZ || pol == POLARIZATION::HY)
            fieldEnd_i[0] = 1;
        if(pol == POLARIZATION::EY || pol == POLARIZATION::HX || pol == POLARIZATION::HZ)
            fieldEnd_i[1] = 1;
        if(mainGrid->n_vec(2) != 1 && (pol == POLARIZATION::EZ || pol == POLARIZATION::HX || pol == POLARIZATION::HY))
            fieldEnd_i[2] = 1;

        // Used to determine where to start for the incd field calculations
        if(polIncd == POLARIZATION::EX || polIncd == POLARIZATION::HZ || polIncd == POLARIZATION::HY)
        {
            mOffSet  += static_cast<double>(m_[0])/2.0;
            mOffHalf += static_cast<double>(m_[0]%2)/2.0;
        }
        if(polIncd == POLARIZATION::EY || polIncd == POLARIZATION::HX || polIncd == POLARIZATION::HZ)
        {
            mOffSet  += static_cast<double>(m_[1])/2.0;
            mOffHalf += static_cast<double>(m_[1]%2)/2.0;
        }
        if(polIncd == POLARIZATION::EZ || polIncd == POLARIZATION::HX || polIncd == POLARIZATION::HY)
        {
            mOffSet  += static_cast<double>(m_[2])/2.0;
            mOffHalf += static_cast<double>(m_[2]%2)/2.0;
        }

        // Initialize structures
        bool isJ = true;
        bool plProp = false;
        int cor = -1; // the index in the loc/sz vectors corresponding to the i direction
        int transCor1 = -1; // the index in the loc/sz vectors corresponding to the j or k direction (not z)
        int transCor2 = -1; // the index in the loc/sz vectors corresponding to the j or k direction (z if possible)
        std::array<int,3> addVec1;
        std::array<int,3> addVec2;
        paramStoreTFSF sur;
        sur.incdField_ = incdField;
        sur.szTrans_ = {{-1, -1}};
        // Set prefactors for updating H field (used on E_inc) use of XOR since sign flips for E/H and pl/mn
        if(dir ==DIRECTION::X)
        {
            if (pol == POLARIZATION::HY || pol == POLARIZATION::EY)
                isJ = false;
            if(H_[2] && E_[2] && mInd != 2 )
            {
                cor = 0; transCor1 = 2; transCor2 = 1;
                addVec1 = {{ 0, 0, 1 }};
                addVec2 = {{ 0, 1, 0 }};
                sur.strideMain_ = mainGrid->local_x();
            }
            else
            {
                cor = 0; transCor1 = 1; transCor2 = 2;
                addVec1 = {{ 0, 1, 0 }};
                addVec2 = {{ 0, 0, 1 }};
                sur.strideMain_ = mainGrid->local_x()*mainGrid->local_z();
            }
        }
        else if( dir == DIRECTION::Y)
        {
            if (pol == POLARIZATION::HZ || pol == POLARIZATION::EZ)
                isJ = false;
            if(mInd != 0)
            {
                cor = 1; transCor1 = 0; transCor2 = 2;
                addVec1 = {{ 1, 0, 0 }};
                addVec2 = {{ 0, 0, 1 }};
                sur.strideMain_ = 1;
            }
            else
            {
                cor = 1; transCor1 = 2; transCor2 = 0;
                addVec1 = {{ 0, 0, 1 }};
                addVec2 = {{ 1, 0, 0 }};
                sur.strideMain_ = mainGrid->local_x();
            }
        }
        else if( dir == DIRECTION::Z)
        {
            if (pol == POLARIZATION::HX || pol == POLARIZATION::EX)
                isJ = false;
            if(mInd != 0)
            {
                cor = 2; transCor1 = 0; transCor2 = 1;
                addVec1 = {{ 1, 0, 0 }};
                addVec2 = {{ 0, 1, 0 }};
                sur.strideMain_ = 1;
            }
            else
            {
                cor = 2; transCor1 = 1; transCor2 = 0;
                addVec1 = {{ 0, 1, 0 }};
                addVec2 = {{ 1, 0, 0 }};
                sur.strideMain_ = mainGrid->local_x()*mainGrid->local_z();
            }
        }
        sur.prefactor_ = ( ( E ^ isJ ^ pl ) ) ? dt_/d_[cor] : -1.0*dt_/d_[cor]; // Prefactors are the same, but sign dependent on if it's an E/H field, pl or mn surface and if it is j/k

        sur.strideIncd_  = m_[transCor1]; // Incident streides based on m values
        sur.addIncdProp_ = m_[transCor2]; // Incident streides based on m values
        // Where does the plane start?
        // locTemp used because the plane for the H field updates does not have to correspond to the E field surface (one point outside surface)
        std::array<int, 3> locTemp(loc_);
        std::array<int, 3> szTemp(sz_);
        locTemp[cor] = pl ? (loc_[cor] + sz_[cor]-1) : (E ? loc_[cor] : loc_[cor] - 1);
        szTemp[cor] = 1;
        locTemp = getSurLoc(locTemp, szTemp, mainGrid);
        if(!H_[2] || !E_[2])
            locTemp[2] = 0; // 2D calculs the location in z is always 0
        // If any of the locations are -1 then the prcess does not contain the surface
        if( std::all_of(locTemp.begin(), locTemp.end(), [](int ii){return ii != -1;} ) )
        {
            // Get the transverse sizes
            int tarnsSz0Max = getSzVal(transCor1, fieldEnd_i[transCor1], locTemp[transCor1], loc_, szTemp, mainGrid);
            sur.szTrans_[1] = getSzVal(transCor2, fieldEnd_i[transCor2], locTemp[transCor2], loc_, szTemp, mainGrid);
            // If 2D szTrans[1] = 1 by default
            if(!H_[2] || !E_[2])
                sur.szTrans_[1] = 1;
            // Incd start based on where the sur location is realtive to the origin (distance from the point of propagation in grid points)
            std::array<int,3> locRelOr(locTemp);
            if(!pl && !E )
                locRelOr[cor] += 1; // H fields on the mn surface have one addtional point away from teh orgin
            locRelOr = getLocRelOr(locRelOr, mainGrid);
            // ir Based on loc realtive to Origin + source + buffer + mOffSet for field in eqution - stride condtions (negative strides need the position to be the end point of the negative negative in mkl)
            // M offsets added since the E field needs incident of the H field outside the surface (why there is a buffer)
            if(E && m_[cor] > 0 && locRelOr[cor] == 0)
                mOffSet -= m_[cor];
            else if( E && m_[cor] < 0 && -1*locRelOr[cor] == sz_[cor]-1 && sz_[cor] != 1 )
                mOffSet -= m_[cor];
            mOff = static_cast<int>(floor(mOffSet-mOffHalf)); // The -mOffHalf is to handle the half integer offset problems
            for(int ll = 0; ll < sur.szTrans_[1]; ll+=1)
            {
                int sz0 = 0;
                while(sz0 < tarnsSz0Max)
                {
                    sur.incdStart_ = getIr(locRelOr) + pmlThick_ + mOff + 6*mTot_ + ll*sur.addIncdProp_ + sz0*sur.strideIncd_;
                    int mainInd =  mainGrid->getInd(locTemp[0]+ll*addVec2[0]+ sz0*addVec1[0], locTemp[1]+ll*addVec2[1]+ sz0*addVec1[1], locTemp[2]+ll*addVec2[2]+ sz0*addVec1[2]);
                    int initSz0 = sz0;
                    ++sz0;
                    while( ( !E || objArr[physField->point(sur.incdStart_+ sz0*sur.strideIncd_)]->gamma().size() == objArr[physField->point(sur.incdStart_)]->gamma().size() ) && ( E || objArr[physField->point(sur.incdStart_+ sz0*sur.strideIncd_)]->magGamma().size() == objArr[physField->point(sur.incdStart_)]->magGamma().size() ) && sz0 < tarnsSz0Max )
                        ++sz0;

                    sur.szTrans_[0] = sz0 - initSz0;
                    if(sur.strideIncd_ < 0)
                        sur.incdStart_ += sur.strideIncd_*(sur.szTrans_[0]-1) ;
                    if( ( E && objArr[physField->point(sur.incdStart_)]->alpha().size() > 0 ) || ( !E && objArr[physField->point(sur.incdStart_)]->magAlpha().size() > 0 ) || objArr[physField->point(sur.incdStart_)]->chiAlpha().size() > 0)
                    {
                        sur.indsD_.push_back(sur.incdStart_);
                        sur.indsD_.push_back(mainInd);
                    }
                    else
                    {
                        sur.indsU_.push_back(sur.incdStart_);
                        sur.indsU_.push_back(mainInd);
                    }
                }
            }
            return std::make_shared<paramStoreTFSF>(sur);
        }
        else
            return nullptr;
    }

    /**
     * @brief      Gets the surface location
     *
     * @param[in]  loc       Location of the surface in global grid coordinates
     * @param[in]  sz        The size of the surface
     * @param[in]  mainGrid  The main grid field ptr
     *
     * @return     The sur location in local coordinates, any value of -1  means process does not contain surface.
     */
    std::array<int,3> getSurLoc(std::array<int,3> loc, std::array<int,3> sz, pgrid_ptr mainGrid)
    {
        std::array<int, 3> surLoc = {-1,-1,-1};
        for(int ii = 0; ii < 3; ++ii)
        {
            // Does the plane start this process? -> Yes: Loc is set to the location of the plane - the process start loc :: Is the plane inside this process but starts before it? -> Yes: location is the start point for the process :: No process does not store the plane loc = -1
            if( ( loc[ii] >= mainGrid->procLoc(ii) ) && ( loc[ii] < (mainGrid->procLoc(ii) + mainGrid->ln_vec(ii) - 2) ) )
                surLoc[ii] =  loc[ii] - mainGrid->procLoc(ii) + 1;
            else if( ( loc[ii] < mainGrid->procLoc(ii) ) && ( ( loc[ii] + sz[ii] ) > mainGrid->procLoc(ii) ) )
                surLoc[ii] =  1;
        }
        return surLoc;
    }
    /**
     * @brief      Gets the size of the surface in a particlar direction .
     *
     * @param[in]  cor       Index of direction needed to get the size
     * @param[in]  fieldEnd  Size modifier for the grids in a yee cell for that field
     * @param[in]  surLoc    The location of the surface in local coordiantes
     * @param[in]  mainGrid  The main grid field ptr
     *
     * @return     The size of the surface in direction cor.
     */
    int getSzVal(int cor, int fieldEnd, int surLoc, std::array<int,3> loc, std::array<int, 3> sz, pgrid_ptr mainGrid)
    {
        // Does the plane go out of the process? -> Yes: then set size to be size of process - loc :: No: tThen size is total size - amount in process - (0 or 1 depending on field)
        if(sz[cor] + loc[cor] > mainGrid->procLoc(cor) + mainGrid->ln_vec(cor) - 2)
            return mainGrid->ln_vec(cor) - 2 - surLoc + 1;
        else
            return loc[cor] + sz[cor] - (mainGrid->procLoc(cor) + surLoc - 1) - fieldEnd;
    }

    /**
     * @brief      Gets the location of the surface relative to the origin.
     *
     * @param[in]  loc       The of the surface in local coordinates
     * @param[in]  mainGrid  The main grid field ptr
     *
     * @return     The location of the surface realtive to the origin .
     */
    std::array<int,3> getLocRelOr(std::array<int,3> loc, pgrid_ptr mainGrid)
    {
        std::array<int,3> relOr = {0,0,0};
        for(int ii = 0; ii < 3; ++ii)
            relOr[ii] = (loc[ii]-1) + mainGrid->procLoc(ii) - originLoc_[ii];
        return relOr;
    }
    /**
     * @brief      Gets the ir for the point. (mx*i + my*j + mz*k)
     *
     * @param[in]  loc   The location of the point realtive to the origin
     *
     * @return     The ir = mx*i + my*j + mz*k.
     */
    inline int getIr(std::array<int,3> loc) { return std::abs(m_[0]*loc[0]) + std::abs(m_[1]*loc[1]) + std::abs(m_[2]*loc[2]); }
    /**
     * @brief      updates the the fields using update functors
     */
    void updateFields()
    {
        for(int ii = 0; ii < 3; ++ii )
        {
            for(auto sur : hSurfaces_[ii])
            {
                addHIncd_[ii](H_[ii], B_[ii], sur,  mu_[ii], scratchEpMu_.data() );
            }
        }
        step();
        for(int ii = 0; ii < 3; ++ii)
        {
            for(auto sur : eSurfaces_[ii])
                addEIncd_[ii](E_[ii], D_[ii], sur, eps_[ii], scratchEpMu_.data() );
        }
    }
    /**
     * @brief  Accessor function for loc_
     *
     * @return origin Location
     */
    inline std::vector<int> & loc()  {return loc_;}
    /**
     * @brief  Accessor function for sz_
     *
     * @return size of total field region
     */
    inline std::vector<int> & size() {return sz_;}
    /**
     * @brief  Accessor function for phi_
     *
     * @return angle of incidence of the light
     */
    inline double phi(){return phi_;}

    /**
     * @brief      returns prefactor phi factor
     *
     * @return     phiPrefactCalc_
     */
    inline double phiPreFact(){return phiPrefactCalc_;}

    /**
     * @brief  Accessor function for theta_
     *
     * @return angle of incidence of the light
     */
    inline double theta(){return theta_;}

    /**
     * @brief  Accessor function for psi_
     *
     * @return angle of polarization of the light
     */
    inline double psi(){return psi_;}

    /**
     * @brief      returns prefactor psi factor
     *
     * @return     psiPrefactCalc_
     */
    inline double psiPreFact(){return psiPrefactCalc_;}

    /**
     * @brief  Accessor function for alpha_
     *
     * @return phase angle of incidence of the light if circularly polarized
     */
    inline double alpha(){return alpha_;}
    /**
     * @brief  Accessor function for originQuadrent_
     *
     * @return origin quadrant
     */
    inline int quadrant(){return originQuadrent_;}
    /**
     * @brief  Accessor function for gridLen_
     *
     * @return length of 1D auxiliary fields
     */
    inline int gridLen(){return gridLen_;}

    inline const tfsfSurfaceAddParamVec& getChiHCorrect(int ii) { return chiHCorrectSurfaces_[ii]; }

    inline const tfsfSurfaceAddParamVec& getChiECorrect(int ii) { return chiECorrectSurfaces_[ii]; }

    /**
     * @brief Moves the incident fields forward one time step
     * @details Uses the 1D FDTD equations to propagate the fields in time
     */
    void step()
    {
        // Update all incident H fields
        for(int ii = 0; ii < 3; ++ii)
        {
            pulAddH_[ii](distVecH_[ii], pul_, t_step_*dt_, pulseVec_);

            incdUpM_   [ii](upIncdParamH_[ii], H_incd_[ii],                     M_incd_[ii],    prevM_incd_[ii], scratchPIncd_.data()                       );

            incdUpH_[ii](pulseVec_, pmlThick_ + 3*mTot_, prefactH_[ii],  upIncdParamH_[ii], B_incd_[ii] , E_incd_ [(ii+1)%3], E_incd_ [(ii+2)%3], pmlH_incd_ [ii] );
            incdUpH_[ii](pulseVec_, pmlThick_ + 3*mTot_, prefactH_[ii], upPulseParamH_[ii], H_pulse_[ii], E_pulse_[(ii+1)%3], E_pulse_[(ii+2)%3], pmlH_pulse_[ii] );

            B2H_[ii](-1.0, H_incd_[ii], B_incd_[ii], mu_[ii], M_incd_[ii], scratchPIncd_.data() );
        }
        // Once incident E fields are updated then update the incident E fields
        for(int ii = 0; ii < 3; ++ii)
        {
            pulAddE_[ii](distVecE_[ii], pul_, t_step_*dt_, pulseVec_);
            incdUpP_   [ii](upIncdParamE_[ii], E_incd_[ii],                     P_incd_[ii],    prevP_incd_[ii], scratchPIncd_.data()                       );

            incdUpE_[ii](pulseVec_, pmlThick_ + 3*mTot_, prefactE_[ii],  upIncdParamE_[ii], D_incd_ [ii], H_incd_ [(ii+1)%3], H_incd_ [(ii+2)%3], pmlE_incd_ [ii] );
            incdUpE_[ii](pulseVec_, pmlThick_ + 3*mTot_, prefactE_[ii], upPulseParamE_[ii], E_pulse_[ii], H_pulse_[(ii+1)%3], H_pulse_[(ii+2)%3], pmlE_pulse_[ii] );

            D2E_[ii]( 1.0, E_incd_[ii], D_incd_[ii], eps_[ii], P_incd_[ii], scratchPIncd_.data());
        }
        ++t_step_;
    }
};

namespace tfsfUpdateFxnReal
{
    /**
     * @brief      Adds incd fields.
     *
     * @param[in]  grid_Ui    grid pointers to the main grid U field
     * @param[in]  grid_Di    grid pointers to the main grid D field
     * @param[in]  sur        The surface parameter struct
     * @param[in]  ep_mu      The grid pointer to the high frequency permeability or permivitty function
     * @param      epMuStore  Temporary storage or the field/eps or mu
     */
    void addIncdFields(real_pgrid_ptr grid_Ui, real_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, double* epMuStore);


    /**
     * @brief      Adds incd fields when there is a change in material parameters.
     *
     * @param[in]  grid_Ui    grid pointers to the main grid U field
     * @param[in]  grid_Di    grid pointers to the main grid D field
     * @param[in]  sur        The surface parameter struct
     * @param[in]  ep_mu      The grid pointer to the high frequency permeability or permivitty function
     * @param      epMuStore  Temporary storage or the field/eps or mu
     */
    void addIncdFieldsEPChange(real_pgrid_ptr grid_Ui, real_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, double* tempEpMuStore);

    void addIncdFieldsChiCorrect(real_pgrid_ptr grid_Ui, std::shared_ptr<paramStoreTFSF> sur, double prefactMult);
    /**
     * @brief      does grid->gridTransfer() for grid pointers
     *
     * @param[in]  grid grid that needs grid->gridTransfer to it
     */
    void transferDat(real_pgrid_ptr grid);
}

namespace tfsfUpdateFxnCplx
{
    /**
     * @brief      Adds incd fields.
     *
     * @param[in]  grid_Ui    grid pointers to the main grid U field
     * @param[in]  grid_Di    grid pointers to the main grid D field
     * @param[in]  sur        The surface parameter struct
     * @param[in]  ep_mu      The grid pointer to the high frequency permeability or permivitty function
     * @param      epMuStore  Temporary storage or the field/eps or mu
     */
    void addIncdFields(cplx_pgrid_ptr grid_Ui, cplx_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, cplx* epMuStore);

    /**
     * @brief      Adds incd fields when there is a change in material parameters.
     *
     * @param[in]  grid_Ui    grid pointers to the main grid U field
     * @param[in]  grid_Di    grid pointers to the main grid D field
     * @param[in]  sur        The surface parameter struct
     * @param[in]  ep_mu      The grid pointer to the high frequency permeability or permivitty function
     * @param      epMuStore  Temporary storage or the field/eps or mu
     */
    void addIncdFieldsEPChange(cplx_pgrid_ptr grid_Ui, cplx_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, cplx* tempEpMuStore);

    void addIncdFieldsChiCorrect(cplx_pgrid_ptr grid_Ui, std::shared_ptr<paramStoreTFSF> sur, double prefactMult);

    /**
     * @brief      does grid->gridTransfer() for grid pointers
     *
     * @param[in]  grid grid that needs grid->gridTransfer to it
     */
    void transferDat(cplx_pgrid_ptr);
}

class parallelTFSFReal : public parallelTFSFBase<double>
{
    typedef std::shared_ptr<parallelGrid<double>> pgrid_ptr;
public:

    /**
     * @brief      Construct a TFSF surface
     *
     * @param[in]  gridComm     The mpiInterface of the grids
     * @param[in]  loc          location of TFSF origin
     * @param[in]  sz           size of the total field region
     * @param[in]  theta        polar angle of the plane wave's k-vector
     * @param[in]  phi          angle of plane wave's k-vector in the xy plane
     * @param[in]  psi          angle describing the polarization of the plane wave from the vector \vec{k}\times e_{z} (if k is along z psi_ = phi_)
     * @param[in]  circPol      POLARIZATION::R if R polarized, L if L polarized, linear if anything else
     * @param[in]  kLenRelJ     ratio between the size of the axis oriented along psi to that perpendicular to it for elliptically polarized light
     * @param[in]  d            Array storing the grid spacing in all directions
     * @param[in]  m            Array storing the values of m for the direction of the tfsf propagation
     * @param[in]  dt           time step of the tfsf surface
     * @param[in]  pul          The pulse list used for the surface
     * @param[in]  E            array of pgrid_ptrs corresponding to the E field
     * @param[in]  H            array of pgrid_ptrs corresponding to the H field
     * @param[in]  physE        array of pgrid_ptrs to the map of materials for the E field
     * @param[in]  physH        array of pgrid_ptrs to the map of materials for the H field
     * @param[in]  objArr       A vector storing all the objects in the field
     * @param[in]  nomPMLThick  The nominal thickness for the incd pmls will be affected by sum of the m vector
     * @param[in]  pmlM         The incd pml scaling value for sigma and kapa
     * @param[in]  pmlMA        The incd pml scaling value for a
     * @param[in]  pmlAMax      The incd pml's maximum a value
     */
    parallelTFSFReal(std::shared_ptr<mpiInterface> gridComm, std::array<int,3> loc, std::array<int,3> sz, double theta, double phi, double psi, POLARIZATION circPol, double kLenRelJ, std::array<double,3> d, std::array<int,3> m, double dt, std::vector<std::shared_ptr<Pulse>> pul, std::array<pgrid_ptr,3> E, std::array<pgrid_ptr,3> H, std::array<pgrid_ptr,3> D, std::array<pgrid_ptr,3> B, std::array<int_pgrid_ptr,3> physE, std::array<int_pgrid_ptr,3> physH, std::vector<std::shared_ptr<Obj>> objArr, int nomPMLThick=20, double pmlM=3.0, double pmlMA=1.0, double pmlAMax=0.25);
};
class parallelTFSFCplx : public parallelTFSFBase<cplx>
{
    typedef std::shared_ptr<parallelGrid<cplx>> pgrid_ptr;
public:

    /**
     * @brief      Construct a TFSF surface
     *
     * @param[in]  gridComm     The mpiInterface of the grids
     * @param[in]  loc          location of TFSF origin
     * @param[in]  sz           size of the total field region
     * @param[in]  theta        polar angle of the plane wave's k-vector
     * @param[in]  phi          angle of plane wave's k-vector in the xy plane
     * @param[in]  psi          angle describing the polarization of the plane wave from the vector \vec{k}\times e_{z} (if k is along z psi_ = phi_)
     * @param[in]  circPol      POLARIZATION::R if R polarized, L if L polarized, linear if anything else
     * @param[in]  kLenRelJ     ratio between the size of the axis oriented along psi to that perpendicular to it for elliptically polarized light
     * @param[in]  d            Array storing the grid spacing in all directions
     * @param[in]  m            Array storing the values of m for the direction of the tfsf propagation
     * @param[in]  dt           time step of the tfsf surface
     * @param[in]  pul          The pulse list used for the surface
     * @param[in]  E            array of pgrid_ptrs corresponding to the E field
     * @param[in]  H            array of pgrid_ptrs corresponding to the H field
     * @param[in]  physE        array of pgrid_ptrs to the map of materials for the E field
     * @param[in]  physH        array of pgrid_ptrs to the map of materials for the H field
     * @param[in]  objArr       A vector storing all the objects in the field
     * @param[in]  nomPMLThick  The nominal thickness for the incd pmls will be affected by sum of the m vector
     * @param[in]  pmlM         The incd pml scaling value for sigma and kapa
     * @param[in]  pmlMA        The incd pml scaling value for a
     * @param[in]  pmlAMax      The incd pml's maximum a value
     */
    parallelTFSFCplx(std::shared_ptr<mpiInterface> gridComm, std::array<int,3> loc, std::array<int,3> sz, double theta, double phi, double psi, POLARIZATION circPol, double kLenRelJ, std::array<double,3> d, std::array<int,3> m, double dt, std::vector<std::shared_ptr<Pulse>> pul, std::array<pgrid_ptr,3> E, std::array<pgrid_ptr,3> H, std::array<pgrid_ptr,3> D, std::array<pgrid_ptr,3> B, std::array<int_pgrid_ptr,3> physE, std::array<int_pgrid_ptr,3> physH, std::vector<std::shared_ptr<Obj>> objArr, int nomPMLThick=20, double pmlM=3.0, double pmlMA=1.0, double pmlAMax=0.25);
};

#endif