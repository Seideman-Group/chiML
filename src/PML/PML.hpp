/** @file PML/PML.hpp
 *  @brief Stores/updates the CPML fields and adds them to the incident FDTD grids
 *
 *  A class that store and updates the CPML fields and add them the the incident FDTD grids for the TFSF surfaces
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_PML
#define FDTD_PML

#include <src/OBJECTS/Obj.hpp>

/**
 * @brief      parameters to update the $\psi$ fields for IncdCPMLs
 */
struct updatePsiParamsIncd
{
    int loc_; //!< starting point for the pml updates for grid_j
    int locOff_; //!< starting point for the pml updates for grid_j offset (negative )
    std::vector<double> b_; //!< the b parameter as defined in chapter 7 of Taflove
    std::vector<double> c_; //!< the c parameter as defined in chapter 7 of Taflove
    std::vector<double> cOff_; //!< the c parameter as defined in chapter 7 of Taflove, but with the opposite sign of c
};
/**
 * @brief      Parameters to update the fields using the $\psi$ fields
 */
struct updateGridParamsIncd
{
    int nAx_; //!< size of blas operator
    int loc_; //!< starting point of MKL operations
    double Db_; //!< psi factor add on prefactor
};


template <typename T> class IncdCPML
{
    typedef std::shared_ptr<Grid<T>> grid_ptr;
protected:
    POLARIZATION pol_i_; //!< Polarization of the Field that the IncdCPML is acting on
    DIRECTION i_; //!< Direction corresponding to polarization of the field the IncdCPML is acting on
    DIRECTION j_; //!< Direction next in the cycle from i_; i.e. if i_ is DIRECTION::Y then j_ is DIRECTION::Z
    DIRECTION k_; //!< Direction last one in the cycle from i_; i.e. if i_ is DIRECTION::Y then j_ is DIRECTION::X
    int pmlThick_; //!< Thickness of the PML region
    int cor_ii_; //!< index of the ii coordinate (0 if ii = x, 1 if ii = y, and 2 if ii = z)
    int cor_jj_; //!< index of the jj coordinate (0 if jj = x, 1 if jj = y, and 2 if jj = z)
    int cor_kk_; //!< index of the kk coordinate (0 if kk = x, 1 if kk = y, and 2 if kk = z)
    double m_; //!< Exponent for calculating IncdCPML constants sigma and kappa
    double ma_; //!< Exponent for calculating IncdCPML constant a
    double sigmaMax_; //!< Maximum value of sigma in the X direction
    double kappaMax_; //!< Maximum value of kappa
    double aMax_; //!< Maximum value of a
    double dt_; //!< time step
    double dr_; //!< TFSF incident field grid spacing = d[ii]/m[ii]
    double epsMn_; //!< value of the eps_0 of the material inside PML region
    double muMn_; //!< value of the mu_0 of the material inside PML region
    double epsPl_; //!< value of the eps_0 of the material inside PML region
    double muPl_; //!< value of the mu_0 of the material inside PML region
    std::array<int,3> mainGridSz_; //!< vector storing the size of the TFSF surface
    std::array<int,3> tfsfMs_; //!< vector storing the size of the TFSF surface
    std::array<double,3> d_; //!< vector storing the step size in all directions
    std::vector<cplx> tempStore_; //!< scratch space vector
    std::function<void(const updateGridParamsIncd&, const updatePsiParamsIncd&, cplx*, cplx_grid_ptr, cplx_grid_ptr, cplx_grid_ptr)> upPsi_j_; //!< update function for the $\\psi_j$ field
    std::function<void(const updateGridParamsIncd&, const updatePsiParamsIncd&, cplx*, cplx_grid_ptr, cplx_grid_ptr, cplx_grid_ptr)> upPsi_k_; //!< update function for the $\\psi_k$ field

public:
    grid_ptr grid_i_; //!< FDTD field polarized in the i direction (E/H) is the same as it is in pol_i
    grid_ptr grid_j_; //!< FDTD field polarized in the j direction (E/H) is the opposite as it is in pol_i
    grid_ptr grid_k_; //!< FDTD field polarized in the k direction (E/H) is the opposite as it is in pol_i
    std::vector<grid_ptr> psi_j_; //!< IncdCPML helper field $\\psi$ polarized in the j direction
    std::vector<grid_ptr> psi_k_; //!< IncdCPML helper field $\\psi$ polarized in the j direction
    std::vector<updatePsiParamsIncd>  updateListPsi_j_; //!< update parameters for $\\psi_j$ field
    std::vector<updatePsiParamsIncd>  updateListPsi_k_; //!< update parameters for $\\psi_k$ field
    std::vector<updateGridParamsIncd> updateListGrid_j_; //!< update parameters for the updates in the j direction
    std::vector<updateGridParamsIncd> updateListGrid_k_; //!< update parameters for the updates in the k direction

    /**
     * @brief      Constructor
     *
     * @param[in]  grid_i       shared pointer to grid_i (field PML is being applied to)
     * @param[in]  grid_j       shared pointer to grid_j (field polarized in the j direction of the PML; nullptr if none)
     * @param[in]  grid_k       shared pointer to grid_k (field polarized in the k direction of the PML; nullptr if none)
     * @param[in]  pol_i        polarization of grid_i
     * @param[in]  pmlThick     The PML thickness
     * @param[in]  tfsfMs       The tfsf m vector
     * @param[in]  mainGridSz   The main grid size
     * @param[in]  m            scalling factor for sigma
     * @param[in]  ma           scaling factor for a
     * @param[in]  kappaMax     The kappa maximum
     * @param[in]  aMax         max a value
     * @param[in]  d            vector storing the step sizes in all directions
     * @param[in]  dr           step size of the TFSF incd fields
     * @param[in]  dt           time step
     */
    IncdCPML(grid_ptr grid_i, grid_ptr grid_j, grid_ptr grid_k, POLARIZATION pol_i, int pmlThick, std::array<int,3> tfsfMs, std::array<int,3> mainGridSz, bool useMn, double epsPl, double muPl, double epsMn, double muMn, double m, double ma, double kappaMax, double aMax, std::array<double,3> d, double dr, double dt) :
        pol_i_(pol_i),
        cor_ii_(-1),
        cor_jj_(-1),
        cor_kk_(-1),
        pmlThick_(pmlThick),
        m_(m),
        ma_(ma),
        sigmaMax_( 0.8*(m+1)/std::abs(dr) ),
        kappaMax_(kappaMax),
        aMax_(aMax),
        d_(d),
        tempStore_(pmlThick_, 0.0),
        dt_(dt),
        dr_(dr),
        epsPl_(epsPl),
        muPl_(muPl),
        epsMn_(epsMn),
        muMn_(muMn),
        mainGridSz_(mainGridSz),
        tfsfMs_(tfsfMs),
        grid_i_(grid_i),
        grid_j_(grid_j),
        grid_k_(grid_k),
        psi_j_(useMn ? 2 : 1),
        psi_k_(useMn ? 2 : 1)
    {
        if(!grid_i)
            throw std::logic_error("PMLs have to have a real field to be associated with, grid_i_ can't be a nullptr.");
        // Determine the PML's i,j,k directions based on the polarization of the field, direction i shares the direction of the field polarization
        double mOffHalf = 0; // The half integer changes will determine the offset distance for the fields
        if(pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EX)
        {
            mOffHalf = ( pol_i_ == POLARIZATION::HX ? tfsfMs_[1]%2+tfsfMs_[2]%2 : tfsfMs_[0]%2 )/2.0;
            cor_ii_ = 0; cor_jj_ = 1; cor_kk_ = 2;
            i_ = DIRECTION::X;
            j_ = DIRECTION::Y;
            k_ = DIRECTION::Z;
        }
        else if(pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EY)
        {
            mOffHalf = ( pol_i_ == POLARIZATION::HY ? tfsfMs_[2]%2+tfsfMs_[0]%2 : tfsfMs_[1]%2 )/2.0;
            cor_ii_ = 1; cor_jj_ = 2; cor_kk_ = 0;
            i_ = DIRECTION::Y;
            j_ = DIRECTION::Z;
            k_ = DIRECTION::X;
        }
        else
        {
            mOffHalf = ( pol_i_ == POLARIZATION::HZ ? tfsfMs_[0]%2+tfsfMs_[1]%2 : tfsfMs_[2]%2 )/2.0;
            cor_ii_ = 2; cor_jj_ = 0; cor_kk_ = 1;
            i_ = DIRECTION::Z;
            j_ = DIRECTION::X;
            k_ = DIRECTION::Y;
        }
        std::array<int, 3> psiSize = {pmlThick_, 1, 1};
        // In 2D z direction is assumed to be isotropic, but 3D it is not
        if(grid_k_ && (mainGridSz_[2] != 1 || j_ != DIRECTION::Z) )
        {
            psi_j_[0] = std::make_shared<Grid<T>>( psiSize, grid_k_->d() );
            if(useMn)
                psi_j_[1] = std::make_shared<Grid<T>>( psiSize, grid_k_->d() );
        }

        // In 2D z direction is assumed to be isotropic, but in 3D it is not
        if(grid_j_ && (mainGridSz_[2] != 1 || k_ != DIRECTION::Z) )
        {
            psi_k_[0] = std::make_shared<Grid<T>>( psiSize, grid_j_->d() );
            if(useMn)
                psi_k_[1] = std::make_shared<Grid<T>>( psiSize, grid_j_->d() );
        }

        if(psi_k_.size() == 0 && psi_j_.size() == 0)
            throw std::logic_error("Both PML auxiliary fields are undefined this likely means that there is an issue with grid_i or grid_k");
        // Initialize all update lists
        updatePsiParamsIncd tempPsi;
        updateGridParamsIncd tempGrid;
        if(grid_k_ && tfsfMs_[cor_jj_] != 0)
        {
            std::tie(tempPsi, tempGrid)  = getUpList (j_, mOffHalf, true , grid_i_->x()-1-static_cast<int>( tempStore_.size() )-std::accumulate(tfsfMs_.begin(), tfsfMs_.end(), 0, [](int tot, int mm){return tot + std::abs(mm);}), tempStore_.size());
            updateListPsi_j_.push_back(tempPsi);
            updateListGrid_j_.push_back(tempGrid);
            if(useMn)
            {
                std::tie(tempPsi, tempGrid)  = getUpList (j_, mOffHalf, false, std::abs( std::accumulate(tfsfMs_.begin(), tfsfMs_.end(), 0, [](int tot, int mm){return tot + std::abs(mm);} ) ), tempStore_.size());
                updateListPsi_j_.push_back(tempPsi);
                updateListGrid_j_.push_back(tempGrid);
            }
        }
        if(grid_j_ && tfsfMs_[cor_kk_] != 0)
        {
            std::tie(tempPsi, tempGrid)  = getUpList (k_, mOffHalf, true , grid_i_->x()-1-static_cast<int>( tempStore_.size() )-std::accumulate(tfsfMs_.begin(), tfsfMs_.end(), 0, [](int tot, int mm){return tot + std::abs(mm);}), tempStore_.size());
            updateListPsi_k_.push_back(tempPsi);
            updateListGrid_k_.push_back(tempGrid);
            if(useMn)
            {
                std::tie(tempPsi, tempGrid)  = getUpList (k_, mOffHalf, false, std::abs( std::accumulate(tfsfMs_.begin(), tfsfMs_.end(), 0, [](int tot, int mm){return tot + std::abs(mm);} ) ), tempStore_.size());
                updateListPsi_k_.push_back(tempPsi);
                updateListGrid_k_.push_back(tempGrid);
            }
        }
    }

    /**
     * @brief      Generate the psi and main grid updates
     *
     * @param[in]  dir      Direction of the psi field.
     * @param[in]  distOff  Offset for the distance based on the remainders in m/2
     * @param[in]  pl       True if top, right or front.
     * @param[in]  startPt  Where the PMLs starts.
     * @param[in]  dirMax   How far to iterate over.
     *
     * @return     The PML up lists.
     */
    std::tuple<updatePsiParamsIncd, updateGridParamsIncd> getUpList(DIRECTION dir, double distOff, bool pl, int startPt, int dirMax)
    {
        updatePsiParamsIncd psiParam;
        updateGridParamsIncd gridParam;

        int cor_ii = -1;
        // Determine the coordinate system for the PML update lists compared to x,y,z
        if(dir == DIRECTION::X )
            cor_ii = 0;
        else if(dir == DIRECTION::Y )
            cor_ii = 1;
        else
            cor_ii = 2;

        bool E = (pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::EY || pol_i_ == POLARIZATION::EZ) ? true :false;
        if(pl)
            gridParam.Db_ = dt_ / ( d_[cor_ii] * (E ? epsPl_ : muPl_ ) );
        else
            gridParam.Db_ = dt_ / ( d_[cor_ii] * (E ? epsMn_ : muMn_ ) );

        gridParam.loc_ = startPt;
        gridParam.nAx_ = dirMax;
        // Get the constants for all directions by looping over the PML thickness
        if( (dir == j_) ^ !E)
        {
            psiParam.loc_    = startPt + (E ?    (tfsfMs_[cor_ii]                    )/2 : -1*( tfsfMs_[cor_ii]                    )/2);
            psiParam.locOff_ = startPt + (E ? -1*(tfsfMs_[cor_ii] + tfsfMs_[cor_ii]%2)/2 :    ( tfsfMs_[cor_ii] + tfsfMs_[cor_ii]%2)/2);
        }
        else
        {
            psiParam.loc_    = startPt + (E ? -1*(tfsfMs_[cor_ii] + tfsfMs_[cor_ii]%2)/2 :    ( tfsfMs_[cor_ii] + tfsfMs_[cor_ii]%2)/2);
            psiParam.locOff_ = startPt + (E ?    (tfsfMs_[cor_ii]                    )/2 : -1*( tfsfMs_[cor_ii]                    )/2);
        }
        // std::cout << psiParam.loc_ << '\t' << psiParam.locOff_ << std::endl;
        std::vector<double> dist(dirMax, 0.0);
        for(int dd = 0; dd < dirMax; ++dd)
            dist[dd] = pl ? dirMax - 1 - (static_cast<double>(dd) + distOff) : static_cast<double>(dd) + distOff;

        std::vector<double> sig(dirMax, 0.0);
        std::vector<double> kap(dirMax, 0.0);
        std::vector<double> a(dirMax, 0.0);

        std::transform(dist.begin(), dist.end(), sig.begin(), [&](double dist){return sigma(dist, static_cast<double>(dirMax-1), sigmaMax_ / (pl ? std::sqrt(epsPl_ * muPl_) : std::sqrt(epsMn_ * muMn_) ) ); } );
        std::transform(dist.begin(), dist.end(), kap.begin(), [&](double dist){return kappa(dist, static_cast<double>(dirMax-1)           ); } );
        std::transform(dist.begin(), dist.end(),   a.begin(), [&](double dist){return aVal (dist, static_cast<double>(dirMax-1)           ); } );

        psiParam.b_    = std::vector<double>(dirMax, 0.0);
        psiParam.c_    = std::vector<double>(dirMax, 0.0);
        psiParam.cOff_ = std::vector<double>(dirMax, 0.0);
        for(int dd = 0; dd < dirMax; ++dd)
        {
            psiParam.    b_[dd] = b(sig[dd], a[dd], kap[dd]);
            psiParam.    c_[dd] = c(sig[dd], a[dd], kap[dd]);
            psiParam.cOff_[dd] = c(sig[dd], a[dd], kap[dd]);
            (!E) ? psiParam.c_[dd] *= -1.0 : psiParam.cOff_[dd] *= -1.0;
        }
        return std::make_tuple(psiParam, gridParam);
    }

    /**
     * @brief updates the girds
     */
    void updateGrid()
    {
        for(int ii = 0; ii < updateListGrid_j_.size(); ++ii)
            upPsi_j_(updateListGrid_j_[ii], updateListPsi_j_[ii], tempStore_.data(), grid_i_, psi_j_[ii], grid_k_);
        for(int ii = 0; ii < updateListGrid_k_.size(); ++ii)
            upPsi_k_(updateListGrid_k_[ii], updateListPsi_k_[ii], tempStore_.data(), grid_i_, psi_k_[ii], grid_j_);
    }

    /**
     * @brief      calculates b from Taflove chapter 7
     *
     * @param[in]  sig   sigma value at that point
     * @param[in]  a     value at that point
     * @param[in]  kap   kappa value at that point
     *
     * @return     b
     */
    inline double b(double sig, double a, double kap)
    {
        return std::exp(-1.0 * (sig/kap + a) * dt_);
    }

    /**
     * @brief      calculates c from Taflove chapter 7
     *
     * @param[in]  sigma value at taht point
     * @param[in]  value at that point
     * @param[in]  kappa value at that point
     *
     * @return     c
     */
    inline double c(double sig, double a, double kap)
    {
        return (sig == 0 && a == 0) ? 0 : sig / (sig*kap + std::pow(kap,2.0)*a) * (std::exp(-1.0 * (sig/kap + a) * dt_) - 1.0);
    }

    /**
     * @brief      calculates a from Taflove chapter 7
     *
     * @param[in]  ii     current point location
     * @param[in]  iiMax  maximum ii value
     *
     * @return     a
     */
    inline double aVal(double ii, double iiMax)
    {
        return (0.0 <= ii && ii <= iiMax) ? aMax_ * std::pow( (ii) / iiMax, ma_) : 0.0;
    }

    /**
     * @brief      calculates kappa from Taflove chapter 7
     *
     * @param[in]  ii current point location
     * @param[in]  iiMax maximum ii value
     *
     * @return     kappa
     */
    inline double kappa(double ii, double iiMax)
    {
        return (0.0 <= ii && ii <= iiMax) ? 1.0 + (kappaMax_ - 1.0) * std::pow((iiMax - ii) / iiMax , m_) : 1.0;
    }

    /**
     * @brief      {calculates sigma from Taflove chapter 7
     *
     * @param[in]  ii       ii current point location
     * @param[in]  iiMax    iiMax maximum ii value
     * @param[in]  eta_eff  The effective $\\eta$ for the point
     *
     * @return     sigma
     */
    inline double sigma(double ii, double iiMax, double sigmaMax)
    {
        return (0.0 <= ii && ii <= iiMax) ? sigmaMax * pow((iiMax - ii) / iiMax , m_) : 0.0;
    }

};
/**
 * @brief functions for psi updates
 *
 */
namespace pmlUpdateFxnIncdCplx
{
    /**
     * @brief      Adds a psi.
     *
     * @param[in]  gridParamList  list of parameters to update grid_i_
     * @param[in]  psiParamList   ist of parameters to update Psi field
     * @param      tempStore      Scratch space for the calculation
     * @param[in]  grid_i         Grid to be updated
     * @param[in]  psi            psi field to update, sued to update grid_i
     * @param[in]  grid           grid used to update psi
     */
    void addPsi(const updateGridParamsIncd& gridParamList, const updatePsiParamsIncd& psiParamList, cplx* tempStore, cplx_grid_ptr grid_i, cplx_grid_ptr psi, cplx_grid_ptr grid);

    /**
     * @brief      update psi field
     *
     * @param      list to loop over to update psi
     * @param[in]  psi field to update
     * @param[in]  grid used to update psi
     */
    void updatePsiField(const updatePsiParamsIncd& paramList, cplx* tempStore, cplx_grid_ptr psi , cplx_grid_ptr grid);
}

class IncdCPMLCplx : public IncdCPML<cplx>
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  grid_i       shared pointer to grid_i (field PML is being applied to)
     * @param[in]  grid_j       shared pointer to grid_j (field polarized in the j direction of the PML; nullptr if none)
     * @param[in]  grid_k       shared pointer to grid_k (field polarized in the k direction of the PML; nullptr if none)
     * @param[in]  pol_i        polarization of grid_i
     * @param[in]  pmlThick     The PML thickness
     * @param[in]  tfsfMs       The tfsf m vector
     * @param[in]  mainGridSz   The main grid size
     * @param[in]  m            scalling factor for sigma
     * @param[in]  ma           scaling factor for a
     * @param[in]  kappaMax     The kappa maximum
     * @param[in]  aMax         max a value
     * @param[in]  d            vector storing the step sizes in all directions
     * @param[in]  dr           step size of the TFSF incd fields
     * @param[in]  dt           time step
     */
    IncdCPMLCplx(cplx_grid_ptr grid_i, cplx_grid_ptr grid_j, cplx_grid_ptr grid_k, POLARIZATION pol_i, int pmlThick, std::array<int,3> tfsfMs, std::array<int,3> mainGridSz, bool useMn, double epsPl, double muPl, double epsMn, double muMn, double m, double ma, double kappaMax, double aMax, std::array<double,3> d, double dr, double dt);
};

#endif