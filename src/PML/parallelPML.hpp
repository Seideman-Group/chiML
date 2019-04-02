/** @file PML/parallelPML.hpp
 *  @brief Stores/updates the CPML fields and adds them to the FDTD grids
 *
 *  A class that store and updates the CPML fields and add them the the FDTD grids for the TFSF surfaces
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef PARALLEL_FDTD_PML
#define PARALLEL_FDTD_PML

#include <src/OBJECTS/Obj.hpp>

/**
 * @brief      parameters to update the $\psi$ fields for CPMLs
 */
struct updatePsiParams
{
    int transSz_; //!< size in the transverse direction (how many points to do blas operations on)
    int stride_; //!< stride for the blas operations
    int ind_; //!< index of the initial blas point in the grids
    int indOff_; //!< index of the offset blas point in the grids
    double b_; //!< the b parameter as defined in chapter 7 of Taflove
    double c_; //!< the c parameter as defined in chapter 7 of Taflove
};
/**
 * @brief      Parameters to update the fields using the $\psi$ fields
 */
struct updateGridParams
{
    int nAx_; //!< size of blas operator
    int stride_; //!< stride for blas operator
    int ind_; //!< index of the initial blas point in the grids
    int indOff_; //!< index of the offset blas point in the grids
    double Db_; //!< prefactor for psi field terms
    double DbField_; //!< prefactor for direct field terms
};


template <typename T> class parallelCPML
{
    typedef std::shared_ptr<parallelGrid<T>> pgrid_ptr;
protected:
    std::shared_ptr<mpiInterface> gridComm_; //!< MPI Communicator
    POLARIZATION pol_i_; //!< Polarization of the Field that the CPML is acting on
    DIRECTION i_; //!< Direction corresponding to polarization of the field the CPML is acting on
    DIRECTION j_; //!< Direction next in the cycle from i_; i.e. if i_ is DIRECTION::Y then j_ is DIRECTION::Z
    DIRECTION k_; //!< Direction last one in the cycle from i_; i.e. if i_ is DIRECTION::Y then j_ is DIRECTION::X
    double m_; //!< Exponent for calculating CPML constants sigma and kappa
    double ma_; //!< Exponent for calculating CPML constant a
    double sigmaMaxX_; //!< Maximum value of sigma
    double sigmaMaxY_; //!< Maximum value of sigma
    double sigmaMaxZ_; //!< Maximum value of sigma
    double kappaMax_; //!< Maximum value of kappa
    double aMax_; //!< Maximum value of a
    double dt_; //!< time step
    std::array<int,3> n_vec_; //!< vector storing the thickness of the PML in all directions
    std::array<int,3> ln_vec_pl_; //!< vector storing the local thickness of the PMLs in the positive directions (top, right, and front)
    std::array<int,3> ln_vec_mn_; //!< vector storing the local thickness of the PMLs in the positive directions (bottom, left, and back)
    std::array<double,3> d_; //!< vector storing the step size in all directions
    std::vector<double> eta_eff_top_; //!< eta effective along top PML
    std::vector<double> eta_eff_bot_; //!< eta effective along bottom PML
    std::vector<double> eta_eff_left_; //!< eta effective along left PML
    std::vector<double> eta_eff_right_; //!< eta effective along right PML
    std::vector<double> eta_eff_front_; //!< eta effective along right PML
    std::vector<double> eta_eff_back_; //!< eta effective along right PML
    std::function<void(const std::vector<updateGridParams>&, const std::vector<updatePsiParams>&, pgrid_ptr, pgrid_ptr, pgrid_ptr)> addGrid_j_; //!< update function for the $\\psi_j$ field
    std::function<void(const std::vector<updateGridParams>&, const std::vector<updatePsiParams>&, pgrid_ptr, pgrid_ptr, pgrid_ptr)> addGrid_k_; //!< update function for the $\\psi_k$ field

public:
    pgrid_ptr grid_i_; //!< FDTD field polarized in the i direction (E/H) is the same as it is in pol_i
    pgrid_ptr grid_j_; //!< FDTD field polarized in the j direction (E/H) is the opposite as it is in pol_i
    pgrid_ptr grid_k_; //!< FDTD field polarized in the k direction (E/H) is the opposite as it is in pol_i
    pgrid_ptr psi_j_; //!< CPML helper field $\\psi$ polarized in the j direction
    pgrid_ptr psi_k_; //!< CPML helper field $\\psi$ polarized in the k direction
    std::vector<updatePsiParams>  updateListPsi_j_; //!< update parameters for $\\psi_j$ field
    std::vector<updatePsiParams>  updateListPsi_k_; //!< update parameters for $\\psi_k$ field
    std::vector<updateGridParams> updateListGrid_j_; //!< update parameters for the updates in the j direction
    std::vector<updateGridParams> updateListGrid_k_; //!< update parameters for the updates in the k direction

    /**
     * @brief      Constructor
     *
     * @param[in]  gridComm      mpi communicator
     * @param[in]  weights       The weights
     * @param[in]  grid_i        shared pointer to grid_i (field PML is being applied to)
     * @param[in]  grid_j        shared pointer to grid_j (field polarized in the j direction of the PML; nullptr if none)
     * @param[in]  grid_k        shared pointer to grid_k (field polarized in the k direction of the PML; nullptr if none)
     * @param[in]  pol_i         polarization of grid_o
     * @param[in]  n_vec         Vector storing thickness of the PMLs in all directions
     * @param[in]  m             scalling factor for sigma
     * @param[in]  ma            scaling factor for a
     * @param[in]  sigOptMaxRat  The ratio between the optimal sigma value in Taflove 2005 ch 7 and sigma max
     * @param[in]  kappaMax      The maximum kappa value
     * @param[in]  aMax          max a value
     * @param[in]  d             vector storing the step sizes in all directions
     * @param[in]  dt            time step
     * @param[in]  physGrid      The physical grid
     * @param[in]  objArr        The object arr
     */
    parallelCPML(std::shared_ptr<mpiInterface> gridComm, std::vector<real_grid_ptr> weights, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k, POLARIZATION pol_i, std::array<int,3> n_vec, double m, double ma, double sigOptMaxRat, double kappaMax, double aMax, std::array<double,3> d, double dt, bool matInPML, int_pgrid_ptr physGrid, std::vector<std::shared_ptr<Obj>> objArr) :
        gridComm_(gridComm),
        pol_i_(pol_i),
        n_vec_(n_vec),
        ln_vec_mn_({{ 0,0,0,}}),
        ln_vec_pl_({{ 0,0,0,}}),
        m_(m),
        ma_(ma),
        sigmaMaxX_(sigOptMaxRat*0.8*(m+1)/d[0]),
        sigmaMaxY_(sigOptMaxRat*0.8*(m+1)/d[1]),
        sigmaMaxZ_(sigOptMaxRat*0.8*(m+1)/d[2]),
        kappaMax_(kappaMax),
        aMax_(aMax),
        d_(d),
        dt_(dt),
        grid_i_(grid_i),
        grid_j_(grid_j),
        grid_k_(grid_k)
    {
        if(!grid_i)
            throw std::logic_error("PMLs have to have a real field to be associated with, grid_i_ can't be a nullptr.");
        // Determine the PML's i,j,k directions based on the polarization of the field, direction i shares the direction of the field polarization
        if(pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EX)
        {
            i_ = DIRECTION::X;
            j_ = DIRECTION::Y;
            k_ = DIRECTION::Z;
        }
        else if(pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EY)
        {
            i_ = DIRECTION::Y;
            j_ = DIRECTION::Z;
            k_ = DIRECTION::X;
        }
        else
        {
            i_ = DIRECTION::Z;
            j_ = DIRECTION::X;
            k_ = DIRECTION::Y;
        }
        // In 2D z direction is assumed to be isotropic, but 3D it is not
        if(grid_k_ && (grid_i_->local_z() != 1 || j_ != DIRECTION::Z) )
        {
            psi_j_ = std::make_shared<parallelGrid<T>>(gridComm, grid_k_->PBC(), weights, grid_k_->n_vec(), grid_k_->d() );
        }
        else
            psi_j_ = nullptr;

        // In 2D z direction is assumed to be isotropic, but in 3D it is not
        if(grid_j_ && (grid_i_->local_z() != 1 || k_ != DIRECTION::Z) )
        {
            psi_k_ = std::make_shared<parallelGrid<T>>(gridComm, grid_j_->PBC(), weights, grid_j_->n_vec(), grid_j_->d() );
        }
        else
            psi_k_ = nullptr;

        if(!psi_k_ && ! psi_j_)
            throw std::logic_error("Both PML auxiliary fields are undefined this likely means that there is an issue with grid_i or grid_k");

        findLnVecs();
        // Calculate the mean $\eta_r$ values for each PML
        genEtaEff(physGrid, DIRECTION::X, false, matInPML, objArr, eta_eff_left_);
        genEtaEff(physGrid, DIRECTION::X,  true, matInPML, objArr, eta_eff_right_);

        genEtaEff(physGrid, DIRECTION::Y, false, matInPML, objArr, eta_eff_bot_);
        genEtaEff(physGrid, DIRECTION::Y,  true, matInPML, objArr, eta_eff_top_);


        if(grid_i_->z() != 1)
        {
            genEtaEff(physGrid, DIRECTION::Z, false, matInPML, objArr, eta_eff_back_);
            genEtaEff(physGrid, DIRECTION::Z,  true, matInPML, objArr, eta_eff_front_);
        }
        // Initialize all update lists
        initalizeLists(ln_vec_mn_[0], DIRECTION::X, false, grid_i_->procLoc(0)                       , n_vec_[0]                      , n_vec_[0], ln_vec_mn_[0]);
        initalizeLists(ln_vec_pl_[0], DIRECTION::X,  true, grid_i_->procLoc(0) + grid_i_->local_x()-2, grid_i_->x()-2*gridComm_->npX(), n_vec_[0], ln_vec_pl_[0]);
        initalizeLists(ln_vec_mn_[1], DIRECTION::Y, false, grid_i_->procLoc(1)                       , n_vec_[1]                      , n_vec_[1], ln_vec_mn_[1]);
        initalizeLists(ln_vec_pl_[1], DIRECTION::Y,  true, grid_i_->procLoc(1) + grid_i_->local_y()-2, grid_i_->y()-2*gridComm_->npY(), n_vec_[1], ln_vec_pl_[1]);
        initalizeLists(ln_vec_mn_[2], DIRECTION::Z, false, grid_i_->procLoc(2)                       , n_vec_[2]                      , n_vec_[2], ln_vec_mn_[2]);
        initalizeLists(ln_vec_pl_[2], DIRECTION::Z,  true, grid_i_->procLoc(2) + grid_i_->local_z()-2, grid_i_->z()-2*gridComm_->npZ(), n_vec_[2], ln_vec_pl_[2]);
    }

    /**
     * @brief      calculates the effective mean value of $\eta_r$ for a plane in the FDTD cell
     *
     * @param[in]  physVals        Slice from the object map grid
     * @param[in]  objArr          The object arr
     *
     * @return     value of $\sqrt{\varepsilon_{r,eff} \mu_{r,eff}}$
     */
    double genEtaEffVal(const std::vector<int>& physVals, std::vector<std::shared_ptr<Obj>> objArr)
    {
        double eps_sum = 0.0;
        double mu_sum = 0.0;
        // Sum up all the dielectric constants in the plane, if it is a boundary region don't add it
        for(auto& val : physVals)
        {
            if(val != -1)
            {
                eps_sum += objArr[val]->epsInfty() / static_cast<double>( physVals.size() );
                mu_sum += objArr[val]->muInfty() / static_cast<double>( physVals.size() );
            }
        }
        double etaSum = eps_sum* mu_sum;
        // Find the average dielectric constant if outside the boundary regions
        return (etaSum != 0.00) ? etaSum : 0;
    }

    /**
     * @brief      For the entire thickness of the PML find the values of the effective wave impedence
     *
     * @param[in]  physGrid      The object map grid
     * @param[in]  planeNormDir  Direction of the normal vector to the PML
     * @param[in]  pl            True if a pl PML
     * @param[in]  objArr        The object arr
     * @param      eta_eff       Vector storing the effective wave impedance for the PML
     */
    void genEtaEff(int_pgrid_ptr physGrid, DIRECTION planeNormDir, bool pl, bool matInPML, std::vector<std::shared_ptr<Obj>> objArr, std::vector<double>& eta_eff)
    {
        int cor_ii = -1, cor_jj = -1, cor_kk = -1;
        std::array<int, 2> planeLoc = {0, 0};
        std::array<int, 3> planeSzTemp  = {grid_i_->x()-2*gridComm_->npArr(0), grid_i_->y()-2*gridComm_->npArr(1), grid_i_->z()-2*gridComm_->npArr(2)};
        if(pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::HZ )
            planeSzTemp[0] -= 1;
        if(pol_i_ == POLARIZATION::EY || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HZ )
            planeSzTemp[1] -= 1;
        if(pol_i_ == POLARIZATION::EZ || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HY )
            planeSzTemp[2] -= 1;

        if(grid_i_->z() == 1)
            planeSzTemp[2] = 1;

        std::array<int,2> planeSz;

        if(planeNormDir == DIRECTION::X)
        {
            cor_ii = 0; cor_jj = 1; cor_kk = 2;
            planeSz = { planeSzTemp[2], planeSzTemp[1] };
        }
        else if(planeNormDir == DIRECTION::Y)
        {
            cor_ii = 1; cor_jj = 2; cor_kk = 0;
            planeSz = { planeSzTemp[0], planeSzTemp[2] };
        }
        else if(planeNormDir == DIRECTION::Z)
        {
            cor_ii = 2; cor_jj = 0; cor_kk = 1;
            planeSz = { planeSzTemp[0], planeSzTemp[1] };
        }

        // find values to loop over
        int min = pl ? physGrid->n_vec(cor_ii) - 2 * gridComm_->npArr(cor_ii) - n_vec_[cor_ii] : 0;
        int max = pl ? physGrid->n_vec(cor_ii) - 2 * gridComm_->npArr(cor_ii)                  : n_vec_[cor_ii] ;
        // loop over those values
        for(int ii = min; ii < max; ++ii)
        {
            double etaVal = 1.0;
            if(planeNormDir == DIRECTION::X)
            {
                etaVal = genEtaEffVal( physGrid->getPlaneYZ(ii, planeLoc, planeSz), objArr);
            }
            else if(planeNormDir == DIRECTION::Y)
            {
                etaVal = genEtaEffVal( physGrid->getPlaneXZ(ii, planeLoc, planeSz), objArr);
            }
            else if(planeNormDir == DIRECTION::Z)
            {
                etaVal = genEtaEffVal( physGrid->getPlaneXY(ii, planeLoc, planeSz), objArr);
            }
            // Find the average dielectric constant if outside the boundary regions
            if(eta_eff.size() < n_vec_[cor_ii] && etaVal != 0.0)
                eta_eff.push_back( etaVal );
        }
    }

    /**
     * @brief      Finds each process's local PML thicknesses
     */
    void findLnVecs()
    {
        // 1) Does the process inside the left PML? yes-> 2) Does the PML extend outside the process's region yes-> ln_vec_mn_[0] = local size of process, no-> ln_vec_mn_[0] = PML thickness-where the process started
        if(grid_i_->procLoc(0) < n_vec_[0])
            ln_vec_mn_[0] = grid_i_->procLoc(0) + grid_i_->local_x()-2 < n_vec_[0] ? grid_i_->local_x()-2 : n_vec_[0] - grid_i_->procLoc(0);

        // 1) Does the right PML begin before the process's region end? yes-> 2) Does the process begin after the PML begins? yes->ln_vec_pl_[0] = local size of process, no-> ln_vec_pl_[0] = the process's size - PML thickness
        if(grid_i_->procLoc(0) + grid_i_->local_x()-2 > grid_i_->x() - 2*gridComm_->npX() - n_vec_[0])
            ln_vec_pl_[0] = grid_i_->procLoc(0) > grid_i_->x() - 2*gridComm_->npX() - n_vec_[0] ? grid_i_->local_x()-2 : grid_i_->procLoc(0) + grid_i_->local_x() - 2 - (grid_i_->x() - 2*gridComm_->npX() - n_vec_[0]);;

        // 1) Does the process inside the bottom PML? yes-> 2) Does the PML extend outside the process's region yes-> ln_vec_mn_[1] = local size of process, no-> ln_vec_mn_[1] = PML thickness-where the process started
        if(grid_i_->procLoc(1) < n_vec_[1])
            ln_vec_mn_[1] = grid_i_->procLoc(1) + grid_i_->local_y()-2 < n_vec_[1] ? grid_i_->local_y()-2 : n_vec_[1] - grid_i_->procLoc(1);

        // 1) Does the top PML begin before the process's region end? yes-> 2) Does the process begin after the PML begins? yes->ln_vec_pl_[1] = local size of process, no-> ln_vec_pl_[1] = the process's size - PML thickness
        if(grid_i_->procLoc(1) + grid_i_->local_y()-2 > grid_i_->y() - gridComm_->npY()*2 - n_vec_[1])
            ln_vec_pl_[1] = grid_i_->procLoc(1) > grid_i_->y() - gridComm_->npY()*2 - n_vec_[1] ? grid_i_->local_y()-2 : grid_i_->procLoc(1) + grid_i_->local_y() - 2 - (grid_i_->y() - 2*gridComm_->npY() - n_vec_[1]);

        if(grid_i_->local_z() != 1)
        {
            // 1) Does the process inside the back PML? yes-> 2) Does the PML extend outside the process's region yes-> ln_vec_mn_[2] = local size of process, no-> ln_vec_mn_[2] = PML thickness-where the process started
            if(grid_i_->procLoc(2) < n_vec_[2])
                ln_vec_mn_[2] = grid_i_->procLoc(2) + grid_i_->local_z()-2 < n_vec_[2] ? grid_i_->local_z()-2 : n_vec_[2] - grid_i_->procLoc(2);

            // 1) Does the front PML begin before the process's region end? yes-> 2) Does the process begin after the PML begins? yes->ln_vec_pl_[2] = local size of process, no-> ln_vec_pl_[2] = the process's size - PML thickness
            if(grid_i_->procLoc(2) + grid_i_->local_z()-2 > grid_i_->z() - gridComm_->npZ()*2 - n_vec_[2])
                ln_vec_pl_[2] = grid_i_->procLoc(2) > grid_i_->z() - gridComm_->npZ()*2 - n_vec_[2] ? grid_i_->local_z()-2 : grid_i_->procLoc(2) + grid_i_->local_z() - 2 - (grid_i_->z() - 2*gridComm_->npZ() - n_vec_[2]);
        }
    }

    /**
     * @brief      Generate the psi field update lists.
     *
     * @param[in]  dir      Direction of the psi field.
     * @param[in]  pl       True if top, right or front.
     * @param[in]  startPt  Where the PMLs starts.
     * @param[in]  pmlEdge  Where the PML ends
     * @param[in]  nDir     Thickness in that direction.
     * @param[in]  dirMax   How far to iterate over.
     * @param      psiList  The psi up list.
     */
    void getPsiUpList(DIRECTION dir, bool pl, int startPt,  int pmlEdge, int nDir, int dirMax, std::vector<updatePsiParams>& psiList)
    {
        updatePsiParams param; // Base param object that will be modified and added to psiList

        bool E = ( pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::EY || pol_i_ == POLARIZATION::EZ ) ? true : false;
        int transSz2 = 1;
        std::vector<double> *eta_eff;

        double sigmaMax; // Maximum sigma value based on the value of d_[dir]
        int cor_norm = -1; // Index for all spatial arrays corresponding to the direction normal to the PML surface
        int cor_trans1 = -1; // Index for all spatial arrays corresponding to the direction of blas commands
        int cor_trans2 = -1; // Index for all spatial arrays corresponding to the transverse direction that blas commands are not acting on
        int trans1FieldOff = 0; // offset to blas operation size due to grid sizes not being the same for a yee cell
        int trans2FieldOff = 0; // offset to transverse grid size due to grid sizes not being the same for a yee cell
        int ccStart = 0; // where to start the loop over pmls
        // Determine the coordinate system for the PML update lists compared to x,y,z
        if(dir == DIRECTION::X)
        {
            sigmaMax = sigmaMaxX_;
            cor_norm = 0;
            // 3D calculation do blas operations over z coordinate; for 2D do blas operations over y coordinate
            cor_trans1 = (grid_i_->local_z() == 1) ? 1 : 2;
            cor_trans2 = (grid_i_->local_z() == 1) ? 2 : 1;

            param.stride_ = grid_i_->local_x();
            // What eta effective vector should be used
            eta_eff = !pl ? &eta_eff_left_ : &eta_eff_right_;

            // Ey, Hx, and Hz fields have one less point in the y direction
            if( (pol_i_ == POLARIZATION::EY || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HZ ) && gridComm_->rank() == gridComm_->size() - 1)
                (grid_i_->local_z() == 1) ? trans1FieldOff = 1 : trans2FieldOff = 1;

            // Ez, Hx, and Hy fields have one less point in the z direction
            if( (pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EZ ) )
                (grid_i_->local_z() == 1) ? trans2FieldOff = 1 : trans1FieldOff = 1;

            // PML start / Max conditions depending on where in the map they are
            if( pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EX)
                pl ? ccStart = 1 : dirMax -= 1;
        }
        else if(dir == DIRECTION::Y)
        {
            sigmaMax = sigmaMaxY_;
            cor_norm = 1;
            cor_trans1 = 0;
            cor_trans2 = 2;

            param.stride_ = 1;
            // What eta effective vector should be used
            eta_eff = !pl ? &eta_eff_bot_ : &eta_eff_top_;
            // Ex, Hy, and Hz fields have one less point in the x direction
            if(pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::HZ )
                trans1FieldOff = 1;
            // Ez, Hx, and Hy fields have one less point in the z direction
            if( (pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EZ ) )
                trans2FieldOff = 1;
            // Hx, Ey, and Hz fields all have one less point in the z direction
            if( ( pl && gridComm_->rank() == gridComm_->size()-1) && (pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EY) )
                ccStart = 1;
            if( ( !pl && startPt + dirMax == nDir)  && (pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EY) )
                dirMax -= 1;
        }
        else if(dir == DIRECTION::Z)
        {
            sigmaMax = sigmaMaxZ_;
            cor_norm = 2;
            cor_trans1 = 0;
            cor_trans2 = 1;
            param.stride_ = 1;
            // What eta effective vector should be used
            eta_eff = !pl ? &eta_eff_back_ : &eta_eff_front_;
            // Ex, Hy, and Hz fields have one less point in the x direction
            if(pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::HZ )
                trans1FieldOff = 1;
            // Ey, Hx, and Hz fields have one less point in the y direction
            if( (pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::EY ) && gridComm_->rank() == gridComm_->size() - 1)
                trans2FieldOff = 1;
            // Hx, Hy, and Ez fields all have one less point in the z direction
            if( pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EZ )
                pl ? ccStart = 1 : dirMax -= 1;
        }
        else
            throw std::logic_error("PMLs need to be in X,Y, or Z direction it can't be NONE");

        param.transSz_ = grid_i_->ln_vec(cor_trans1) - 2 - trans1FieldOff;
        transSz2       = grid_i_->ln_vec(cor_trans2) - 2 - trans2FieldOff;
        if(grid_i_->local_z() == 1 && cor_trans2 == 2)
            transSz2 = 1;

        // Magnetic fields are all 0.5 units off the main grid points in a direction not along its polarization (which are not included.)
        double distOff = E ? 0.0 : 0.5;

        // initialize the loc and locOff to -1, -1, -1
        std::array<int, 3> loc   = {-1, -1, -1};
        // blas operations will always start at 1 (transSz)
        loc[cor_trans1] = 1;

        // Get the constants for all directions by looping over the PML thickness
        for(int cc = ccStart; cc < dirMax; ++cc)
        {
            // loc[cor_norm] is based on the distance and the locOff_[cor_norm] would be + 1 if an E field polarization, and - 1 if an H field polarization
            loc   [cor_norm] = pl ? grid_i_->ln_vec(cor_norm) - 2 - cc : cc+1;
            // calculate distance from edge (facing into the cell)
            double dist = grid_i_->procLoc(cor_norm) + (loc[cor_norm]-1) + distOff;
            if(pl)
                dist = grid_i_->n_vec(cor_norm)-2*gridComm_->npArr(cor_norm) - 1 - dist;

            double sig = sigma( dist, static_cast<double>(nDir-1), eta_eff->at(dist), sigmaMax);
            double kap = kappa( dist, static_cast<double>(nDir-1) );
            double a   = aVal ( dist, static_cast<double>(nDir-1) );

            param.b_    = b( sig, a, kap );
            param.c_    = c( sig, a, kap ) / d_[cor_norm];
            if(!E)
                param.c_ *= -1.0;
            // along the transSz2 direction  get the offset location from the update and then put the param set at the end of the vector
            for(int jj = 0; jj < transSz2 ; ++jj)
            {
                // looping parameters start at 1+jj
                loc   [cor_trans2] = 1+jj;
                if(grid_i_->local_z() == 1)
                    loc[2] = 0;
                std::array<int, 3> locOff(loc);
                locOff[cor_norm] += E ? -1 : 1;
                // if 2D z coordinate will be 0 not one so subtract 1
                param.ind_    = grid_i_->getInd(loc);
                param.indOff_ = grid_i_->getInd(locOff);
                psiList.push_back(param);
            }
        }
    }

    /**
     * @brief generate the lists that add psi to grid
     *
     * @param[in] min     an array describing minimum values in each direction
     * @param[in] max     an array describing maximum values in each direction
     * @param[in] stride  stride for blas operations

     * @return ax lists to generate the lists that add psi to grid
     */
    std::vector<std::array<int,5>> getAxLists(std::array<int,3> min, std::array<int,3> max, int stride)
    {
        // Loop over the PMLS and add the PML updates are independent of the object maps since they act on the D field if not in vacuum, the code is commented out in case this needs to change.
        std::vector<std::array<int,5>> axLists;
        if(stride == 1)
        {
            for(int kk = min[2]; kk < max[2]; ++kk)
                for(int jj = min[1]; jj < max[1]; ++jj)
                    axLists.push_back( {{ min[0], jj, kk, max[0]-min[0], 0 }} );
        }
        else
        {
            // If 2D do the blas operations on the Y direction, if 3D do blas operations in the Z direction
            if(grid_i_->local_z()==1)
            {
                for(int kk = min[2]; kk < max[2]; ++kk)
                    for(int ii = min[0]; ii < max[0]; ++ii)
                        axLists.push_back( {{ ii, min[1], kk, max[1]-min[1], 0 }} );
            }
            else
            {
                for(int jj = min[1]; jj < max[1]; ++jj)
                    for(int ii = min[0]; ii < max[0]; ++ii)
                        axLists.push_back( {{ ii, jj, min[2], max[2]-min[2], 0 }} );
            }
        }
        return axLists;
    }

    /**
     * @brief      Generates the lists that add psi fields to grid_i
     *
     * @param[in]  dir       direction of pml
     * @param[in]  derivDir  The direction of the derivative
     * @param[in]  pl        true if top or right
     * @param[in]  grid      The grid that is used to update grid_i_
     * @param      gridList  The grid up list.
     */
    void getGridUpList(DIRECTION dir, DIRECTION derivDir, bool pl, pgrid_ptr grid, std::vector<updateGridParams>& gridList)
    {
        updateGridParams param; // Base param object that will be used to add to gridList
        std::array<int,3> min = {0,0,0}; // minimum grid point value in all directions
        std::array<int,3> max = {0,0,0}; // maximum grid point value in all direction
        int cor_ii = 0; // Index of the spatial arrays corresponding to the direction dir
        int cor_jj = 0; // Index of the spatial arrays corresponding to the direction j if i = dir
        int cor_kk = 0; // Index of the spatial arrays corresponding to the direction k if i = dir
        int off_ii = 0; // offset of the grid size in direction ii based of the yee grid sizes
        int corDeriv_ii = 0; // Index of the spatial arrays corresponding to the direction derivDir
        int corDeriv_jj = 0; // Index of the spatial arrays corresponding to the direction j if i = derivDir
        int corDeriv_kk = 0; // Index of the spatial arrays corresponding to the direction k if i = derivDir
        // Initialize the Db constants to dt_
        param.Db_ = dt_;
        double DbFieldBase = dt_;
        // Initialize the offset array
        std::array<int,3> offset = {0,0,0};
        // Create a bool the check E
        bool E = (pol_i_ == POLARIZATION::EZ || pol_i_ == POLARIZATION::EY || pol_i_ == POLARIZATION::EX) ? true : false;
        // Determine the x, y, and z min/max for each direction. For transverse directions it is the field size limitations and the normal direction is based on the PML thickness
        if(dir == DIRECTION::X)
        {
            cor_ii = 0; cor_jj = 1; cor_kk = 2;
            param.stride_ = grid_i_->local_x();
            // offset in the ii direction is based on taking the mirror image of the point in the opposite PML
            if(pl && ( pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::HY ) )
                off_ii = -1;
            else if( !pl )
                off_ii = 1;
        }
        else if(dir == DIRECTION::Y)
        {
            cor_ii = 1; cor_jj = 2; cor_kk = 0;
            param.stride_ = 1;
            // offset in the ii direction is based on taking the mirror image of the point in the opposite PML
            if( ( pl && gridComm_->rank() == gridComm_->size()-1 ) && ( pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EY ) )
                off_ii = -1;
            if( !pl  ) //&& ( grid_i_->local_y()-2 == ln_vec_mn_[1] ) )
                off_ii = 1;

        }
        else if(dir == DIRECTION::Z)
        {
            cor_ii = 2; cor_jj = 0; cor_kk = 1;
            param.stride_ = 1;
            // offset in the ii direction is based on taking the mirror image of the point in the opposite PML
            if(pl && ( pol_i_ == POLARIZATION::EZ || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HY ) )
                off_ii = -1;
            else if( !pl )
                off_ii = 1;

        }
        else
            throw std::logic_error("Direction for grid updates has to be X, Y or Z not NONE");

        if(derivDir == DIRECTION::X)
        {
            corDeriv_ii = 0; corDeriv_jj = 1; corDeriv_kk = 2;
            // Prefactor sign can be determined by looking at Taflove Chapter 7
            if(pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::EY)
                param.Db_*=-1.0;
            if(pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EY)
                DbFieldBase *= -1.0;
        }
        else if(derivDir == DIRECTION::Y)
        {
            corDeriv_ii = 1; corDeriv_jj = 2; corDeriv_kk = 0;
            // Prefactor sign can be determined by looking at Taflove Chapter 7
            if(pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EZ)
                param.Db_*=-1.0;
            if(pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::EZ)
                DbFieldBase *= -1.0;
        }
        else if(derivDir == DIRECTION::Z)
        {
            corDeriv_ii = 2; corDeriv_jj = 0; corDeriv_kk = 1;
            // Prefactor sign can be determined by looking at Taflove Chapter 7
            if(pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EX)
                param.Db_*=-1.0;
            if(pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EX)
                DbFieldBase *= -1.0;
        }
        else
            throw std::logic_error("Direction for grid updates has to be X, Y or Z not NONE");
        // Base value for Db_ for field additions scaled by grid spacing in the derivative direction
        DbFieldBase /= d_[corDeriv_ii];
        offset[corDeriv_ii] = ( pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::EY || pol_i_ == POLARIZATION::EZ ) ? -1 : 1;

        // Set min and max values for every direction based on ii, jj, and kk
        min[cor_ii] += pl ? grid_i_->ln_vec(cor_ii) - ln_vec_pl_[cor_ii] - 1  : 1 ;
        max[cor_ii] += off_ii + (pl ? grid_i_->ln_vec(cor_ii) -1 : ln_vec_mn_[cor_ii]);

        min[cor_jj] = 1;
        max[cor_jj] = grid_i_->ln_vec(cor_jj)-1;

        min[cor_kk] = 1;
        max[cor_kk] = grid_i_->ln_vec(cor_kk)-1;

        // To avoid updating fields twice subtract out PML thicknesses in transverse directions (condition is due to each PML list corresponding to a single j/k update)
        if( dir == i_ )
        {
            min[cor_jj] += ln_vec_mn_[cor_jj];
            max[cor_jj] -= ln_vec_pl_[cor_jj] - ( (E || (cor_jj == 1 && gridComm_->rank() != gridComm_->size()-1 ) ) ? 0 : 1 );

            min[cor_kk] += ln_vec_mn_[cor_kk];
            max[cor_kk] -= ln_vec_pl_[cor_kk] - ( (E || (cor_kk == 1 && gridComm_->rank() != gridComm_->size()-1 ) ) ? 0 : 1 );
        }
        else if( corDeriv_ii != cor_ii )
        {
            min[corDeriv_ii] += ln_vec_mn_[corDeriv_ii];
            max[corDeriv_ii] -= ln_vec_pl_[corDeriv_ii] - ( (E || (corDeriv_ii == 1 && gridComm_->rank() != gridComm_->size()-1 ) ) ? 0 : 1 );
        }
        // if(corDeriv_ii != cor_ii)
        // {
        //     min[corDeriv_jj] += ln_vec_mn_[corDeriv_jj];
        //     max[corDeriv_jj] -= ln_vec_pl_[corDeriv_jj];
        // }
        // if(corDeriv_kk == cor_ii)
        // {
        //     min[corDeriv_kk] += ln_vec_mn_[corDeriv_kk];
        //     max[corDeriv_kk] -= ln_vec_pl_[corDeriv_kk];
        // }

        // Ex, Hy, Hz fields have one less point in the x direction
        if( cor_ii != 0 && ( pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::EX || pol_i_ == POLARIZATION::HY ) )
            max[0] -= 1;
        // Hx, Ey, Hz fields have one less point in the y direction
        if( cor_ii != 1 && ( gridComm_->rank() == gridComm_->size()-1 ) && (pol_i_ == POLARIZATION::HZ || pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::EY) )
            max[1] -= 1;
        // Hx, Hy, Ez fields have one less point in the z direction
        if( cor_ii != 2 && grid_i_->local_z() != 1 && ( pol_i_ == POLARIZATION::HX || pol_i_ == POLARIZATION::HY || pol_i_ == POLARIZATION::EZ ) )
            max[2] -= 1;

        if( grid_i_->n_vec(2) == 1 )
        {
            min[2] = 0;
            max[2] = 1;
        }

        std::vector<std::array<int,5>> axParams = getAxLists(min, max, param.stride_);
        // Distance offset by 1/2 of grid spacing for H fieds
        double distOff = E ? 0.0 : 0.5;
        for(auto & axList : axParams)
        {
            // Calculate kappa in i direction
            double dist = grid_i_->procLoc(corDeriv_ii) + axList[corDeriv_ii] - 1 + distOff;
            if(pl)
                dist = grid_i_->n_vec(corDeriv_ii)-2*gridComm_->npArr(corDeriv_ii) - 1 - dist;
            double kap = kappa( dist, static_cast<double>(n_vec_[corDeriv_ii]-1) );
            // scale DbField value by kappa
            param.DbField_ = DbFieldBase / kap;
            param.ind_ = grid->getInd(axList[0], axList[1], axList[2]);
            param.indOff_ = grid->getInd(axList[0] + offset[0], axList[1] + offset[1], axList[2] + offset[2]);
            param.nAx_ = axList[3];
            gridList.push_back(param);
        }
    }

    /**
     * @brief      Initializes the update list with the correct parameters
     *
     * @param[in]  ln_pml   local size of the PML thickness
     * @param[in]  dir      The direction of polarization of the $\psi$ field
     * @param[in]  pl       True if top, right or front.
     * @param[in]  startPt  Where the PMLs starts.
     * @param[in]  pmlEdge  Where the PML ends
     * @param[in]  nDir     Thickness in that direction.
     * @param[in]  dirMax   How far to iterate over.
     */
    void initalizeLists(int ln_pml, DIRECTION dir, bool pl, int startPt, int pmlEdge, int nDir, int dirMax)
    {
        if(ln_pml == 0)
            return;
        if(grid_k_)
        {
            if( dir == j_)
            {
                getPsiUpList(dir, pl, startPt, pmlEdge, nDir, dirMax, updateListPsi_j_);
            }
            getGridUpList(dir, j_, pl, grid_k_, updateListGrid_k_);
        }
        if(grid_j_)
        {
            if( dir == k_)
            {
                getPsiUpList(dir, pl, startPt, pmlEdge, nDir, dirMax, updateListPsi_k_);
            }
            getGridUpList(dir, k_, pl, grid_j_, updateListGrid_j_);
        }
    }
    /**
     * @brief updates the grids
     */
    void updateGrid()
    {
        addGrid_j_(updateListGrid_k_, updateListPsi_j_, grid_i_, psi_j_, grid_k_);
        addGrid_k_(updateListGrid_j_, updateListPsi_k_, grid_i_, psi_k_, grid_j_);
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
        return (sig == 0 && a == 0) ? 0 : sig * ( b(sig, a, kap) - 1.0 ) / ( kap * (sig + kap*a) );
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
        return (0.0 <= ii && ii <= iiMax) ? aMax_ * std::pow( ii / iiMax, ma_) : 0.0;
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
    inline double sigma(double ii, double iiMax, double eta_eff, double sigmaMax)
    {
        return (0.0 <= ii && ii <= iiMax) ? sigmaMax / eta_eff * pow((iiMax - ii) / iiMax , m_) : 0.0;
    }
    /**
     * @brief      ln_vec_right accessor function
     *
     * @return     ln_vec_right
     */
    inline int lnx_right() {return ln_vec_pl_[0];}
    /**
     * @brief      ln_vec_left accessor function
     *
     * @return     ln_vec_left
     */
    inline int lnx_left()  {return ln_vec_mn_[0] ;}
    /**
     * @brief      ln_vec_top accessor function
     *
     * @return     ln_vec_top
     */
    inline int lny_top()   {return ln_vec_pl_[1]  ;}
    /**
     * @brief      ln_vec_bot accessor function
     *
     * @return     ln_vec_bot
     */
    inline int lny_bot()   {return ln_vec_mn_[1]  ;}

     /**
     * @brief      ln_vec_top accessor function
     *
     * @return     ln_vec_top
     */
    inline int lnz_back()   {return ln_vec_pl_[2]  ;}
    /**
     * @brief      ln_vec_bot accessor function
     *
     * @return     ln_vec_bot
     */
    inline int lnz_front()   {return ln_vec_mn_[2]  ;}
};
/**
 * @brief functions for psi updates
 *
 */
namespace pmlUpdateFxnReal
{
    /**
     * @brief      Adds psi to grid_i_
     *
     * @param[in]  gridParamList  list of parameters to update grid_i_
     * @param[in]  psiParamList   ist of parameters to update Psi field
     * @param[in]  grid_i         grid_i_
     * @param[in]  psi            psi field to update
     * @param[in]  grid           grid used to update psi
     */
    void addPsi(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, real_pgrid_ptr grid_i, real_pgrid_ptr psi, real_pgrid_ptr grid);

    /**
     * @brief      Updates the gird in the PML region excluding the psi fields
     *
     * @param[in]  gridParamList  list of parameters to update grid_i_
     * @param[in]  psiParamList   ist of parameters to update Psi field
     * @param[in]  grid_i         grid_i_
     * @param[in]  psi            psi field to update
     * @param[in]  grid           grid used to update psi
     */
    void addGridOnly(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, real_pgrid_ptr grid_i, real_pgrid_ptr psi, real_pgrid_ptr grid);

    /**
     * @brief      update psi field
     *
     * @param[in]  list to loop over to update psi
     * @param[in]  psi field to update
     * @param[in]  grid used to update psi
     */
    void updatePsiField(const std::vector<updatePsiParams>&paramList, real_pgrid_ptr psi , real_pgrid_ptr grid);
}
namespace pmlUpdateFxnCplx
{
    /**
     * @brief      Adds psi to grid_i_
     *
     * @param      gridParamList  list of parameters to update grid_i_
     * @param      psiParamList   ist of parameters to update Psi field
     * @param[in]  grid_i         grid_i_
     * @param[in]  psi            psi field to update
     * @param[in]  grid           grid used to update psi
     */
    void addPsi(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, cplx_pgrid_ptr grid_i, cplx_pgrid_ptr psi, cplx_pgrid_ptr grid);

    /**
     * @brief      Updates the gird in the PML region excluding the psi fields
     *
     * @param[in]  gridParamList  list of parameters to update grid_i_
     * @param[in]  psiParamList   ist of parameters to update Psi field
     * @param[in]  grid_i         grid_i_
     * @param[in]  psi            psi field to update
     * @param[in]  grid           grid used to update psi
     */
    void addGridOnly(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, cplx_pgrid_ptr grid_i, cplx_pgrid_ptr psi, cplx_pgrid_ptr grid);

    /**
     * @brief      update psi field
     *
     * @param      list to loop over to update psi
     * @param[in]  psi field to update
     * @param[in]  grid used to update psi
     */
    void updatePsiField(const std::vector<updatePsiParams>&paramList , cplx_pgrid_ptr psi , cplx_pgrid_ptr grid);
}

class parallelCPMLReal : public parallelCPML<double>
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  gridComm      mpi communicator
     * @param[in]  weights       The weights
     * @param[in]  grid_i        shared pointer to grid_i (field PML is being applied to)
     * @param[in]  grid_j        shared pointer to grid_j (field polarized in the j direction of the PML; nullptr if none)
     * @param[in]  grid_k        shared pointer to grid_k (field polarized in the k direction of the PML; nullptr if none)
     * @param[in]  pol_i         polarization of grid_o
     * @param[in]  n_vec         Vector storing thickness of the PMLs in all directions
     * @param[in]  m             scalling factor for sigma
     * @param[in]  ma            scaling factor for a
     * @param[in]  sigOptMaxRat  The ratio between the optimal sigma value in Taflove 2005 ch 7 and sigma max
     * @param[in]  kappaMax      The maximum kappa value
     * @param[in]  aMax          max a value
     * @param[in]  d             vector storing the step sizes in all directions
     * @param[in]  dt            time step
     * @param[in]  physGrid      The physical grid
     * @param[in]  objArr        The object arr
     */
    parallelCPMLReal(std::shared_ptr<mpiInterface> gridComm, std::vector<real_grid_ptr> weights, real_pgrid_ptr grid_i, real_pgrid_ptr grid_j, real_pgrid_ptr grid_k, POLARIZATION pol_i, std::array<int,3> n_vec, double m, double ma, double sigOptMaxRat, double kappaMax, double aMax, std::array<double,3> d, double dt, bool matInPML, int_pgrid_ptr physGrid, std::vector<std::shared_ptr<Obj>> objArr);
};

class parallelCPMLCplx : public parallelCPML<cplx>
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  gridComm      mpi communicator
     * @param[in]  weights       The weights
     * @param[in]  grid_i        shared pointer to grid_i (field PML is being applied to)
     * @param[in]  grid_j        shared pointer to grid_j (field polarized in the j direction of the PML; nullptr if none)
     * @param[in]  grid_k        shared pointer to grid_k (field polarized in the k direction of the PML; nullptr if none)
     * @param[in]  pol_i         polarization of grid_o
     * @param[in]  n_vec         Vector storing thickness of the PMLs in all directions
     * @param[in]  m             scalling factor for sigma
     * @param[in]  ma            scaling factor for a
     * @param[in]  sigOptMaxRat  The ratio between the optimal sigma value in Taflove 2005 ch 7 and sigma max
     * @param[in]  kappaMax      The maximum kappa value
     * @param[in]  aMax          max a value
     * @param[in]  d             vector storing the step sizes in all directions
     * @param[in]  dt            time step
     * @param[in]  physGrid      The physical grid
     * @param[in]  objArr        The object arr
     */
    parallelCPMLCplx(std::shared_ptr<mpiInterface> gridComm, std::vector<real_grid_ptr> weights, std::shared_ptr<parallelGrid<cplx > > grid_i, std::shared_ptr<parallelGrid<cplx > > grid_j, std::shared_ptr<parallelGrid<cplx > > grid_k, POLARIZATION pol_i, std::array<int,3> n_vec, double m, double ma, double sigOptMaxRat, double kappaMax, double aMax, std::array<double,3> d, double dt, bool matInPML, int_pgrid_ptr physGrid, std::vector<std::shared_ptr<Obj>> objArr);
};
#endif