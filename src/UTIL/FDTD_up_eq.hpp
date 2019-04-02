/** @file UTIL/FDTD_up_eq.hpp
 *  @brief A group of function that various function pointers in FDTDField and parallelQE use to update grids
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#ifndef PARALLEL_FDTD_UPEQ
#define PARALLEL_FDTD_UPEQ

#include <PML/parallelPML.hpp>
#include <UTIL/ml_consts.hpp>

namespace FDTDCompUpdateFxnReal
{
    typedef real_pgrid_ptr pgrid_ptr;
    /**
     * @brief      Updates the fields by taking the curl and assuming the k field is not there
     *
     * @param[in]  upParams    array storing the blas operation parameters
     * @param[in]  prefactors  array storing the prefactors
     * @param[in]  grid_i      shared_ptr to the field to be updated
     * @param[in]  grid_j      shared_ptr to the jth field with ijk notation: (i.e. if grid_i is Ey, grid_j is Hz)
     * @param[in]  grid_k      shared_ptr to the kth field with ijk notation: (i.e. if grid_i is Ex, grid_j is Hz)
     */
    void OneCompCurlJ (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k);

    /**
     * @brief      Updates the fields by taking the curl and assuming the j field is not there
     *
     * @param[in]  upParams    array storing the blas operation parameters
     * @param[in]  prefactors  array storing the prefactors
     * @param[in]  grid_i      shared_ptr to the field to be updated
     * @param[in]  grid_j      shared_ptr to the jth field with ijk notation: (i.e. if grid_i is Ey, grid_j is Hz)
     * @param[in]  grid_k      shared_ptr to the kth field with ijk notation: (i.e. if grid_i is Ex, grid_j is Hz)
     */
    void OneCompCurlK (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k);

    /**
     * @brief      Updates the fields by taking the curl and assuming that both fields are there
     *
     * @param[in]  upParams    array storing the blas operation parameters
     * @param[in]  prefactors  array storing the prefactors
     * @param[in]  grid_i      shared_ptr to the field to be updated
     * @param[in]  grid_j      shared_ptr to the jth field with ijk notation: (i.e. if grid_i is Ey, grid_j is Hz)
     * @param[in]  grid_k      shared_ptr to the kth field with ijk notation: (i.e. if grid_i is Ex, grid_j is Hz)
     */
    void TwoCompCurl (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k);

    /**
     * @brief      Updates the electric polarizations from the H field in chiral media
     *
     * @param[in]  upParams       list of parameters to update field (locations, sizes, strides, what material)
     * @param[in]  oppGrid_i      The H field in the i direction (produces the polarization)
     * @param[in]  prefOppGrid_i  The H field in the i direction (produces the polarization)
     * @param[in]  lorPMi         Stores the polarization
     * @param[in]  prevLorPMi     stores the polarization fields at previous time step
     * @param[in]  alpha          A vector storing the alpha parameters for the objects
     * @param[in]  xi             A vector storing the xi parameters for the objects
     * @param[in]  gamma          A vector storing the gamma parameters for the objects
     * @param[in]  jstore         scratch to transfer current to previous
     */
    void UpdateChiral(const std::array<int,6>& upParams, pgrid_ptr oppGrid_i, pgrid_ptr prevOppGrid_i, std::vector<pgrid_ptr>& lorHPi, std::vector<pgrid_ptr> & prevLorHPi, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, const std::vector<double>& gammaPrev, double* jstore);

        /**
     * @brief      Updates chiral material with oriented dipoles
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  oppGrid_x             The field used to update the polarization/magnetization in the x direction
     * @param[in]  prevOppGrid_x         The field at the previous time step used to update the polarization/magnetization x direction
     * @param[in]  oppGrid_y             The field used to update the polarization/magnetization in the y direction
     * @param[in]  prevOppGrid_y         The field at the previous time step used to update the polarization/magnetization in y direction
     * @param[in]  oppGrid_z             The field used to update the polarization/magnetization in the z direction
     * @param[in]  prevOppGrid_z         The field at the previous time step used to update the polarization/magnetization in the z direction
     * @param[in]  lorChiPx              A vector storing shared_ptrs to the P/Mx fields
     * @param[in]  lorChiPy              A vector storing shared_ptrs to the P/My fields
     * @param[in]  lorChiPz              A vector storing shared_ptrs to the P/Mz fields
     * @param[in]  prevLorChiPx          A vector storing shared_ptrs to the P/Mx fields at the previous time step
     * @param[in]  prevLorChiPy          A vector storing shared_ptrs to the P/My fields at the previous time step
     * @param[in]  prevLorChiPz          A vector storing shared_ptrs to the P/Mz fields at the previous time step
     * @param[in]  dipChi                The grid describing dipole moment corresponding to lorHPi field at all grid points
     * @param[in]  oppDipChi_x           The grid describing dipole moment corresponding to oppUx field at all grid points
     * @param[in]  oppDipChi_y           The grid describing dipole moment corresponding to oppUy field at all grid points
     * @param[in]  oppDipChi_z           The grid describing dipole moment corresponding to oppUz field at all grid points
     * @param[in]  dip_x                 The grid describing dipole moment corresponding to Ux field at all grid points
     * @param[in]  dip_y                 The grid describing dipole moment corresponding to Uy field at all grid points
     * @param[in]  dip_z                 The grid describing dipole moment corresponding to Uz field at all grid points
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               The temporary storage for the prevLorChiPx grids
     * @param      jstorey               The temporary storage for the prevLorChiPy grids
     * @param      jstorez               The temporary storage for the prevLorChiPz grids
     * @param      tempStoreUDeriv       The temporary storage for the chiral point averaging
     * @param      tempStoreDipDotField  Temporary storage for the dipole scaled U fields
     */
    void UpdateChiralOrDip( const std::array<int,6>& upParams, pgrid_ptr oppGrid_x, pgrid_ptr prevOppGrid_x, pgrid_ptr oppGrid_y, pgrid_ptr prevOppGrid_y, pgrid_ptr oppGrid_z, pgrid_ptr prevOppGrid_z, std::vector<pgrid_ptr>& lorChiPx, std::vector<pgrid_ptr>& lorChiPy, std::vector<pgrid_ptr>& lorChiPz, std::vector<pgrid_ptr> & prevLorChiPx, std::vector<pgrid_ptr> & prevLorChiPy, std::vector<pgrid_ptr> & prevLorChiPz, const std::vector<real_pgrid_ptr>& oppDipChi_x, const std::vector<real_pgrid_ptr>& oppDipChi_y, const std::vector<real_pgrid_ptr>& oppDipChi_z, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, const std::vector<double>& gammaPrev, double* jstorex, double* jstorey, double* jstorez, double* tempStoreUDeriv, double* tempStoreDipDotField);

    /**
     * @brief      Updates an isotropic polarization/magnetization
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  grid_i     A shared_ptr to the Ui grid used to update the polarizations
     * @param      lorPi      A vector storing a shared_ptr to the polarizations/magnetizations being updated
     * @param      prevLorPi  A vector storing a shared_ptr to the polarizations/magnetizations being updated at the previous time step
     * @param[in]  alpha      A vector storing the alpha parameters for the objects
     * @param[in]  xi         A vector storing the xi parameters for the objects
     * @param[in]  gamma      A vector storing the gamma parameters for the objects
     * @param      jstore     Temporary storage for the polarization/magnetization fields
     */
    void UpdateLorPol( const std::array<int,6>& upParams, pgrid_ptr grid_i, std::vector<pgrid_ptr> & lorPi, std::vector<pgrid_ptr> & prevLorPi, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, double* jstore);

    /**
     * @brief      Updates an anisotropic polarization/magnetization
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  grid_x                A shared_ptr to the Ux grid used to update the polarizations
     * @param[in]  grid_y                A shared_ptr to the Uy grid used to update the polarizations
     * @param[in]  grid_z                A shared_ptr to the Ux grid used to update the polarizations
     * @param      lorPx                 A vector storing shared_ptrs to the P/Mx grids
     * @param      lorPy                 A vector storing shared_ptrs to the P/My grids
     * @param      lorPz                 A vector storing shared_ptrs to the P/Mx grids
     * @param      prevLorPx             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param      prevLorPy             A vector storing shared_ptrs to the P/My grids at the previous time step
     * @param      prevLorPz             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param[in]  dip_x                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  dip_y                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the y direction
     * @param[in]  dip_z                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               Temporary storage for the P/Mx grids
     * @param      jstorey               Temporary storage for the P/My grids
     * @param      jstorez               Temporary storage for the P/Mx grids
     * @param      tempStoreOppU         The temporary storage to keep the averaging of the j/k fields
     * @param      tempStoreDipDotField  The temporary storage keeping the dot product of dipole moment and fields
     */
    void UpdateLorPolOrDip( const std::array<int,6>& upParams, pgrid_ptr grid_x, pgrid_ptr grid_y, pgrid_ptr grid_z, std::vector<pgrid_ptr>& lorPx, std::vector<pgrid_ptr>& lorPy, std::vector<pgrid_ptr>& lorPz, std::vector<pgrid_ptr>& prevLorPx, std::vector<pgrid_ptr>& prevLorPy, std::vector<pgrid_ptr>& prevLorPz, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, double* jstorex, double* jstorey, double* jstorez, double* tempStoreDipDotU, double* tempStoreDipDotField);

    /**
     * @brief      Updates an anisotropic polarization/magnetization
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  grid_x                A shared_ptr to the Ux grid used to update the polarizations
     * @param[in]  grid_y                A shared_ptr to the Uy grid used to update the polarizations
     * @param[in]  grid_z                A shared_ptr to the Ux grid used to update the polarizations
     * @param      lorPx                 A vector storing shared_ptrs to the P/Mx grids
     * @param      lorPy                 A vector storing shared_ptrs to the P/My grids
     * @param      lorPz                 A vector storing shared_ptrs to the P/Mx grids
     * @param      prevLorPx             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param      prevLorPy             A vector storing shared_ptrs to the P/My grids at the previous time step
     * @param      prevLorPz             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param[in]  dip_x                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  dip_y                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the y direction
     * @param[in]  dip_z                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               Temporary storage for the P/Mx grids
     * @param      jstorey               Temporary storage for the P/My grids
     * @param      jstorez               Temporary storage for the P/Mx grids
     * @param      tempStoreOppU         The temporary storage to keep the averaging of the j/k fields
     * @param      tempStoreDipDotField  The temporary storage keeping the dot product of dipole moment and fields
     */
    void UpdateLorPolOrDipXY( const std::array<int,6>& upParams, pgrid_ptr grid_x, pgrid_ptr grid_y, pgrid_ptr grid_z, std::vector<pgrid_ptr>& lorPx, std::vector<pgrid_ptr>& lorPy, std::vector<pgrid_ptr>& lorPz, std::vector<pgrid_ptr>& prevLorPx, std::vector<pgrid_ptr>& prevLorPy, std::vector<pgrid_ptr>& prevLorPz, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, double* jstorex, double* jstorey, double* jstorez, double* tempStoreDipDotU, double* tempStoreDipDotField);

    /**
     * @brief      Updates an anisotropic polarization/magnetization
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  grid_x                A shared_ptr to the Ux grid used to update the polarizations
     * @param[in]  grid_y                A shared_ptr to the Uy grid used to update the polarizations
     * @param[in]  grid_z                A shared_ptr to the Ux grid used to update the polarizations
     * @param      lorPx                 A vector storing shared_ptrs to the P/Mx grids
     * @param      lorPy                 A vector storing shared_ptrs to the P/My grids
     * @param      lorPz                 A vector storing shared_ptrs to the P/Mx grids
     * @param      prevLorPx             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param      prevLorPy             A vector storing shared_ptrs to the P/My grids at the previous time step
     * @param      prevLorPz             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param[in]  dip_x                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  dip_y                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the y direction
     * @param[in]  dip_z                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               Temporary storage for the P/Mx grids
     * @param      jstorey               Temporary storage for the P/My grids
     * @param      jstorez               Temporary storage for the P/Mx grids
     * @param      tempStoreOppU         The temporary storage to keep the averaging of the j/k fields
     * @param      tempStoreDipDotField  The temporary storage keeping the dot product of dipole moment and fields
     */
    void UpdateLorPolOrDipZ( const std::array<int,6>& upParams, pgrid_ptr grid_x, pgrid_ptr grid_y, pgrid_ptr grid_z, std::vector<pgrid_ptr>& lorPx, std::vector<pgrid_ptr>& lorPy, std::vector<pgrid_ptr>& lorPz, std::vector<pgrid_ptr>& prevLorPx, std::vector<pgrid_ptr>& prevLorPy, std::vector<pgrid_ptr>& prevLorPz, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, double* jstorex, double* jstorey, double* jstorez, double* tempStoreDipDotU, double* tempStoreDipDotField);

    /**
     * @brief      Updates a U field from the D field by adding polarization/magnetizations
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param      lorPi      A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void DtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr> & lorPi);

    /**
     * @brief      Updates a U field from the D field by adding polarization/magnetizations
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param[in]  dip_i      A vector storing shared_ptrs to the dipole fields
     * @param      lorPi      A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void orDipDtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorPi, double* tempPolStore, double* tempDotStore);

    /**
     * @brief      Updates a U field from the D field by adding polarization/magnetizations
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param[in]  dip_i      A vector storing shared_ptrs to the dipole fields
     * @param      lorPi      A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void orDipDtoUZ(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorPi, double* tempPolStore, double* tempDotStore);

    void chiOrDipDtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorPi, double* tempPolStore, double* tempDotStore);

    /**
     * @brief      Adds the chiral polarizations/magnetizations to the U field
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param      lorChiPi   A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void chiDtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr>& lorChiPi);

    /**
     * @brief      Transfers data and applies periodic boundary conditions to the field for process 0
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCProc0  (pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

     /**
     * @brief      Transfers data and applies periodic boundary conditions to the field for the last process
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCProcMax(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

     /**
     * @brief      applies periodic boundary conditions to the field
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCProcMid(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

    /**
     * @brief      Transfers data and applies periodic boundary conditions to the field for non-endcap process
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBC1Proc(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

    /**
     * @brief      Transfers grid data to other processes
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCNonPer(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

}

namespace FDTDCompUpdateFxnCplx
{
    typedef cplx_pgrid_ptr pgrid_ptr;

    /**
     * @brief      Updates the fields by taking the curl and assuming the k field is not there
     *
     * @param[in]  upParams    array storing the blas operation parameters
     * @param[in]  prefactors  array storing the prefactors
     * @param[in]  grid_i      shared_ptr to the field to be updated
     * @param[in]  grid_j      shared_ptr to the jth field with ijk notation: (i.e. if grid_i is Ey, grid_j is Hz)
     * @param[in]  grid_k      shared_ptr to the kth field with ijk notation: (i.e. if grid_i is Ex, grid_j is Hz)
     */
    void OneCompCurlJ (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k);

    /**
     * @brief      Updates the fields by taking the curl and assuming the j field is not there
     *
     * @param[in]  upParams    array storing the blas operation parameters
     * @param[in]  prefactors  array storing the prefactors
     * @param[in]  grid_i      shared_ptr to the field to be updated
     * @param[in]  grid_j      shared_ptr to the jth field with ijk notation: (i.e. if grid_i is Ey, grid_j is Hz)
     * @param[in]  grid_k      shared_ptr to the kth field with ijk notation: (i.e. if grid_i is Ex, grid_j is Hz)
     */
    void OneCompCurlK (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k);

    /**
     * @brief      Updates the fields by taking the curl and assuming that both fields are there
     *
     * @param[in]  upParams    array storing the blas operation parameters
     * @param[in]  prefactors  array storing the prefactors
     * @param[in]  grid_i      shared_ptr to the field to be updated
     * @param[in]  grid_j      shared_ptr to the jth field with ijk notation: (i.e. if grid_i is Ey, grid_j is Hz)
     * @param[in]  grid_k      shared_ptr to the kth field with ijk notation: (i.e. if grid_i is Ex, grid_j is Hz)
     */
    void TwoCompCurl (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k);

    /**
     * @brief      Updates the electric polarizations from the H field in chiral media
     *
     * @param[in]  upParams       list of parameters to update field (locations, sizes, strides, what material)
     * @param[in]  oppGrid_i      The H field in the i direction (produces the polarization)
     * @param[in]  prefOppGrid_i  The H field in the i direction (produces the polarization)
     * @param[in]  lorPMi         Stores the polarization
     * @param[in]  prevLorPMi     stores the polarization fields at previous time step
     * @param[in]  alpha          A vector storing the alpha parameters for the objects
     * @param[in]  xi             A vector storing the xi parameters for the objects
     * @param[in]  gamma          A vector storing the gamma parameters for the objects
     * @param[in]  jstore         scratch to transfer current to previous
     */
    void UpdateChiral(const std::array<int,6>& upParams, pgrid_ptr oppGrid_i, pgrid_ptr prevOppGrid_i, std::vector<pgrid_ptr>& lorHPi, std::vector<pgrid_ptr> & prevLorHPi, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gam, const std::vector<double>& gamPrevma, cplx* jstore);

    /**
     * @brief      Updates chiral material with oriented dipoles
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  oppGrid_x             The field used to update the polarization/magnetization in the x direction
     * @param[in]  prevOppGrid_x         The field at the previous time step used to update the polarization/magnetization x direction
     * @param[in]  oppGrid_y             The field used to update the polarization/magnetization in the y direction
     * @param[in]  prevOppGrid_y         The field at the previous time step used to update the polarization/magnetization in y direction
     * @param[in]  oppGrid_z             The field used to update the polarization/magnetization in the z direction
     * @param[in]  prevOppGrid_z         The field at the previous time step used to update the polarization/magnetization in the z direction
     * @param[in]  lorChiPx              A vector storing shared_ptrs to the P/Mx fields
     * @param[in]  lorChiPy              A vector storing shared_ptrs to the P/My fields
     * @param[in]  lorChiPz              A vector storing shared_ptrs to the P/Mz fields
     * @param[in]  prevLorChiPx          A vector storing shared_ptrs to the P/Mx fields at the previous time step
     * @param[in]  prevLorChiPy          A vector storing shared_ptrs to the P/My fields at the previous time step
     * @param[in]  prevLorChiPz          A vector storing shared_ptrs to the P/Mz fields at the previous time step
     * @param[in]  dipChi                The grid describing dipole moment corresponding to lorHPi field at all grid points
     * @param[in]  oppDipChi_x           The grid describing dipole moment corresponding to oppUx field at all grid points
     * @param[in]  oppDipChi_y           The grid describing dipole moment corresponding to oppUy field at all grid points
     * @param[in]  oppDipChi_z           The grid describing dipole moment corresponding to oppUz field at all grid points
     * @param[in]  dip_x                 The grid describing dipole moment corresponding to Ux field at all grid points
     * @param[in]  dip_y                 The grid describing dipole moment corresponding to Uy field at all grid points
     * @param[in]  dip_z                 The grid describing dipole moment corresponding to Uz field at all grid points
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               The temporary storage for the prevLorChiPx grids
     * @param      jstorey               The temporary storage for the prevLorChiPy grids
     * @param      jstorez               The temporary storage for the prevLorChiPz grids
     * @param      tempStoreUDeriv       The temporary storage for the chiral point averaging
     * @param      tempStoreDipDotField  Temporary storage for the dipole scaled U fields
     */
    void UpdateChiralOrDip( const std::array<int,6>& upParams, pgrid_ptr oppGrid_x, pgrid_ptr prevOppGrid_x, pgrid_ptr oppGrid_y, pgrid_ptr prevOppGrid_y, pgrid_ptr oppGrid_z, pgrid_ptr prevOppGrid_z, std::vector<pgrid_ptr>& lorChiPx, std::vector<pgrid_ptr>& lorChiPy, std::vector<pgrid_ptr>& lorChiPz, std::vector<pgrid_ptr> & prevLorChiPx, std::vector<pgrid_ptr> & prevLorChiPy, std::vector<pgrid_ptr> & prevLorChiPz, const std::vector<real_pgrid_ptr>& oppDipChi_x, const std::vector<real_pgrid_ptr>& oppDipChi_y, const std::vector<real_pgrid_ptr>& oppDipChi_z, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, const std::vector<double>& gammaPrev, cplx* jstorex, cplx* jstorey, cplx* jstorez, cplx* tempStoreUDeriv, cplx* tempStoreDipDotField);

    /**
     * @brief      Updates an isotropic polarization/magnetization
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  grid_i     A shared_ptr to the Ui grid used to update the polarizations
     * @param      lorPi      A vector storing a shared_ptr to the polarizations/magnetizations being updated
     * @param      prevLorPi  A vector storing a shared_ptr to the polarizations/magnetizations being updated at the previous time step
     * @param[in]  alpha      A vector storing the alpha parameters for the objects
     * @param[in]  xi         A vector storing the xi parameters for the objects
     * @param[in]  gamma      A vector storing the gamma parameters for the objects
     * @param      jstore     Temporary storage for the polarization/magnetization fields
     */
    void UpdateLorPol( const std::array<int,6>& upParams, pgrid_ptr grid_i, std::vector<pgrid_ptr> & lorPi, std::vector<pgrid_ptr> & prevLorPi, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, cplx* jstore);

    /**
     * @brief      Updates an anisotropic polarization/magnetization
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  grid_x                A shared_ptr to the Ux grid used to update the polarizations
     * @param[in]  grid_y                A shared_ptr to the Uy grid used to update the polarizations
     * @param[in]  grid_z                A shared_ptr to the Ux grid used to update the polarizations
     * @param      lorPx                 A vector storing shared_ptrs to the P/Mx grids
     * @param      lorPy                 A vector storing shared_ptrs to the P/My grids
     * @param      lorPz                 A vector storing shared_ptrs to the P/Mx grids
     * @param      prevLorPx             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param      prevLorPy             A vector storing shared_ptrs to the P/My grids at the previous time step
     * @param      prevLorPz             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param[in]  dip_x                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  dip_y                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the y direction
     * @param[in]  dip_z                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               Temporary storage for the P/Mx grids
     * @param      jstorey               Temporary storage for the P/My grids
     * @param      jstorez               Temporary storage for the P/Mx grids
     * @param      tempStoreOppU         The temporary storage to keep the averaging of the j/k fields
     * @param      tempStoreDipDotField  The temporary storage keeping the dot product of dipole moment and fields
     */
    void UpdateLorPolOrDip( const std::array<int,6>& upParams, pgrid_ptr grid_x, pgrid_ptr grid_y, pgrid_ptr grid_z, std::vector<pgrid_ptr>& lorPx, std::vector<pgrid_ptr>& lorPy, std::vector<pgrid_ptr>& lorPz, std::vector<pgrid_ptr>& prevLorPx, std::vector<pgrid_ptr>& prevLorPy, std::vector<pgrid_ptr>& prevLorPz, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, cplx* jstorex, cplx* jstorey, cplx* jstorez, cplx* tempStoreDipDotU, cplx* tempStoreDipDotField);

    /**
     * @brief      Updates an anisotropic polarization/magnetization
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  grid_x                A shared_ptr to the Ux grid used to update the polarizations
     * @param[in]  grid_y                A shared_ptr to the Uy grid used to update the polarizations
     * @param[in]  grid_z                A shared_ptr to the Ux grid used to update the polarizations
     * @param      lorPx                 A vector storing shared_ptrs to the P/Mx grids
     * @param      lorPy                 A vector storing shared_ptrs to the P/My grids
     * @param      lorPz                 A vector storing shared_ptrs to the P/Mx grids
     * @param      prevLorPx             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param      prevLorPy             A vector storing shared_ptrs to the P/My grids at the previous time step
     * @param      prevLorPz             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param[in]  dip_x                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  dip_y                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the y direction
     * @param[in]  dip_z                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               Temporary storage for the P/Mx grids
     * @param      jstorey               Temporary storage for the P/My grids
     * @param      jstorez               Temporary storage for the P/Mx grids
     * @param      tempStoreOppU         The temporary storage to keep the averaging of the j/k fields
     * @param      tempStoreDipDotField  The temporary storage keeping the dot product of dipole moment and fields
     */
    void UpdateLorPolOrDipXY( const std::array<int,6>& upParams, pgrid_ptr grid_x, pgrid_ptr grid_y, pgrid_ptr grid_z, std::vector<pgrid_ptr>& lorPx, std::vector<pgrid_ptr>& lorPy, std::vector<pgrid_ptr>& lorPz, std::vector<pgrid_ptr>& prevLorPx, std::vector<pgrid_ptr>& prevLorPy, std::vector<pgrid_ptr>& prevLorPz, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, cplx* jstorex, cplx* jstorey, cplx* jstorez, cplx* tempStoreDipDotU, cplx* tempStoreDipDotField);

    /**
     * @brief      Updates an anisotropic polarization/magnetization
     *
     * @param[in]  upParams              Array of parameters used to update the field
     * @param[in]  grid_x                A shared_ptr to the Ux grid used to update the polarizations
     * @param[in]  grid_y                A shared_ptr to the Uy grid used to update the polarizations
     * @param[in]  grid_z                A shared_ptr to the Ux grid used to update the polarizations
     * @param      lorPx                 A vector storing shared_ptrs to the P/Mx grids
     * @param      lorPy                 A vector storing shared_ptrs to the P/My grids
     * @param      lorPz                 A vector storing shared_ptrs to the P/Mx grids
     * @param      prevLorPx             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param      prevLorPy             A vector storing shared_ptrs to the P/My grids at the previous time step
     * @param      prevLorPz             A vector storing shared_ptrs to the P/Mx grids at the previous time step
     * @param[in]  dip_x                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  dip_y                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the y direction
     * @param[in]  dip_z                 A vector storing shared_ptrs to the grids storing the value of the dipole moment of the material in the x direction
     * @param[in]  alpha                 A vector storing the alpha parameters for the objects
     * @param[in]  xi                    A vector storing the xi parameters for the objects
     * @param[in]  gamma                 A vector storing the gamma parameters for the objects
     * @param      jstorex               Temporary storage for the P/Mx grids
     * @param      jstorey               Temporary storage for the P/My grids
     * @param      jstorez               Temporary storage for the P/Mx grids
     * @param      tempStoreOppU         The temporary storage to keep the averaging of the j/k fields
     * @param      tempStoreDipDotField  The temporary storage keeping the dot product of dipole moment and fields
     */
    void UpdateLorPolOrDipZ( const std::array<int,6>& upParams, pgrid_ptr grid_x, pgrid_ptr grid_y, pgrid_ptr grid_z, std::vector<pgrid_ptr>& lorPx, std::vector<pgrid_ptr>& lorPy, std::vector<pgrid_ptr>& lorPz, std::vector<pgrid_ptr>& prevLorPx, std::vector<pgrid_ptr>& prevLorPy, std::vector<pgrid_ptr>& prevLorPz, const std::vector<real_pgrid_ptr>& dip_x, const std::vector<real_pgrid_ptr>& dip_y, const std::vector<real_pgrid_ptr>& dip_z, const std::vector<double>& alpha, const std::vector<double>& xi, const std::vector<double>& gamma, cplx* jstorex, cplx* jstorey, cplx* jstorez, cplx* tempStoreDipDotU, cplx* tempStoreDipDotField);

    /**
     * @brief      Updates a U field from the D field by adding polarization/magnetizations
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param      lorPi      A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void DtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr> & lorPi);

    /**
     * @brief      Adds the chiral polarizations/magnetizations to the U field
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param      lorChiPi   A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void chiDtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr>& lorChiPi);

    /**
     * @brief      Adds the chiral polarizations/magnetizations to the U field
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param      lorChiPi   A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void orDipDtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorPi, cplx* tempPolStore, cplx* tempDotStore);

    void orDipDtoUZ(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorPi, cplx* tempPolStore, cplx* tempDotStore);

    void chiOrDipDtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorPi, cplx* tempPolStore, cplx* tempDotStore);

    /**
     * @brief      Adds the chiral polarizations/magnetizations to the U field
     *
     * @param[in]  upParams   Array of parameters used to update the field
     * @param[in]  epMuInfty  The high frequnency vacuum permittivity/permeability
     * @param[in]  Di         A shared_ptr to the D field
     * @param[in]  Ui         A shared_ptr to the U field
     * @param      lorChiPi   A vector storing shared_ptrs to the polarizations/magnetizations
     */
    void chiDtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr>& lorChiPi);

    /**
     * @brief      Transfers data and applies periodic boundary conditions to the field for process 0
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCProc0  (pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

     /**
     * @brief      Transfers data and applies periodic boundary conditions to the field for the last process
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCProcMax(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

     /**
     * @brief      applies periodic boundary conditions to the field
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCProcMid(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

    /**
     * @brief      Transfers data and applies periodic boundary conditions to the field for non-endcap process
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBC1Proc(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

    /**
     * @brief      Transfers grid data to other processes
     *
     * @param[in]  fUp        shared_ptr to the field that is being updated
     * @param      tempStore  Temporary storage for X/Z edge values
     * @param[in]  k_point    vector representing the k_point of light
     * @param[in]  nx         right boundary of the field
     * @param[in]  ny         top boundary of the cell
     * @param[in]  dx         grid spacing in the x direction
     * @param[in]  dy         grid spacing in the y direction
     */
    void applyBCNonPer(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz);

}
namespace MLUpdateFxnReal
{
    typedef real_pgrid_ptr pgrid_ptr;
    /**
     * @brief      updates The QE polarization
     *
     * @param[in]  denDeriv    vector containing $\frac{\partial \rho}{\partial t$
     * @param[in]  xx          x location of the QE point
     * @param[in]  yy          y location of the QE point
     * @param[in]  expectation expectation value of the dipole operator
     * @param[in]  P           shared_ptr to P field
     * @param[in]  na          molecular density
     * @param[in]  nlevel_     number of levels in the system
     * @param[in]  P_vec       vector storing the polarization vector
     *
     */
    inline void updateQEPol(cplx* denDeriv, int xx, int yy, int zz, cplx* expectation, real_grid_ptr P, double na, int szMat) { P->point(xx+1, yy +1, zz+1) += na * std::real(zdotc_(szMat, denDeriv, 1, expectation, 1) ); return;}

    /**
     * @brief      Moves the E field to previous E field
     *
     * @param[in]  E      shared_ptr to E field
     * @param[in]  prevE  shared_ptr to previous E field
     */
    inline void updatePrev(real_grid_ptr E, real_grid_ptr prevE) { for(int ii = 0; ii < static_cast<int>(E->y()); ii++) dcopy_(E->x()*E->z(), &E->point(0,ii,0), 1, &prevE->point(0,ii,0), 1); }

    /**
     * @brief      Adds a P field to the E field
     *
     * @param[in]  P     shared_ptr to polarization field
     * @param[in]  E     shared_ptr to E field
     * @param[in]  dt    time-step
     * @param[in]  xoff  offset in x direction
     * @param[in]  yoff  offset in y direction
     */
    void addP(real_grid_ptr P, pgrid_ptr E, real_pgrid_ptr eps, double* tempScaled, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff);

    void transferE(real_pgrid_ptr inField, real_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart);

    void updateRadPol(cplx* den, cplx* denDeriv, real_grid_ptr P, int xx, int yy, int zz, int n0, int nf, cplx& ampPreFact, cplx& omg, double t);

    void addRadPol( double na, real_grid_ptr radPol, real_grid_ptr prevRadPol, real_grid_ptr P );
    // void transferP(real_grid_ptr inField, real_grid_ptr outField, int xmin, int ymin, int& nx, int& ny, int& ystart);

    void getE_TE(real_grid_ptr inField, cplx_grid_ptr outField, real_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff);

    void getE_TM(real_grid_ptr inField, cplx_grid_ptr outField, real_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff);

    void getE05(cplx_grid_ptr half, cplx_grid_ptr cur, cplx_grid_ptr past);

    void getE_Continuum(real_grid_ptr inField, real_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart);

    void getE05_Continuum(real_grid_ptr half, real_grid_ptr cur, real_grid_ptr past);
}

namespace MLUpdateFxnCplx
{
    typedef cplx_pgrid_ptr pgrid_ptr;

    /**
     * @brief      updates The QE polarization
     *
     * @param[in]  denDeriv    vector containing $\frac{\partial \rho}{\partial t$
     * @param[in]  xx          x location of the QE point
     * @param[in]  yy          y location of the QE point
     * @param[in]  expectation expectation value of the dipole operator
     * @param[in]  P           shared_ptr to P field
     * @param[in]  na          molecular density
     * @param[in]  nlevel_     number of levels in the system
     * @param[in]  P_vec       vector storing the polarization vector
     *
     */
    inline void updateQEPol(cplx* denDeriv, int xx, int yy, int zz, cplx* expectation, cplx_grid_ptr P, double na, int  szMat) { P->point(xx+1, yy +1, zz+1) += na * zdotc_(szMat, expectation, 1, denDeriv, 1); return;}

    /**
     * @brief      Moves the E field to previous E field
     *
     * @param[in]  E      shared_ptr to E field
     * @param[in]  prevE  shared_ptr to previous E field
     */
    inline void updatePrev(cplx_grid_ptr E, cplx_grid_ptr prevE) { for(int ii = 0; ii < static_cast<int>(E->y()); ii++) zcopy_(E->x()*E->z(), &E->point(0,ii,0), 1, &prevE->point(0,ii,0), 1); }

    /**
     * @brief      Adds a P field to the E field
     *
     * @param[in]  P     shared_ptr to polarization field
     * @param[in]  E     shared_ptr to E field
     * @param[in]  dt    time-step
     * @param[in]  xoff  offset in x direction
     * @param[in]  yoff  offset in y direction
     */
    void addP(cplx_grid_ptr P, pgrid_ptr E, real_pgrid_ptr eps, cplx* tempScaled, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff);

    void transferE(pgrid_ptr inField, cplx_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart);

    void updateRadPol(cplx* den, cplx* denDeriv, cplx_grid_ptr P, int xx, int yy, int zz, int n0, int nf, cplx& ampPreFact, cplx& omg, double t);

    void addRadPol( double na, cplx_grid_ptr radPol, cplx_grid_ptr prevRadPol, cplx_grid_ptr P );
    // void transferP(cplx_grid_ptr inField, cplx_grid_ptr outField, int xmin, int ymin, int& nx, int& ny, int& ystart);

    void getE_TE(cplx_grid_ptr inField, cplx_grid_ptr outField, cplx_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff);

    void getE_TM(cplx_grid_ptr inField, cplx_grid_ptr outField, cplx_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff);

    void getE05(cplx_grid_ptr half, cplx_grid_ptr cur, cplx_grid_ptr past);

    void getE_Continuum(cplx_grid_ptr inField, cplx_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart);

    void getE05_Continuum(cplx_grid_ptr half, cplx_grid_ptr cur, cplx_grid_ptr past);
}

#endif