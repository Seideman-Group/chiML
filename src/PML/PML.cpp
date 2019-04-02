/** @file PML/PML.cpp
 *  @brief Stores/updates the CPML fields and adds them to the incident FDTD grids
 *
 *  A class that store and updates the CPML fields and add them the the incident FDTD grids for the TFSF surfaces
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

 #include <PML/PML.hpp>

void pmlUpdateFxnIncdCplx::addPsi(const updateGridParamsIncd& gridParam, const updatePsiParamsIncd& psiParam, cplx* tempStore, cplx_grid_ptr grid_i, cplx_grid_ptr psi, cplx_grid_ptr grid)
{
    updatePsiField(psiParam, tempStore, psi, grid);
    zaxpy_(psi->size(), gridParam.Db_, psi->data(), 1, &grid_i->point(gridParam.loc_, 0), 1);
}

void pmlUpdateFxnIncdCplx::updatePsiField(const updatePsiParamsIncd& param, cplx* tempStore, cplx_grid_ptr psi, cplx_grid_ptr grid)
{
    std::transform( param.b_.begin(), param.b_.end(), &psi->point(0, 0),  &psi->point(0, 0), std::multiplies<cplx>() );

    std::transform( param.c_   .begin(), param.c_.end()              , &grid->point(param.loc_   , 0), tempStore       , std::multiplies<cplx>() );
    std::transform( &psi->point(0,0)   , &psi->point(0,0)+psi->size(), tempStore                     , &psi->point(0,0), std::plus      <cplx>() );

    std::transform( param.cOff_.begin(), param.cOff_.end()           , &grid->point(param.locOff_, 0), tempStore       , std::multiplies<cplx>() ) ;
    std::transform( &psi->point(0,0)   , &psi->point(0,0)+psi->size(), tempStore                     , &psi->point(0,0), std::plus      <cplx>() );
}

IncdCPMLCplx::IncdCPMLCplx(cplx_grid_ptr grid_i, cplx_grid_ptr grid_j, cplx_grid_ptr grid_k, POLARIZATION pol_i, int pmlThick, std::array<int,3> tfsfMs, std::array<int,3> maingGridSz, bool useMn, double epsPl, double muPl, double epsMn, double muMn, double m, double ma, double kappaMax, double aMax, std::array<double,3> d, double dr, double dt) :
    IncdCPML<cplx>(grid_i, grid_j, grid_k, pol_i, pmlThick, tfsfMs, maingGridSz, useMn, epsPl, muPl, epsMn, muMn, m, ma, kappaMax, aMax, d, dr, dt)
{
    if(psi_j_.size() > 0 && tfsfMs_[cor_jj_] != 0)
        upPsi_j_ = pmlUpdateFxnIncdCplx::addPsi;
    else
        upPsi_j_ = [](const updateGridParamsIncd&, const updatePsiParamsIncd&, cplx*, cplx_grid_ptr, cplx_grid_ptr, cplx_grid_ptr){return;};

    if(psi_k_.size() > 0 && tfsfMs_[cor_kk_] != 0)
        upPsi_k_ = pmlUpdateFxnIncdCplx::addPsi;
    else
        upPsi_k_ = [](const updateGridParamsIncd&, const updatePsiParamsIncd&, cplx*, cplx_grid_ptr, cplx_grid_ptr, cplx_grid_ptr){return;};
}
