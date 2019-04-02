/** @file PML/parallelPML.cpp
 *  @brief Stores/updates the CPML fields and adds them to the FDTD grids
 *
 *  A class that store and updates the CPML fields and add them the the FDTD grids for the TFSF surfaces
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <PML/parallelPML.hpp>

void pmlUpdateFxnReal::addPsi(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, real_pgrid_ptr grid_i, real_pgrid_ptr psi, real_pgrid_ptr grid)
{
    updatePsiField(psiParamList, psi, grid);
    for(auto& param : gridParamList)
    {
        daxpy_(param.nAx_,      param.DbField_, &grid->point(param.ind_   ), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
        daxpy_(param.nAx_, -1.0*param.DbField_, &grid->point(param.indOff_), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
        daxpy_(param.nAx_,      param.Db_     ,  &psi->point(param.ind_   ), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
    }
}

void pmlUpdateFxnReal::addGridOnly(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, real_pgrid_ptr grid_i, real_pgrid_ptr psi, real_pgrid_ptr grid)
{
    for(auto& param : gridParamList)
    {
        daxpy_(param.nAx_,      param.DbField_, &grid->point(param.ind_   ), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
        daxpy_(param.nAx_, -1.0*param.DbField_, &grid->point(param.indOff_), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
    }
}

void pmlUpdateFxnReal::updatePsiField(const std::vector<updatePsiParams> &paramList, real_pgrid_ptr psi, real_pgrid_ptr grid)
{
    for (auto& param : paramList)
    {
        dscal_(param.transSz_,      param.b_,  &psi->point(param.ind_   ), param.stride_);
        daxpy_(param.transSz_,      param.c_, &grid->point(param.ind_   ), param.stride_, &psi->point(param.ind_), param.stride_);
        daxpy_(param.transSz_, -1.0*param.c_, &grid->point(param.indOff_), param.stride_, &psi->point(param.ind_), param.stride_);
    }
}

void pmlUpdateFxnCplx::addPsi(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, cplx_pgrid_ptr grid_i, cplx_pgrid_ptr psi, cplx_pgrid_ptr grid)
{
   updatePsiField(psiParamList, psi, grid);
    for(auto& param : gridParamList)
    {
        zaxpy_(param.nAx_,      param.DbField_, &grid->point(param.ind_   ), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
        zaxpy_(param.nAx_, -1.0*param.DbField_, &grid->point(param.indOff_), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
        zaxpy_(param.nAx_,      param.Db_     ,  &psi->point(param.ind_   ), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
    }
}

void pmlUpdateFxnCplx::updatePsiField(const std::vector<updatePsiParams> &paramList, cplx_pgrid_ptr psi, cplx_pgrid_ptr grid)
{
    for(auto& param : paramList)
    {
        zscal_(param.transSz_,      param.b_,  &psi->point(param.ind_   ), param.stride_);
        zaxpy_(param.transSz_,      param.c_, &grid->point(param.ind_   ), param.stride_, &psi->point(param.ind_), param.stride_);
        zaxpy_(param.transSz_, -1.0*param.c_, &grid->point(param.indOff_), param.stride_, &psi->point(param.ind_), param.stride_);
    }
}

void pmlUpdateFxnCplx::addGridOnly(const std::vector<updateGridParams> &gridParamList, const std::vector<updatePsiParams> &psiParamList, cplx_pgrid_ptr grid_i, cplx_pgrid_ptr psi, cplx_pgrid_ptr grid)
{
    for(auto& param : gridParamList)
    {
        zaxpy_(param.nAx_,      param.DbField_, &grid->point(param.ind_   ), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
        zaxpy_(param.nAx_, -1.0*param.DbField_, &grid->point(param.indOff_), param.stride_, &grid_i->point(param.ind_ ), param.stride_);
    }
}

parallelCPMLReal::parallelCPMLReal(std::shared_ptr<mpiInterface> gridComm, std::vector<real_grid_ptr> weights, real_pgrid_ptr grid_i, real_pgrid_ptr grid_j, real_pgrid_ptr grid_k, POLARIZATION pol_i, std::array<int,3> n_vec, double m, double ma, double sigOptMaxRat, double kappaMax, double aMax, std::array<double,3> d, double dt, bool matInPML, int_pgrid_ptr physGrid, std::vector<std::shared_ptr<Obj>> objArr) :
    parallelCPML<double>(gridComm, weights, grid_i, grid_j, grid_k, pol_i, n_vec, m, ma,sigOptMaxRat, kappaMax, aMax, d, dt, matInPML, physGrid, objArr)
{
    if(psi_j_)
        addGrid_j_ = pmlUpdateFxnReal::addPsi;
    else if(grid_k_)
        addGrid_j_ = pmlUpdateFxnReal::addGridOnly;
    else
        addGrid_j_ = [](const std::vector<updateGridParams>&, const std::vector<updatePsiParams>&, real_pgrid_ptr, real_pgrid_ptr, real_pgrid_ptr){return;};

    if(psi_k_)
        addGrid_k_ = pmlUpdateFxnReal::addPsi;
    else if(grid_j_)
        addGrid_k_ = pmlUpdateFxnReal::addGridOnly;
    else
        addGrid_k_ = [](const std::vector<updateGridParams>&, const std::vector<updatePsiParams>&, real_pgrid_ptr, real_pgrid_ptr, real_pgrid_ptr){return;};
}
parallelCPMLCplx::parallelCPMLCplx(std::shared_ptr<mpiInterface> gridComm, std::vector<real_grid_ptr> weights, std::shared_ptr<parallelGrid<cplx > > grid_i, std::shared_ptr<parallelGrid<cplx > > grid_j, std::shared_ptr<parallelGrid<cplx > > grid_k, POLARIZATION pol_i, std::array<int,3> n_vec, double m, double ma, double sigOptMaxRat, double kappaMax, double aMax, std::array<double,3> d, double dt, bool matInPML, int_pgrid_ptr physGrid, std::vector<std::shared_ptr<Obj>> objArr) :
    parallelCPML<cplx>(gridComm, weights, grid_i, grid_j, grid_k, pol_i, n_vec, m, ma,sigOptMaxRat, kappaMax, aMax, d, dt, matInPML, physGrid, objArr)
{
    if(psi_j_)
        addGrid_j_ = pmlUpdateFxnCplx::addPsi;
    else if(grid_k_)
        addGrid_j_ = pmlUpdateFxnCplx::addGridOnly;
    else
        addGrid_j_ = [](const std::vector<updateGridParams>&, const std::vector<updatePsiParams>&, cplx_pgrid_ptr, cplx_pgrid_ptr, cplx_pgrid_ptr){return;};

    if(psi_k_)
        addGrid_k_ = pmlUpdateFxnCplx::addPsi;
    else if(grid_j_)
        addGrid_k_ = pmlUpdateFxnCplx::addGridOnly;
    else
        addGrid_k_ = [](const std::vector<updateGridParams>&, const std::vector<updatePsiParams>&, cplx_pgrid_ptr, cplx_pgrid_ptr, cplx_pgrid_ptr){return;};
}