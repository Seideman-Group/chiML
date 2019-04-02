#include <ML/parallelQEContinuum.hpp>
parallelQEContinuumReal::parallelQEContinuumReal(std::shared_ptr<mpiInterface> gridComm, std::array<Grid<double>,3> weights, int nlevel, std::vector<double> gam, std::vector<double> gamP, std::vector<std::array<int,2>> locs, real_pgrid_ptr E, double omgGap, double dOmg, double mu, double a, double I0, double dx, double dy, double dt, double na) :
    parallelQEContinuum(gridComm, weights, nlevel, gam, gamP, locs, E, omgGap, dOmg, mu, a, I0, dx, dy, dt, na)
{
    // getE_    = MLUpdateFxnReal::getE_Continuum;
    // getE_05_ = MLUpdateFxnReal::getE05_Continuum;

    // upPrev_  = MLUpdateFxnReal::updatePrev;
    // addP_    = MLUpdateFxnReal::addP;

    // sendP_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, real_grid_ptr Pz, std::shared_ptr<RecvEFieldSendPFieldCont> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+2, &Pz->point(process_struct->loc_[0], process_struct->loc_[1]), process_struct->szP_ ); return; };
    // recvP_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, real_grid_ptr Pz, std::shared_ptr<SendEFieldRecvPFieldCont> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+2,  Pz->data()                                                   , Pz->size()           ); return; };

    // sendE_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, real_grid_ptr Ez, std::shared_ptr<SendEFieldRecvPFieldCont> process_struct)  {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+2,  Ez->data()                                                 , Ez->size()          ); return; };
    // recvE_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, real_grid_ptr Ez, std::shared_ptr<RecvEFieldSendPFieldCont> process_struct)  {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+2, &Ez->point(process_struct->loc_[0], process_struct->loc_[1]), process_struct->sz_ ); return; };

    // transferE_    = MLUpdateFxnReal::transferE;
}

parallelQEContinuumCplx::parallelQEContinuumCplx(std::shared_ptr<mpiInterface> gridComm, std::array<Grid<double>,3> weights, int nlevel, std::vector<double> gam, std::vector<double> gamP, std::vector<std::array<int,2>> locs, cplx_pgrid_ptr E, double omgGap, double dOmg, double mu, double a, double I0, double dx, double dy, double dt, double na) :
    parallelQEContinuum(gridComm, weights, nlevel, gam, gamP, locs, E, omgGap, dOmg, mu, a, I0, dx, dy, dt, na)
{

    // getE_    = MLUpdateFxnCplx::getE_Continuum;
    // getE_05_ = MLUpdateFxnCplx::getE05_Continuum;

    // upPrev_  = MLUpdateFxnReal::updatePrev;
    // addP_    = MLUpdateFxnCplx::addP;

    // sendP_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, cplx_grid_ptr Pz, std::shared_ptr<RecvEFieldSendPFieldCont> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+2, &Pz->point(process_struct->loc_[0], process_struct->loc_[1]), process_struct->szP_ ); return; };
    // recvP_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, cplx_grid_ptr Pz, std::shared_ptr<SendEFieldRecvPFieldCont> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+2,  Pz->data()                                                   , Pz->size()           ); return; };

    // sendE_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, cplx_grid_ptr Ez, std::shared_ptr<SendEFieldRecvPFieldCont> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+2,  Ez->data()                                                 , Ez->size()          ); return; };
    // recvE_ = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, cplx_grid_ptr Ez, std::shared_ptr<RecvEFieldSendPFieldCont> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+2, &Ez->point(process_struct->loc_[0], process_struct->loc_[1]), process_struct->sz_ ); return; };

    // transferE_    = MLUpdateFxnCplx::transferE;
}

parallelQEContinuumReal::parallelQEContinuumReal(const parallelQEContinuumReal & o) : parallelQEContinuum(o)
{}


parallelQEContinuumCplx::parallelQEContinuumCplx(const parallelQEContinuumCplx & o) : parallelQEContinuum(o)
{}