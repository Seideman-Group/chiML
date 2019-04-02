/** @file ML/parallelQE.cpp
 *  @brief Class that stores and updates quantum emitter's density matrix and Polarizations
 *
 *  Stores and transfers relevant electric field information across all processes to update
 *  quantum emitter density matrices, updates the density matrices at all points
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <ML/parallelQE.hpp>

parallelQEReal::parallelQEReal(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<relaxParams> relaxStateTransitions, std::vector<std::array<int,3>> locs, std::array<real_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na) :
    parallelQEBase(gridComm, hams, eWeights, dtcPopArr, basis, relaxStateTransitions, locs, E, eps, accumulateP, outputP_, totTime, a, I0, dt, na)
{
    if(e_[0] && E_[0])
    {
        if(!E_[1] || !e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");

        getE_[0]    = MLUpdateFxnReal::getE_TE;
        getE_[1]    = MLUpdateFxnReal::getE_TE;

        upP_[0]     = MLUpdateFxnReal::updateQEPol;
        upP_[1]     = MLUpdateFxnReal::updateQEPol;

        addP_[0]    = MLUpdateFxnReal::addP;
        addP_[1]    = MLUpdateFxnReal::addP;

        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_  , &Px->point(0)                                                                        , Px->size()           ); return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+1, &Py->point(0)                                                                        , Py->size()           ); return; };
        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_  , &Px->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+1, &Py->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };

        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_  , &Ex->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+1, &Ey->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };
        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_  ,  Ex->data()                                                                          , Ex->size()          ); return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+1,  Ey->data()                                                                          , Ey->size()          ); return; };

        transferE_[0] = MLUpdateFxnReal::transferE;
        transferE_[1] = MLUpdateFxnReal::transferE;
        zeroP_[0] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };
        zeroP_[1] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };

        if(accumulateP )
        {
            totP_[0].reserve(totTime);
            totP_[1].reserve(totTime);
            accumP_[0] = [](grid_ptr P, std::vector<double>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), 0.0) ); };
            accumP_[1] = [](grid_ptr P, std::vector<double>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), 0.0) ); };
        }
        else
        {
            accumP_[0] = [](grid_ptr P, std::vector<double>& totVec){return; };
            accumP_[1] = [](grid_ptr P, std::vector<double>& totVec){return; };
        }
    }
    else
    {
        if(E_[1] && e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");

        getE_[0]    = [] (grid_ptr, cplx_grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;
        getE_[1]    = [] (grid_ptr, cplx_grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[0]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        upP_[1]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};

        addP_[0]    = [] (grid_ptr, real_pgrid_ptr, real_pgrid_ptr, double*, int, int, int, int, int, int, int, int, int, int) {return;};
        addP_[1]    = [] (grid_ptr, real_pgrid_ptr, real_pgrid_ptr, double*, int, int, int, int, int, int, int, int, int, int) {return;};

        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };
        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };

        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };

        transferE_[0] = [](real_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };
        transferE_[1] = [](real_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[0] = [](grid_ptr P){return; };
        zeroP_[1] = [](grid_ptr P){return; };

        accumP_[0] = [](grid_ptr P, std::vector<double>& totVec){return; };
        accumP_[1] = [](grid_ptr P, std::vector<double>& totVec){return; };

    }
    if(e_[2] && E_[2])
    {
        if(!E_[0])
            getE_[2]    = MLUpdateFxnReal::getE_TM;
        else
            getE_[2]    = MLUpdateFxnReal::getE_TE;


        upP_[2]     = MLUpdateFxnReal::updateQEPol;
        addP_[2]    = MLUpdateFxnReal::addP;

        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+2, &Pz->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };
        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+2,  Pz->data()                                                                          , Pz->size()           ); return; };

        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+2,  Ez->data()                                                                          , Ez->size()          ); return; };
        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+2, &Ez->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };

        transferE_[2]    = MLUpdateFxnReal::transferE;
        zeroP_[2] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };
        if(accumulateP )
        {
            totP_[2].reserve(totTime);
            accumP_[2] = [](grid_ptr P, std::vector<double>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), 0.0) ); };
        }
        else
        {
            accumP_[2] = [](grid_ptr P, std::vector<double>& totVec){return; };
        }
    }
    else
    {
        getE_[2]    = [] (grid_ptr, cplx_grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[2]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        addP_[2]    = [] (grid_ptr, real_pgrid_ptr, real_pgrid_ptr, double*, int, int, int, int, int, int, int, int, int, int) {return;};

        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) { return; };
        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) { return; };

        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct)  { return; };
        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct)  { return; };
        transferE_[2]  = [](real_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[2] = [](grid_ptr P){ return; };
        accumP_[2] = [](grid_ptr P, std::vector<double>& totVec){return; };
    }
}

parallelQEReal::parallelQEReal(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<std::vector<double>> gam, std::vector<std::array<int,3>> locs, std::array<real_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na) :
    parallelQEBase(gridComm, hams, eWeights, dtcPopArr, basis, gam, locs, E, eps, accumulateP, outputP_, totTime, a, I0, dt, na)
{
    if(e_[0] && E_[0])
    {
        if(!E_[1] || !e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");

        getE_[0]    = MLUpdateFxnReal::getE_TE;
        getE_[1]    = MLUpdateFxnReal::getE_TE;

        upP_[0]     = MLUpdateFxnReal::updateQEPol;
        upP_[1]     = MLUpdateFxnReal::updateQEPol;

        addP_[0]    = MLUpdateFxnReal::addP;
        addP_[1]    = MLUpdateFxnReal::addP;

        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_  ,  Px->data()                                                                          , Px->size()           ); return; };
        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_  , &Px->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+1,  Py->data()                                                                          , Py->size()           ); return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+1, &Py->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };

        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_  ,  Ex->data()                                                                          , Ex->size()          ); return; };
        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_  , &Ex->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+1,  Ey->data()                                                                          , Ey->size()          ); return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+1, &Ey->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };

        transferE_[0] = MLUpdateFxnReal::transferE;
        transferE_[1] = MLUpdateFxnReal::transferE;
        zeroP_[0] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };
        zeroP_[1] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };

        if(accumulateP )
        {
            totP_[0].reserve(totTime);
            totP_[1].reserve(totTime);
            accumP_[0] = [](grid_ptr P, std::vector<double>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), 0.0) ); };
            accumP_[1] = [](grid_ptr P, std::vector<double>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), 0.0) ); };
        }
        else
        {
            accumP_[0] = [](grid_ptr P, std::vector<double>& totVec){return; };
            accumP_[1] = [](grid_ptr P, std::vector<double>& totVec){return; };
        }
    }
    else
    {
        if(E_[1] && e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");

        getE_[0]    = [] (grid_ptr, cplx_grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;
        getE_[1]    = [] (grid_ptr, cplx_grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[0]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        upP_[1]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};


        addP_[0]    = [] (grid_ptr, real_pgrid_ptr, real_pgrid_ptr, double*, int, int, int, int, int, int, int, int, int, int) {return;};
        addP_[1]    = [] (grid_ptr, real_pgrid_ptr, real_pgrid_ptr, double*, int, int, int, int, int, int, int, int, int, int) {return;};

        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };
        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };

        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };

        transferE_[0] = [](real_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };
        transferE_[1] = [](real_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[0] = [](grid_ptr P){return; };
        zeroP_[1] = [](grid_ptr P){return; };

        accumP_[0] = [](grid_ptr P, std::vector<double>& totVec){return; };
        accumP_[1] = [](grid_ptr P, std::vector<double>& totVec){return; };

    }
    if(e_[2] && E_[2])
    {
        if(!E_[0])
            getE_[2]    = MLUpdateFxnReal::getE_TM;
        else
            getE_[2]    = MLUpdateFxnReal::getE_TE;

        upP_[2]     = MLUpdateFxnReal::updateQEPol;
        addP_[2]    = MLUpdateFxnReal::addP;

        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+2, &Pz->point(process_struct->loc_[0], process_struct->loc_[1]), process_struct->szP_ ); return; };
        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+2,  Pz->data()                                                   , Pz->size()           ); return; };

        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct)  {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+2,  Ez->data()                                                 , Ez->size()          ); return; };
        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct)  {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+2, &Ez->point(process_struct->loc_[0], process_struct->loc_[1]), process_struct->sz_ ); return; };

        transferE_[2]    = MLUpdateFxnReal::transferE;
        zeroP_[2] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };
        if(accumulateP )
        {
            totP_[2].reserve(totTime);
            accumP_[2] = [](grid_ptr P, std::vector<double>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), 0.0) ); };
        }
        else
        {
            accumP_[2] = [](grid_ptr P, std::vector<double>& totVec){return; };
        }
    }
    else
    {
        getE_[2]    = [] (grid_ptr, cplx_grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[2]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        addP_[2]    = [] (grid_ptr, real_pgrid_ptr, real_pgrid_ptr, double*, int, int, int, int, int, int, int, int, int, int) {return;};

        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) { return; };
        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) { return; };

        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct)  { return; };
        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct)  { return; };
        transferE_[2]  = [](real_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[2] = [](grid_ptr P){ return; };
        accumP_[2] = [](grid_ptr P, std::vector<double>& totVec){return; };
    }
}

parallelQECplx::parallelQECplx(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<relaxParams> relaxStateTransitions, std::vector<std::array<int,3>> locs, std::array<cplx_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na) :
    parallelQEBase(gridComm, hams, eWeights, dtcPopArr, basis, relaxStateTransitions, locs, E, eps, accumulateP, outputP_, totTime, a, I0, dt, na)
{
    if(e_[0] && E_[0])
    {
        if(!E_[1] || !e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");
        getE_[0]    = MLUpdateFxnCplx::getE_TE;
        getE_[1]    = MLUpdateFxnCplx::getE_TE;

        upP_[0]     = MLUpdateFxnCplx::updateQEPol;
        upP_[1]     = MLUpdateFxnCplx::updateQEPol;

        addP_[0]    = MLUpdateFxnCplx::addP;
        addP_[1]    = MLUpdateFxnCplx::addP;

        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_  ,  Px->data()                                                                          , Px->size()           ); return; };
        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_  , &Px->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+1,  Py->data()                                                                          , Py->size()           ); return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+1, &Py->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };

        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_  ,  Ex->data()                                                                          , Ex->size()         ); return; };
        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_  , &Ex->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_); return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+1,  Ey->data()                                                                          , Ey->size()          ); return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+1, &Ey->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };

        transferE_[0] = MLUpdateFxnCplx::transferE;
        transferE_[1] = MLUpdateFxnCplx::transferE;

        zeroP_[0] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };
        zeroP_[1] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };

        if(accumulateP)
        {
            totP_[0].reserve(totTime);
            totP_[1].reserve(totTime);
            accumP_[0] = [](grid_ptr P, std::vector<cplx>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), cplx(0.0) ) ); };
            accumP_[1] = [](grid_ptr P, std::vector<cplx>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), cplx(0.0) ) ); };
        }
        else
        {
            accumP_[0] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
            accumP_[1] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
        }
    }
    else
    {
        if(E_[1] && e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");

        getE_[0]    = [] (grid_ptr, grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;
        getE_[1]    = [] (grid_ptr, grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[0]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        upP_[1]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        addP_[0]    = [] (grid_ptr, cplx_pgrid_ptr, real_pgrid_ptr, cplx*, int, int, int, int, int, int, int, int, int, int) {return;};
        addP_[1]    = [] (grid_ptr, cplx_pgrid_ptr, real_pgrid_ptr, cplx*, int, int, int, int, int, int, int, int, int, int) {return;};

        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };
        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };

        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };

        transferE_[0] = [](cplx_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };
        transferE_[1] = [](cplx_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[0] = [](grid_ptr P){ return; };
        zeroP_[1] = [](grid_ptr P){ return; };

        accumP_[0] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
        accumP_[1] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
    }
    if(e_[2] && E_[2])
    {
        if(!E_[0])
            getE_[2]    = MLUpdateFxnCplx::getE_TM;
        else
            getE_[2]    = MLUpdateFxnCplx::getE_TE;

        upP_[2]     = MLUpdateFxnCplx::updateQEPol;
        addP_[2]    = MLUpdateFxnCplx::addP;

        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+2, &Pz->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };
        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+2,  Pz->data()                                                                          , Pz->size()           ); return; };

        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+2, &Ez->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };
        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+2,  Ez->data()                                                                          , Ez->size()          ); return; };
        transferE_[2]    = MLUpdateFxnCplx::transferE;

        zeroP_[2] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };

        if(accumulateP )
        {
            totP_[2].reserve(totTime);
            accumP_[2] = [](grid_ptr P, std::vector<cplx>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), cplx(0.0) ) ); };
        }
        else
        {
            accumP_[2] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
        }
    }
    else
    {
        getE_[2]    = [] (grid_ptr, grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[2]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        addP_[2]    = [] (grid_ptr, cplx_pgrid_ptr, real_pgrid_ptr, cplx*, int, int, int, int, int, int, int, int, int, int) {return;};

        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) { return; };
        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) { return; };

        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct)  { return; };
        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct)  { return; };

        transferE_[2] = [](cplx_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[2] = [](grid_ptr P){return; };
        accumP_[2] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
    }
}

parallelQECplx::parallelQECplx(std::shared_ptr<mpiInterface> gridComm, std::vector<Hamiltonian> hams, std::vector<std::pair<std::vector<double>, double>> eWeights, std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr, BasisSet basis, std::vector<std::vector<double>> gam, std::vector<std::array<int,3>> locs, std::array<cplx_pgrid_ptr,3> E, real_pgrid_ptr eps, bool accumulateP, std::string outputP_, int totTime, double a, double I0, double dt, double na) :
    parallelQEBase(gridComm, hams, eWeights, dtcPopArr, basis, gam, locs, E, eps, accumulateP, outputP_, totTime, a, I0, dt, na)
{
    if(e_[0] && E_[0])
    {
        if(!E_[1] || !e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");
        getE_[0]    = MLUpdateFxnCplx::getE_TE;
        getE_[1]    = MLUpdateFxnCplx::getE_TE;

        upP_[0]     = MLUpdateFxnCplx::updateQEPol;
        upP_[1]     = MLUpdateFxnCplx::updateQEPol;

        addP_[0]    = MLUpdateFxnCplx::addP;
        addP_[1]    = MLUpdateFxnCplx::addP;

        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_  ,  Px->data()                                                                          , Px->size()           ); return; };
        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_  , &Px->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+1,  Py->data()                                                                          , Py->size()           ); return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+1, &Py->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };

        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_  ,  Ex->data()                                                                          , Ex->size()          ); return; };
        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_  , &Ex->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+1,  Ey->data()                                                                          , Ey->size()          ); return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+1, &Ey->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };

        transferE_[0] = MLUpdateFxnCplx::transferE;
        transferE_[1] = MLUpdateFxnCplx::transferE;

        zeroP_[0] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };
        zeroP_[1] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };

        if(accumulateP)
        {
            totP_[0].reserve(totTime);
            totP_[1].reserve(totTime);
            accumP_[0] = [](grid_ptr P, std::vector<cplx>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), cplx(0.0) ) ); };
            accumP_[1] = [](grid_ptr P, std::vector<cplx>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), cplx(0.0) ) ); };
        }
        else
        {
            accumP_[0] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
            accumP_[1] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
        }
    }
    else
    {
        if(E_[1] && e_[1])
            throw std::logic_error("If Ex field is in the qe then there must be an Ey film, please try to reconstruct with both");

        getE_[0]    = [] (grid_ptr, grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;
        getE_[1]    = [] (grid_ptr, grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[0]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        upP_[1]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};

        addP_[0]    = [] (grid_ptr, cplx_pgrid_ptr, real_pgrid_ptr, cplx*, int, int, int, int, int, int, int, int, int, int) {return;};
        addP_[1]    = [] (grid_ptr, cplx_pgrid_ptr, real_pgrid_ptr, cplx*, int, int, int, int, int, int, int, int, int, int) {return;};

        recvP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };
        sendP_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Px, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };
        recvP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<SendEFieldRecvPField> process_struct) {return; };
        sendP_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Py, std::shared_ptr<RecvEFieldSendPField> process_struct) {return; };

        sendE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[0] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ex, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };
        sendE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<SendEFieldRecvPField> process_struct)  {return; };
        recvE_[1] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ey, std::shared_ptr<RecvEFieldSendPField> process_struct)  {return; };

        transferE_[0] = [](cplx_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };
        transferE_[1] = [](cplx_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[0] = [](grid_ptr P){ return; };
        zeroP_[1] = [](grid_ptr P){ return; };

        accumP_[0] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
        accumP_[1] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
    }
    if(e_[2] && E_[2])
    {
        if(!E_[0])
            getE_[2]    = MLUpdateFxnCplx::getE_TM;
        else
            getE_[2]    = MLUpdateFxnCplx::getE_TE;

        upP_[2]     = MLUpdateFxnCplx::updateQEPol;
        addP_[2]    = MLUpdateFxnCplx::addP;

        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procCalcQE_  , process_struct->tagRecv_+2,  Pz->data()                                                                          , Pz->size()           ); return; };
        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procMainGrid_, process_struct->tagSend_+2, &Pz->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->szP_ ); return; };

        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct) {reqs[index] = gridComm->isend(process_struct->procCalcQE_  , process_struct->tagSend_+2,  Ez->data()                                                                          , Ez->size()          ); return; };
        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct) {reqs[index] = gridComm->irecv(process_struct->procMainGrid_, process_struct->tagRecv_+2, &Ez->point(process_struct->loc_[0], process_struct->loc_[1], process_struct->loc_[2]), process_struct->sz_ ); return; };
        transferE_[2]    = MLUpdateFxnCplx::transferE;

        zeroP_[2] = [](grid_ptr P){std::fill_n(P->data(), P->size(), 0.0); return; };

        if(accumulateP )
        {
            totP_[2].reserve(totTime);
            accumP_[2] = [](grid_ptr P, std::vector<cplx>& totVec){totVec.push_back( std::accumulate(P->data(), P->data()+P->size(), cplx(0.0) ) ); };
        }
        else
        {
            accumP_[2] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
        }
    }
    else
    {
        getE_[2]    = [] (grid_ptr, grid_ptr, grid_ptr, int, int, int, int, int, int, int, int, int, int) {return;} ;

        upP_[2]     = [] (cplx*, int, int, int, cplx*, grid_ptr, double, int) {return;};
        addP_[2]    = [] (grid_ptr, cplx_pgrid_ptr, real_pgrid_ptr, cplx*, int, int, int, int, int, int, int, int, int, int) {return;};

        sendP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<RecvEFieldSendPField> process_struct) { return; };
        recvP_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Pz, std::shared_ptr<SendEFieldRecvPField> process_struct) { return; };

        sendE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<SendEFieldRecvPField> process_struct)  { return; };
        recvE_[2] = [](std::shared_ptr<mpiInterface> gridComm, std::vector<mpi::request>& reqs, int index, grid_ptr Ez, std::shared_ptr<RecvEFieldSendPField> process_struct)  { return; };

        transferE_[2] = [](cplx_pgrid_ptr, grid_ptr, int, int, int, int, int, int, int){ return; };

        zeroP_[2] = [](grid_ptr P){return; };
        accumP_[2] = [](grid_ptr P, std::vector<cplx>& totVec){return; };
    }
}

void parallelQEReal::outputPol()
{
    if(gridComm_->rank() == outputPAccuProc_)
    {
        std::ofstream outPol;
        outPol.open(outputPfname_);
        std::vector<std::vector<double>> allPx;
        std::vector<std::vector<double>> allPy;
        std::vector<std::vector<double>> allPz;
        mpi::gather(*gridComm_, totP_[0], allPx, outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[1], allPy, outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[2], allPz, outputPAccuProc_);
        int iterMax = std::max(allPx[0].size(), allPz[0].size() );
        for(int ii = 0; ii < iterMax; ++ii)
        {
            outPol << dt_*ii;
            double px = 0; double py = 0; double pz = 0;
            if(allPx[0].size() > 0)
            {
                for(int pp = 0; pp < allPx.size(); ++pp)
                {
                    px += allPx[pp][ii];
                    py += allPy[pp][ii];
                }
                outPol << '\t' << px << '\t' << py;
            }
            if(allPz[0].size() > 0)
            {
                for(int pp = 0; pp < allPz.size(); ++pp)
                {
                    pz += allPz[pp][ii];
                }
                outPol << '\t' << pz;
            }
            outPol << '\t'<< sqrt( pow(px,2) + pow(py,2) + pow(pz,2) ) << '\n';
        }
        outPol.close();
    }
    else
    {
        mpi::gather(*gridComm_, totP_[0], outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[1], outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[2], outputPAccuProc_);
    }
}

void parallelQECplx::outputPol()
{
    if(gridComm_->rank() == outputPAccuProc_)
    {
        std::ofstream outPol;
        outPol.open(outputPfname_);
        std::vector<std::vector<cplx>> allPx;
        std::vector<std::vector<cplx>> allPy;
        std::vector<std::vector<cplx>> allPz;
        mpi::gather(*gridComm_, totP_[0], allPx, outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[1], allPy, outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[2], allPz, outputPAccuProc_);
        int iterMax = std::max(allPx[0].size(), allPz[0].size() );
        for(int ii = 0; ii < iterMax; ++ii)
        {
            outPol << dt_*ii;
            cplx px = 0; cplx py = 0; ; cplx pz = 0;
            if(allPx[0].size() > 0)
            {
                for(int pp = 0; pp < allPx.size(); ++pp)
                {
                    px += allPx[pp][ii];
                    py += allPy[pp][ii];
                }
                outPol << '\t' << std::real(px) << '\t' << std::imag(px) << '\t' << std::real(py) << '\t' << std::imag(py);
            }
            if(allPz[0].size() > 0)
            {
                for(int pp = 0; pp < allPz.size(); ++pp)
                {
                    pz += allPz[pp][ii];
                }
                outPol << '\t' << std::real(pz) << '\t' << std::imag(pz);
            }
            outPol << '\t' <<  sqrt( pow(std::abs(px), 2.0) + pow(std::abs(py), 2.0) + pow(std::abs(pz), 2.0) ) << '\n';
        }
        outPol.close();
    }
    else
    {
        mpi::gather(*gridComm_, totP_[0], outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[1], outputPAccuProc_);
        mpi::gather(*gridComm_, totP_[2], outputPAccuProc_);
    }
}