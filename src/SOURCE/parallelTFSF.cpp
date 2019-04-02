/** @file SOURCE/parallelTFSF.cpp
 *  @brief Class creates a TFSF surface to introduce an incident pulse
 *
 *  A class used to introduce a plane wave from a TFSF surface centered at loc in a box the size of sz
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#include <SOURCE/parallelTFSF.hpp>
void updateIncdFieldJ(const std::vector<cplx>& pulseVec, int pulAddStart, cplx prefact, const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incd_i, cplx_grid_ptr incd_j, cplx_grid_ptr incd_k, std::shared_ptr<IncdCPMLCplx> pml_incd )
{
    std::transform(pulseVec.data(), pulseVec.data()+pulseVec.size(), &incd_i->point(pulAddStart), &incd_i->point(pulAddStart), [&](cplx pul, cplx incd){return incd + prefact * pul;} );
    for(auto& param : upList)
    {
        zaxpy_(param.nSz_,      param.prefactor_j_, &incd_j->point(param.indPosK_), 1, &incd_i->point(param.indI_), 1);
        zaxpy_(param.nSz_, -1.0*param.prefactor_j_, &incd_j->point(param.indNegK_), 1, &incd_i->point(param.indI_), 1);
    }
    pml_incd->updateGrid();
}

void updateIncdFieldK(const std::vector<cplx>& pulseVec, int pulAddStart, cplx prefact, const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incd_i, cplx_grid_ptr incd_j, cplx_grid_ptr incd_k, std::shared_ptr<IncdCPMLCplx> pml_incd )
{
    std::transform(pulseVec.data(), pulseVec.data()+pulseVec.size(), &incd_i->point(pulAddStart), &incd_i->point(pulAddStart), [&](cplx pul, cplx incd){return incd + prefact * pul;} );
    for(auto& param : upList)
    {
        zaxpy_(param.nSz_,      param.prefactor_k_, &incd_k->point(param.indPosJ_), 1, &incd_i->point(param.indI_), 1);
        zaxpy_(param.nSz_, -1.0*param.prefactor_k_, &incd_k->point(param.indNegJ_), 1, &incd_i->point(param.indI_), 1);
    }
    pml_incd->updateGrid();
}

void updateIncdFieldJK(const std::vector<cplx>& pulseVec, int pulAddStart, cplx prefact, const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incd_i, cplx_grid_ptr incd_j, cplx_grid_ptr incd_k, std::shared_ptr<IncdCPMLCplx> pml_incd )
{
    std::transform(pulseVec.data(), pulseVec.data()+pulseVec.size(), &incd_i->point(pulAddStart), &incd_i->point(pulAddStart), [&](cplx pul, cplx incd){return incd + prefact * pul;} );
    for(auto& param : upList)
    {
        zaxpy_(param.nSz_,      param.prefactor_j_, &incd_j->point(param.indPosK_), 1, &incd_i->point(param.indI_), 1);
        zaxpy_(param.nSz_, -1.0*param.prefactor_j_, &incd_j->point(param.indNegK_), 1, &incd_i->point(param.indI_), 1);
        zaxpy_(param.nSz_,      param.prefactor_k_, &incd_k->point(param.indPosJ_), 1, &incd_i->point(param.indI_), 1);
        zaxpy_(param.nSz_, -1.0*param.prefactor_k_, &incd_k->point(param.indNegJ_), 1, &incd_i->point(param.indI_), 1);
    }
    pml_incd->updateGrid();
}

void updateIncdPols(const std::vector<paramUpIncdField>& upList, cplx_grid_ptr incdU , std::vector<cplx_grid_ptr>& incdP, std::vector<cplx_grid_ptr>& incdPrevP, cplx* scratch )
{
    for(auto& param : upList)
    {
        for(int pp = 0; pp  < param.aChiXi_.size(); ++pp)
        {
            zcopy_(param.nSz_, &incdP[pp]->point(param.indI_), 1, scratch, 1);

            zscal_(param.nSz_, param.aChiAlpha_[pp],     &incdP[pp]->point(param.indI_),  1);
            zaxpy_(param.nSz_, param.aChiXi_   [pp], &incdPrevP[pp]->point(param.indI_),  1, &incdP[pp]->point(param.indI_), 1);
            zaxpy_(param.nSz_, param.aChiGamma_[pp],         &incdU->point(param.indI_),  1, &incdP[pp]->point(param.indI_), 1);

            zcopy_(param.nSz_, scratch, 1, &incdPrevP[pp]->point(param.indI_), 1);
        }
    }
}

void incdD2U( double chiFact, cplx_grid_ptr incdU, cplx_grid_ptr incdD, real_grid_ptr ep_mu, const std::vector<cplx_grid_ptr>& incdP, cplx* scratch )
{
    std::transform(incdD->data(), incdD->data()+incdD->size(), ep_mu->data(), incdU->data(), [](cplx a, double b){return a/b; });
    for(int pp = 0; pp < incdP.size(); ++pp)
    {
        std::transform(incdP[pp]->data(), incdP[pp]->data()+incdP[pp]->size(), ep_mu->data(), scratch, [](cplx a, double b){return -1.0*a/b; });
        zaxpy_(incdU->size(), 1.0, scratch, 1, incdU->data(), 1);
    }
}

void fillPulseVec(const std::vector<double>& distVec, std::vector<std::shared_ptr<Pulse>> pulses, double tt, std::vector<cplx>& pulseVec)
{
    std::fill_n(pulseVec.begin(), pulseVec.size(), 0.0);
    for(auto& p : pulses)
        std::transform(distVec.begin(), distVec.end(), pulseVec.begin(), pulseVec.begin(), [&](double dist, cplx pVal){ return pVal + p->pulse(tt - dist); } );
}

void tfsfUpdateFxnReal::addIncdFields(real_pgrid_ptr grid_Ui, real_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, double* tempEpMuStore)
{
    for(int ll = 0; ll < sur->indsD_.size(); ll+=2)
        daxpy_(sur->szTrans_[0], sur->prefactor_, reinterpret_cast<double*>(&sur->incdField_->point( sur->indsD_[ll] ) ), 2*sur->strideIncd_, &grid_Di->point( sur->indsD_[ll+1] ), sur->strideMain_);
    for(int ll = 0; ll < sur->indsU_.size(); ll+=2)
        daxpy_(sur->szTrans_[0], sur->prefactor_, reinterpret_cast<double*>(&sur->incdField_->point( sur->indsU_[ll] ) ), 2*sur->strideIncd_, &grid_Ui->point( sur->indsU_[ll+1] ), sur->strideMain_);
}

void tfsfUpdateFxnCplx::addIncdFields(cplx_pgrid_ptr grid_Ui, cplx_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, cplx* tempEpMuStore)
{
    for(int ll = 0; ll < sur->indsD_.size(); ll+=2)
        zaxpy_(sur->szTrans_[0], sur->prefactor_, &sur->incdField_->point( sur->indsD_[ll] ), sur->strideIncd_, &grid_Di->point( sur->indsD_[ll+1] ), sur->strideMain_);
    for(int ll = 0; ll < sur->indsU_.size(); ll+=2)
        zaxpy_(sur->szTrans_[0], sur->prefactor_, &sur->incdField_->point( sur->indsU_[ll] ), sur->strideIncd_, &grid_Ui->point( sur->indsU_[ll+1] ), sur->strideMain_);
}

void tfsfUpdateFxnReal::addIncdFieldsEPChange(real_pgrid_ptr grid_Ui, real_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, double* tempEpMuStore)
{
    for(int ll = 0; ll < sur->indsD_.size(); ll+=2)
        daxpy_(sur->szTrans_[0], sur->prefactor_, reinterpret_cast<double*>(&sur->incdField_->point( sur->indsD_[ll] ) ), 2*sur->strideIncd_, &grid_Di->point( sur->indsD_[ll+1] ), sur->strideMain_);
    for(int ll = 0; ll < sur->indsU_.size(); ll+=2)
    {
        dcopy_(sur->szTrans_[0], reinterpret_cast<double*>(&sur->incdField_->point( sur->indsU_[ll] ) ), 2*sur->strideIncd_, tempEpMuStore                 , 1);
        dcopy_(sur->szTrans_[0],                                     &ep_mu->point( sur->indsU_[ll] )  ,   sur->strideIncd_, tempEpMuStore+sur->szTrans_[0], 1);
        std::transform(tempEpMuStore, tempEpMuStore+sur->szTrans_[0], tempEpMuStore+sur->szTrans_[0], tempEpMuStore, std::divides<double>() );
        daxpy_(sur->szTrans_[0], sur->prefactor_, tempEpMuStore, 1, &grid_Ui->point( sur->indsU_[ll+1] ), sur->strideMain_);
    }
}

void tfsfUpdateFxnCplx::addIncdFieldsEPChange(cplx_pgrid_ptr grid_Ui, cplx_pgrid_ptr grid_Di, std::shared_ptr<paramStoreTFSF> sur, real_grid_ptr ep_mu, cplx* tempEpMuStore)
{
    for(int ll = 0; ll < sur->indsD_.size(); ll+=2)
        zaxpy_(sur->szTrans_[0], sur->prefactor_, &sur->incdField_->point( sur->indsD_[ll] ), sur->strideIncd_, &grid_Di->point( sur->indsD_[ll+1] ), sur->strideMain_);
    for(int ll = 0; ll < sur->indsU_.size(); ll+=2)
    {
        zcopy_(sur->szTrans_[0], &sur->incdField_->point( sur->indsU_[ll] ), sur->strideIncd_,                           tempEpMuStore                  , 1);
        dcopy_(sur->szTrans_[0],           &ep_mu->point( sur->indsU_[ll] ), sur->strideIncd_, reinterpret_cast<double*>(tempEpMuStore+sur->szTrans_[0]), 2);
        std::transform(tempEpMuStore, tempEpMuStore+sur->szTrans_[0], tempEpMuStore+sur->szTrans_[0], tempEpMuStore, std::divides<cplx>() );
        zaxpy_(sur->szTrans_[0], sur->prefactor_, tempEpMuStore, 1, &grid_Ui->point( sur->indsU_[ll+1] ), sur->strideMain_);
    }
}

parallelTFSFReal::parallelTFSFReal(std::shared_ptr<mpiInterface> gridComm, std::array<int,3> loc, std::array<int,3> sz, double theta, double phi, double psi, POLARIZATION circPol, double kLenRelJ, std::array<double,3> d, std::array<int,3> m, double dt, std::vector<std::shared_ptr<Pulse>> pul, std::array<pgrid_ptr,3> E, std::array<pgrid_ptr,3> H, std::array<pgrid_ptr,3> D, std::array<pgrid_ptr,3> B, std::array<int_pgrid_ptr,3> physE, std::array<int_pgrid_ptr,3> physH, std::vector<std::shared_ptr<Obj>> objArr, int nomPMLThick, double pmlM, double pmlMA, double pmlAMax) :
    parallelTFSFBase<double>(gridComm, loc, sz, theta, phi, psi, circPol, kLenRelJ, d, m, dt, pul, E, H, D, B, physE, physH, objArr, nomPMLThick, pmlM, pmlMA, pmlAMax)
{
    if(E_[0])
    {
        if( std::any_of(eps_[0]->data(), eps_[0]->data() + eps_[0]->size(), [=](double a){ return a != eps_[0]->point(0); } ) )
            addEIncd_[0] = tfsfUpdateFxnReal::addIncdFieldsEPChange;
        else
            addEIncd_[0] = tfsfUpdateFxnReal::addIncdFields;
    }
    else
    {
        addEIncd_[0] = [](real_pgrid_ptr, real_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, double*){ return; };
    }

    if(E_[1])
    {
        if( std::any_of(eps_[1]->data(), eps_[1]->data() + eps_[1]->size(), [=](double a){ return a != eps_[1]->point(0); } ) )
            addEIncd_[1] = tfsfUpdateFxnReal::addIncdFieldsEPChange;
        else
            addEIncd_[1] = tfsfUpdateFxnReal::addIncdFields;
    }
    else
    {
        addEIncd_[1] = [](real_pgrid_ptr, real_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, double*){ return; };
    }

    if(E_[2])
    {
        if( std::any_of(eps_[2]->data(), eps_[2]->data() + eps_[2]->size(), [=](double a){ return a != eps_[2]->point(0); } ) )
            addEIncd_[2] = tfsfUpdateFxnReal::addIncdFieldsEPChange;
        else
            addEIncd_[2] = tfsfUpdateFxnReal::addIncdFields;
    }
    else
    {
        addEIncd_[2] = [](real_pgrid_ptr, real_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, double*){ return; };
    }

    if(H_[0])
    {
        if( std::any_of(mu_[0]->data(), mu_[0]->data() + mu_[0]->size(), [=](double a){ return a != mu_[0]->point(0); } ) )
            addHIncd_[0] = tfsfUpdateFxnReal::addIncdFieldsEPChange;
        else
            addHIncd_[0] = tfsfUpdateFxnReal::addIncdFields;
    }
    else
    {
        addHIncd_[0] = [](real_pgrid_ptr, real_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, double*){ return; };
    }

    if(H_[1])
    {
        if( std::any_of(mu_[1]->data(), mu_[1]->data() + mu_[1]->size(), [=](double a){ return a != mu_[1]->point(0); } ) )
            addHIncd_[1] = tfsfUpdateFxnReal::addIncdFieldsEPChange;
        else
            addHIncd_[1] = tfsfUpdateFxnReal::addIncdFields;
    }
    else
    {
        addHIncd_[1] = [](real_pgrid_ptr, real_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, double*){ return; };
    }

    if(H_[2])
    {
        if( std::any_of(mu_[2]->data(), mu_[2]->data() + mu_[2]->size(), [=](double a){ return a != mu_[2]->point(0); } ) )
            addHIncd_[2] = tfsfUpdateFxnReal::addIncdFieldsEPChange;
        else
            addHIncd_[2] = tfsfUpdateFxnReal::addIncdFields;
    }
    else
    {
        addHIncd_[2] = [](real_pgrid_ptr, real_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, double*){ return; };
    }
}
parallelTFSFCplx::parallelTFSFCplx(std::shared_ptr<mpiInterface> gridComm, std::array<int,3> loc, std::array<int,3> sz, double theta, double phi, double psi, POLARIZATION circPol, double kLenRelJ, std::array<double,3> d, std::array<int,3> m, double dt, std::vector<std::shared_ptr<Pulse>> pul, std::array<pgrid_ptr,3> E, std::array<pgrid_ptr,3> H, std::array<pgrid_ptr,3> D, std::array<pgrid_ptr,3> B, std::array<int_pgrid_ptr,3> physE, std::array<int_pgrid_ptr,3> physH, std::vector<std::shared_ptr<Obj>> objArr, int nomPMLThick, double pmlM, double pmlMA, double pmlAMax) :
    parallelTFSFBase<cplx>(gridComm, loc, sz, theta, phi, psi, circPol, kLenRelJ, d, m, dt, pul, E, H, D, B, physE, physH, objArr, nomPMLThick, pmlM, pmlMA, pmlAMax)
{
    if(E_[0])
    {
        if( std::any_of(eps_[0]->data(), eps_[0]->data() + eps_[0]->size(), [=](double a){ return a != eps_[0]->point(0); } ) )
            addEIncd_[0] = tfsfUpdateFxnCplx::addIncdFieldsEPChange;
        else
            addEIncd_[0] = tfsfUpdateFxnCplx::addIncdFields;
    }
    else
    {
        addEIncd_[0] = [](cplx_pgrid_ptr, cplx_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, cplx*){ return; };
    }

    if(E_[1])
    {
        if( std::any_of(eps_[1]->data(), eps_[1]->data() + eps_[1]->size(), [=](double a){ return a != eps_[1]->point(0); } ) )
            addEIncd_[1] = tfsfUpdateFxnCplx::addIncdFieldsEPChange;
        else
            addEIncd_[1] = tfsfUpdateFxnCplx::addIncdFields;
    }
    else
    {
        addEIncd_[1] = [](cplx_pgrid_ptr, cplx_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, cplx*){ return; };
    }

    if(E_[2])
    {
        if( std::any_of(eps_[2]->data(), eps_[2]->data() + eps_[2]->size(), [=](double a){ return a != eps_[2]->point(0); } ) )
            addEIncd_[2] = tfsfUpdateFxnCplx::addIncdFieldsEPChange;
        else
            addEIncd_[2] = tfsfUpdateFxnCplx::addIncdFields;
    }
    else
    {
        addEIncd_[2] = [](cplx_pgrid_ptr, cplx_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, cplx*){ return; };
    }

    if(H_[0])
    {
        if( std::any_of(mu_[0]->data(), mu_[0]->data() + mu_[0]->size(), [=](double a){ return a != mu_[0]->point(0); } ) )
            addHIncd_[0] = tfsfUpdateFxnCplx::addIncdFieldsEPChange;
        else
            addHIncd_[0] = tfsfUpdateFxnCplx::addIncdFields;
    }
    else
    {
        addHIncd_[0] = [](cplx_pgrid_ptr, cplx_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, cplx*){ return; };
    }

    if(H_[1])
    {
        if( std::any_of(mu_[1]->data(), mu_[1]->data() + mu_[1]->size(), [=](double a){ return a != mu_[1]->point(0); } ) )
            addHIncd_[1] = tfsfUpdateFxnCplx::addIncdFieldsEPChange;
        else
            addHIncd_[1] = tfsfUpdateFxnCplx::addIncdFields;
    }
    else
    {
        addHIncd_[1] = [](cplx_pgrid_ptr, cplx_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, cplx*){ return; };
    }

    if(H_[2])
    {
        if( std::any_of(mu_[2]->data(), mu_[2]->data() + mu_[2]->size(), [=](double a){ return a != mu_[2]->point(0); } ) )
            addHIncd_[2] = tfsfUpdateFxnCplx::addIncdFieldsEPChange;
        else
            addHIncd_[2] = tfsfUpdateFxnCplx::addIncdFields;
    }
    else
    {
        addHIncd_[2] = [](cplx_pgrid_ptr, cplx_pgrid_ptr, std::shared_ptr<paramStoreTFSF>, real_grid_ptr, cplx*){ return; };
    }
}