/** @file FDTD_MANAGER/parallelFDTDField.cpp
 *  @brief Manager that stores the FDTD grids and updates them in time
 *
 *  Class that stores the FDTD grids and all necessary components to update them forward
 *  in time.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <FDTD_MANAGER/parallelFDTDField.hpp>

parallelFDTDFieldReal::parallelFDTDFieldReal(parallelProgramInputs &IP, std::shared_ptr<mpiInterface> gridComm) :
    parallelFDTDFieldBase<double>(IP, gridComm)
{
    if(E_[2] && H_[2])
    {
        upLorOrDipP_ = &FDTDCompUpdateFxnReal::UpdateLorPolOrDip;
        upLorOrDipM_ = &FDTDCompUpdateFxnReal::UpdateLorPolOrDip;
    }
    else if(E_[2])
    {
        upLorOrDipP_ = &FDTDCompUpdateFxnReal::UpdateLorPolOrDipZ;
        upLorOrDipM_ = &FDTDCompUpdateFxnReal::UpdateLorPolOrDipXY;
    }
    else
    {
        upLorOrDipP_ = &FDTDCompUpdateFxnReal::UpdateLorPolOrDipXY;
        upLorOrDipM_ = &FDTDCompUpdateFxnReal::UpdateLorPolOrDipZ;
    }

    upChiLorOrDipP_ = &FDTDCompUpdateFxnReal::UpdateChiralOrDip;
    upChiLorOrDipM_ = &FDTDCompUpdateFxnReal::UpdateChiralOrDip;
    if(IP.periodic_ && gridComm_->size() > 1)
    {
        if(gridComm_->rank() !=0 || gridComm_->rank() != gridComm_->size() - 1)
            applBCOrDip_ = &FDTDCompUpdateFxnReal::applyBCProcMid;
        else
            applBCOrDip_ = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnReal::applyBCProc0 : &FDTDCompUpdateFxnReal::applyBCProcMax;
    }
    else if(IP.periodic_)
    {
        applBCOrDip_ = &FDTDCompUpdateFxnReal::applyBC1Proc;
    }
    else if(gridComm_->size() > 1)
    {
        applBCOrDip_ = &FDTDCompUpdateFxnReal::applyBCNonPer;
    }
    else
    {
        applBCOrDip_ = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
    }
    // If there is an Hz field set up all TE mode functions otherwise set them to do nothing
    std::array<std::vector<std::array<int,5>>,3> chiBLocs;
    std::array<std::vector<std::array<int,5>>,3> chiDLocs;
    std::vector<std::array<int,5>> chiOrDipLocs;
    if(H_[2])
    {
        // Initialize the PMLs
        if(magMatInPML_)
        {
            HPML_[2]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, B_[2], E_[0], E_[1], POLARIZATION::HZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[2], objArr_);
        }
        else
        {
            HPML_[2]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, H_[2], E_[0], E_[1], POLARIZATION::HZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[2], objArr_);
        }
        if(dielectricMatInPML_)
        {
            EPML_[0]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, D_[0], H_[1], H_[2], POLARIZATION::EX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[0], objArr_);
            EPML_[1]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, D_[1], H_[2], H_[0], POLARIZATION::EY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[1], objArr_);
        }
        else
        {
            EPML_[0]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, E_[0], H_[1], H_[2], POLARIZATION::EX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[0], objArr_);
            EPML_[1]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, E_[1], H_[2], H_[0], POLARIZATION::EY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[1], objArr_);
        }
        // Fill the update Lists
        gridComm_->barrier();
        initializeList(physH_[2],  muRel_[2], HPML_[2], false, std::array<int,3>( {{ 1,  0,  0 }} ), std::array<int,3>( {{ 1, 1, 0 }} ), d_[0], d_[1], upH_[2], upB_[2], upLorB_[2], upChiB_[2], upOrDipB_[2], upOrDipChiB_[2], chiBLocs[2]);
        initializeList(physE_[0], epsRel_[0], EPML_[0],  true, std::array<int,3>( {{ 0, -1,  0 }} ), std::array<int,3>( {{ 1, 0, 0 }} ), d_[1], d_[2], upE_[0], upD_[0], upLorD_[0], upChiD_[0], upOrDipD_[0], upOrDipChiD_[0], chiDLocs[0]);
        initializeList(physE_[1], epsRel_[1], EPML_[1],  true, std::array<int,3>( {{ 0,  0, -1 }} ), std::array<int,3>( {{ 0, 1, 0 }} ), d_[2], d_[0], upE_[1], upD_[1], upLorD_[1], upChiD_[1], upOrDipD_[1], upOrDipChiD_[1], chiDLocs[1]);
        if(dipP_[0].size() > 0)
        {
            upLists tempNonOrDip;
            initializeList(physPOrDip_, epsRelOrDip_, HPML_[2],  true, std::array<int,3>( {{-1, -1, -1 }} ), std::array<int,3>( {{ 0, 0, 0 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipP_, upChiOrDipP_, chiOrDipLocs, true);
        }
        if(dipM_[0].size() > 0)
        {
            upLists tempNonOrDip;
            initializeList(physMOrDip_,  muRelOrDip_, HPML_[2], false, std::array<int,3>( {{ 1,  1,  1 }} ), std::array<int,3>( {{ 1, 1, 1 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipM_, upChiOrDipM_, chiOrDipLocs, true);
        }

        if(!E_[2])
        {
            upEFxn_[0] = &FDTDCompUpdateFxnReal::OneCompCurlK;
            upEFxn_[1] = &FDTDCompUpdateFxnReal::OneCompCurlJ;
        }
        else
        {
            upEFxn_[0] = &FDTDCompUpdateFxnReal::TwoCompCurl;
            upEFxn_[1] = &FDTDCompUpdateFxnReal::TwoCompCurl;
        }

        upHFxn_[2] = &FDTDCompUpdateFxnReal::TwoCompCurl;

        updateEPML_[0] = [](pml_ptr pml){pml->updateGrid();};
        updateEPML_[1] = [](pml_ptr pml){pml->updateGrid();};
        updateHPML_[2] = [](pml_ptr pml){pml->updateGrid();};

        upLorPFxn_[0] = &FDTDCompUpdateFxnReal::UpdateLorPol;
        upLorPFxn_[1] = &FDTDCompUpdateFxnReal::UpdateLorPol;
        upLorMFxn_[2] = &FDTDCompUpdateFxnReal::UpdateLorPol;

        upChiE_[0] = &FDTDCompUpdateFxnReal::UpdateChiral;
        upChiE_[1] = &FDTDCompUpdateFxnReal::UpdateChiral;
        upChiH_[2] = &FDTDCompUpdateFxnReal::UpdateChiral;

        D2EFxn_[0] = &FDTDCompUpdateFxnReal::DtoU;
        D2EFxn_[1] = &FDTDCompUpdateFxnReal::DtoU;
        B2HFxn_[2] = &FDTDCompUpdateFxnReal::DtoU;

        chiD2EFxn_[0] = &FDTDCompUpdateFxnReal::chiDtoU;
        chiD2EFxn_[1] = &FDTDCompUpdateFxnReal::chiDtoU;
        chiB2HFxn_[2] = &FDTDCompUpdateFxnReal::chiDtoU;

        orDipD2EFxn_[0] = &FDTDCompUpdateFxnReal::orDipDtoU;
        orDipD2EFxn_[1] = &FDTDCompUpdateFxnReal::orDipDtoU;
        if(!E_[2])
            orDipB2HFxn_[2] = &FDTDCompUpdateFxnReal::orDipDtoUZ;
        else
            orDipB2HFxn_[2] = &FDTDCompUpdateFxnReal::orDipDtoU;

        chiOrDipD2EFxn_[0] = &FDTDCompUpdateFxnReal::chiOrDipDtoU;
        chiOrDipD2EFxn_[1] = &FDTDCompUpdateFxnReal::chiOrDipDtoU;
        chiOrDipB2HFxn_[2] = &FDTDCompUpdateFxnReal::chiOrDipDtoU;

        if(IP.periodic_ && gridComm_->size() > 1)
        {
            if(gridComm_->rank() != gridComm_->size()-1 || gridComm_->rank() != 0)
            {
                applBCE_[0] = &FDTDCompUpdateFxnReal::applyBCProcMid;
                applBCE_[1] = &FDTDCompUpdateFxnReal::applyBCProcMid;
                applBCH_[2] = &FDTDCompUpdateFxnReal::applyBCProcMid;
            }
            else
            {
                applBCE_[0] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnReal::applyBCProc0 : &FDTDCompUpdateFxnReal::applyBCProcMax;
                applBCE_[1] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnReal::applyBCProc0 : &FDTDCompUpdateFxnReal::applyBCProcMax;
                applBCH_[2] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnReal::applyBCProc0 : &FDTDCompUpdateFxnReal::applyBCProcMax;
            }

            yEPBC_[0] = ln_vec_[1]+1;
            yEPBC_[1] = ln_vec_[1]+1;
            yHPBC_[2] = ln_vec_[1]+1;
            // Field definition buffer
            if(gridComm_->size()-1 == gridComm_->rank())
            {
                yHPBC_[2] -= 1;
                yEPBC_[1] -= 1;
            }
        }
        else if(IP.periodic_)
        {
            applBCE_[0] = &FDTDCompUpdateFxnReal::applyBC1Proc;
            applBCE_[1] = &FDTDCompUpdateFxnReal::applyBC1Proc;
            applBCH_[2] = &FDTDCompUpdateFxnReal::applyBC1Proc;

            yHPBC_[2] = ln_vec_[1];
            yEPBC_[0] = ln_vec_[1]+1;
            yEPBC_[1] = ln_vec_[1];
        }
        else if(gridComm_->size() > 1)
        {
            applBCE_[0] = &FDTDCompUpdateFxnReal::applyBCNonPer;
            applBCE_[1] = &FDTDCompUpdateFxnReal::applyBCNonPer;
            applBCH_[2] = &FDTDCompUpdateFxnReal::applyBCNonPer;
        }
        else
        {
            applBCE_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCE_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCH_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        }
    }
    else
    {
        upHFxn_[2] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};

        upEFxn_[0] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};;
        upEFxn_[1] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};;

        updateEPML_[0] = [](pml_ptr pml){return;};
        updateEPML_[1] = [](pml_ptr pml){return;};
        updateHPML_[2] = [](pml_ptr pml){return;};

        upLorPFxn_[0] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upLorPFxn_[1] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upLorMFxn_[2] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};

        upChiE_[0] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upChiE_[1] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upChiH_[2] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};

        D2EFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        D2EFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        B2HFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        chiD2EFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiD2EFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiB2HFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        orDipD2EFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        orDipD2EFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        orDipB2HFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};

        chiOrDipD2EFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        chiOrDipD2EFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        chiOrDipB2HFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};

        applBCE_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCE_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCH_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
    }
    // If there is an Ez field set up all TM functions otherwise set them to do nothing
    if(E_[2])
    {
        //initialize the PMLs
        if(magMatInPML_)
        {
            HPML_[0]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, B_[0], E_[1], E_[2], POLARIZATION::HX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[0], objArr_);
            HPML_[1]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, B_[1], E_[2], E_[0], POLARIZATION::HY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[1], objArr_);
        }
        else
        {
            HPML_[0]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, H_[0], E_[1], E_[2], POLARIZATION::HX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[0], objArr_);
            HPML_[1]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, H_[1], E_[2], E_[0], POLARIZATION::HY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[1], objArr_);
        }
        if(dielectricMatInPML_)
        {
            EPML_[2]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, D_[2], H_[0], H_[1], POLARIZATION::EZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[2], objArr_);
        }
        else
        {
            EPML_[2]   = std::make_shared<parallelCPMLReal>(gridComm_, weights_, E_[2], H_[0], H_[1], POLARIZATION::EZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[2], objArr_);
        }
        // Fill update lists
        initializeList(physE_[2], epsRel_[2], EPML_[2],  true, std::array<int,3>( {{-1,  0,  0 }} ), std::array<int,3>( {{ 0, 0, 1 }} ), d_[0], d_[1], upE_[2], upD_[2], upLorD_[2], upChiD_[2], upOrDipD_[2], upOrDipChiD_[2], chiDLocs[2]);
        initializeList(physH_[0],  muRel_[0], HPML_[0], false, std::array<int,3>( {{ 0,  1,  0 }} ), std::array<int,3>( {{ 0, 1, 1 }} ), d_[1], d_[2], upH_[0], upB_[0], upLorB_[0], upChiB_[0], upOrDipB_[0], upOrDipChiB_[0], chiBLocs[0]);
        initializeList(physH_[1],  muRel_[1], HPML_[1], false, std::array<int,3>( {{ 0,  0,  1 }} ), std::array<int,3>( {{ 1, 0, 1 }} ), d_[2], d_[0], upH_[1], upB_[1], upLorB_[1], upChiB_[1], upOrDipB_[1], upOrDipChiB_[1], chiBLocs[1]);

        if(dipP_[2].size() > 0 && !H_[2])
        {
            upLists tempNonOrDip;
            initializeList(physPOrDip_, epsRelOrDip_, EPML_[2],  true, std::array<int,3>( {{-1, -1, -1 }} ), std::array<int,3>( {{ 0, 0, 0 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipP_, upChiOrDipM_, chiOrDipLocs, true);
        }
        if(dipM_[0].size() > 0 && !H_[2])
        {
            upLists tempNonOrDip;
            initializeList(physMOrDip_,  muRelOrDip_, EPML_[2], false, std::array<int,3>( {{ 1,  1,  1 }} ), std::array<int,3>( {{ 1, 1, 1 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipM_, upChiOrDipP_, chiOrDipLocs, true);
        }

        if(!H_[2])
        {
            upHFxn_[0] = &FDTDCompUpdateFxnReal::OneCompCurlK;
            upHFxn_[1] = &FDTDCompUpdateFxnReal::OneCompCurlJ;
        }
        else
        {
            upHFxn_[0] = &FDTDCompUpdateFxnReal::TwoCompCurl;
            upHFxn_[1] = &FDTDCompUpdateFxnReal::TwoCompCurl;
        }

        upEFxn_[2] = &FDTDCompUpdateFxnReal::TwoCompCurl;

        upLorMFxn_[0] = &FDTDCompUpdateFxnReal::UpdateLorPol;
        upLorMFxn_[1] = &FDTDCompUpdateFxnReal::UpdateLorPol;
        upLorPFxn_[2] = &FDTDCompUpdateFxnReal::UpdateLorPol;

        upChiH_[0] = &FDTDCompUpdateFxnReal::UpdateChiral;
        upChiH_[1] = &FDTDCompUpdateFxnReal::UpdateChiral;
        upChiE_[2] = &FDTDCompUpdateFxnReal::UpdateChiral;

        B2HFxn_[0] = &FDTDCompUpdateFxnReal::DtoU;
        B2HFxn_[1] = &FDTDCompUpdateFxnReal::DtoU;
        D2EFxn_[2] = &FDTDCompUpdateFxnReal::DtoU;

        chiB2HFxn_[0] = &FDTDCompUpdateFxnReal::chiDtoU;
        chiB2HFxn_[1] = &FDTDCompUpdateFxnReal::chiDtoU;
        chiD2EFxn_[2] = &FDTDCompUpdateFxnReal::chiDtoU;

        orDipB2HFxn_[0] = &FDTDCompUpdateFxnReal::orDipDtoU;
        orDipB2HFxn_[1] = &FDTDCompUpdateFxnReal::orDipDtoU;
        if(!H_[2])
            orDipD2EFxn_[2] = &FDTDCompUpdateFxnReal::orDipDtoUZ;
        else
            orDipD2EFxn_[2] = &FDTDCompUpdateFxnReal::orDipDtoU;

        chiOrDipB2HFxn_[0] = &FDTDCompUpdateFxnReal::chiOrDipDtoU;
        chiOrDipB2HFxn_[1] = &FDTDCompUpdateFxnReal::chiOrDipDtoU;
        chiOrDipD2EFxn_[2] = &FDTDCompUpdateFxnReal::chiOrDipDtoU;

        updateHPML_[0] = [](pml_ptr pml){pml->updateGrid();};
        updateHPML_[1] = [](pml_ptr pml){pml->updateGrid();};
        updateEPML_[2] = [](pml_ptr pml){pml->updateGrid();};

        if(IP.periodic_ && gridComm_->size() > 1)
        {
            if(gridComm_->rank() != gridComm_->size()-1 || gridComm_->rank() != 0)
            {
                applBCH_[0] = &FDTDCompUpdateFxnReal::applyBCProcMid;
                applBCH_[1] = &FDTDCompUpdateFxnReal::applyBCProcMid;
                applBCE_[2] = &FDTDCompUpdateFxnReal::applyBCProcMid;
            }
            else
            {
                applBCH_[0] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnReal::applyBCProc0 : &FDTDCompUpdateFxnReal::applyBCProcMax;
                applBCH_[1] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnReal::applyBCProc0 : &FDTDCompUpdateFxnReal::applyBCProcMax;
                applBCE_[2] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnReal::applyBCProc0 : &FDTDCompUpdateFxnReal::applyBCProcMax;
            }

            yHPBC_[0] = ln_vec_[1]+1;
            yHPBC_[1] = ln_vec_[1]+1;
            yEPBC_[2] = ln_vec_[1]+1;
            //Field definition buffer
            if(gridComm_->size()-1 == gridComm_->rank())
                yHPBC_[0] = ln_vec_[1];
        }
        else if(IP.periodic_)
        {
            applBCH_[0] = &FDTDCompUpdateFxnReal::applyBC1Proc;
            applBCH_[1] = &FDTDCompUpdateFxnReal::applyBC1Proc;
            applBCE_[2] = &FDTDCompUpdateFxnReal::applyBC1Proc;
            yHPBC_[0] = ln_vec_[1];
            yHPBC_[1] = ln_vec_[1]+1;
            yEPBC_[2] = ln_vec_[1]+1;
        }
        else if(gridComm_->size() > 1)
        {
            applBCH_[0] = &FDTDCompUpdateFxnReal::applyBCNonPer;
            applBCH_[1] = &FDTDCompUpdateFxnReal::applyBCNonPer;
            applBCE_[2] = &FDTDCompUpdateFxnReal::applyBCNonPer;
        }
        else
        {
            applBCH_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCH_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCE_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        }
    }
    else
    {
        upHFxn_[0] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};;
        upHFxn_[1] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};

        upEFxn_[2] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};

        upLorMFxn_[0] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upLorMFxn_[1] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upLorPFxn_[2] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};

        upChiH_[0] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upChiH_[1] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};
        upChiE_[2] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, double*){return;};

        B2HFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        B2HFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        D2EFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        chiB2HFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiB2HFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiD2EFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        orDipB2HFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        orDipB2HFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        orDipD2EFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};

        chiOrDipB2HFxn_[0] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        chiOrDipB2HFxn_[1] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};
        chiOrDipD2EFxn_[2] = [](const std::array<int,6>&, double, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, double*, double*){return;};

        updateHPML_[0] = [](pml_ptr pml){return;};
        updateHPML_[1] = [](pml_ptr pml){return;};
        updateEPML_[2] = [](pml_ptr pml){return;};

        applBCH_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCH_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCE_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
    }

    // Find region to copy fields to prev fields
    for(int oo = 0; oo < objArr_.size(); ++oo)
    {
        if(objArr_[oo]->chiGamma().size() > 0)
        {
            int xChiMin = n_vec_[0], yChiMin = n_vec_[1], zChiMin = n_vec_[2];
            int xChiMax = 0, yChiMax = 0, zChiMax = 0;
            // Find Min and Max across all field updates for that object
            for(int ii = 0; ii < 3; ++ii)
            {
                findMinMaxChiLocsObj(chiDLocs[ii], oo, xChiMin, yChiMin, zChiMin, xChiMax, yChiMax, zChiMax);
                findMinMaxChiLocsObj(chiBLocs[ii], oo, xChiMin, yChiMin, zChiMin, xChiMax, yChiMax, zChiMax);
            }
            // Give a +/- buffer to copy over
            int sz = xChiMax - xChiMin + 3;
            for(int yy = yChiMin-1; yy <= yChiMax+1; ++yy)
                for(int zz = zChiMin-1; zz <= zChiMax+1; ++zz)
                    copy2PrevFields_.push_back(std::array<int,4>( {{ sz, xChiMin-1, yy, zz }} ) );
        }
    }
    // Construct QE objects
    for(int qq = 0; qq < IP.qeLoc_.size(); qq++)
    {
        // std::shared_ptr<parallelQEBase<double>> temp;
        // Construct BasisSet and Hamiltonian
        BasisSet basis(IP.qeBasis_[qq]);
        std::vector<Hamiltonian> hams;
        std::vector<double> current;
        double weight = 1.0;
        std::vector<std::pair<std::vector<double>, double>> eWeightPairs;
        std::vector<double> eWeights;
        eWeights.reserve(eWeightPairs.size());
        GenerateAllELevCombos(IP.qeELevs_, eWeightPairs, 0, current, weight);
        for(auto& eWeightpair : eWeightPairs)
        {
            hams.push_back( Hamiltonian( IP.qeBasis_[qq].size(), std::get<0>(eWeightpair),IP.qeCouplings_[qq], basis) );
            eWeights.push_back( std::get<1>(eWeightpair) );
        }
        std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr;
        for( int dd=0; dd < IP.qePopDtcLevs_[qq].size(); dd++)
            dtcPopArr.push_back( std::make_shared<QEPopDtc>(gridComm_, IP.qePopDtcLevs_[qq][dd], IP.qeBasis_[qq].size(), IP.qeLoc_[qq].size(), eWeights, IP.qeDtcPopOutFile_[qq][dd], IP.qeDtcPopTimeInt_[qq], static_cast<int>( ceil( IP.tMax() / (dt_ * IP.qeDtcPopTimeInt_[qq]) ) ), dt_) );

        // If the input file sets up the relaxation operators use that otherwise set up the relaxation operator storage operator for the qe constrcutor and then construct it
        if( IP.qeGam_[qq].size() > 0 )
        {
            qeArr_.push_back(std::make_shared<parallelQEReal>(gridComm_, hams, eWeightPairs, dtcPopArr, basis, IP.qeGam_[qq], IP.qeLoc_[qq], E_, epsRelOrDip_, IP.qeAccumP_[qq], IP.qeOutPolFname_[qq], ceil(IP.tMax_ / dt_ ) + 1, IP.a_, IP.I0_, dt_, IP.qeDen_[qq]*pow(IP.a_,3) ) ) ;
        }
        else
        {
            if(IP.qeRelaxTransitonStates_[qq].size() != IP.qeRelaxRates_[qq].size() )
                throw std::logic_error("All relaxation transition states must have a list of rates and state definitions");
            std::vector<relaxParams> relax;
            for(int rr = 0; rr < IP.qeRelaxTransitonStates_[qq].size(); rr++)
            {
                relaxParams tempRelax;
                tempRelax.n0_ = IP.qeRelaxTransitonStates_[qq][rr][0];
                tempRelax.nf_ = IP.qeRelaxTransitonStates_[qq][rr][1];
                tempRelax.rate_ = IP.qeRelaxRates_[qq][rr];
                tempRelax.dephasingRate_ = IP.qeRelaxDephasingRate_[qq][rr];
                relax.push_back(tempRelax);
            }
            qeArr_.push_back( std::make_shared<parallelQEReal>(gridComm_, hams, eWeightPairs, dtcPopArr, basis, relax, IP.qeLoc_[qq], E_, epsRelOrDip_, IP.qeAccumP_[qq], IP.qeOutPolFname_[qq], ceil(IP.tMax_ / dt_ ) + 1, IP.a_, IP.I0_, dt_, IP.qeDen_[qq]*pow(IP.a_,3) ) );
        }
    }

    // Construct all soft sources
    for(int ss = 0; ss < IP.srcPol_.size(); ss++)
    {
        // Make the pulse (including all pulses to be used)
        std::vector<std::shared_ptr<Pulse>> pul;
        for(int pp = 0; pp < IP.srcPulShape_[ss].size(); pp ++)
        {
            std::function<const cplx(double, const std::vector<cplx>&)> pulseFxn;
            if(IP.srcPulShape_[ss][pp] == PLSSHAPE::CONTINUOUS)
                pulseFxn = contPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::BH)
                pulseFxn = bhPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RECT)
                pulseFxn = rectPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::GAUSSIAN)
                pulseFxn = gaussPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RICKER)
                pulseFxn = rickerPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RAMP_CONT)
                pulseFxn = rampContPulse;
            IP.srcFxn_[ss][pp].push_back( IP.srcEmax_[ss][pp] );
            pul.push_back(std::make_shared<Pulse>(pulseFxn, IP.srcFxn_[ss][pp], dt_));
        }
        //  Make the source act on any of the fields
        if(IP.srcPol_[ss] == POLARIZATION::EX)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul, E_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, E_[0], POLARIZATION::EX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, E_[1], POLARIZATION::EY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::EY)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul, E_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, E_[0], POLARIZATION::EX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, E_[1], POLARIZATION::EY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::EZ)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul, E_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, E_[2], POLARIZATION::EZ, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::HX)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul, H_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, H_[1], POLARIZATION::HY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, H_[0], POLARIZATION::HX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::HY)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul, H_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, H_[0], POLARIZATION::HX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, H_[1], POLARIZATION::HY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::HZ)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul, H_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueReal >(gridComm_, pul, H_[2], POLARIZATION::HZ, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else
        {
            double axRat = IP.srcEllipticalKratio_[ss];
            double psi = IP.srcPsi_[ss];
            double psiPrefactCalc = psi;
            double alphaOff = 0.0;
            double prefactor_k_ = 1.0;
            double prefactor_j_ = 1.0;
            double c = pow(axRat, 2.0);

            // phi/psi control the light polarization angle
            psiPrefactCalc = 0.5 * asin( sqrt( ( pow(cos(2.0*psi),2.0)*4.0*c + pow( (1.0+c)*sin(2.0*psi), 2.0) ) / pow(1.0+c, 2.0) ) );
            alphaOff = acos( ( (c - 1.0)*sin(2.0*psi) ) / sqrt( pow(cos(2.0*psi),2.0)*4.0*c + pow( (1.0+c)*sin(2.0*psi), 2.0) ) );
            if(std::abs( std::tan(psi) ) > 1)
                psiPrefactCalc = M_PI/2.0 - psiPrefactCalc;
            if(IP.srcPol_[ss] == POLARIZATION::R)
                alphaOff *= -1.0;

            if( isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 0 )
            {
                prefactor_j_ *= -1.0 * cos(psiPrefactCalc);
                prefactor_k_ *= -1.0 * sin(psiPrefactCalc);
            }
            else if( isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 1 )
            {
                prefactor_j_ *=  -1.0 * cos(psiPrefactCalc);
                prefactor_k_ *=         sin(psiPrefactCalc);
            }
            else if( isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 2 )
            {
                prefactor_j_ *= -1.0*sin(psiPrefactCalc);
                prefactor_k_ *=      cos(psiPrefactCalc);
            }
            std::vector<std::shared_ptr<Pulse>> pul_j;
            std::vector<std::shared_ptr<Pulse>> pul_k;
            for(int pp = 0; pp < IP.srcPulShape_[ss].size(); pp ++)
            {
                std::function<const cplx(double, const std::vector<cplx>&)> pulseFxn;
                if(IP.srcPulShape_[ss][pp] == PLSSHAPE::CONTINUOUS)
                    pulseFxn = contPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::BH)
                    pulseFxn = bhPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RECT)
                    pulseFxn = rectPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::GAUSSIAN)
                    pulseFxn = gaussPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RICKER)
                    pulseFxn = rickerPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RAMP_CONT)
                    pulseFxn = rampContPulse;

                IP.srcFxn_[ss][pp].push_back( IP.srcEmax_[ss][pp]*prefactor_j_ );
                pul_j.push_back(std::make_shared<Pulse>(pulseFxn, IP.srcFxn_[ss][pp], dt_));

                IP.srcFxn_[ss][pp][IP.srcFxn_[ss][pp].size()-1] *= prefactor_k_ / prefactor_j_ * cplx(0.0, alphaOff);
                pul_k.push_back(std::make_shared<Pulse>(pulseFxn, IP.srcFxn_[ss][pp], dt_));
            }
            if(isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 0 )
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul_j, E_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul_k, E_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            if(isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 1 )
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul_j, E_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul_k, E_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            if(isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 2 )
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul_j, E_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceNormalReal>(gridComm_, pul_k, E_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
        }
    }

    // Construct all TFSF surfaces
    for(int tt = 0; tt < IP.tfsfSize_.size(); tt++)
    {
        if(IP.tfsfSize_[tt][0] != 0.0 || IP.tfsfSize_[tt][1] != 0.0)
        {
        // Make the pulse (including all pulses to be used)
            std::vector<std::shared_ptr<Pulse>> pul;
            for(int pp = 0; pp < IP.tfsfPulShape_[tt].size(); pp ++)
            {
                std::function<const cplx(double, const std::vector<cplx>&)> pulseFxn;
                if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::CONTINUOUS)
                    pulseFxn = contPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::BH)
                    pulseFxn = bhPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::RECT)
                    pulseFxn = rectPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::GAUSSIAN)
                    pulseFxn = gaussPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::RICKER)
                    pulseFxn = rickerPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::RAMP_CONT)
                    pulseFxn = rampContPulse;
                IP.tfsfPulFxn_[tt][pp].push_back(cplx(0.0,IP.tfsfEmax_[tt][pp]));
                pul.push_back(std::make_shared<Pulse>(pulseFxn, IP.tfsfPulFxn_[tt][pp], dt_));
            }
            // TFSF will determine the correct polarization
            tfsfArr_.push_back(std::make_shared<parallelTFSFReal>(gridComm_, IP.tfsfLoc_[tt], IP.tfsfSize_[tt], IP.tfsfTheta_[tt], IP.tfsfPhi_[tt], IP.tfsfPsi_[tt], IP.tfsfCircPol_[tt], IP.tfsfEllipticalKratio_[tt], d_, IP.tfsfM_[tt], dt_, pul, E_, H_, D_, B_, physE_, physH_, objArr_, IP.tfsfPMLThick_[tt], IP.tfsfPMLM_[tt], IP.tfsfPMLMa_[tt], IP.tfsfPMLAMax_[tt])  );
        }
    }
    // Polarization matters here since the z field always forms the continous box for the spatial offset (TE uses H, TM uses E)
    if(IP.fluxName_.size() > 0)
    {
        DIRECTION propDir;
        if(tfsfArr_.size() > 0 && (tfsfArr_.back()->theta() == 0.0 || tfsfArr_.back()->theta() == M_PI ) )
            propDir = DIRECTION::Z;
        else if(tfsfArr_.size() > 0 && (std::abs( tfsfArr_.back()->quadrant() ) == 2 || std::abs( tfsfArr_.back()->quadrant() ) == 4 ) )
            propDir = DIRECTION::X;
        else if(tfsfArr_.size() > 0 && (std::abs( tfsfArr_.back()->quadrant() ) == 1 || std::abs( tfsfArr_.back()->quadrant() ) == 3 ) )
            propDir = DIRECTION::Y;
        else if(srcArr_.size() > 0 && (srcArr_.back()->sz(0) >= srcArr_.back()->sz(1) ) && (srcArr_.back()->sz(2) >= srcArr_.back()->sz(1) ) )
            propDir = DIRECTION::Y;
        else if(srcArr_.size() > 0 && (srcArr_.back()->sz(0) <= srcArr_.back()->sz(1) ) && (srcArr_.back()->sz(0) <= srcArr_.back()->sz(2) ) )
            propDir = DIRECTION::X;
        else if(srcArr_.size() > 0)
            propDir = DIRECTION::Z;
        else
            propDir = DIRECTION::NONE;

        double theta = 0; double phi = 0; double psi = 0; double alpha = 0;
        if(tfsfArr_.size() > 0)
        {
            theta = std::atan( std::abs( std::tan( tfsfArr_.back()->theta() ) ) );
            phi   = std::atan( std::abs( std::tan( tfsfArr_.back()->phiPreFact() ) ) );
            psi   = std::atan( std::abs( std::tan( tfsfArr_.back()->psiPreFact() ) ) );
            alpha = tfsfArr_.back()->alpha();
        }
        for(int ff = 0; ff < IP.fluxLoc_.size(); ff ++)
            fluxArr_.push_back(std::make_shared<parallelFluxDTCReal>(gridComm_, IP.fluxName_[ff], IP.fluxWeight_[ff], E_, H_, IP.fluxLoc_[ff], IP.fluxSz_[ff], IP.fluxCrossSec_[ff], IP.fluxSave_[ff], IP.fluxLoad_[ff], IP.fluxTimeInt_[ff], IP.fluxFreqList_[ff], propDir, d_, dt_, theta, phi, psi, alpha, IP.fluxIncdFieldsFilename_[ff], IP.fluxSI_[ff], IP.I0_, IP.a_) );
    }
    // Construct all DTC based on types (all it changes is the list of fields it passes)
    for(int dd = 0; dd < IP.dtcType_.size(); dd++)
    {
        std::vector<std::pair<pgrid_ptr, std::array<int,3> > > fields;
        // Fill the fields vector with the appropriate field values
        if(IP.dtcType_[dd] == DTCTYPE::EX)
        {
            fields.push_back( std::make_pair(E_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EY)
        {
            fields.push_back( std::make_pair(E_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EZ)
        {
            fields.push_back( std::make_pair(E_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HX)
        {
            fields.push_back( std::make_pair(H_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HY)
        {
            fields.push_back( std::make_pair(H_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HZ)
        {
            fields.push_back( std::make_pair(H_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::DX)
        {
            fields.push_back( std::make_pair(D_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::DY)
        {
            fields.push_back( std::make_pair(D_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::DZ)
        {
            fields.push_back( std::make_pair(D_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::BX)
        {
            fields.push_back( std::make_pair(B_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::BY)
        {
            fields.push_back( std::make_pair(B_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::BZ)
        {
            fields.push_back( std::make_pair(B_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::PX)
        {
            for(auto& ff : lorP_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::PY)
        {
            for(auto& ff : lorP_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::PZ)
        {
            for(auto& ff : lorP_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::MX)
        {
            for(auto& ff : lorM_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::MY)
        {
            for(auto& ff : lorM_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::MZ)
        {
            for(auto& ff : lorM_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIPX)
        {
            for(auto& ff : lorChiHP_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIPY)
        {
            for(auto& ff : lorChiHP_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIPZ)
        {
            for(auto& ff : lorChiHP_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIMX)
        {
            for(auto& ff : lorChiEM_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIMY)
        {
            for(auto& ff : lorChiEM_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIMZ)
        {
            for(auto& ff : lorChiEM_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EPOW && H_[2] && E_[2])
        {
            fields.push_back( std::make_pair(E_[0], std::array<int, 3> ( {-1,  0,  0} ) ) );
            fields.push_back( std::make_pair(E_[1], std::array<int, 3> ( { 0, -1,  0} ) ) );
            fields.push_back( std::make_pair(E_[2], std::array<int, 3> ( { 0,  0, -1} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HPOW && H_[2] && E_[2])
        {
            fields.push_back( std::make_pair(H_[0], std::array<int, 3> ( { 1,  0,  0} ) ) );
            fields.push_back( std::make_pair(H_[1], std::array<int, 3> ( { 0,  1,  0} ) ) );
            fields.push_back( std::make_pair(H_[2], std::array<int, 3> ( { 0,  0,  1} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EPOW && H_[2])
        {
            fields.push_back( std::make_pair(E_[0], std::array<int, 3> ( {-1,  0,  0} ) ) );
            fields.push_back( std::make_pair(E_[1], std::array<int, 3> ( { 0, -1,  0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HPOW && H_[2])
        {
            fields.push_back( std::make_pair(H_[2], std::array<int, 3> ( { 0,  0,  0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EPOW && E_[2])
        {
            fields.push_back( std::make_pair(E_[2], std::array<int, 3> ( { 0,  0, 0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HPOW && E_[2])
        {
            fields.push_back( std::make_pair(H_[0], std::array<int, 3> ( { 1,  0,  0} ) ) );
            fields.push_back( std::make_pair(H_[1], std::array<int, 3> ( { 0,  1,  0} ) ) );
        }
        else
            throw std::logic_error("DTC TYPE IS NOT DEFINED");
        coustructDTC(IP.dtcClass_[dd], fields, IP.dtcSI_[dd], IP.dtcLoc_[dd], IP.dtcSz_[dd], IP.dtcName_[dd], IP.dtcOutBMPFxnType_[dd], IP.dtcOutBMPOutType_[dd], IP.dtcType_[dd], IP.dtcTStart_[dd], IP.dtcTEnd_[dd], IP.dtcOutputAvg_[dd], IP.dtcFreqList_[dd], IP.dtcTimeInt_[dd], IP.a_, IP.I0_, IP.tMax_, IP.dtcOutputMaps_[dd]);
    }
    // Initialze all detectors to time 0
    for(auto& dtc : dtcArr_)
        dtc->output(tcur_);
    for(auto& dtc : dtcFreqArr_)
        dtc->output(tcur_);
    for(auto& flux : fluxArr_)
        flux->fieldIn(tcur_);
    // Incd done twice to account for offset
    H_incd_[0].push_back(0.0);
    H_incd_[1].push_back(0.0);
    H_incd_[2].push_back(0.0);
    E_incd_[0].push_back(0.0);
    E_incd_[1].push_back(0.0);
    E_incd_[2].push_back(0.0);

    H_incd_[0].push_back(0.0);
    H_incd_[1].push_back(0.0);
    H_incd_[2].push_back(0.0);
    E_incd_[0].push_back(0.0);
    E_incd_[1].push_back(0.0);
    E_incd_[2].push_back(0.0);
}

parallelFDTDFieldCplx::parallelFDTDFieldCplx(parallelProgramInputs &IP, std::shared_ptr<mpiInterface> gridComm) :
    parallelFDTDFieldBase<cplx>(IP, gridComm)
{
    if(E_[2] && H_[2])
    {
        upLorOrDipP_ = &FDTDCompUpdateFxnCplx::UpdateLorPolOrDip;
        upLorOrDipM_ = &FDTDCompUpdateFxnCplx::UpdateLorPolOrDip;
    }
    else if(E_[2])
    {
        upLorOrDipP_ = &FDTDCompUpdateFxnCplx::UpdateLorPolOrDipZ;
        upLorOrDipM_ = &FDTDCompUpdateFxnCplx::UpdateLorPolOrDipXY;
    }
    else
    {
        upLorOrDipP_ = &FDTDCompUpdateFxnCplx::UpdateLorPolOrDipXY;
        upLorOrDipM_ = &FDTDCompUpdateFxnCplx::UpdateLorPolOrDipZ;
    }

    upChiLorOrDipP_ = &FDTDCompUpdateFxnCplx::UpdateChiralOrDip;
    upChiLorOrDipM_ = &FDTDCompUpdateFxnCplx::UpdateChiralOrDip;
    if(IP.periodic_ && gridComm_->size() > 1)
    {
        if(gridComm_->rank() !=0 || gridComm_->rank() != gridComm_->size() - 1)
            applBCOrDip_ = &FDTDCompUpdateFxnCplx::applyBCProcMid;
        else
            applBCOrDip_ = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnCplx::applyBCProc0 : &FDTDCompUpdateFxnCplx::applyBCProcMax;
    }
    else if(IP.periodic_)
    {
        applBCOrDip_ = &FDTDCompUpdateFxnCplx::applyBC1Proc;
    }
    else if(gridComm_->size() > 1)
    {
        applBCOrDip_ = &FDTDCompUpdateFxnCplx::applyBCNonPer;
    }
    else
    {
        applBCOrDip_ = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
    }
    // If there is an Hz field set up all TE mode functions otherwise set them to do nothing
    std::array<std::vector<std::array<int,5>>,3> chiBLocs;
    std::array<std::vector<std::array<int,5>>,3> chiDLocs;
    std::vector<std::array<int,5>> chiOrDipLocs;
    if(H_[2])
    {
        // Initialize the PMLs
        if(magMatInPML_)
        {
            HPML_[2]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, B_[2], E_[0], E_[1], POLARIZATION::HZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[2], objArr_);
        }
        else
        {
            HPML_[2]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, H_[2], E_[0], E_[1], POLARIZATION::HZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[2], objArr_);
        }
        if(dielectricMatInPML_)
        {
            EPML_[0]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, D_[0], H_[1], H_[2], POLARIZATION::EX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[0], objArr_);
            EPML_[1]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, D_[1], H_[2], H_[0], POLARIZATION::EY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[1], objArr_);
        }
        else
        {
            EPML_[0]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, E_[0], H_[1], H_[2], POLARIZATION::EX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[0], objArr_);
            EPML_[1]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, E_[1], H_[2], H_[0], POLARIZATION::EY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[1], objArr_);
        }
        // Fill the update Lists
        gridComm_->barrier();
        initializeList(physH_[2],  muRel_[2], HPML_[2], false, std::array<int,3>( {{ 1,  0,  0 }} ), std::array<int,3>( {{1, 1, 0 }} ), d_[0], d_[1], upH_[2], upB_[2], upLorB_[2], upChiB_[2], upOrDipB_[2], upOrDipChiB_[2], chiBLocs[2]);
        initializeList(physE_[0], epsRel_[0], EPML_[0],  true, std::array<int,3>( {{ 0, -1,  0 }} ), std::array<int,3>( {{1, 0, 0 }} ), d_[1], d_[2], upE_[0], upD_[0], upLorD_[0], upChiD_[0], upOrDipD_[0], upOrDipChiD_[0], chiDLocs[0]);
        initializeList(physE_[1], epsRel_[1], EPML_[1],  true, std::array<int,3>( {{ 0,  0, -1 }} ), std::array<int,3>( {{0, 1, 0 }} ), d_[2], d_[0], upE_[1], upD_[1], upLorD_[1], upChiD_[1], upOrDipD_[1], upOrDipChiD_[1], chiDLocs[1]);
        if(dipP_[0].size() > 0)
        {
            upLists tempNonOrDip;
            initializeList(physPOrDip_, epsRelOrDip_, HPML_[2],  true, std::array<int,3>( {{-1, -1, -1 }} ), std::array<int,3>( {{ 0, 0, 0 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipP_, upChiOrDipP_, chiOrDipLocs, true);
        }
        if(dipM_[0].size() > 0)
        {
            upLists tempNonOrDip;
            initializeList(physMOrDip_,  muRelOrDip_, HPML_[2], false, std::array<int,3>( {{ 1,  1,  1 }} ), std::array<int,3>( {{ 1, 1, 1 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipM_, upChiOrDipM_, chiOrDipLocs, true);
        }

        if(!E_[2])
        {
            upEFxn_[0] = &FDTDCompUpdateFxnCplx::OneCompCurlK;
            upEFxn_[1] = &FDTDCompUpdateFxnCplx::OneCompCurlJ;
        }
        else
        {
            upEFxn_[0] = &FDTDCompUpdateFxnCplx::TwoCompCurl;
            upEFxn_[1] = &FDTDCompUpdateFxnCplx::TwoCompCurl;
        }

        upHFxn_[2] = &FDTDCompUpdateFxnCplx::TwoCompCurl;

        updateEPML_[0] = [](pml_ptr pml){pml->updateGrid();};
        updateEPML_[1] = [](pml_ptr pml){pml->updateGrid();};
        updateHPML_[2] = [](pml_ptr pml){pml->updateGrid();};

        upLorPFxn_[0] = &FDTDCompUpdateFxnCplx::UpdateLorPol;
        upLorPFxn_[1] = &FDTDCompUpdateFxnCplx::UpdateLorPol;
        upLorMFxn_[2] = &FDTDCompUpdateFxnCplx::UpdateLorPol;

        upChiE_[0] = &FDTDCompUpdateFxnCplx::UpdateChiral;
        upChiE_[1] = &FDTDCompUpdateFxnCplx::UpdateChiral;
        upChiH_[2] = &FDTDCompUpdateFxnCplx::UpdateChiral;

        D2EFxn_[0] = &FDTDCompUpdateFxnCplx::DtoU;
        D2EFxn_[1] = &FDTDCompUpdateFxnCplx::DtoU;
        B2HFxn_[2] = &FDTDCompUpdateFxnCplx::DtoU;

        chiD2EFxn_[0] = &FDTDCompUpdateFxnCplx::chiDtoU;
        chiD2EFxn_[1] = &FDTDCompUpdateFxnCplx::chiDtoU;
        chiB2HFxn_[2] = &FDTDCompUpdateFxnCplx::chiDtoU;

        orDipD2EFxn_[0] = &FDTDCompUpdateFxnCplx::orDipDtoU;
        orDipD2EFxn_[1] = &FDTDCompUpdateFxnCplx::orDipDtoU;
        if(!E_[2])
            orDipB2HFxn_[2] = &FDTDCompUpdateFxnCplx::orDipDtoUZ;
        else
            orDipB2HFxn_[2] = &FDTDCompUpdateFxnCplx::orDipDtoU;

        chiOrDipD2EFxn_[0] = &FDTDCompUpdateFxnCplx::chiOrDipDtoU;
        chiOrDipD2EFxn_[1] = &FDTDCompUpdateFxnCplx::chiOrDipDtoU;
        chiOrDipB2HFxn_[2] = &FDTDCompUpdateFxnCplx::chiOrDipDtoU;

        if(IP.periodic_ && gridComm_->size() > 1)
        {
            if(gridComm_->rank() != gridComm_->size()-1 || gridComm_->rank() != 0)
            {
                applBCE_[0] = &FDTDCompUpdateFxnCplx::applyBCProcMid;
                applBCE_[1] = &FDTDCompUpdateFxnCplx::applyBCProcMid;
                applBCH_[2] = &FDTDCompUpdateFxnCplx::applyBCProcMid;
            }
            else
            {
                applBCE_[0] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnCplx::applyBCProc0 : &FDTDCompUpdateFxnCplx::applyBCProcMax;
                applBCE_[1] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnCplx::applyBCProc0 : &FDTDCompUpdateFxnCplx::applyBCProcMax;
                applBCH_[2] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnCplx::applyBCProc0 : &FDTDCompUpdateFxnCplx::applyBCProcMax;
            }

            yEPBC_[0] = ln_vec_[1]+1;
            yEPBC_[1] = ln_vec_[1]+1;
            yHPBC_[2] = ln_vec_[1]+1;
            // Field definition buffer
            if(gridComm_->size()-1 == gridComm_->rank())
            {
                yHPBC_[2] -= 1;
                yEPBC_[1] -= 1;
            }
        }
        else if(IP.periodic_)
        {
            applBCE_[0] = &FDTDCompUpdateFxnCplx::applyBC1Proc;
            applBCE_[1] = &FDTDCompUpdateFxnCplx::applyBC1Proc;
            applBCH_[2] = &FDTDCompUpdateFxnCplx::applyBC1Proc;

            yHPBC_[2] = ln_vec_[1];
            yEPBC_[0] = ln_vec_[1]+1;
            yEPBC_[1] = ln_vec_[1];
        }
        else if(gridComm_->size() > 1)
        {
            applBCE_[0] = &FDTDCompUpdateFxnCplx::applyBCNonPer;
            applBCE_[1] = &FDTDCompUpdateFxnCplx::applyBCNonPer;
            applBCH_[2] = &FDTDCompUpdateFxnCplx::applyBCNonPer;
        }
        else
        {
            applBCE_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCE_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCH_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        }
    }
    else
    {
        upHFxn_[2] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};

        upEFxn_[0] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};;
        upEFxn_[1] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};;

        updateEPML_[0] = [](pml_ptr pml){return;};
        updateEPML_[1] = [](pml_ptr pml){return;};
        updateHPML_[2] = [](pml_ptr pml){return;};

        upLorPFxn_[0] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upLorPFxn_[1] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upLorMFxn_[2] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};

        upChiE_[0] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upChiE_[1] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upChiH_[2] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};

        D2EFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        D2EFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        B2HFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        chiD2EFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiD2EFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiB2HFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        orDipD2EFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        orDipD2EFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        orDipB2HFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};

        chiOrDipD2EFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        chiOrDipD2EFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        chiOrDipB2HFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};

        applBCE_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCE_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCH_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};

    }
    // If there is an Ez field set up all TM functions otherwise set them to do nothing
    if(E_[2])
    {
        //initialize the PMLs
        if(magMatInPML_)
        {
            HPML_[0]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, B_[0], E_[1], E_[2], POLARIZATION::HX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[0], objArr_);
            HPML_[1]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, B_[1], E_[2], E_[0], POLARIZATION::HY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[1], objArr_);
        }
        else
        {
            HPML_[0]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, H_[0], E_[1], E_[2], POLARIZATION::HX, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[0], objArr_);
            HPML_[1]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, H_[1], E_[2], E_[0], POLARIZATION::HY, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physH_[1], objArr_);
        }
        if(dielectricMatInPML_)
        {
            EPML_[2]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, D_[2], H_[0], H_[1], POLARIZATION::EZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[2], objArr_);
        }
        else
        {
            EPML_[2]   = std::make_shared<parallelCPMLCplx>(gridComm_, weights_, E_[2], H_[0], H_[1], POLARIZATION::EZ, pmlThickness_, IP.pmlM_, IP.pmlMa_, IP.pmlSigOptRat_, IP.pmlKappaMax_, IP.pmlAMax_, d_, dt_, ( dielectricMatInPML_ || magMatInPML_ ), physE_[2], objArr_);
        }
        // Fill update lists
        initializeList(physH_[0],  muRel_[0], HPML_[0], false, std::array<int,3>( {{ 0,  1,  0 }} ), std::array<int,3>( {{ 0, 1, 1 }} ), d_[1], d_[2], upH_[0], upB_[0], upLorB_[0], upChiB_[0], upOrDipB_[0], upOrDipChiB_[0], chiBLocs[0]);
        initializeList(physH_[1],  muRel_[1], HPML_[1], false, std::array<int,3>( {{ 0,  0,  1 }} ), std::array<int,3>( {{ 1, 0, 1 }} ), d_[2], d_[0], upH_[1], upB_[1], upLorB_[1], upChiB_[1], upOrDipB_[1], upOrDipChiB_[1], chiBLocs[1]);
        initializeList(physE_[2], epsRel_[2], EPML_[2],  true, std::array<int,3>( {{-1,  0,  0 }} ), std::array<int,3>( {{ 0, 0, 1 }} ), d_[0], d_[1], upE_[2], upD_[2], upLorD_[2], upChiD_[2], upOrDipD_[2], upOrDipChiD_[2], chiDLocs[2]);

        if(dipP_[2].size() > 0 && !H_[2])
        {
            upLists tempNonOrDip;
            initializeList(physPOrDip_, epsRelOrDip_, EPML_[2],  true, std::array<int,3>( {{-1, -1, -1 }} ), std::array<int,3>( {{ 0, 0, 0 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipP_, upChiOrDipM_, chiOrDipLocs, true);
        }
        if(dipM_[0].size() > 0 && !H_[2])
        {
            upLists tempNonOrDip;
            initializeList(physMOrDip_,  muRelOrDip_, EPML_[2], false, std::array<int,3>( {{ 1,  1,  1 }} ), std::array<int,3>( {{ 1, 1, 1 }} ), d_[0], d_[0], tempNonOrDip, tempNonOrDip, tempNonOrDip, tempNonOrDip, upOrDipM_, upChiOrDipP_, chiOrDipLocs, true);
        }

        if(!H_[2])
        {
            upHFxn_[0] = &FDTDCompUpdateFxnCplx::OneCompCurlK;
            upHFxn_[1] = &FDTDCompUpdateFxnCplx::OneCompCurlJ;
        }
        else
        {
            upHFxn_[0] = &FDTDCompUpdateFxnCplx::TwoCompCurl;
            upHFxn_[1] = &FDTDCompUpdateFxnCplx::TwoCompCurl;
        }

        upEFxn_[2] = &FDTDCompUpdateFxnCplx::TwoCompCurl;

        upLorMFxn_[0] = &FDTDCompUpdateFxnCplx::UpdateLorPol;
        upLorMFxn_[1] = &FDTDCompUpdateFxnCplx::UpdateLorPol;
        upLorPFxn_[2] = &FDTDCompUpdateFxnCplx::UpdateLorPol;

        upChiH_[0] = &FDTDCompUpdateFxnCplx::UpdateChiral;
        upChiH_[1] = &FDTDCompUpdateFxnCplx::UpdateChiral;
        upChiE_[2] = &FDTDCompUpdateFxnCplx::UpdateChiral;

        B2HFxn_[0] = &FDTDCompUpdateFxnCplx::DtoU;
        B2HFxn_[1] = &FDTDCompUpdateFxnCplx::DtoU;
        D2EFxn_[2] = &FDTDCompUpdateFxnCplx::DtoU;

        chiB2HFxn_[0] = &FDTDCompUpdateFxnCplx::chiDtoU;
        chiB2HFxn_[1] = &FDTDCompUpdateFxnCplx::chiDtoU;
        chiD2EFxn_[2] = &FDTDCompUpdateFxnCplx::chiDtoU;

        orDipB2HFxn_[0] = &FDTDCompUpdateFxnCplx::orDipDtoU;
        orDipB2HFxn_[1] = &FDTDCompUpdateFxnCplx::orDipDtoU;
        if(!H_[2])
            orDipD2EFxn_[2] = &FDTDCompUpdateFxnCplx::orDipDtoUZ;
        else
            orDipD2EFxn_[2] = &FDTDCompUpdateFxnCplx::orDipDtoU;

        chiOrDipB2HFxn_[0] = &FDTDCompUpdateFxnCplx::chiOrDipDtoU;
        chiOrDipB2HFxn_[1] = &FDTDCompUpdateFxnCplx::chiOrDipDtoU;
        chiOrDipD2EFxn_[2] = &FDTDCompUpdateFxnCplx::chiOrDipDtoU;

        updateHPML_[0] = [](pml_ptr pml){pml->updateGrid();};
        updateHPML_[1] = [](pml_ptr pml){pml->updateGrid();};
        updateEPML_[2] = [](pml_ptr pml){pml->updateGrid();};

        if(IP.periodic_ && gridComm_->size() > 1)
        {
            if(gridComm_->rank() != gridComm_->size()-1 || gridComm_->rank() != 0)
            {
                applBCH_[0] = &FDTDCompUpdateFxnCplx::applyBCProcMid;
                applBCH_[1] = &FDTDCompUpdateFxnCplx::applyBCProcMid;
                applBCE_[2] = &FDTDCompUpdateFxnCplx::applyBCProcMid;
            }
            else
            {
                applBCH_[0] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnCplx::applyBCProc0 : &FDTDCompUpdateFxnCplx::applyBCProcMax;
                applBCH_[1] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnCplx::applyBCProc0 : &FDTDCompUpdateFxnCplx::applyBCProcMax;
                applBCE_[2] = gridComm_->rank() == 0 ? &FDTDCompUpdateFxnCplx::applyBCProc0 : &FDTDCompUpdateFxnCplx::applyBCProcMax;
            }

            yHPBC_[0] = ln_vec_[1]+1;
            yHPBC_[1] = ln_vec_[1]+1;
            yEPBC_[2] = ln_vec_[1]+1;
            //Field definition buffer
            if(gridComm_->size()-1 == gridComm_->rank())
                yHPBC_[0] = ln_vec_[1];
        }
        else if(IP.periodic_)
        {
            applBCH_[0] = &FDTDCompUpdateFxnCplx::applyBC1Proc;
            applBCH_[1] = &FDTDCompUpdateFxnCplx::applyBC1Proc;
            applBCE_[2] = &FDTDCompUpdateFxnCplx::applyBC1Proc;
            yHPBC_[0] = ln_vec_[1];
            yHPBC_[1] = ln_vec_[1]+1;
            yEPBC_[2] = ln_vec_[1]+1;
        }
        else if(gridComm_->size() > 1)
        {
            applBCH_[0] = &FDTDCompUpdateFxnCplx::applyBCNonPer;
            applBCH_[1] = &FDTDCompUpdateFxnCplx::applyBCNonPer;
            applBCE_[2] = &FDTDCompUpdateFxnCplx::applyBCNonPer;
        }
        else
        {
            applBCH_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCH_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
            applBCE_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        }
    }
    else
    {
        upHFxn_[0] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};;
        upHFxn_[1] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};

        upEFxn_[2] = [](const std::array<int,6>&, const std::array<double,4>&, pgrid_ptr, pgrid_ptr, pgrid_ptr){return;};

        upLorMFxn_[0] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upLorMFxn_[1] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upLorPFxn_[2] = []( const std::array<int,6>&, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};

        upChiH_[0] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upChiH_[1] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};
        upChiE_[2] = [](const std::array<int,6>&, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&, std::vector<pgrid_ptr>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, const std::vector<double>&, cplx*){return;};

        B2HFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        B2HFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        D2EFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        chiB2HFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiB2HFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};
        chiD2EFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, std::vector<pgrid_ptr>&){return;};

        orDipB2HFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        orDipB2HFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        orDipD2EFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};

        chiOrDipB2HFxn_[0] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        chiOrDipB2HFxn_[1] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};
        chiOrDipD2EFxn_[2] = [](const std::array<int,6>&, cplx, pgrid_ptr, pgrid_ptr, int, std::vector<pgrid_ptr>&, cplx*, cplx*){return;};

        updateHPML_[0] = [](pml_ptr pml){return;};
        updateHPML_[1] = [](pml_ptr pml){return;};
        updateEPML_[2] = [](pml_ptr pml){return;};

        applBCH_[0] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCH_[1] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};
        applBCE_[2] = [](pgrid_ptr, std::array<double,3>&, int, int, int, int, int, int, int, double&, double&, double&){return;};

    }

    // Find region to copy fields to prev fields
    for(int oo = 0; oo < objArr_.size(); ++oo)
    {
        if(objArr_[oo]->chiGamma().size() > 0)
        {
            int xChiMin = n_vec_[0], yChiMin = n_vec_[1], zChiMin = n_vec_[2];
            int xChiMax = 0, yChiMax = 0, zChiMax = 0;
            // Find Min and Max across all field updates for that object
            for(int ii = 0; ii < 3; ++ii)
            {
                findMinMaxChiLocsObj(chiDLocs[ii], oo, xChiMin, yChiMin, zChiMin, xChiMax, yChiMax, zChiMax);
                findMinMaxChiLocsObj(chiBLocs[ii], oo, xChiMin, yChiMin, zChiMin, xChiMax, yChiMax, zChiMax);
            }
            // Give a +/- buffer to copy over
            int sz = xChiMax - xChiMin + 3;
            for(int yy = yChiMin-1; yy <= yChiMax+1; ++yy)
                for(int zz = zChiMin-1; zz <= zChiMax+1; ++zz)
                    copy2PrevFields_.push_back(std::array<int,4>( {{ sz, xChiMin-1, yy, zz }} ) );
        }
    }
    // Construct QE objects
    for(int qq = 0; qq < IP.qeLoc_.size(); qq++)
    {
        // std::shared_ptr<parallelQEBase<double>> temp;
        // Construct BasisSet and Hamiltonian
        BasisSet basis(IP.qeBasis_[qq]);
        std::vector<Hamiltonian> hams;
        std::vector<double> current;
        double weight = 1.0;
        std::vector<std::pair<std::vector<double>, double>> eWeightPairs;
        std::vector<double> eWeights;
        eWeights.reserve(eWeightPairs.size());
        GenerateAllELevCombos(IP.qeELevs_, eWeightPairs, 0, current, weight);
        for(auto& eWeightpair : eWeightPairs)
        {
            hams.push_back( Hamiltonian( IP.qeBasis_[qq].size(), std::get<0>(eWeightpair),IP.qeCouplings_[qq], basis) );
            eWeights.push_back( std::get<1>(eWeightpair) );
        }
        std::vector<std::shared_ptr<QEPopDtc>> dtcPopArr;
        for( int dd=0; dd < IP.qePopDtcLevs_[qq].size(); dd++)
            dtcPopArr.push_back( std::make_shared<QEPopDtc>(gridComm_, IP.qePopDtcLevs_[qq][dd], IP.qeBasis_[qq].size(), IP.qeLoc_[qq].size(), eWeights, IP.qeDtcPopOutFile_[qq][dd], IP.qeDtcPopTimeInt_[qq], static_cast<int>( ceil( IP.tMax() / (dt_* IP.qeDtcPopTimeInt_[qq]) ) ), dt_) );

        // If the input file sets up the relaxation operators use that otherwise set up the relaxation operator storage operator for the qe constrcutor and then construct it
        if( IP.qeGam_[qq].size() > 0 )
        {
            qeArr_.push_back(std::make_shared<parallelQECplx>(gridComm_, hams, eWeightPairs, dtcPopArr, basis, IP.qeGam_[qq], IP.qeLoc_[qq], E_, epsRelOrDip_, IP.qeAccumP_[qq], IP.qeOutPolFname_[qq], ceil(IP.tMax_ / dt_ ) + 1, IP.a_, IP.I0_, dt_, IP.qeDen_[qq]*pow(IP.a_,3) ) ) ;
        }
        else
        {
            if(IP.qeRelaxTransitonStates_[qq].size() != IP.qeRelaxRates_[qq].size() )
                throw std::logic_error("All relaxation transition states must have a list of rates and state definitions");
            std::vector<relaxParams> relax;
            for(int rr = 0; rr < IP.qeRelaxTransitonStates_[qq].size(); rr++)
            {
                relaxParams tempRelax;
                tempRelax.n0_ = IP.qeRelaxTransitonStates_[qq][rr][0];
                tempRelax.nf_ = IP.qeRelaxTransitonStates_[qq][rr][1];
                tempRelax.rate_ = IP.qeRelaxRates_[qq][rr];
                tempRelax.dephasingRate_ = IP.qeRelaxDephasingRate_[qq][rr];
                relax.push_back(tempRelax);
            }
            qeArr_.push_back( std::make_shared<parallelQECplx>(gridComm_, hams, eWeightPairs, dtcPopArr, basis, relax, IP.qeLoc_[qq], E_, epsRelOrDip_, IP.qeAccumP_[qq], IP.qeOutPolFname_[qq], ceil(IP.tMax_ / dt_ ) + 1, IP.a_, IP.I0_, dt_, IP.qeDen_[qq]*pow(IP.a_,3) ) );
        }
    }

    // Construct all soft sources
    for(int ss = 0; ss < IP.srcPol_.size(); ss++)
    {
        // Make the pulse (including all pulses to be used)
        std::vector<std::shared_ptr<Pulse>> pul;
        for(int pp = 0; pp < IP.srcPulShape_[ss].size(); pp ++)
        {
            std::function<const cplx(double, const std::vector<cplx>&)> pulseFxn;
            if(IP.srcPulShape_[ss][pp] == PLSSHAPE::CONTINUOUS)
                pulseFxn = contPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::BH)
                pulseFxn = bhPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RECT)
                pulseFxn = rectPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::GAUSSIAN)
                pulseFxn = gaussPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RICKER)
                pulseFxn = rickerPulse;
            else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RAMP_CONT)
                pulseFxn = rampContPulse;
            IP.srcFxn_[ss][pp].push_back( IP.srcEmax_[ss][pp] );
            pul.push_back(std::make_shared<Pulse>(pulseFxn, IP.srcFxn_[ss][pp], dt_));
        }
        //  Make the source act on any of the fields
        if(IP.srcPol_[ss] == POLARIZATION::EX)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul, E_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, E_[0], POLARIZATION::EX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, E_[1], POLARIZATION::EY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::EY)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul, E_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, E_[0], POLARIZATION::EX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, E_[1], POLARIZATION::EY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::EZ)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul, E_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, E_[2], POLARIZATION::EZ, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::HX)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul, H_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, H_[1], POLARIZATION::HY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, H_[0], POLARIZATION::HX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::HY)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul, H_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, H_[0], POLARIZATION::HX, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, H_[1], POLARIZATION::HY, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else if(IP.srcPol_[ss] == POLARIZATION::HZ)
        {
            if(int( round(IP.srcPhi_[ss]) ) % 90 == 0)
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul, H_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            else
            {
                srcArr_.push_back( std::make_shared<parallelSourceObliqueCplx >(gridComm_, pul, H_[2], POLARIZATION::HZ, dt_, IP.srcLoc_[ss], IP.srcSz_[ss], IP.srcPhi_[ss], IP.srcTheta_[ss] ) );
            }
        }
        else
        {
            double axRat = IP.srcEllipticalKratio_[ss];
            double psi = IP.srcPsi_[ss];
            double psiPrefactCalc = psi;
            double alphaOff = 0.0;
            double prefactor_k_ = 1.0;
            double prefactor_j_ = 1.0;
            double c = pow(axRat, 2.0);

            // phi/psi control the light polarization angle
            psiPrefactCalc = 0.5 * asin( sqrt( ( pow(cos(2.0*psi),2.0)*4.0*c + pow( (1.0+c)*sin(2.0*psi), 2.0) ) / pow(1.0+c, 2.0) ) );
            alphaOff = acos( ( (c - 1.0)*sin(2.0*psi) ) / sqrt( pow(cos(2.0*psi),2.0)*4.0*c + pow( (1.0+c)*sin(2.0*psi), 2.0) ) );
            if(std::abs( std::tan(psi) ) > 1)
                psiPrefactCalc = M_PI/2.0 - psiPrefactCalc;
            if(IP.srcPol_[ss] == POLARIZATION::R)
                alphaOff *= -1.0;

            if( isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 0 )
            {
                prefactor_j_ *= -1.0 * cos(psiPrefactCalc);
                prefactor_k_ *= -1.0 * sin(psiPrefactCalc);
            }
            else if( isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 1 )
            {
                prefactor_j_ *=  -1.0 * cos(psiPrefactCalc);
                prefactor_k_ *=         sin(psiPrefactCalc);
            }
            else if( isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 2 )
            {
                prefactor_j_ *= -1.0*sin(psiPrefactCalc);
                prefactor_k_ *=      cos(psiPrefactCalc);
            }
            std::vector<std::shared_ptr<Pulse>> pul_j;
            std::vector<std::shared_ptr<Pulse>> pul_k;
            for(int pp = 0; pp < IP.srcPulShape_[ss].size(); pp ++)
            {
                std::function<const cplx(double, const std::vector<cplx>&)> pulseFxn;
                if(IP.srcPulShape_[ss][pp] == PLSSHAPE::CONTINUOUS)
                    pulseFxn = contPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::BH)
                    pulseFxn = bhPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RECT)
                    pulseFxn = rectPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::GAUSSIAN)
                    pulseFxn = gaussPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RICKER)
                    pulseFxn = rickerPulse;
                else if(IP.srcPulShape_[ss][pp] == PLSSHAPE::RAMP_CONT)
                    pulseFxn = rampContPulse;

                IP.srcFxn_[ss][pp].push_back( IP.srcEmax_[ss][pp]*prefactor_j_ );
                pul_j.push_back(std::make_shared<Pulse>(pulseFxn, IP.srcFxn_[ss][pp], dt_));

                IP.srcFxn_[ss][pp][IP.srcFxn_[ss][pp].size()-1] *= prefactor_k_ / prefactor_j_ * cplx(0.0, alphaOff);
                pul_k.push_back(std::make_shared<Pulse>(pulseFxn, IP.srcFxn_[ss][pp], dt_));
            }
            if(isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 0 )
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul_j, E_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul_k, E_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            if(isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 1 )
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul_j, E_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul_k, E_[2], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
            if(isamin_(IP.srcSz_[ss].size(), IP.srcSz_[ss].data(), 1)-1 == 2 )
            {
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul_j, E_[0], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
                srcArr_.push_back( std::make_shared<parallelSourceNormalCplx>(gridComm_, pul_k, E_[1], dt_, IP.srcLoc_[ss], IP.srcSz_[ss] ) );
            }
        }
    }

    // Construct all TFSF surfaces
    for(int tt = 0; tt < IP.tfsfSize_.size(); tt++)
    {
        if(IP.tfsfSize_[tt][0] != 0.0 || IP.tfsfSize_[tt][1] != 0.0)
        {
        // Make the pulse (including all pulses to be used)
            std::vector<std::shared_ptr<Pulse>> pul;
            for(int pp = 0; pp < IP.tfsfPulShape_[tt].size(); pp ++)
            {
                std::function<const cplx(double, const std::vector<cplx>&)> pulseFxn;
                if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::CONTINUOUS)
                    pulseFxn = contPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::BH)
                    pulseFxn = bhPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::RECT)
                    pulseFxn = rectPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::GAUSSIAN)
                    pulseFxn = gaussPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::RICKER)
                    pulseFxn = rickerPulse;
                else if(IP.tfsfPulShape_[tt][pp] == PLSSHAPE::RAMP_CONT)
                    pulseFxn = rampContPulse;
                IP.tfsfPulFxn_[tt][pp].push_back(IP.tfsfEmax_[tt][pp]);
                pul.push_back(std::make_shared<Pulse>(pulseFxn, IP.tfsfPulFxn_[tt][pp], dt_));
            }
            // TFSF will determine the correct polarization
            tfsfArr_.push_back(std::make_shared<parallelTFSFCplx>(gridComm_, IP.tfsfLoc_[tt], IP.tfsfSize_[tt], IP.tfsfTheta_[tt], IP.tfsfPhi_[tt], IP.tfsfPsi_[tt], IP.tfsfCircPol_[tt], IP.tfsfEllipticalKratio_[tt], d_, IP.tfsfM_[tt], dt_, pul, E_, H_, D_, B_, physE_, physH_, objArr_, IP.tfsfPMLThick_[tt], IP.tfsfPMLM_[tt], IP.tfsfPMLMa_[tt], IP.tfsfPMLAMax_[tt])  );
        }
    }
    // Polarization matters here since the z field always forms the continous box for the spatial offset (TE uses H, TM uses E)
    if(IP.fluxName_.size() > 0)
    {
        DIRECTION propDir;
        if(tfsfArr_.size() > 0 && (tfsfArr_.back()->theta() == 0.0 || tfsfArr_.back()->theta() == M_PI ) )
            propDir = DIRECTION::Z;
        else if(tfsfArr_.size() > 0 && (std::abs( tfsfArr_.back()->quadrant() ) == 2 || std::abs( tfsfArr_.back()->quadrant() ) == 4 ) )
            propDir = DIRECTION::X;
        else if(tfsfArr_.size() > 0 && (std::abs( tfsfArr_.back()->quadrant() ) == 1 || std::abs( tfsfArr_.back()->quadrant() ) == 3 ) )
            propDir = DIRECTION::Y;
        else if(srcArr_.size() > 0 && (srcArr_.back()->sz(0) >= srcArr_.back()->sz(1) ) && (srcArr_.back()->sz(2) >= srcArr_.back()->sz(1) ) )
            propDir = DIRECTION::Y;
        else if(srcArr_.size() > 0 && (srcArr_.back()->sz(0) <= srcArr_.back()->sz(1) ) && (srcArr_.back()->sz(0) <= srcArr_.back()->sz(2) ) )
            propDir = DIRECTION::X;
        else if(srcArr_.size() > 0)
            propDir = DIRECTION::Z;
        else
            propDir = DIRECTION::NONE;

        double theta = 0; double phi = 0; double psi = 0; double alpha = 0;
        if(tfsfArr_.size() > 0)
        {
            theta = std::atan( std::abs( std::tan( tfsfArr_.back()->theta() ) ) );
            phi   = std::atan( std::abs( std::tan( tfsfArr_.back()->phiPreFact() ) ) );
            psi   = std::atan( std::abs( std::tan( tfsfArr_.back()->psiPreFact() ) ) );
            alpha = tfsfArr_.back()->alpha();
        }
        for(int ff = 0; ff < IP.fluxLoc_.size(); ff ++)
            fluxArr_.push_back(std::make_shared<parallelFluxDTCCplx>(gridComm_, IP.fluxName_[ff], IP.fluxWeight_[ff], E_, H_, IP.fluxLoc_[ff], IP.fluxSz_[ff], IP.fluxCrossSec_[ff], IP.fluxSave_[ff], IP.fluxLoad_[ff], IP.fluxTimeInt_[ff], IP.fluxFreqList_[ff], propDir, d_, dt_, theta, phi, psi, alpha, IP.fluxIncdFieldsFilename_[ff], IP.fluxSI_[ff], IP.I0_, IP.a_) );
    }
    // Construct all DTC based on types (all it changes is the list of fields it passes)
    for(int dd = 0; dd < IP.dtcType_.size(); dd++)
    {
        std::vector<std::pair<pgrid_ptr, std::array<int,3> > > fields;
        // Fill the fields vector with the appropriate field values
        if(IP.dtcType_[dd] == DTCTYPE::EX)
        {
            fields.push_back( std::make_pair(E_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EY)
        {
            fields.push_back( std::make_pair(E_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EZ)
        {
            fields.push_back( std::make_pair(E_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HX)
        {
            fields.push_back( std::make_pair(H_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HY)
        {
            fields.push_back( std::make_pair(H_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HZ)
        {
            fields.push_back( std::make_pair(H_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::DX)
        {
            fields.push_back( std::make_pair(D_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::DY)
        {
            fields.push_back( std::make_pair(D_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::DZ)
        {
            fields.push_back( std::make_pair(D_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::BX)
        {
            fields.push_back( std::make_pair(B_[0], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::BY)
        {
            fields.push_back( std::make_pair(B_[1], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::BZ)
        {
            fields.push_back( std::make_pair(B_[2], std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::PX)
        {
            for(auto& ff : lorP_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::PY)
        {
            for(auto& ff : lorP_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::PZ)
        {
            for(auto& ff : lorP_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::MX)
        {
            for(auto& ff : lorM_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::MY)
        {
            for(auto& ff : lorM_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::MZ)
        {
            for(auto& ff : lorM_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIPX)
        {
            for(auto& ff : lorChiHP_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIPY)
        {
            for(auto& ff : lorChiHP_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIPZ)
        {
            for(auto& ff : lorChiHP_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIMX)
        {
            for(auto& ff : lorChiEM_[0])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIMY)
        {
            for(auto& ff : lorChiEM_[1])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::CHIMZ)
        {
            for(auto& ff : lorChiEM_[2])
                fields.push_back( std::make_pair(ff, std::array<int,3>( {0,0,0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EPOW && H_[2] && E_[2])
        {
            fields.push_back( std::make_pair(E_[0], std::array<int, 3> ( {-1,  0,  0} ) ) );
            fields.push_back( std::make_pair(E_[1], std::array<int, 3> ( { 0, -1,  0} ) ) );
            fields.push_back( std::make_pair(E_[2], std::array<int, 3> ( { 0,  0, -1} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HPOW && H_[2] && E_[2])
        {
            fields.push_back( std::make_pair(H_[0], std::array<int, 3> ( { 1,  0,  0} ) ) );
            fields.push_back( std::make_pair(H_[1], std::array<int, 3> ( { 0,  1,  0} ) ) );
            fields.push_back( std::make_pair(H_[2], std::array<int, 3> ( { 0,  0,  1} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EPOW && H_[2])
        {
            fields.push_back( std::make_pair(E_[0], std::array<int, 3> ( {-1,  0,  0} ) ) );
            fields.push_back( std::make_pair(E_[1], std::array<int, 3> ( { 0, -1,  0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HPOW && H_[2])
        {
            fields.push_back( std::make_pair(H_[2], std::array<int, 3> ( { 0,  0,  0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::EPOW && E_[2])
        {
            fields.push_back( std::make_pair(E_[2], std::array<int, 3> ( { 0,  0, 0} ) ) );
        }
        else if(IP.dtcType_[dd] == DTCTYPE::HPOW && E_[2])
        {
            fields.push_back( std::make_pair(H_[0], std::array<int, 3> ( { 1,  0,  0} ) ) );
            fields.push_back( std::make_pair(H_[1], std::array<int, 3> ( { 0,  1,  0} ) ) );
        }
        else
            throw std::logic_error("DTC TYPE IS NOT DEFINED");
        coustructDTC(IP.dtcClass_[dd], fields, IP.dtcSI_[dd], IP.dtcLoc_[dd], IP.dtcSz_[dd], IP.dtcName_[dd], IP.dtcOutBMPFxnType_[dd], IP.dtcOutBMPOutType_[dd], IP.dtcType_[dd], IP.dtcTStart_[dd], IP.dtcTEnd_[dd], IP.dtcOutputAvg_[dd], IP.dtcFreqList_[dd], IP.dtcTimeInt_[dd], IP.a_, IP.I0_, IP.tMax_, IP.dtcOutputMaps_[dd]);
    }
    // Initialze all detectors to time 0
    for(auto& dtc : dtcArr_)
        dtc->output(tcur_);
    for(auto& dtc : dtcFreqArr_)
        dtc->output(tcur_);
    for(auto& flux : fluxArr_)
        flux->fieldIn(tcur_);

    //Done twice for offset
    H_incd_[0].push_back(0.0);
    H_incd_[1].push_back(0.0);
    H_incd_[2].push_back(0.0);
    E_incd_[0].push_back(0.0);
    E_incd_[1].push_back(0.0);
    E_incd_[2].push_back(0.0);

    H_incd_[0].push_back(0.0);
    H_incd_[1].push_back(0.0);
    H_incd_[2].push_back(0.0);
    E_incd_[0].push_back(0.0);
    E_incd_[1].push_back(0.0);
    E_incd_[2].push_back(0.0);
}

void GenerateAllELevCombos(std::vector<EnergyLevelDiscriptor> AllELevs, std::vector<std::pair<std::vector<double>, double>>& eLevs, int depth, std::vector<double> current, double weight)
{
    if(depth == AllELevs.size())
    {
        eLevs.push_back( std::make_pair(current, weight) );
        return;
    }

    for(int ii = 0; ii < AllELevs[depth].energyStates_.size(); ++ii)
    {
        for(int ee = 0; ee < AllELevs[depth].levDescribed_; ++ee)
            current.push_back(AllELevs[depth].energyStates_[ii]);

        GenerateAllELevCombos(AllELevs, eLevs, depth + 1, current, weight*AllELevs[depth].weights_[ii]);
        for(int ee = 0; ee < AllELevs[depth].levDescribed_; ++ee)
            current.pop_back();
    }
}

void parallelFDTDFieldReal::coustructDTC(DTCCLASS c, std::vector< std::pair <pgrid_ptr, std::array<int,3>> > grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double t_start, double t_end, bool outputAvg, std::vector<double> freqList, double timeInterval, double a, double I0, double t_max, bool outputMaps)
{
    if(c == DTCCLASS::BIN)
        dtcArr_.push_back( std::make_shared<parallelDetectorBINReal>( grid, SI, loc, sz, out_name, type, timeInterval, a, I0, dt_) );
    else if(c == DTCCLASS::BMP)
        dtcArr_.push_back( std::make_shared<parallelDetectorBMPReal>( grid, SI, loc, sz, outputAvg, t_start, t_end, out_name, fxn, txtType, type, timeInterval, a, I0, dt_) );
    else if(c == DTCCLASS::TXT)
        dtcArr_.push_back( std::make_shared<parallelDetectorTXTReal>(  grid, SI, loc, sz, out_name, type, timeInterval, a, I0, dt_ ) );
    else if(c == DTCCLASS::COUT)
        dtcArr_.push_back( std::make_shared<parallelDetectorCOUTReal>(grid, SI, loc, sz, out_name, type, timeInterval, a, I0, dt_) );
    else if(c == DTCCLASS::FREQ)
        dtcFreqArr_.push_back(std::make_shared<parallelDetectorFREQReal>(out_name, grid, loc, sz, type, static_cast<int>(std::floor(timeInterval/dt_+0.5) ), outputMaps, freqList, d_, dt_, SI, I0, a) );
    else
        throw std::logic_error("The detector class is undefined.");
}

void parallelFDTDFieldCplx::coustructDTC(DTCCLASS c, std::vector< std::pair <pgrid_ptr, std::array<int,3>> > grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double t_start, double t_end, bool outputAvg, std::vector<double> freqList, double timeInterval, double a, double I0, double t_max, bool outputMaps)
{
    if(c == DTCCLASS::BIN)
        dtcArr_.push_back( std::make_shared<parallelDetectorBINCplx>( grid, SI, loc, sz, out_name, type, timeInterval, a, I0, dt_) );
    else if(c == DTCCLASS::BMP)
        dtcArr_.push_back( std::make_shared<parallelDetectorBMPCplx>( grid, SI, loc, sz, outputAvg, t_start, t_end, out_name, fxn, txtType, type, timeInterval, a, I0, dt_) );
    else if(c == DTCCLASS::TXT)
        dtcArr_.push_back( std::make_shared<parallelDetectorTXTCplx>( grid, SI, loc, sz, out_name, type, timeInterval, a, I0, dt_ ) );
    else if(c == DTCCLASS::COUT)
        dtcArr_.push_back( std::make_shared<parallelDetectorCOUTCplx>(grid, SI, loc, sz, out_name, type, timeInterval, a, I0, dt_) );
    else if(c == DTCCLASS::FREQ)
        dtcFreqArr_.push_back(std::make_shared<parallelDetectorFREQCplx>(out_name, grid, loc, sz, type, static_cast<int>(std::floor(timeInterval/dt_+0.5) ), outputMaps, freqList, d_, dt_, SI, I0, a) );
    else
        throw std::logic_error("The detector class is undefined.");
}