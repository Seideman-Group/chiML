/** @file UTIL/FDTD_up_eq.cpp
 *  @brief A group of function that various function pointers in FDTDField and parallelQE use to update grids
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#include <UTIL/FDTD_up_eq.hpp>

void FDTDCompUpdateFxnReal::OneCompCurlJ (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k)
{
    // Finite difference of the j derivative components in the curl
    daxpy_(upParams[0],      prefactors[2], &grid_j->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    daxpy_(upParams[0], -1.0*prefactors[2], &grid_j->point(upParams[4]), 1, &grid_i->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnReal::OneCompCurlK (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k)
{
    // Finite differnce of the k derivatives components in the curl
    daxpy_(upParams[0], -1.0*prefactors[1], &grid_k->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    daxpy_(upParams[0],      prefactors[1], &grid_k->point(upParams[3]), 1, &grid_i->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnReal::TwoCompCurl (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k)
{
    // Finite difference of the j derivative components in the curl
    daxpy_(upParams[0],      prefactors[2], &grid_j->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    daxpy_(upParams[0], -1.0*prefactors[2], &grid_j->point(upParams[4]), 1, &grid_i->point(upParams[1]), 1);
    // Finite difference of the k derivative components in the curl
    daxpy_(upParams[0], -1.0*prefactors[1], &grid_k->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    daxpy_(upParams[0],      prefactors[1], &grid_k->point(upParams[3]), 1, &grid_i->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnCplx::OneCompCurlJ (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k)
{
    // Finite difference of the j derivative components in the curl
    zaxpy_(upParams[0],      prefactors[2], &grid_j->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    zaxpy_(upParams[0], -1.0*prefactors[2], &grid_j->point(upParams[4]), 1, &grid_i->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnCplx::OneCompCurlK (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k)
{
    // Finite differnce of the k derivatives components in the curl
    zaxpy_(upParams[0], -1.0*prefactors[1], &grid_k->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    zaxpy_(upParams[0],      prefactors[1], &grid_k->point(upParams[3]), 1, &grid_i->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnCplx::TwoCompCurl (const std::array<int,6>& upParams, const std::array<double,4>& prefactors, pgrid_ptr grid_i, pgrid_ptr grid_j, pgrid_ptr grid_k)
{
    // Finite difference of the j derivative components in the curl
    zaxpy_(upParams[0],      prefactors[2], &grid_j->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    zaxpy_(upParams[0], -1.0*prefactors[2], &grid_j->point(upParams[4]), 1, &grid_i->point(upParams[1]), 1);
    // Finite difference of the k derivative components in the curl
    zaxpy_(upParams[0], -1.0*prefactors[1], &grid_k->point(upParams[1]), 1, &grid_i->point(upParams[1]), 1);
    zaxpy_(upParams[0],      prefactors[1], &grid_k->point(upParams[3]), 1, &grid_i->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnReal::UpdateChiral(
    const std::array<int,6>& upParams,
    pgrid_ptr oppGrid_i,
    pgrid_ptr prevOppGrid_i,
    std::vector<pgrid_ptr>& lorChiPi,
    std::vector<pgrid_ptr> & prevLorPi,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    const std::vector<double>& gammaPrev,
    double* jstore)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        dcopy_(upParams[0], &lorChiPi[pp]->point(upParams[1]),  1, jstore, 1);

        // Update the polarizations as done in Taflove Ch. 7
        dscal_(upParams[0],      alpha[pp],   &lorChiPi[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],         xi[pp], &prevLorPi[pp]->point(upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[1]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(            upParams[3]                          ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(                        upParams[4]              ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(            upParams[3]+upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]+upParams[3]            -  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]+            upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]+upParams[3]+upParams[4]-2*upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[1]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(            upParams[3]                          ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(                        upParams[4]              ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(            upParams[3]+upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]+upParams[3]            -  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]+            upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]+upParams[3]+upParams[4]-2*upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        // reset prevLorP with the previously stored values in jstore
        dcopy_(upParams[0], jstore,1, &prevLorPi[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::UpdateChiralOrDip(
    const std::array<int,6>& upParams,
    pgrid_ptr oppGrid_x,
    pgrid_ptr prevOppGrid_x,
    pgrid_ptr oppGrid_y,
    pgrid_ptr prevOppGrid_y,
    pgrid_ptr oppGrid_z,
    pgrid_ptr prevOppGrid_z,
    std::vector<pgrid_ptr>& lorChiPx,
    std::vector<pgrid_ptr>& lorChiPy,
    std::vector<pgrid_ptr>& lorChiPz,
    std::vector<pgrid_ptr> & prevLorChiPx,
    std::vector<pgrid_ptr> & prevLorChiPy,
    std::vector<pgrid_ptr> & prevLorChiPz,
    const std::vector<real_pgrid_ptr>& oppDipChi_x,
    const std::vector<real_pgrid_ptr>& oppDipChi_y,
    const std::vector<real_pgrid_ptr>& oppDipChi_z,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    const std::vector<double>& gammaPrev,
    double* jstorex,
    double* jstorey,
    double* jstorez,
    double* tempStoreUDeriv,
    double* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        dcopy_(upParams[0], &lorChiPx[pp]->point(upParams[1]),  1, jstorex, 1);
        dcopy_(upParams[0], &lorChiPy[pp]->point(upParams[1]),  1, jstorey, 1);
        dcopy_(upParams[0], &lorChiPz[pp]->point(upParams[1]),  1, jstorez, 1);

        // Update the polarizations as done in Taflove Ch. 7
        dscal_(upParams[0], alpha[pp],     &lorChiPx[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorChiPx[pp]->point(upParams[1]),  1, &lorChiPx[pp] ->point(upParams[1]), 1);

        dscal_(upParams[0], alpha[pp],     &lorChiPy[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorChiPy[pp]->point(upParams[1]),  1, &lorChiPy[pp] ->point(upParams[1]), 1);

        dscal_(upParams[0], alpha[pp],     &lorChiPz[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorChiPz[pp]->point(upParams[1]),  1, &lorChiPz[pp] ->point(upParams[1]), 1);

        // Add dip_x * Ex to dotProduct
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[1]                        ), tempStoreUDeriv     , std::multiplies<double>() );
        dscal_(upParams[0],  0.25, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[3]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_y * Ey to dotProduct
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[2]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_z * Ez to dotProduct
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[3]+upParams[2]-upParams[1]), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Scale the dot product by dip_i*gamma[pp] and add it to the lorPi
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorChiPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorChiPy[pp] ->point(upParams[1]), 1);

        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorChiPz[pp] ->point(upParams[1]), 1);

        // Add oppDipChi_y * prevEx to dotProduct
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[1]                        ), tempStoreUDeriv     , std::multiplies<double>() );
        dscal_(upParams[0],  0.25, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[3]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_y * prevEy to dotProduct
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[2]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_z * prevEz to dotProduct
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[3]+upParams[2]-upParams[1]), tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Scale the dot product by gamma[pp] and add it to the lorChiPx
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gammaPrev[pp], tempStoreDipDotField,  1, &lorChiPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gammaPrev[pp], tempStoreDipDotField,  1, &lorChiPy[pp] ->point(upParams[1]), 1);

        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gammaPrev[pp], tempStoreDipDotField,  1, &lorChiPz[pp] ->point(upParams[1]), 1);

        // reset prevLorP with the previously stored values in jstorex
        dcopy_(upParams[0], jstorex,1, &prevLorChiPx[pp]->point(upParams[1]), 1);
        dcopy_(upParams[0], jstorey,1, &prevLorChiPy[pp]->point(upParams[1]), 1);
        dcopy_(upParams[0], jstorez,1, &prevLorChiPz[pp]->point(upParams[1]), 1);
    }
    return;
}
void FDTDCompUpdateFxnCplx::UpdateChiral(
    const std::array<int,6>& upParams,
    pgrid_ptr oppGrid_i,
    pgrid_ptr prevOppGrid_i,
    std::vector<pgrid_ptr>& lorChiPi,
    std::vector<pgrid_ptr> & prevLorPi,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    const std::vector<double>& gammaPrev,
    cplx* jstore)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        zcopy_(upParams[0], &lorChiPi[pp]->point(upParams[1]),  1, jstore, 1);

        // Update the polarizations as done in Taflove Ch. 7
        zscal_(upParams[0],      alpha[pp],  &lorChiPi[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],         xi[pp], &prevLorPi[pp]->point(upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[1]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(            upParams[3]                          ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(                        upParams[4]              ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(            upParams[3]+upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]+upParams[3]            -  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]+            upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0],     gamma[pp]/8.0,     &oppGrid_i->point(upParams[2]+upParams[3]+upParams[4]-2*upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[1]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(            upParams[3]                          ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(                        upParams[4]              ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(            upParams[3]+upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);

        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]                                      ),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]+upParams[3]            -  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]+            upParams[4]-  upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gammaPrev[pp]/8.0, &prevOppGrid_i->point(upParams[2]+upParams[3]+upParams[4]-2*upParams[1]),  1, &lorChiPi[pp] ->point(upParams[1]), 1);
        // reset prevLorP with the previously stored values in jstore
        zcopy_(upParams[0], jstore,1, &prevLorPi[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::UpdateChiralOrDip(
    const std::array<int,6>& upParams,
    pgrid_ptr oppGrid_x,
    pgrid_ptr prevOppGrid_x,
    pgrid_ptr oppGrid_y,
    pgrid_ptr prevOppGrid_y,
    pgrid_ptr oppGrid_z,
    pgrid_ptr prevOppGrid_z,
    std::vector<pgrid_ptr>& lorChiPx,
    std::vector<pgrid_ptr>& lorChiPy,
    std::vector<pgrid_ptr>& lorChiPz,
    std::vector<pgrid_ptr> & prevLorChiPx,
    std::vector<pgrid_ptr> & prevLorChiPy,
    std::vector<pgrid_ptr> & prevLorChiPz,
    const std::vector<real_pgrid_ptr>& oppDipChi_x,
    const std::vector<real_pgrid_ptr>& oppDipChi_y,
    const std::vector<real_pgrid_ptr>& oppDipChi_z,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    const std::vector<double>& gammaPrev,
    cplx* jstorex,
    cplx* jstorey,
    cplx* jstorez,
    cplx* tempStoreUDeriv,
    cplx* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        zcopy_(upParams[0], &lorChiPx[pp]->point(upParams[1]),  1, jstorex, 1);
        zcopy_(upParams[0], &lorChiPy[pp]->point(upParams[1]),  1, jstorey, 1);
        zcopy_(upParams[0], &lorChiPz[pp]->point(upParams[1]),  1, jstorez, 1);

        // Update the polarizations as done in Taflove Ch. 7
        zscal_(upParams[0], alpha[pp],     &lorChiPx[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorChiPx[pp]->point(upParams[1]),  1, &lorChiPx[pp] ->point(upParams[1]), 1);

        zscal_(upParams[0], alpha[pp],     &lorChiPy[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorChiPy[pp]->point(upParams[1]),  1, &lorChiPy[pp] ->point(upParams[1]), 1);

        zscal_(upParams[0], alpha[pp],     &lorChiPz[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorChiPz[pp]->point(upParams[1]),  1, &lorChiPz[pp] ->point(upParams[1]), 1);

        // Add dip_x * Ex to dotProduct
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[1]                        ), tempStoreUDeriv     , std::multiplies<cplx>() );
        zscal_(upParams[0],  0.25, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &oppGrid_x->point(upParams[3]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_y * Ey to dotProduct
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &oppGrid_y->point(upParams[2]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_z * Ez to dotProduct
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &oppGrid_z->point(upParams[3]+upParams[2]-upParams[1]), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Scale the dot product by dip_i*gamma[pp] and add it to the lorPi
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorChiPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorChiPy[pp] ->point(upParams[1]), 1);

        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorChiPz[pp] ->point(upParams[1]), 1);

        // Add oppDipChi_y * prevEx to dotProduct
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[1]                        ), tempStoreUDeriv     , std::multiplies<cplx>() );
        zscal_(upParams[0],  0.25, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_x[pp]->point(upParams[1]), &oppDipChi_x[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_x->point(upParams[3]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_y * prevEy to dotProduct
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[4]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_y[pp]->point(upParams[1]), &oppDipChi_y[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_y->point(upParams[2]+upParams[4]-upParams[1]), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Add oppDipChi_z * prevEz to dotProduct
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[1]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[3]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[2]                        ), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);
        std::transform(&oppDipChi_z[pp]->point(upParams[1]), &oppDipChi_z[pp]->point(upParams[1]) + upParams[0], &prevOppGrid_z->point(upParams[3]+upParams[2]-upParams[1]), tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0],  0.25, tempStoreDipDotField, 1, tempStoreUDeriv, 1);

        // Scale the dot product by gamma[pp] and add it to the lorChiPx
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gammaPrev[pp], tempStoreDipDotField,  1, &lorChiPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gammaPrev[pp], tempStoreDipDotField,  1, &lorChiPy[pp] ->point(upParams[1]), 1);

        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], tempStoreUDeriv, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gammaPrev[pp], tempStoreDipDotField,  1, &lorChiPz[pp] ->point(upParams[1]), 1);

        // reset prevLorP with the previously stored values in jstorex
        zcopy_(upParams[0], jstorex,1, &prevLorChiPx[pp]->point(upParams[1]), 1);
        zcopy_(upParams[0], jstorey,1, &prevLorChiPy[pp]->point(upParams[1]), 1);
        zcopy_(upParams[0], jstorez,1, &prevLorChiPz[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::UpdateLorPol(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_i,
    std::vector<pgrid_ptr> & lorPi,
    std::vector<pgrid_ptr> & prevLorPi,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    double* jstore)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        dcopy_(upParams[0], &lorPi[pp]->point(upParams[1]),  1, jstore, 1);

        // Update the polarizations as done in Taflove Ch. 7
        dscal_(upParams[0], alpha[pp],     &lorPi[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorPi[pp]->point(upParams[1]),  1, &lorPi[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gamma[pp],        &grid_i->point(upParams[1]),  1, &lorPi[pp] ->point(upParams[1]), 1);
        // reset prevLorP with the previously stored values in jstore
        dcopy_(upParams[0], jstore,1, &prevLorPi[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::UpdateLorPolOrDip(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_x,
    pgrid_ptr grid_y,
    pgrid_ptr grid_z,
    std::vector<pgrid_ptr>& lorPx,
    std::vector<pgrid_ptr>& lorPy,
    std::vector<pgrid_ptr>& lorPz,
    std::vector<pgrid_ptr>& prevLorPx,
    std::vector<pgrid_ptr>& prevLorPy,
    std::vector<pgrid_ptr>& prevLorPz,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    double* jstorex,
    double* jstorey,
    double* jstorez,
    double* tempStoreDipDotU,
    double* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        dcopy_(upParams[0], &lorPx[pp]->point(upParams[1]),  1, jstorex, 1);
        dcopy_(upParams[0], &lorPy[pp]->point(upParams[1]),  1, jstorey, 1);
        dcopy_(upParams[0], &lorPz[pp]->point(upParams[1]),  1, jstorez, 1);

        // Update the polarizations as done in Taflove Ch. 7
        dscal_(upParams[0], alpha[pp],     &lorPx[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorPx[pp]->point(upParams[1]),  1, &lorPx[pp] ->point(upParams[1]), 1);

        dscal_(upParams[0], alpha[pp],     &lorPy[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorPy[pp]->point(upParams[1]),  1, &lorPy[pp] ->point(upParams[1]), 1);

        dscal_(upParams[0], alpha[pp],     &lorPz[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorPz[pp]->point(upParams[1]),  1, &lorPz[pp] ->point(upParams[1]), 1);

        // Add dip_x * Ux to dotProduct
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[1]), tempStoreDipDotU    , multAvg<double,double>() );
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[2]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Add dip_y * Uy to dotProduct
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[1]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[3]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Add dip_z * Ez to dotProduct
        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], &grid_z->point(upParams[1]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);
        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], &grid_z->point(upParams[4]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Scale the dot product by dip_i*gamma[pp] and add it to the lorPi
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPy[pp] ->point(upParams[1]), 1);

        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPz[pp] ->point(upParams[1]), 1);
        // std::cout << *tempStoreDipDotU << '\t' << *tempStoreDipDotField << '\t' << lorPx[pp]->point(upParams[1]) << '\t' << lorPy[pp]->point(upParams[1]) << '\t' << lorPz[pp]->point(upParams[1]) << '\t' << grid_x->point(upParams[1]) << '\t' << grid_y->point(upParams[1]) << '\t' << grid_z->point(upParams[1]) << std::endl;

        // reset prevLorPi with the previously stored values in jstorei
        dcopy_(upParams[0], jstorex,1, &prevLorPx[pp]->point(upParams[1]), 1);
        dcopy_(upParams[0], jstorey,1, &prevLorPy[pp]->point(upParams[1]), 1);
        dcopy_(upParams[0], jstorez,1, &prevLorPz[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::UpdateLorPolOrDipXY(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_x,
    pgrid_ptr grid_y,
    pgrid_ptr grid_z,
    std::vector<pgrid_ptr>& lorPx,
    std::vector<pgrid_ptr>& lorPy,
    std::vector<pgrid_ptr>& lorPz,
    std::vector<pgrid_ptr>& prevLorPx,
    std::vector<pgrid_ptr>& prevLorPy,
    std::vector<pgrid_ptr>& prevLorPz,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    double* jstorex,
    double* jstorey,
    double* jstorez,
    double* tempStoreDipDotU,
    double* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        dcopy_(upParams[0], &lorPx[pp]->point(upParams[1]),  1, jstorex, 1);
        dcopy_(upParams[0], &lorPy[pp]->point(upParams[1]),  1, jstorey, 1);

        // Update the polarizations as done in Taflove Ch. 7
        dscal_(upParams[0], alpha[pp],     &lorPx[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorPx[pp]->point(upParams[1]),  1, &lorPx[pp] ->point(upParams[1]), 1);

        dscal_(upParams[0], alpha[pp],     &lorPy[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorPy[pp]->point(upParams[1]),  1, &lorPy[pp] ->point(upParams[1]), 1);


        // Add dip_x * Ux to dotProduct
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[1]), tempStoreDipDotU    , multAvg<double,double>() );
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[2]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Add dip_y * Uy to dotProduct
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[1]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[3]), tempStoreDipDotField, multAvg<double,double>() );
        daxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Scale the dot product by dip_i*gamma[pp] and add it to the lorPi
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<double>() );
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPy[pp] ->point(upParams[1]), 1);

        // reset prevLorPi with the previously stored values in jstorei
        dcopy_(upParams[0], jstorex,1, &prevLorPx[pp]->point(upParams[1]), 1);
        dcopy_(upParams[0], jstorey,1, &prevLorPy[pp]->point(upParams[1]), 1);

    }
    return;
}

void FDTDCompUpdateFxnReal::UpdateLorPolOrDipZ(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_x,
    pgrid_ptr grid_y,
    pgrid_ptr grid_z,
    std::vector<pgrid_ptr>& lorPx,
    std::vector<pgrid_ptr>& lorPy,
    std::vector<pgrid_ptr>& lorPz,
    std::vector<pgrid_ptr>& prevLorPx,
    std::vector<pgrid_ptr>& prevLorPy,
    std::vector<pgrid_ptr>& prevLorPz,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    double* jstorex,
    double* jstorey,
    double* jstorez,
    double* tempStoreDipDotU,
    double* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        dcopy_(upParams[0], &lorPz[pp]->point(upParams[1]),  1, jstorez, 1);

        // Update the polarizations as done in Taflove Ch. 7

        dscal_(upParams[0], alpha[pp],     &lorPz[pp]->point(upParams[1]),  1);
        daxpy_(upParams[0],    xi[pp], &prevLorPz[pp]->point(upParams[1]),  1, &lorPz[pp] ->point(upParams[1]), 1);

        // Add dip_z * Ez to dotProduct
        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], &grid_z->point(upParams[1]), tempStoreDipDotU, std::multiplies<double>() );

        // Scale the dot product by gamma[pp] and add it to the lo4P0], gamma[pp], tempStoreDipDotU,  1, &lorP[pp] ->point(upParams[1]), 1);
        daxpy_(upParams[0], gamma[pp], tempStoreDipDotU,  1, &lorPz[pp] ->point(upParams[1]), 1);
        // reset prevLorP with the previously stored values in jstore
        dcopy_(upParams[0], jstorez,1, &prevLorPz[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::UpdateLorPol(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_i,
    std::vector<pgrid_ptr> & lorPi,
    std::vector<pgrid_ptr> & prevLorPi,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    cplx* jstore)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        zcopy_(upParams[0], &lorPi[pp]->point(upParams[1]),  1, jstore, 1);

        // Update the polarizations as done in Taflove Ch. 7
        zscal_(upParams[0], alpha[pp],     &lorPi[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorPi[pp]->point(upParams[1]),  1, &lorPi[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gamma[pp],        &grid_i->point(upParams[1]),  1, &lorPi[pp] ->point(upParams[1]), 1);
        // reset prevLorP with the previously stored values in jstore
        zcopy_(upParams[0], jstore,1, &prevLorPi[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::UpdateLorPolOrDip(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_x,
    pgrid_ptr grid_y,
    pgrid_ptr grid_z,
    std::vector<pgrid_ptr>& lorPx,
    std::vector<pgrid_ptr>& lorPy,
    std::vector<pgrid_ptr>& lorPz,
    std::vector<pgrid_ptr>& prevLorPx,
    std::vector<pgrid_ptr>& prevLorPy,
    std::vector<pgrid_ptr>& prevLorPz,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    cplx* jstorex,
    cplx* jstorey,
    cplx* jstorez,
    cplx* tempStoreDipDotU,
    cplx* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        zcopy_(upParams[0], &lorPx[pp]->point(upParams[1]),  1, jstorex, 1);
        zcopy_(upParams[0], &lorPy[pp]->point(upParams[1]),  1, jstorey, 1);
        zcopy_(upParams[0], &lorPz[pp]->point(upParams[1]),  1, jstorez, 1);

        // Update the polarizations as done in Taflove Ch. 7
        zscal_(upParams[0], alpha[pp],     &lorPx[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorPx[pp]->point(upParams[1]),  1, &lorPx[pp] ->point(upParams[1]), 1);

        zscal_(upParams[0], alpha[pp],     &lorPy[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorPy[pp]->point(upParams[1]),  1, &lorPy[pp] ->point(upParams[1]), 1);

        zscal_(upParams[0], alpha[pp],     &lorPz[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorPz[pp]->point(upParams[1]),  1, &lorPz[pp] ->point(upParams[1]), 1);

        // Add dip_x * Ux to dotProduct
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[1]), tempStoreDipDotU    , multAvg<double,cplx>() );
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[2]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Add dip_y * Uy to dotProduct
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[1]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[3]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Add dip_z * Ez to dotProduct
        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], &grid_z->point(upParams[1]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);
        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], &grid_z->point(upParams[4]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Scale the dot product by dip_i*gamma[pp] and add it to the lorPi
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPy[pp] ->point(upParams[1]), 1);

        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPz[pp] ->point(upParams[1]), 1);
        // reset prevLorPi with the previously stored values in jstorei
        zcopy_(upParams[0], jstorex,1, &prevLorPx[pp]->point(upParams[1]), 1);
        zcopy_(upParams[0], jstorey,1, &prevLorPy[pp]->point(upParams[1]), 1);
        zcopy_(upParams[0], jstorez,1, &prevLorPz[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::UpdateLorPolOrDipXY(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_x,
    pgrid_ptr grid_y,
    pgrid_ptr grid_z,
    std::vector<pgrid_ptr>& lorPx,
    std::vector<pgrid_ptr>& lorPy,
    std::vector<pgrid_ptr>& lorPz,
    std::vector<pgrid_ptr>& prevLorPx,
    std::vector<pgrid_ptr>& prevLorPy,
    std::vector<pgrid_ptr>& prevLorPz,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    cplx* jstorex,
    cplx* jstorey,
    cplx* jstorez,
    cplx* tempStoreDipDotU,
    cplx* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        zcopy_(upParams[0], &lorPx[pp]->point(upParams[1]),  1, jstorex, 1);
        zcopy_(upParams[0], &lorPy[pp]->point(upParams[1]),  1, jstorey, 1);

        // Update the polarizations as done in Taflove Ch. 7
        zscal_(upParams[0], alpha[pp],     &lorPx[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorPx[pp]->point(upParams[1]),  1, &lorPx[pp] ->point(upParams[1]), 1);

        zscal_(upParams[0], alpha[pp],     &lorPy[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],    xi[pp], &prevLorPy[pp]->point(upParams[1]),  1, &lorPy[pp] ->point(upParams[1]), 1);


        // Add dip_x * Ux to dotProduct
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[1]), tempStoreDipDotU    , multAvg<double,cplx>() );
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], &grid_x->point(upParams[2]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);

        // Add dip_y * Uy to dotProduct
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[1]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);
        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], &grid_y->point(upParams[3]), tempStoreDipDotField, multAvg<double,cplx>() );
        zaxpy_(upParams[0],  1.0, tempStoreDipDotField, 1, tempStoreDipDotU, 1);


        // Scale the dot product by dip_i*gamma[pp] and add it to the lorPi
        std::transform(&dip_x[pp]->point(upParams[1]), &dip_x[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPx[pp] ->point(upParams[1]), 1);

        std::transform(&dip_y[pp]->point(upParams[1]), &dip_y[pp]->point(upParams[1]) + upParams[0], tempStoreDipDotU, tempStoreDipDotField, std::multiplies<cplx>() );
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotField,  1, &lorPy[pp] ->point(upParams[1]), 1);

        // Restet prevLor_i with jstorei
        zcopy_(upParams[0], jstorex,1, &prevLorPx[pp]->point(upParams[1]), 1);
        zcopy_(upParams[0], jstorey,1, &prevLorPy[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::UpdateLorPolOrDipZ(
    const std::array<int,6>& upParams,
    pgrid_ptr grid_x,
    pgrid_ptr grid_y,
    pgrid_ptr grid_z,
    std::vector<pgrid_ptr>& lorPx,
    std::vector<pgrid_ptr>& lorPy,
    std::vector<pgrid_ptr>& lorPz,
    std::vector<pgrid_ptr>& prevLorPx,
    std::vector<pgrid_ptr>& prevLorPy,
    std::vector<pgrid_ptr>& prevLorPz,
    const std::vector<real_pgrid_ptr>& dip_x,
    const std::vector<real_pgrid_ptr>& dip_y,
    const std::vector<real_pgrid_ptr>& dip_z,
    const std::vector<double>& alpha,
    const std::vector<double>& xi,
    const std::vector<double>& gamma,
    cplx* jstorex,
    cplx* jstorey,
    cplx* jstorez,
    cplx* tempStoreDipDotU,
    cplx* tempStoreDipDotField)
{
    for(int pp = 0; pp < alpha.size(); ++pp)
    {
        // Store current values of the Lorentzian polarizations into jstore
        zcopy_(upParams[0], &lorPz[pp]->point(upParams[1]),  1, jstorez, 1);

        // Update the polarizations as done in Taflove Ch. 7
        zscal_(upParams[0],      alpha[pp],     &lorPz[pp]->point(upParams[1]),  1);
        zaxpy_(upParams[0],         xi[pp], &prevLorPz[pp]->point(upParams[1]),  1, &lorPz[pp] ->point(upParams[1]), 1);

        // Add dip_z * Ez to dotProduct
        std::transform(&dip_z[pp]->point(upParams[1]), &dip_z[pp]->point(upParams[1]) + upParams[0], &grid_z->point(upParams[1]), tempStoreDipDotU, std::multiplies<cplx>() );

        // Scale the dot product by gamma[pp] and add it to the lo4P0], gamma[pp], tempStoreDipDotU,  1, &lorP[pp] ->point(upParams[1]), 1);
        zaxpy_(upParams[0], gamma[pp], tempStoreDipDotU,  1, &lorPz[pp] ->point(upParams[1]), 1);
        // reset prevLorP with the previously stored values in jstorez
        zcopy_(upParams[0], jstorez,1, &prevLorPz[pp]->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::DtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr> & lorPi)
{
    // Set the E field to the D field
    dcopy_(upParams[0], &Di->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    // Scale E field by 1/epMuInfty (Done because E = 1/epMuInfty D)
    dscal_(upParams[0], 1.0/epMuInfty, &Ui->point(upParams[1]), 1);
    // Add all Polarizations
    for(int pp = 0; pp < lorPi.size(); ++pp)
        daxpy_(upParams[0], -1.0/epMuInfty, &lorPi[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnCplx::DtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr> & lorPi)
{
    // Set the E field to the D field
    zcopy_(upParams[0], &Di->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    // Scale E field by 1/epMuInfty (Done because E = 1/epMuInfty D)
    zscal_(upParams[0], 1.0/epMuInfty, &Ui->point(upParams[1]), 1);
    // Add all Polarizations
    for(int pp = 0; pp < lorPi.size(); ++pp)
        zaxpy_(upParams[0], -1.0/epMuInfty, &lorPi[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnReal::orDipDtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorP, double* tempPolStore, double* tempDotStore)
{
    // Set the E field to the D field
    dcopy_(upParams[0], &Di->point(upParams[1]), 1, &Ui->point(upParams[1]), 1);
    // Scale E field by 1/epMuInfty (Done because E = 1/epMuInfty D)
    dscal_(upParams[0], 1.0/epMuInfty, &Ui->point(upParams[1]), 1);
    // Add all Polarizations
    for(int pp = 0; pp < nPols; ++pp)
    {
        daxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[1]), 1, &Ui->point(upParams[1]), 1);
        daxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[2]), 1, &Ui->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::orDipDtoUZ(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorP, double* tempPolStore, double* tempDotStore)
{
    // Set the E field to the D field
    dcopy_(upParams[0], &Di->point(upParams[1]), 1, &Ui->point(upParams[1]), 1);
    // Scale E field by 1/epMuInfty (Done because E = 1/epMuInfty D)
    dscal_(upParams[0], 1.0/epMuInfty, &Ui->point(upParams[1]), 1);
    // Add all Polarizations
    for(int pp = 0; pp < nPols; ++pp)
    {
        daxpy_(upParams[0], -1.0/epMuInfty, &lorP[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::orDipDtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorP, cplx* tempPolStore, cplx* tempDotStore)
{
    // Set the E field to the D field
    zcopy_(upParams[0], &Di->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    // Scale E field by 1/epMuInfty (Done because E = 1/epMuInfty D)
    zscal_(upParams[0], 1.0/epMuInfty, &Ui->point(upParams[1]), 1);
    // Add all Polarizations
    for(int pp = 0; pp < nPols; ++pp)
    {
        zaxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
        zaxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[2]), 1,&Ui->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::orDipDtoUZ(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorP, cplx* tempPolStore, cplx* tempDotStore)
{
    // Set the E field to the D field
    zcopy_(upParams[0], &Di->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    // Scale E field by 1/epMuInfty (Done because E = 1/epMuInfty D)
    zscal_(upParams[0], 1.0/epMuInfty, &Ui->point(upParams[1]), 1);
    // Add all Polarizations
    for(int pp = 0; pp < nPols; ++pp)
    {
        zaxpy_(upParams[0], -1.0/epMuInfty, &lorP[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::chiDtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr>& lorChiPi)
{
    for(int pp = 0; pp < lorChiPi.size(); ++pp)
        daxpy_(upParams[0],  -1.0/epMuInfty, &lorChiPi[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnCplx::chiDtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, std::vector<pgrid_ptr>& lorChiPi)
{
    for(int pp = 0; pp < lorChiPi.size(); ++pp)
        zaxpy_(upParams[0],  -1.0/epMuInfty, &lorChiPi[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
    return;
}

void FDTDCompUpdateFxnReal::chiOrDipDtoU(const std::array<int,6>& upParams, double epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorP, double* tempPolStore, double* tempDotStore)
{
    for(int pp = 0; pp < nPols; ++pp)
    {
        daxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
        daxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[2]), 1,&Ui->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnCplx::chiOrDipDtoU(const std::array<int,6>& upParams, cplx epMuInfty, pgrid_ptr Di, pgrid_ptr Ui, int nPols, std::vector<pgrid_ptr> & lorP, cplx* tempPolStore, cplx* tempDotStore)
{
    for(int pp = 0; pp < nPols; ++pp)
    {
        zaxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[1]), 1,&Ui->point(upParams[1]), 1);
        zaxpy_(upParams[0], -0.5/epMuInfty, &lorP[pp]->point(upParams[2]), 1,&Ui->point(upParams[1]), 1);
    }
    return;
}

void FDTDCompUpdateFxnReal::applyBCProc0(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
    // if real fields then the PBC is just copying from one side to the other
    if(zmin != 0)
    {
        for(int jj = 1; jj < ny; ++jj)
        {
            dcopy_(nz, &fUp->point(xmax-1, jj, 1     ), fUp->local_x(), &fUp->point(0   , jj,      1), fUp->local_x() );
            dcopy_(nz, &fUp->point(1     , jj, 1     ), fUp->local_x(), &fUp->point(xmax, jj,      1), fUp->local_x() );

            dcopy_(nx, &fUp->point(1     , jj, zmax-1), 1             , &fUp->point(1   , jj, zmin-1), 1 );
            dcopy_(nx, &fUp->point(1     , jj, zmin  ), 1             , &fUp->point(1   , jj, zmax  ), 1 );
        }
        // X edges
        dcopy_( nx, &fUp->point(1,    0, zmax-1), 1, &fUp->point(1, 0     , zmin-1), 1 );
        dcopy_( nx, &fUp->point(1,    0, zmin  ), 1, &fUp->point(1, 0     , zmax  ), 1 );

        // Z edges
        dcopy_( nz, &fUp->point(xmax-1, 0   , 1), fUp->local_x(), &fUp->point(0   , 0   , 1), fUp->local_x() );
        dcopy_( nz, &fUp->point(1     , 0   , 1), fUp->local_x(), &fUp->point(xmax, 0   , 1), fUp->local_x() );

        // Y edges
        dcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());

        // Corners
        fUp->point(xmax, 0   , zmax  ) = fUp->point(1     , 0, zmin  );
        fUp->point(0   , 0   , zmax  ) = fUp->point(xmax-1, 0, zmin  );
        fUp->point(xmax, 0   , zmin-1) = fUp->point(1     , 0, zmax-1);
        fUp->point(0   , 0   , zmin-1) = fUp->point(xmax-1, 0, zmax-1);
    }
    else
    {
        dcopy_(fUp->local_y(), &fUp->point(xmax-1, 0), fUp->local_x(), &fUp->point(0   , 0), fUp->local_x() );
        dcopy_(fUp->local_y(), &fUp->point(1     , 0), fUp->local_x(), &fUp->point(xmax, 0), fUp->local_x() );
    }
}

void FDTDCompUpdateFxnReal::applyBCProcMax(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
    // if real fields then the PBC is just copying from one side to the other
    if(zmin != 0)
    {
        for(int jj = 1; jj < ny; ++jj)
        {
            dcopy_(nz, &fUp->point(xmax-1, jj, 1     ), fUp->local_x(), &fUp->point(0   , jj,      1), fUp->local_x() );
            dcopy_(nz, &fUp->point(1     , jj, 1     ), fUp->local_x(), &fUp->point(xmax, jj,      1), fUp->local_x() );

            dcopy_(nx, &fUp->point(1     , jj, zmax-1), 1             , &fUp->point(1   , jj, zmin-1), 1 );
            dcopy_(nx, &fUp->point(1     , jj, zmin  ), 1             , &fUp->point(1   , jj, zmax  ), 1 );
        }
        // X edges
        dcopy_( nx, &fUp->point(1, ymax, zmin  ), 1, &fUp->point(1, ymax  , zmax  ), 1 );
        dcopy_( nx, &fUp->point(1, ymax, zmax-1), 1, &fUp->point(1, ymax  , zmin-1), 1 );

        // Z edges
        dcopy_( nz, &fUp->point(1     , ymax, 1), fUp->local_x(), &fUp->point(xmax, ymax, 1), fUp->local_x() );
        dcopy_( nz, &fUp->point(xmax-1, ymax, 1), fUp->local_x(), &fUp->point(0   , ymax, 1), fUp->local_x() );

        // Y edges
        dcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());

        // Corners
        fUp->point(xmax, ymax, zmax  ) = fUp->point(1     , ymax, zmin  );
        fUp->point(0   , ymax, zmax  ) = fUp->point(xmax-1, ymax, zmin  );
        fUp->point(xmax, ymax, zmin-1) = fUp->point(1     , ymax, zmax-1);
        fUp->point(0   , ymax, zmin-1) = fUp->point(xmax-1, ymax, zmax-1);
    }
    else
    {
        dcopy_(fUp->local_y(), &fUp->point(xmax-1, 0), fUp->local_x(), &fUp->point(0   , 0), fUp->local_x() );
        dcopy_(fUp->local_y(), &fUp->point(1     , 0), fUp->local_x(), &fUp->point(xmax, 0), fUp->local_x() );
    }
}

void FDTDCompUpdateFxnReal::applyBCProcMid(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
    // if real fields then the PBC is just copying from one side to the other
    if(zmin != 0)
    {
        for(int jj = 1; jj < ny; ++jj)
        {
            dcopy_(nz, &fUp->point(xmax-1, jj, 1     ), fUp->local_x(), &fUp->point(0   , jj,      1), fUp->local_x() );
            dcopy_(nz, &fUp->point(1     , jj, 1     ), fUp->local_x(), &fUp->point(xmax, jj,      1), fUp->local_x() );

            dcopy_(nx, &fUp->point(1     , jj, zmax-1), 1             , &fUp->point(1   , jj, zmin-1), 1 );
            dcopy_(nx, &fUp->point(1     , jj, zmin  ), 1             , &fUp->point(1   , jj, zmax  ), 1 );
        }
        // Y edges
        dcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
    }
    else
    {
        dcopy_(fUp->local_y(), &fUp->point(xmax-1, 0), fUp->local_x(), &fUp->point(0   , 0), fUp->local_x() );
        dcopy_(fUp->local_y(), &fUp->point(1     , 0), fUp->local_x(), &fUp->point(xmax, 0), fUp->local_x() );
    }
}
void FDTDCompUpdateFxnReal::applyBC1Proc(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    // if real fields then the PBC is just copying from one side to the other
    if(zmin != 0)
    {
        for(int kk = zmin; kk <= nz; ++kk)
        {
            dcopy_(nx, &fUp->point(1   , ymax-1, kk),  1, &fUp->point(1 , 0   , kk), 1 );
            dcopy_(nx, &fUp->point(1   , 1     , kk),  1, &fUp->point(1 , ymax, kk), 1 );
        }
        for(int jj = 1; jj < ny; ++jj)
        {
            dcopy_(nz, &fUp->point(xmax-1, jj, 1), fUp->local_x(), &fUp->point(0   , jj, 1), fUp->local_x() );
            dcopy_(nz, &fUp->point(1     , jj, 1), fUp->local_x(), &fUp->point(xmax, jj, 1), fUp->local_x() );

            dcopy_(nx, &fUp->point(1, jj, zmax-1), 1, &fUp->point(1, jj, zmin-1), 1 );
            dcopy_(nx, &fUp->point(1, jj, zmin  ), 1, &fUp->point(1, jj, zmax  ), 1 );
        }
        // X edges
        dcopy_(nx, &fUp->point(1,      1, zmin  ), 1, &fUp->point(1, ymax  , zmax  ), 1);
        dcopy_(nx, &fUp->point(1, ymax-1, zmin  ), 1, &fUp->point(1, 0     , zmax  ), 1);
        dcopy_(nx, &fUp->point(1,      1, zmax-1), 1, &fUp->point(1, ymax  , zmin-1), 1);
        dcopy_(nx, &fUp->point(1, ymax-1, zmax-1), 1, &fUp->point(1, 0     , zmin-1), 1);

        // Y edges
        dcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        dcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());

        // Z edges
        dcopy_(nz, &fUp->point(1     , 1     , 1), fUp->local_x(), &fUp->point(xmax, ymax, 1), fUp->local_x());
        dcopy_(nz, &fUp->point(xmax-1, 1     , 1), fUp->local_x(), &fUp->point(0   , ymax, 1), fUp->local_x());
        dcopy_(nz, &fUp->point(1     , ymax-1, 1), fUp->local_x(), &fUp->point(xmax, 0   , 1), fUp->local_x());
        dcopy_(nz, &fUp->point(xmax-1, ymax-1, 1), fUp->local_x(), &fUp->point(0   , 0   , 1), fUp->local_x());

        //Corners
        fUp->point(xmax, ymax, zmax  ) = fUp->point(1     , 1     , zmin  );
        fUp->point(0   , ymax, zmax  ) = fUp->point(xmax-1, 1     , zmin  );
        fUp->point(xmax, 0   , zmax  ) = fUp->point(1     , ymax-1, zmin  );
        fUp->point(0   , 0   , zmax  ) = fUp->point(xmax-1, ymax-1, zmin  );

        fUp->point(xmax, ymax, zmin-1) = fUp->point(1     , 1     , zmax-1);
        fUp->point(0   , ymax, zmin-1) = fUp->point(xmax-1, 1     , zmax-1);
        fUp->point(xmax, 0   , zmin-1) = fUp->point(1     , ymax-1, zmax-1);
        fUp->point(0   , 0   , zmin-1) = fUp->point(xmax-1, ymax-1, zmax-1);
    }
    else
    {
        dcopy_(nx, &fUp->point(1     , ymax-1),  1            , &fUp->point(1   , 0   ),              1 );
        dcopy_(nx, &fUp->point(1     , 1     ),  1            , &fUp->point(1   , ymax),              1 );
        dcopy_(ny, &fUp->point(xmax-1, 1     ), fUp->local_x(), &fUp->point(0   , 1   ), fUp->local_x() );
        dcopy_(ny, &fUp->point(1     , 1     ), fUp->local_x(), &fUp->point(xmax, 1   ), fUp->local_x() );
    }
}

void FDTDCompUpdateFxnCplx::applyBCProc0  (pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
    // if real fields then the PBC is just copying from one side to the other
    if(zmin != 0)
    {
        for(int jj = 1; jj < ny; ++jj)
        {
            zcopy_(nz, &fUp->point(xmax-1, jj, 1), fUp->local_x(), &fUp->point(0   , jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax ) ), &fUp->point(0   , jj, 1), fUp->local_x() );
            zcopy_(nz, &fUp->point(1     , jj, 1), fUp->local_x(), &fUp->point(xmax, jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax ) ), &fUp->point(xmax, jj, 1), fUp->local_x() );

            zcopy_(nx, &fUp->point(1, jj, zmax-1), 1, &fUp->point(1, jj, zmin-1), 1 );
            zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmin-1), 1 );
            zcopy_(nx, &fUp->point(1, jj, zmin  ), 1, &fUp->point(1, jj, zmax  ), 1 );
            zscal_(nx, std::exp( cplx( 0.0,      k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmax  ), 1 );
        }
        // Y edges
        zcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());

        // X edges
        zcopy_(nx, &fUp->point(1, ymax-1, zmin  ), 1,                                              &fUp->point(1, 0, zmax  ), 1);
        zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ),  &fUp->point(1, 0, zmax  ), 1);
        zcopy_(nx, &fUp->point(1, ymax-1, zmax-1), 1,                                              &fUp->point(1, 0, zmin-1), 1);
        zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ),  &fUp->point(1, 0, zmin-1), 1);

        // Z edges
        zcopy_(nz, &fUp->point(1     , ymax-1, 1), fUp->local_x(),                                &fUp->point(xmax, 0, 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[1] * dy * ymax ) ), &fUp->point(xmax, 0, 1), fUp->local_x());
        zcopy_(nz, &fUp->point(xmax-1, ymax-1, 1), fUp->local_x(),                                &fUp->point(0   , 0, 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[1] * dy * ymax ) ), &fUp->point(0   , 0, 1), fUp->local_x());

        // Corners
        fUp->point(xmax, 0   , zmax  ) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmin  );
        fUp->point(0   , 0   , zmax  ) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmin  );
        fUp->point(xmax, 0   , zmin-1) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmax-1);
        fUp->point(0   , 0   , zmin-1) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmax-1);
    }
}
void FDTDCompUpdateFxnCplx::applyBCProcMax(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
    // if real fields then the PBC is just copying from one side to the other
    if(zmin != 0)
    {
        for(int jj = 1; jj < ny; ++jj)
        {
            zcopy_(nz, &fUp->point(xmax-1, jj, 1), fUp->local_x(), &fUp->point(0   , jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax ) ), &fUp->point(0   , jj, 1), fUp->local_x() );
            zcopy_(nz, &fUp->point(1     , jj, 1), fUp->local_x(), &fUp->point(xmax, jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax ) ), &fUp->point(xmax, jj, 1), fUp->local_x() );

            zcopy_(nx, &fUp->point(1, jj, zmax-1), 1, &fUp->point(1, jj, zmin-1), 1 );
            zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmin-1), 1 );
            zcopy_(nx, &fUp->point(1, jj, zmin  ), 1, &fUp->point(1, jj, zmax  ), 1 );
            zscal_(nx, std::exp( cplx( 0.0,      k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmax  ), 1 );
        }
        // Y edges
        zcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());

        // X edges
        zcopy_(nx, &fUp->point(1,      1, zmin  ), 1,                                              &fUp->point(1, ymax  , zmax  ), 1);
        zscal_(nx, std::exp( cplx( 0.0,      k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ),  &fUp->point(1, ymax  , zmax  ), 1);
        zcopy_(nx, &fUp->point(1,      1, zmax-1), 1,                                              &fUp->point(1, ymax  , zmin-1), 1);
        zscal_(nx, std::exp( cplx( 0.0,      k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ),  &fUp->point(1, ymax  , zmin-1), 1);

        // Z edges
        zcopy_(nz, &fUp->point(1     , 1     , 1), fUp->local_x(),                                &fUp->point(xmax, ymax, 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[1] * dy * ymax ) ), &fUp->point(xmax, ymax, 1), fUp->local_x());
        zcopy_(nz, &fUp->point(xmax-1, 1     , 1), fUp->local_x(),                                &fUp->point(0   , ymax, 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[1] * dy * ymax ) ), &fUp->point(0   , ymax, 1), fUp->local_x());

        // Corners
        fUp->point(xmax, ymax, zmax  ) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmin  );
        fUp->point(0   , ymax, zmax  ) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmin  );
        fUp->point(xmax, ymax, zmin-1) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmax-1);
        fUp->point(0   , ymax, zmin-1) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmax-1);
    }
}
void FDTDCompUpdateFxnCplx::applyBCProcMid(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
    // if real fields then the PBC is just copying from one side to the other
    if(zmin != 0)
    {
        for(int jj = 1; jj < ny; ++jj)
        {
            zcopy_(nz, &fUp->point(xmax-1, jj, 1), fUp->local_x(), &fUp->point(0   , jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax ) ), &fUp->point(0   , jj, 1), fUp->local_x() );
            zcopy_(nz, &fUp->point(1     , jj, 1), fUp->local_x(), &fUp->point(xmax, jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax ) ), &fUp->point(xmax, jj, 1), fUp->local_x() );

            zcopy_(nx, &fUp->point(1, jj, zmax-1), 1, &fUp->point(1, jj, zmin-1), 1 );
            zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmin-1), 1 );
            zcopy_(nx, &fUp->point(1, jj, zmin  ), 1, &fUp->point(1, jj, zmax  ), 1 );
            zscal_(nx, std::exp( cplx( 0.0,      k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmax  ), 1 );
        }
        // Y edges
        zcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
    }
}

void FDTDCompUpdateFxnCplx::applyBC1Proc(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    if(zmin != 0)
    {
        for(int kk = zmin; kk <= nz; ++kk)
        {
            zcopy_(nx, &fUp->point(1   , ymax-1, kk),  1, &fUp->point(1 , 0   , kk), 1 );
            zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[1] * dy * ymax ) ),&fUp->point(1 , 0   , kk), 1 );
            zcopy_(nx, &fUp->point(1   , 1     , kk),  1, &fUp->point(1 , ymax, kk), 1 );
            zscal_(nx, std::exp( cplx( 0.0,      k_point[1] * dy * ymax ) ),&fUp->point(1 , ymax, kk), 1 );
        }
        for(int jj = 1; jj < ny; ++jj)
        {
            zcopy_(nz, &fUp->point(xmax-1, jj, 1), fUp->local_x(), &fUp->point(0   , jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax ) ), &fUp->point(0   , jj, 1), fUp->local_x() );
            zcopy_(nz, &fUp->point(1     , jj, 1), fUp->local_x(), &fUp->point(xmax, jj, 1), fUp->local_x() );
            zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax ) ), &fUp->point(xmax, jj, 1), fUp->local_x() );

            zcopy_(nx, &fUp->point(1, jj, zmax-1), 1, &fUp->point(1, jj, zmin-1), 1 );
            zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmin-1), 1 );
            zcopy_(nx, &fUp->point(1, jj, zmin  ), 1, &fUp->point(1, jj, zmax  ), 1 );
            zscal_(nx, std::exp( cplx( 0.0,      k_point[2] * dz * zmax ) ), &fUp->point(1, jj, zmax  ), 1 );
        }
        // Y edges
        zcopy_(ny-1, &fUp->point(     1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmin  ), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmax  ), fUp->local_x()*fUp->local_z());

        zcopy_(ny-1, &fUp->point(     1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(xmax  , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zcopy_(ny-1, &fUp->point(xmax-1, 1, zmax-1), fUp->local_x()*fUp->local_z(),                 &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());
        zscal_(ny-1, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[2] * dz * zmax ) ), &fUp->point(0     , 1, zmin-1), fUp->local_x()*fUp->local_z());

        // X edges
        zcopy_(nx, &fUp->point(1,      1, zmin  ), 1,                                              &fUp->point(1, ymax  , zmax  ), 1);
        zscal_(nx, std::exp( cplx( 0.0,      k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ),  &fUp->point(1, ymax  , zmax  ), 1);
        zcopy_(nx, &fUp->point(1,      1, zmax-1), 1,                                              &fUp->point(1, ymax  , zmin-1), 1);
        zscal_(nx, std::exp( cplx( 0.0,      k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ),  &fUp->point(1, ymax  , zmin-1), 1);

        zcopy_(nx, &fUp->point(1, ymax-1, zmin  ), 1,                                              &fUp->point(1, 0     , zmax  ), 1);
        zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ),  &fUp->point(1, 0     , zmax  ), 1);
        zcopy_(nx, &fUp->point(1, ymax-1, zmax-1), 1,                                              &fUp->point(1, 0     , zmin-1), 1);
        zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ),  &fUp->point(1, 0     , zmin-1), 1);

        // Z edges
        zcopy_(nz, &fUp->point(1     , 1     , 1), fUp->local_x(),                                &fUp->point(xmax, ymax, 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[1] * dy * ymax ) ), &fUp->point(xmax, ymax, 1), fUp->local_x());
        zcopy_(nz, &fUp->point(xmax-1, 1     , 1), fUp->local_x(),                                &fUp->point(0   , ymax, 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[1] * dy * ymax ) ), &fUp->point(0   , ymax, 1), fUp->local_x());

        zcopy_(nz, &fUp->point(1     , ymax-1, 1), fUp->local_x(),                                &fUp->point(xmax, 0   , 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[1] * dy * ymax ) ), &fUp->point(xmax, 0   , 1), fUp->local_x());
        zcopy_(nz, &fUp->point(xmax-1, ymax-1, 1), fUp->local_x(),                                &fUp->point(0   , 0   , 1), fUp->local_x());
        zscal_(nz, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[1] * dy * ymax ) ), &fUp->point(0   , 0   , 1), fUp->local_x());


        // Corners
        fUp->point(xmax, ymax, zmax  ) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmin  );
        fUp->point(0   , ymax, zmax  ) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmin  );
        fUp->point(xmax, ymax, zmin-1) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax + k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmax-1);
        fUp->point(0   , ymax, zmin-1) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax + k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmax-1);

        fUp->point(xmax, 0   , zmax  ) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmin  );
        fUp->point(0   , 0   , zmax  ) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[1] * dy * ymax + k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmin  );
        fUp->point(xmax, 0   , zmin-1) = std::exp( cplx( 0.0,      k_point[0] * dx * xmax - k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(1     , ymax, zmax-1);
        fUp->point(0   , 0   , zmin-1) = std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax - k_point[1] * dy * ymax - k_point[2] * dz * zmax ) ) * fUp->point(xmax-1, ymax, zmax-1);
    }
    else
    {
        zcopy_(nx, &fUp->point(1     , ymax-1),  1            , &fUp->point(1   , 0   ),              1 );
        zscal_(nx, std::exp( cplx( 0.0, -1.0*k_point[1] * dy * ymax ) ), &fUp->point(1   , 0   ),              1 );

        zcopy_(nx, &fUp->point(1     , 1     ),  1            , &fUp->point(1   , ymax),              1 );
        zscal_(nx, std::exp( cplx( 0.0,      k_point[1] * dy * ymax ) ), &fUp->point(1   , ymax),              1 );

        zcopy_(ny, &fUp->point(xmax-1, 1     ), fUp->local_x(), &fUp->point(0   , 1   ), fUp->local_x() );
        zscal_(ny, std::exp( cplx( 0.0, -1.0*k_point[0] * dx * xmax ) ), &fUp->point(0   , 1   ), fUp->local_x() );

        zcopy_(ny, &fUp->point(1     , 1     ), fUp->local_x(), &fUp->point(xmax, 1   ), fUp->local_x() );
        zscal_(ny, std::exp( cplx( 0.0,      k_point[0] * dx * xmax ) ), &fUp->point(xmax, 1   ), fUp->local_x() );
    }
}

void FDTDCompUpdateFxnReal::applyBCNonPer(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
}

void FDTDCompUpdateFxnCplx::applyBCNonPer(pgrid_ptr fUp, std::array<double,3> & k_point, int nx, int ny, int nz, int xmax, int ymax, int zmin, int zmax, double & dx, double & dy, double & dz)
{
    fUp->transferDat();
}
void MLUpdateFxnReal::updateRadPol(cplx* den, cplx* denDeriv, real_grid_ptr P, int xx, int yy, int zz, int n0, int nf, cplx& ampPreFact, cplx& omg, double t)
{
    P->point( xx+1, yy+1, zz+1 ) = std::real( ampPreFact * std::exp(omg*t) * ( std::sqrt(den[n0]) ) );
    // P->point( xx+1, yy+1, zz+1 ) = std::real(omg * ampPreFact * (den[n0] - den[nf]) );
    // P->point( xx+1, yy+1, zz+1 ) = std::real( ampPreFact * std::exp(omg*t) * ( 0.5 * 1.0/std::sqrt(0.5 * (den[n0] - den[nf] + 1.0) ) * std::sqrt(0.5 * (denDeriv[n0] - denDeriv[nf]) ) + omg * std::sqrt(0.5 * (den[n0] - den[nf] + 1.0) ) ) );
}

void MLUpdateFxnCplx::updateRadPol(cplx* den, cplx* denDeriv, cplx_grid_ptr P, int xx, int yy, int zz, int n0, int nf, cplx& ampPreFact, cplx& omg, double t)
{
    P->point( xx+1, yy+1, zz+1 ) = std::real( ampPreFact * std::exp(omg*t) * ( std::sqrt(den[n0]) ) );
}

void MLUpdateFxnReal::addRadPol( double na, real_grid_ptr radPol, real_grid_ptr prevRadPol, real_grid_ptr P )
{
    daxpy_(P->size(), 1.0, &radPol->point(0,0,0), 1, &P->point(0,0,0), 1);
    // for(int yy = 0; yy < P->y(); yy++)
    // {
    //     daxpy_(P->x()*P->z(),      1.0,     &radPol->point(0,yy,0), 1, &P->point(0,yy,0), 1);
    //     // daxpy_(P->x(), -1.0*na/2.0, &prevRadPol->point(0,yy), 1, &P->point(0,yy), 1);
    // }
}

void MLUpdateFxnCplx::addRadPol( double na, cplx_grid_ptr radPol, cplx_grid_ptr prevRadPol, cplx_grid_ptr P )
{
    zaxpy_(P->size(), 1.0, &radPol->point(0,0,0), 1, &P->point(0,0,0), 1);
    // for(int yy = 0; yy < P->y(); yy++)
    // {
    //     zaxpy_(P->x()*P->z(),      1.0,     &radPol->point(0,yy,0), 1, &P->point(0,yy,0), 1);
    //     // zaxpy_(P->x(), -1.0*na/2.0, &prevRadPol->point(0,yy), 1, &P->point(0,yy), 1);
    // }
}

void MLUpdateFxnReal::addP(real_grid_ptr P, pgrid_ptr E, real_pgrid_ptr eps, double* tempScaled, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff)
{
    // average the P fields effects on the Electric field (TE Mode offset Polarization to be at grid points away from E fields, TM offs are set to 0)
    for(int jj = 0; jj < nz; ++jj)
    {
        for(int ii = 0; ii < ny; ++ii)
        {
            std::transform(&P->point(0   , ii+ystart     , jj     ), &P->point(0   , ii+ystart     , jj     )+nx, &eps->point(xmin, ymin+ii, zmin+jj), tempScaled, [](double P, double ep){return -0.5*P/ep; } );
            daxpy_(nx, 1.0, tempScaled, 1, &E->point(xmin, ymin+ii, zmin+jj), 1);
            std::transform(&P->point(xoff, ii+ystart+yoff, jj+zoff), &P->point(xoff, ii+ystart+yoff, jj+zoff)+nx, &eps->point(xmin+xoff, ymin+ii+yoff, zmin+jj+zoff), tempScaled, [](double P, double ep){return -0.5*P/ep; } );
            daxpy_(nx, 1.0, tempScaled, 1, &E->point(xmin, ymin+ii, zmin+jj), 1);
        }
    }
}

void MLUpdateFxnCplx::addP(cplx_grid_ptr P, pgrid_ptr E, real_pgrid_ptr eps, cplx* tempScaled, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff)
{
    // average the P fields effects on the Electric field (TE Mode offset Polarization to be at grid points away from E fields, TM offs are set to 0)
    for(int jj = 0; jj < nz; ++jj)
    {
        for(int ii = 0; ii < ny; ++ii)
        {
            std::transform(&P->point(0   , ii+ystart     , jj     ), &P->point(0   , ii+ystart     , jj     )+nx, &eps->point(xmin, ymin+ii, zmin+jj), tempScaled, [](cplx P, double ep){return -0.5*P/ep; } );
            zaxpy_(nx, 1.0, tempScaled, 1, &E->point(xmin, ymin+ii, zmin+jj), 1);
            std::transform(&P->point(xoff, ii+ystart+yoff, jj+zoff), &P->point(xoff, ii+ystart+yoff, jj+zoff)+nx, &eps->point(xmin+xoff, ymin+ii+yoff, zmin+jj+zoff), tempScaled, [](cplx P, double ep){return -0.5*P/ep; } );
            zaxpy_(nx, 1.0, tempScaled, 1, &E->point(xmin, ymin+ii, zmin+jj), 1);
        }
    }
}


void MLUpdateFxnReal::getE_TE(real_grid_ptr inField, cplx_grid_ptr outField, real_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff)
{
    for(int zz = 0; zz < nz-1; ++zz)
    {
        for(int yy = 0; yy < ny-1; ++yy)
        {
            std::transform(&inField->point(xmin, ymin+yy, zmin+zz),  &inField->point(xmin, ymin+yy,  zmin+zz)+nx-1, &inField->point(xmin+xoff, ymin+yy+yoff, zmin+zz+zoff), &outField->point(0,yy+ystart,zz), [](double& a, double& b) {return 0.5*a + 0.5*b;} );
        }
    }
}

void MLUpdateFxnCplx::getE_TE(cplx_grid_ptr inField, cplx_grid_ptr outField, cplx_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff)
{
    for(int zz = 0; zz < nz-1; ++zz)
    {
        for(int yy = 0; yy < ny-1; ++yy)
        {
            std::transform(&inField->point(xmin, ymin+yy, zmin+zz),  &inField->point(xmin, ymin+yy,  zmin+zz)+nx-1, &inField->point(xmin+xoff, ymin+yy+yoff, zmin+zz+zoff), &outField->point(0,yy+ystart,zz), [](cplx& a, cplx& b) {return 0.5*a + 0.5*b;} );
        }
    }
}

void MLUpdateFxnReal::getE_TM(real_grid_ptr inField, cplx_grid_ptr outField, real_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff)
{
    for(int zz = 0; zz < nz-1; ++zz)
    {
        for(int yy = 0; yy < ny-1; ++yy)
        {
            dcopy_(nx-1, &inField->point(xmin, ymin+yy, zmin+zz), 1, reinterpret_cast<double*>(&outField->point(0,yy+ystart,0) ), 2 );
            // daxpy_(nx-1, 1.0/3.0, &PField->point(xmin, yy+ymin), 1, reinterpret_cast<double*>(&outField->point(0,yy+ystart) ), 2 );
        }
    }
}

void MLUpdateFxnCplx::getE_TM(cplx_grid_ptr inField, cplx_grid_ptr outField, cplx_grid_ptr PField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart, int xoff, int yoff, int zoff)
{
    for(int zz = 0; zz < nz-1; ++zz)
    {
        for(int yy = 0; yy < ny-1; ++yy)
        {
            zcopy_(nx-1, &inField->point(xmin, ymin+yy, zmin+zz), 1, &outField->point(0,yy+ystart,0), 1 );
            // zaxpy_(nx-1, 1.0/3.0, &PField->point(xmin, yy+ymin), 1, &outField->point(0,yy+ystart), 1 );
        }
    }
}


void MLUpdateFxnReal::getE_Continuum(real_grid_ptr inField, real_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart)
{
    for(int zz = 0; zz < nz-1; ++zz)
        for(int yy = 0; yy < ny-1; ++yy)
            dcopy_(nx-1,  &inField->point(xmin, ymin+yy, zmin+zz), 1, &outField->point(0,yy+ystart,0), 1 );
}

void MLUpdateFxnCplx::getE_Continuum(cplx_grid_ptr inField, cplx_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart)
{
    for(int zz = 0; zz < nz-1; ++zz)
        for(int yy = 0; yy < ny-1; ++yy)
            zcopy_(nx-1, &inField->point(xmin, ymin+yy, zmin+zz), 1, &outField->point(0,yy+ystart,0), 1 );
}

void MLUpdateFxnCplx::transferE(pgrid_ptr inField, cplx_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart)
{
    for(int zz = 0; zz < nz; ++zz)
        for(int yy = 0; yy < ny; ++yy)
            zcopy_(nx, &inField->point(xmin, ymin+yy, zmin+zz), 1, &outField->point(0, yy+ystart, zz), 1 );
        // std::copy_n(&inField->point(xmin     , ymin+yy), nx, &outField->point(0,yy+ystart) );
}

void MLUpdateFxnReal::transferE(pgrid_ptr inField, real_grid_ptr outField, int xmin, int ymin, int zmin, int nx, int ny, int nz, int ystart)
{
    for(int zz = 0; zz < nz; ++zz)
        for(int yy = 0; yy < ny; ++yy)
            dcopy_(nx, &inField->point(xmin, ymin+yy, zmin+zz), 1, &outField->point(0, yy+ystart, zz), 1 );
    // std::copy_n(&inField->point(xmin     , ymin+yy), nx, &outField->point(0,yy+ystart) );
}

void MLUpdateFxnReal::getE05_Continuum(real_grid_ptr half, real_grid_ptr cur, real_grid_ptr prev)
{
    std::transform(cur->data(), cur->data()+cur->size(), prev->data(), half->data(), [](double& a, double& b){return 0.5*a+0.5*b;});
}

void MLUpdateFxnCplx::getE05_Continuum(cplx_grid_ptr half, cplx_grid_ptr cur, cplx_grid_ptr prev)
{
    std::transform(cur->data(), cur->data()+cur->size(), prev->data(), half->data(), [](cplx& a, cplx& b){return 0.5*a+0.5*b;});
}

void MLUpdateFxnReal::getE05(cplx_grid_ptr half, cplx_grid_ptr cur, cplx_grid_ptr prev)
{
    std::transform(cur->data(), cur->data()+cur->size(), prev->data(), half->data(), [](cplx& a, cplx& b){return 0.5*a+0.5*b;});
}

void MLUpdateFxnCplx::getE05(cplx_grid_ptr half, cplx_grid_ptr cur, cplx_grid_ptr prev)
{
    std::transform(cur->data(), cur->data()+cur->size(), prev->data(), half->data(), [](cplx& a, cplx& b){return 0.5*a+0.5*b;});
}