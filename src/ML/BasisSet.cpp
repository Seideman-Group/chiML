/** @file ML/BasisSet.cpp
 *  @brief Class that stores values for the basis set, and does necessary operations
 *
 *  Stores and calculates values for spherical harmonic basis set
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include "BasisSet.hpp"

BasisSet::BasisSet(std::vector<std::array<int,2>> basis) :
    nbasis_(basis.size()),
    basis_(basis)
{}

std::vector<cplx> BasisSet::expectationVals(DIRECTION op)
{
    std::vector<cplx> expecVals;
    expecVals.reserve(nbasis_*nbasis_);
    std::array<int,2> op0 = {{1, 0}};
    std::array<int,2> op1 = {{1, 1}};
    std::array<int,2> op2 = {{1,-1}};


    if(op == DIRECTION::X) // The x operator = 2pi/3(Y_1_-1 - Y_1_1)
        for(int ii = 0; ii < nbasis_; ii++)
            for(int jj = 0; jj < nbasis_; jj++)
                expecVals.push_back(sphHarmonicIntegrator(basis_[ii],op2,basis_[jj]) - sphHarmonicIntegrator(basis_[ii],op1,basis_[jj]));
    else if(op == DIRECTION::Y) // The x operator = 2pi/3(Y_1_-1 + Y_1_1)
        for(int ii = 0; ii < nbasis_; ii++)
            for(int jj = 0; jj < nbasis_; jj++)
                expecVals.push_back(cplx(0,sphHarmonicIntegrator(basis_[ii],op1,basis_[jj]) + sphHarmonicIntegrator(basis_[ii],op2,basis_[jj])));
    else if(op == DIRECTION::Z)
        for(int ii = 0; ii < nbasis_; ii++) // z operator = 2pi/3(Y_1_0)
            for(int jj = 0; jj < nbasis_; jj++)
                expecVals.push_back(sqrt(2)*sphHarmonicIntegrator(basis_[ii],op0,basis_[jj]));
    else
        throw std::logic_error("DIRECITON in BasisSet (for quantum emitters) is not defined.");

    transform(expecVals.begin(),expecVals.end(),expecVals.begin(),std::bind2nd(std::multiplies<cplx>(),sqrt(2*M_PI/3))) ; // get the correct prefactor
    return expecVals;
}

// Spherical harmonic integration can be calculated with wigner 3 j symbols
double BasisSet::sphHarmonicIntegrator(std::array<int,2> fxn1, std::array<int,2> fxn2, std::array<int,2> fxn3)
{
    //Analytical expression for spherical harmonics integral
    double sph = sqrt( (fxn1[0]*2+1) * (fxn2[0]*2+1) * (fxn3[0]*2+1) / (4*M_PI) );
    sph *= pow(-1.0,fxn1[1])*gsl_sf_coupling_3j(2*fxn1[0],2*fxn2[0],2*fxn3[0],0,0,0) * gsl_sf_coupling_3j(2*fxn1[0],2*fxn2[0],2*fxn3[0],-2*fxn1[1],2*fxn2[1],2*fxn3[1]);
    return sph;
}
