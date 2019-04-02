/** @file ML/BasisSet.hpp
 *  @brief Class that stores values for the basis set, and does necessary operations
 *
 *  Stores and calculates values for spherical harmonic basis set
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef ML_BASIS
#define ML_BASIS

#include <gsl/gsl_sf.h>
#include <UTIL/typedefs.hpp>

/**
 * @brief      Class for describing the quantum emitter basis set.
 */
class BasisSet
{
protected:
    int nbasis_; //!< the size of the basis set
    std::vector<std::array<int,2>> basis_; //!< a vector of 2 integer arrays that describe a spherical harmonic basis set

public:
    /**
     * @brief Constructs an angular momentum basis set
     *
     * @param basis a vector of 2 integer arrays that describe a spherical harmonic basis set, the first element is l and the second element is m
     */
    BasisSet(std::vector<std::array<int,2>> basis);
    /**
     * @brief Constructs the dipole operator for the Hamiltonian
     * @details Constructs a dipole operator by using wigner 3j symbols to calculate the overlap integral between all states
     *
     * @param op Either x, y or z representing each component of the vector
     * @return mu_x, mu_y, mu_z
     */
    std::vector<cplx> expectationVals(DIRECTION op);
    /**
     * @brief Finds the integral for the product of three spherical harmonic functions using wigner 3j symbols
     *
     * @param fxn1 spherical harmonics function 1
     * @param fxn2 spherical harmonics function 2
     * @param fxn3 spherical harmonics function 3
     * @return Integral value from 0<theta <pi and 0 < phi < 2pi
     */
    double sphHarmonicIntegrator(std::array<int,2> fxn1, std::array<int,2> fxn2, std::array<int,2> fxn3);
    /**
     * @brief returns nbasis_
     */
    inline int nbasis() {return nbasis_;}
    /**
     * @brief returns the basis
     */
    inline std::vector<std::array<int,2>> basis(){return basis_;}
};



#endif