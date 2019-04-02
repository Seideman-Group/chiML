/** @file ML/Hamiltonian.hpp
 *  @brief Class that stores and calculates the Hamiltonian for the quantum emitters
 *
 *  Class that stores and calculates the Hamiltonian for the quantum emitters
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef PARALLEL_ML_HAM
#define PARALLEL_ML_HAM

#include "BasisSet.hpp"
/**
 * @brief A class describing the Hamiltonian of a quantum emitter
 * @details Generates the field dependent Hamiltonian for M-L
 *
 */
class Hamiltonian
{
protected:
    //Hamiltonian and raising and lowering operators go here too
    int nstate_; //!< number of states in the Hamiltonian
    int denSz_; //!< size of the density matrix
    std::shared_ptr<BasisSet> basis_; //!< the basis set of the Hamiltonian
    std::vector<double> couplings_; //!< the scaling terms for the off diagonal elements

    std::vector<cplx> Ham_; //!< Hamiltonian at the current time
    std::vector<cplx> h0_; //!< the diagonal energy terms of the Hamiltonian

    std::vector<cplx> neg_x_expectation_; //!< mu_x operator
    std::vector<cplx> neg_y_expectation_; //!< mu_y operator
    std::vector<cplx> neg_z_expectation_; //!< mu_z operator

    std::vector<cplx> x_expectation_; //!< mu_x operator
    std::vector<cplx> y_expectation_; //!< mu_y operator
    std::vector<cplx> z_expectation_; //!< mu_z operator

    std::vector<cplx> neg_x_expectation_conj_; //!< mu_x operator
    std::vector<cplx> neg_z_expectation_conj_; //!< mu_z operator
    std::vector<cplx> neg_y_expectation_conj_; //!< mu_y operator

    std::function<void(int, cplx, cplx*, int, cplx*, int)> addX_; //!< Function that adds the X component of the dipole interaction
    std::function<void(int, cplx, cplx*, int, cplx*, int)> addY_; //!< Function that adds the Y component of the dipole interaction
    std::function<void(int, cplx, cplx*, int, cplx*, int)> addZ_; //!< Function that adds the Z component of the dipole interaction

public:


    /**
     * @brief Constructs a Hamiltonian
     *
     * @param states number of states in the Hamiltonian
     * @param h0 The energy of the states (Diagonal terms in the Hamiltonian)
     * @param couplings Terms control the strength of the coupling between states (Off diagnoal terms)
     * @param basis A BasisSet that describes the states of the system
     */
    Hamiltonian(int states, std::vector<double> &h0, std::vector<double> &couplings, BasisSet &basis);

    /**
     * @brief Returns a Hamiltonian matrix for given electric field components
     * @details Uses electric field compenets to generate a hamiltonian from using the dipole approximation and Long Wavelength approximation
     *
     * @param Ex: Field strength in the x direction (For real fields the imaginary competent has to be 0)
     * @param Ey: Field strength in the y direction (For real fields the imaginary component has to be 0)
     * @param Ez: Field strength in the z direction (For real fields the imaginary component has to be 0)
     * @return Hamiltonian for the given electric field strengths
     */
    cplx* getHam(const cplx Ex, const cplx Ey, const cplx Ez);

    /**
     * @ brief  Accessor function to nstate_
     *
     * @return the number of states
     */
    inline int nstate(){return nstate_;}
    /**
     * @ brief  Accessor function to h0_
     *
     * @return the energies of each state
     */
    inline std::vector<cplx> h0(){return h0_;}

    /**
     * @ brief  Accessor function to couplings_
     *
     * @return the off diagonal scaling terms
     */
    inline std::vector<double>& couplings(){return couplings_;}
    /**
     * @ brief  Accessor function to x_expectation_
     *
     * @return the mu_x vector
     */
    inline cplx* x_expectation(){return x_expectation_.data();}
    /**
     * @ brief  Accessor function to y_expectation_
     *
     * @return the mu_y vector
     */
    inline cplx* y_expectation(){return y_expectation_.data();}
    /**
     * @ brief  Accessor function to z_expectation_
     *
     * @return the mu_z vector
     */
    inline cplx* z_expectation(){return z_expectation_.data();}
};



#endif