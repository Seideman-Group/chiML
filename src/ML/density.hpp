/** @file ML/density.hpp
 *  @brief Class that stores the location and values of the density matrix for necessary time steps
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef ML_PARALLEL_DENSITY
#define ML_PARALLEL_DENSITY

#include <UTIL/typedefs.hpp>
/**
 * @brief Density Matrix class
 * @details Store information of the individual point density matrices
 */
class Density
{
protected:
    //Hamiltonian and raising and lowering operators go here too
    int ind_; //!< index of the location in all the grids
    std::array<int, 3> loc_; //!< location of the density matrix
public:
    std::vector<cplx> density_; //!< vector representing the density matrix  of the cell.
    std::vector<cplx> density_deriv_n_;         //!< vector representing the density time derivative matrix  of the cell at time n.
    std::vector<cplx> density_deriv_n_minus_1_; //!< vector representing the density time derivative matrix  of the cell at time n - 1.
    std::vector<cplx> density_deriv_n_minus_2_; //!< vector representing the density time derivative matrix  of the cell at time n - 2.
    std::vector<cplx> density_deriv_n_minus_3_; //!< vector representing the density time derivative matrix  of the cell at time n - 3.

    /**
     * @brief Constructor
     *
     * @param loc  the location in grid points of the density matrix
     * @param nstates number of states in the system
     */
    inline Density(std::array<int,3> loc, int ind, int nstates) :
        ind_(ind),
        loc_(loc),
        density_(nstates*nstates,0.0),
        density_deriv_n_(nstates*nstates,0.0),
        density_deriv_n_minus_1_(nstates*nstates,0.0),
        density_deriv_n_minus_2_(nstates*nstates,0.0),
        density_deriv_n_minus_3_(nstates*nstates,0.0)
    {};
    /**
     * @brief Initializes the Density Matrix to be 1 in the ground state
     */
    inline void initializeDensity()
    {
        density_[0] = 1.0;
    }

    /**
     * @brief      Initialize the density to a certain value
     *
     * @param[in]  den   The density initialization value
     */
    inline void initializeDensity(double den)
    {
        density_[0] = cplx(den,0.0);
    }

    /**
     * @brief      Update the density matrix and previous density matrix values forward in time
     */
    inline void moveDensity()
    {
        zcopy_(density_deriv_n_minus_2_.size(), density_deriv_n_minus_2_.data(), 1, density_deriv_n_minus_3_.data(), 1);
        zcopy_(density_deriv_n_minus_1_.size(), density_deriv_n_minus_1_.data(), 1, density_deriv_n_minus_2_.data(), 1);
        zcopy_(density_deriv_n_        .size(), density_deriv_n_        .data(), 1, density_deriv_n_minus_1_.data(), 1);
    }


    /**
     * @return the density matrix
     */
    inline cplx* density(){return density_.data();}

    /**
     * @brief      return reference to density matrix storage
     *
     * @return     density_
     */
    inline std::vector<cplx>& density_vec(){return density_;}
    // inline std::vector<cplx> &density_ref(){return &density_;}
    /**
     * @brief  returns the value of the x component of the density matrix's location
     *
     * @return the x location grid point of the density matrix
     */
    inline int& x(){return loc_[0];}
    /**
     * @brief  returns the value of the y component of the density matrix's location
     *
     * @return the y location grid point of the density matrix
     */
    inline int& y(){return loc_[1];}
    /**
     * @brief  returns the value of the z component of the density matrix's location
     *
     * @return the z location grid point of the density matrix
     */
    inline int& z(){return loc_[2];}

    /**
     * @brief      returns the index of the density matrix's location in the grids
     *
     * @return     ind_
     */
    inline int& ind(){return ind_;}

    /**
     * @brief  returns the location of the density matrix
     *
     * @return the location of the density matrix
     */
    inline std::array<int,3> loc() {return loc_;}
};

#endif