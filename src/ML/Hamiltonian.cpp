/** @file ML/Hamiltonian.cpp
 *  @brief Class that stores and calculates the Hamiltonian for the quantum emitters
 *
 *  Class that stores and calculates the Hamiltonian for the quantum emitters
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */
#include "Hamiltonian.hpp"

Hamiltonian::Hamiltonian(int states, std::vector<double> &h0, std::vector<double> &couplings, BasisSet &basis) :
    nstate_(states),
    denSz_(states*states),
    basis_(std::make_shared<BasisSet>(basis)),
    couplings_(couplings),
    Ham_(states*states,0.0),
    h0_(states*states,0.0),
    neg_x_expectation_(basis.expectationVals(DIRECTION::X)),
    neg_y_expectation_(basis.expectationVals(DIRECTION::Y)),
    neg_z_expectation_(basis.expectationVals(DIRECTION::Z)),
    x_expectation_(basis.expectationVals(DIRECTION::X)),
    y_expectation_(basis.expectationVals(DIRECTION::Y)),
    z_expectation_(basis.expectationVals(DIRECTION::Z)),
    neg_x_expectation_conj_( basis.expectationVals(DIRECTION::X) ),
    neg_y_expectation_conj_( basis.expectationVals(DIRECTION::Y) ),
    neg_z_expectation_conj_( basis.expectationVals(DIRECTION::Z) )
{
    for(int ii = 0; ii < h0.size(); ++ii)
        h0_[ii*h0.size()+ii] = h0[ii];

    for(int ii = 0; ii < couplings_.size(); ++ii)
    {
        x_expectation_[ii] *= couplings_[ii];
        y_expectation_[ii] *= couplings_[ii];
        z_expectation_[ii] *= couplings_[ii];

        neg_x_expectation_[ii] *= -1.0*couplings_[ii];
        neg_y_expectation_[ii] *= -1.0*couplings_[ii];
        neg_z_expectation_[ii] *= -1.0*couplings_[ii];

    }
    if(std::all_of(x_expectation_.begin(), x_expectation_.end(), [](cplx i){ return i == 0.0;} ) )
        addX_ = [](int n, cplx a, cplx* x, int incx, cplx* y, int incy){return;};
    else
        addX_ = [](int n, cplx a, cplx* x, int incx, cplx* y, int incy){zaxpy_(n, a, x, incx, y, incy); };

    if(std::all_of(y_expectation_.begin(), y_expectation_.end(), [](cplx i){ return i == 0.0;} ) )
        addY_ = [](int n, cplx a, cplx* x, int incx, cplx* y, int incy){return;};
    else
        addY_ = [](int n, cplx a, cplx* x, int incx, cplx* y, int incy){zaxpy_(n, a, x, incx, y, incy); };

    if(std::all_of(z_expectation_.begin(), z_expectation_.end(), [](cplx i){ return i == 0.0;} ) )
        addZ_ = [](int n, cplx a, cplx* x, int incx, cplx* y, int incy){return;};
    else
        addZ_ = [](int n, cplx a, cplx* x, int incx, cplx* y, int incy){zaxpy_(n, a, x, incx, y, incy); };

}

cplx* Hamiltonian::getHam(const cplx Ex, const cplx Ey, const cplx Ez)
{
    // H = \mu \cdot \vec{E}: scale the my expecetation values and add them to Ham
    zcopy_(denSz_, h0_.data(), 1, Ham_.data(), 1);
    if(Ex == 0.0 && Ey == 0.0 && Ez == 0.0 )
        return Ham_.data();
    addX_(denSz_, Ex, neg_x_expectation_.data(), 1, Ham_.data(), 1);
    addY_(denSz_, Ey, neg_y_expectation_.data(), 1, Ham_.data(), 1);
    addZ_(denSz_, Ez, neg_z_expectation_.data(), 1, Ham_.data(), 1);
    return Ham_.data();
}
