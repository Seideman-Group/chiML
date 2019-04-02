/** @file SOURCE/parallelSourceOblique.cpp
 *  @brief Class to add a obliquely pulse to the FDTD grids
 *
 *  A class used to generate an obliquely incident current source on all processes
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug Only tested in 2D
 */

#include <SOURCE/parallelSourceOblique.hpp>


parallelSourceObliqueReal::parallelSourceObliqueReal(std::shared_ptr<mpiInterface> gridComm, std::vector<std::shared_ptr<Pulse>> pulse, real_pgrid_ptr grid, POLARIZATION pol, double dt, std::array<int,3> loc, std::array<int,3> sz, double phi, double theta) :
    parallelSourceObliqueBase<double>(gridComm, pulse, grid, pol, dt, loc, sz, phi, theta)
{}

void parallelSourceObliqueReal::addPul(double t)
{
    for(auto& pul : pulse_)
        for(auto& param : updateSrcParams_)
            *param.loc_ += param.scalefact_ * std::real(pul->pulse(t - param.t_off_) );
    grid_->transferDat();

}
parallelSourceObliqueCplx::parallelSourceObliqueCplx(std::shared_ptr<mpiInterface> gridComm, std::vector<std::shared_ptr<Pulse>> pulse, cplx_pgrid_ptr grid, POLARIZATION pol, double dt, std::array<int,3> loc, std::array<int,3> sz, double phi, double theta) :
    parallelSourceObliqueBase<cplx>(gridComm, pulse, grid, pol, dt, loc, sz, phi, theta)
{}

void parallelSourceObliqueCplx::addPul(double t)
{
    for(auto& pul : pulse_)
        for(auto& param : updateSrcParams_)
            *param.loc_ += param.scalefact_ * pul->pulse(t - param.t_off_);
    grid_->transferDat();
}
