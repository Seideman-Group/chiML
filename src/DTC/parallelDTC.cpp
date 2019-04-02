/** @file DTC/parallelDTC.cpp
 *  @brief Parent Class for all detectors
 *
 *  This contains the prototypes for the field detector classes
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <DTC/parallelDTC.hpp>

parallelDetectorBaseReal::parallelDetectorBaseReal(std::vector<std::pair< real_pgrid_ptr, std::array<int,3> > > gridsAndOffSets, bool SI, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBase( gridsAndOffSets, SI, loc, sz, type, timeInterval, a, I0, dt)
{
    real_pgrid_ptr gridBase = std::get<0>(gridsAndOffSets[0]);
    for(auto& go : gridsAndOffSets)
    {
        real_pgrid_ptr grid = std::get<0>(go);
        // Check if all the grids are the same size and then construct a storage object for it.
        if( grid->d().size() == grid->d().size() && gridBase->dx() == grid->dx() && gridBase->dy() == grid->dy() && gridBase->dz() == grid->dz() )
            fields_.push_back(std::make_shared<parallelStorageDTCReal>(grid, loc, sz, std::get<1>(go) ) );
        else
            throw std::logic_error("The step sizes of all the grids for a parallel dtc are not the same.");
    }
}
parallelDetectorBaseCplx::parallelDetectorBaseCplx(std::vector<std::pair< cplx_pgrid_ptr, std::array<int,3> > > gridsAndOffSets, bool SI, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBase(gridsAndOffSets, SI, loc, sz, type, timeInterval, a, I0, dt)
{
    cplx_pgrid_ptr gridBase = std::get<0>(gridsAndOffSets[0]);
    for(auto& go : gridsAndOffSets)
    {
        cplx_pgrid_ptr grid = std::get<0>(go);
        // Check if all the grids are the same size and then construct a storage object for it.
        if( grid->d().size() == grid->d().size() && gridBase->dx() == grid->dx() && gridBase->dy() == grid->dy() && gridBase->dz() == grid->dz() )
            fields_.push_back(std::make_shared<parallelStorageDTCCplx>(grid, loc, sz, std::get<1>(go) ) );
        else
            throw std::logic_error("The step sizes of all the grids for a parallel dtc are not the same.");
    }
}