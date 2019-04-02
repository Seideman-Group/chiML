/** @file UTIL/typedefs.hpp
 *  @brief definitions for common typedefs used throughout the code
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#ifndef PARALLEL_FDTD_TYPEDEFS
#define PARALLEL_FDTD_TYPEDEFS

#include <array>
#include <GRID/parallelGrid.hpp>

typedef std::vector<std::pair<std::array<int,6>, std::array<double,4> > > upLists;
typedef std::complex<double> cplx;

typedef std::shared_ptr<Grid<cplx>> cplx_grid_ptr;
typedef std::shared_ptr<Grid<double>> real_grid_ptr;
typedef std::shared_ptr<Grid<int>> int_grid_ptr;

typedef std::shared_ptr<parallelGrid<cplx>> cplx_pgrid_ptr;
typedef std::shared_ptr<parallelGrid<double>> real_pgrid_ptr;
typedef std::shared_ptr<parallelGrid<int>> int_pgrid_ptr;
#endif