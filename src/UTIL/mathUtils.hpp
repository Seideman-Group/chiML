/** @file UTIL/mathUtils.hpp
 *  @brief definitions for mathematical helper functions
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @author Joshua E. Szekely (jeszekely)
 *  @bug No known bugs
 */
#ifndef FDTD_MATHUTILS
#define FDTD_MATHUTILS

#include <type_traits>
#include <stdexcept>
#include <cassert>

template <typename T> T conj(const T& a) { throw std::logic_error("conj");}

template<> inline double conj<double> (const double& a) { return a; }

template<> inline cplx conj(const cplx& a) { return std::conj(a); }


#endif