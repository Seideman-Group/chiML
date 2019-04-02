/** @file UTIL/ml_consts.hpp
 *  @brief definitions for some physical constants
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#ifndef ML_CONST
#define ML_CONST

#include <cmath>

    const double SPEED_OF_LIGHT         = 2.9979245e8;
    const double EPS0                   = 1/(4*M_PI*1.0e-7*pow(SPEED_OF_LIGHT,2.0));
    const double ELEMENTARY_CHARGE      = 1.6021766208000000926392586608069679194921272585907390121132132243531032145256176590919494628906250000e-19;
    const double HBAR                   = 1.0545718001391127086220667148574871851002905804237980726847871017460777904741746884127674612624536721e-34;
    const double FINE_STRUCTURE_CONST   = 7.2973525663999998236430855058642919175326824188232421875e-3;

#endif
