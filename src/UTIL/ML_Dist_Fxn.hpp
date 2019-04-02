/** @file UTIL/ML_Dist_Fxn.hpp
 *  @brief A set of distribution functions used to get weights for inhomogeneous broadening weights
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#ifndef PARALLEL_ML_DIST_FXN
#define PARALLEL_ML_DIST_FXN
#include <algorithm>
#include <cmath>


inline const double normDist(double xx, double cen, double width)
{
    return 1.0/std::sqrt( 2*M_PI*std::pow(width,2.0) ) * std::exp( -1.0 * std::pow(xx-cen, 2.0) / ( 2.0 * std::pow(width,2.0) ) ) ;
}

inline const double skewNormDist(double xx, double cen, double width, double skewFact)
{
    return 1.0/std::sqrt( 2*M_PI*std::pow(width,2.0) ) * std::exp( -1.0 * std::pow(xx-cen, 2.0) / ( 2.0 * std::pow(width,2.0) ) ) * 0.5 * ( 1.0 + erf(skewFact * (xx-cen) / (sqrt(2.0) * width) ) ) ;
}

inline const double logNormDist(double xx, double cen, double width)
{
    return (xx <= cen) ? 0.0 : 1.0/std::sqrt( 2*M_PI*std::pow(width*xx,2.0) ) * std::exp( -1.0 * std::pow(std::log(xx)-cen, 2.0) / ( 2.0 * std::pow(width,2.0) ) ) ;
}

inline const double deltaFxn(double xx, double cen)
{
    return (xx == cen) ? 1.0 : 0.0;
}

inline const double expPowDist(double xx, double cen, double width, double shapeFact)
{
    return shapeFact / (2 * width * std::tgamma(1.0/shapeFact) ) * std::exp(-1.0*std::pow( std::abs(xx-cen)/width, shapeFact ) );
}

inline const double genSkewNormDist(double xx, double cen, double width, double shapeFact)
{
    if(shapeFact == 0.0)
        return normDist(xx, cen, width);
    else
    {
        if( (shapeFact < 0 && xx < cen + width / shapeFact) || (shapeFact > 0 && xx > cen + width / shapeFact) )
            return 0.0;
        double y = -1.0/shapeFact * std::log(1 - shapeFact*( shapeFact*(xx-cen)/width ) );
        return normDist(y, 0.0, 1.0) / ( width - shapeFact*(xx-cen) );
    }
}

inline const double normDist(double xx, std::vector<int> params)
{
    return normDist(xx, params[0], params[1]);
}

inline const double logNormDist(double xx, std::vector<int> params)
{
    return normDist(xx, params[0], params[1]);
}

inline const double skewNormDist(double xx, std::vector<int> params)
{
    return skewNormDist(xx, params[0], params[1], params[2]);
}

inline const double genSkewNormDist(double xx, std::vector<int> params)
{
    return genSkewNormDist(xx, params[0], params[1], params[2]);
}

inline const double expPowDist(double xx, std::vector<int> params)
{
    return expPowDist(xx, params[0], params[1], params[2]);
}
#endif