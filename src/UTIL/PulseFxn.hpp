/** @file UTIL/PulseFxn.hpp
 *  @brief A set of distribution functions used to get generate pulse functions
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */
#ifndef PARALLEL_FDTD_PULSE_FXN
#define PARALLEL_FDTD_PULSE_FXN
#include <algorithm>

/**
 * @brief      Gaussian Pulse
 *
 * @param[in]  tt     current time
 * @param[in]  param  List of parameters to describe the function {center frequency, pulse width, pulse turn off time, peak center time, E0}
 *
 * @return     The value of the pulse at time tt
 */
inline const cplx gaussPulse(double tt, const std::vector<cplx>& param)
{
    return (tt <= std::real(param[2]) ) ? param[4] * exp( param[0]*tt - ( pow( (tt-param[3])/param[1], 2.0)/2.0 ) ) : 0.0;
}

/**
 * @brief      Continuous Pulse
 *
 * @param[in]  tt     current time
 * @param[in]  param  List of parameters to describe the function {center frequency, E0}
 *
 * @return     The value of the pulse at time tt
 */
inline const cplx contPulse(double tt, const std::vector<cplx>& param)
{
    return param[1] * std::exp(param[0]*tt);
}

/**
 * @brief      Ramped continuous pulse
 *
 * @param[in]  tt     current time
 * @param[in]  param  List of parameters to describe the function {center frequency, ramp up value, E0}
 *
 * @return     The value of the pulse at time tt
 */
inline const cplx rampContPulse(double tt, const std::vector<cplx>& param)
{
    return ( std::abs(param[2]*param[1]*tt) <= std::abs(param[2]) ) ? std::abs(param[1] * tt) * param[2] * std::exp(param[0] * tt) : param[2] * std::exp(param[0] * tt);
}

/**
 * @brief      BH4 pulse
 *
 * @param[in]  tt     current time
 * @param[in]  param  List of parameters to describe the function {center frequency, pulse width, peak itme, BH1, BH2, BH3, BH4, E0}
 *
 * @return     The value of the pulse at time tt
 */
inline const cplx bhPulse(double tt, const std::vector<cplx>& param)
{
    return (std::real(param[2]-param[1]/2.0) <= tt && tt <=  std::real(param[2]+param[1]/2.0) ) ? param[7] * std::exp( param[0]*(tt-param[2]) ) * ( param[3] + param[4] * cos( 2.0*M_PI*(tt-param[2]) / param[1] ) + param[5] * cos(4.0*M_PI*(tt-param[2]) / param[1]) + param[6] * cos(6.0*M_PI*(tt-param[2]) / param[1]) ) : 0.0;
}

/**
 * @brief      Rectangle wave approximation
 *
 * @param[in]  tt     current time
 * @param[in]  param  List of parameters to describe the function {pulse widht, center time, n (higher the value the more rectangular), center frequency, E0 }
 *
 * @return     The value of the pulse at time tt
 */
inline const cplx rectPulse(double tt, const std::vector<cplx>& param)
{
    // return ( tt < std::real(param[1]+param[0]/2.0) && tt > std::real(param[1]-param[0]/2.0) ) ? param[4]*std::exp( param[3]*(tt-param[1]-param[0]/2.0 ) ) : 0.0;
    return ( tt < std::real(param[1]+param[0]) && tt > std::real(param[1]-param[0]) ) ? param[4]/(std::pow( 2.0*(tt - param[1]) / param[0], param[2]) + 1.0) * ( std::exp(param[3]*(tt-param[1]) ) ) : 0.0;
    // return ( tt < std::real(param[1]+param[0]) && tt > std::real(param[1]-param[0]) ) ? std::exp(param[3]*(tt-param[1]) ) : 0.0;
}

/**
 * @brief      Creates a Ricker Pulse
 *
 * @param[in]  tt     current time
 * @param[in]  param  List of parameters to describe the function {center frequency, peak width, pulse turn off time. E0}
 *
 * @return     The value of the pulse at time tt
 */
inline const cplx rickerPulse(double tt, const std::vector<cplx>& param)
{
    return ( tt < std::real(param[2] * param[1]/param[0] ) ) ? param[3] * (1.0-2.0*std::pow(M_PI*(tt*param[0] - param[1]),2.0))*std::exp(-std::pow(M_PI*(tt*param[0] - param[1]),2.0)) : 0.0;
}
#endif