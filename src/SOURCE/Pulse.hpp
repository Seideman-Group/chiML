/** @file SOURCE/Pulse.hpp
 *  @brief Calculates the pulse and stores all necessary pulse parameters
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_PULSE
#define FDTD_PULSE

#include <UTIL/typedefs.hpp>
#include <UTIL/PulseFxn.hpp>

class Pulse
{
protected:
    const std::vector<cplx> param_; //!< pulse function parameters
    double dt_; //!< time step
public:
    std::function<const cplx(double, double)> pulse_;
    /**
     * @brief      Constructor
     *
     * @param[in]  pulse  The pulse function base
     * @param[in]  param  The parameter list for the function
     * @param[in]  dt     time step
     */
    Pulse(std::function<const cplx(double, const std::vector<cplx>&)> pulse, std::vector<cplx> param, double dt);

    /**
     * @brief  Accessor function for param_
     *
     * @return the parameters of the pulse
     */
    inline std::vector<cplx> param() {return param_;}


    /**
     * @brief      Calculates the pulse value at time tt and distance rr
     *
     * @param[in]  tt    time of pulse calculation
     * @param[in]  rr    Distance from initial surface
     *
     * @return     Value of the pulse
     */
    inline const cplx pulse(double tt, double rr) { return pulse_(tt, rr); }

    /**
     * @brief      Calculates the pulse value at time tt and distance 0
     *
     * @param[in]  tt    time of pulse calculation
     *
     * @return     Value of the pulse
     */
    inline const cplx pulse(double tt) { return pulse_(tt, 0.0); }

};

#endif