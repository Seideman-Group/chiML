/** @file SOURCE/Pulse.cpp
 *  @brief Calculates the pulse and stores all necessary pulse parameters
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include "Pulse.hpp"
#include <iostream>

Pulse::Pulse(std::function<const cplx(double, const std::vector<cplx>&)> pulse, std::vector<cplx> param, double dt) :
    param_(param),
    dt_(dt)
{
    pulse_ = [=](double tt, double rr){return pulse(tt-rr, param_);};
};