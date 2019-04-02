/** @file DTC/toBitMap.cpp
 *  @brief Defines functions to take in FDTD Grids and Outputs it to a bmp file
 *
 *  Defines functions to take in FDTD Grids and Outputs it to a bmp file
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include "toBitMap.hpp"

int toGValue(double a)
{
    int a_ind = int( std::floor( std::abs(a) * 255 + 0.5) ) ;
    if(a_ind >= 255)
        return int( 255 * G_vals[255]);
    else
        return int( 255 * G_vals[a_ind]);
}

int toRValue(double a)
{
    int a_ind = int( std::floor( std::abs(a) * 255 + 0.5) ) ;
    if(a_ind >= 255)
        return int( 255 * R_vals[255]);
    else
        return int( 255 * R_vals[a_ind]);
}
int toBValue(double a)
{
    int a_ind = int( std::floor( std::abs(a) * 255 + 0.5) ) ;
    if(a_ind >= 255)
        return int( 255 * B_vals[255]);
    else
        return int( 255 * B_vals[a_ind]);
}


