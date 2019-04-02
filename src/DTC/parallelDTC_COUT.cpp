/** @file DTC/parallelDTC_COUT.cpp
 *  @brief Class that outputs FDTD field information to the console
 *
 *  Uses FDTD grids to convert the base data into the form that should be outputted and
 *  prints to the console.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */
#include <src/DTC/parallelDTC_COUT.hpp>

parallelDetectorCOUTReal::parallelDetectorCOUTReal(std::vector< std::pair< real_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseReal(grid, SI, loc, sz, type, timeInterval, a, I0, dt)
{}

void parallelDetectorCOUTReal::output(double t)
{
    // import fields from other processors
    fields_[0]->getField();
    if( !fields_[0]->master() )
        return;
    // Output the time and location
    std::cout << t*tConv_ << "\t" << realSpaceLoc_[0] << "\t" << realSpaceLoc_[1] << '\t' << realSpaceLoc_[2] << '\t' << std::endl;
    double point = 0.0;
    // Loop over all points
    for(int kk = fields_[0]->outGrid()->z()-1; kk >= 0; --kk )
    {
        for(int jj = fields_[0]->outGrid()->y()-1; jj >= 0; --jj)
        {
            for(int ii = 0; ii < fields_[0]->outGrid()->x(); ++ii)
            {
                // Calculate point and output it
                point = 0.0;
                for(auto & field :fields_)
                {
                    outputCollectFunction_(&field->outGrid()->point(ii                 ,jj                 ,kk                 ), &field->outGrid()->point(ii                 ,jj                 ,kk                 )+1, &point, convFactor_/2.0);
                    outputCollectFunction_(&field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2)), &field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2))+1, &point, convFactor_/2.0);
                }
                std::cout << "\t" << point;
            }
            std::cout << std::endl;
            // Format the output
        }
    }
}

parallelDetectorCOUTCplx::parallelDetectorCOUTCplx(std::vector< std::pair< cplx_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseCplx(grid, SI, loc, sz, type, timeInterval, a, I0, dt)
{}

void parallelDetectorCOUTCplx::output(double t)
{
    // import fields from other processors
    fields_[0]->getField();
    if( !fields_[0]->master() )
        return;
    // Output the time and location
    std::cout << t*tConv_ << "\t" << realSpaceLoc_[0] << "\t" << realSpaceLoc_[1] << '\t' << realSpaceLoc_[2] << std::endl;
    cplx point = 0.0;
    // Loop over all points
    for(int kk = fields_[0]->outGrid()->z()-1; kk >= 0; --kk )
    {
        for(int jj = fields_[0]->outGrid()->y()-1; jj >= 0; --jj)
        {
            for(int ii = 0; ii < fields_[0]->outGrid()->x(); ++ii)
            {
                // Calculate point and output it
                point = 0.0;
                for(auto & field :fields_)
                {
                    outputCollectFunction_(&field->outGrid()->point(ii                 ,jj                 ,kk                 ), &field->outGrid()->point(ii                 ,jj                 ,kk                 )+1, &point, convFactor_/2.0);
                    outputCollectFunction_(&field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2)), &field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2))+1, &point, convFactor_/2.0);
                }
                // Format the output
                std::cout << "\t" << point;
            }
            std::cout << std::endl;
        }
    }
}