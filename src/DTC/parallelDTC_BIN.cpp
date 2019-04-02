/** @file DTC/parallelDTC_BIN.cpp
 *  @brief Class that outputs FDTD field information into a binary file
 *
 *  Uses FDTD grids to convert the base data into the form that should be outputted and prints to a binary file.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <src/DTC/parallelDTC_BIN.hpp>

parallelDetectorBINReal::parallelDetectorBINReal(std::vector< std::pair< real_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseReal(grid, SI, loc, sz, type, timeInterval, a, I0, dt),
    outFile_(out_name),
    gridVals_(sz_[0], 0.0)
{
    //Set up the file and export the size and location array information
    std::ofstream dat(outFile_.c_str(), std::ios::out | std::ios::binary);
    dat.write(reinterpret_cast<const char*>(&sz_[0]), sz_.size()*sizeof(int) );
    dat.write(reinterpret_cast<const char*>(&loc_[0]), loc_.size()*sizeof(int) );
    dat.close();
}
void parallelDetectorBINReal::output(double t)
{
    // Import fields to Master
    for(auto & field : fields_)
        field->getField();
    // If not master return out
    if( !fields_[0]->master())
        return;
    // Open the file
    std::ofstream dat(outFile_, std::ios::app | std::ios::binary);
    // Calculate the time in the right units and write it to the file
    double tt = t*tConv_;
    dat.write(reinterpret_cast<char *>(&tt), sizeof(tt));
    // Go through the file by sizes and output field at each point
    for(int kk = 0; kk < sz_[2]; ++kk)
    {
        for(int jj = 0; jj < sz_[1]; ++jj)
        {
            // Use transforms to do multiple points at onece
            std::fill_n(gridVals_.begin(), gridVals_.size(), 0.0);
            for(auto & field : fields_)
            {
                outputCollectFunction_(&field->outGrid()->point(0,                 jj,                   kk                   ), &field->outGrid()->point(0                 ,jj                   ,kk                   )+sz_[0], gridVals_.data(), convFactor_/2.0);
                outputCollectFunction_(&field->outGrid()->point(field->offSet(0),jj+field->offSet(1),kk+field->offSet(2)), &field->outGrid()->point(field->offSet(0),jj+field->offSet(1),kk+field->offSet(2))+sz_[0], gridVals_.data(), convFactor_/2.0);
            }
            // Write contiguously
            dat.write(reinterpret_cast<char *>(&gridVals_[0]), gridVals_.size()*sizeof(double));
        }
    }
    dat.close();
}

parallelDetectorBINCplx::parallelDetectorBINCplx(std::vector< std::pair< cplx_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseCplx(grid, SI, loc, sz, type, timeInterval, a, I0, dt),
    outFile_(out_name),
    gridVals_(sz_[0], 0.0)
{
    //Set up the file and export the size and location array information
    std::ofstream dat(outFile_.c_str(), std::ios::out | std::ios::binary);
    dat.write(reinterpret_cast<const char*>(&sz_[0]), sz_.size()*sizeof(int) );
    dat.write(reinterpret_cast<const char*>(&loc_[0]), loc_.size()*sizeof(int) );
    dat.close();
}
void parallelDetectorBINCplx::output(double t)
{
    // Import fields to Master
    for(auto & field : fields_)
        field->getField();
    // If not master return out
    if( !fields_[0]->master())
        return;
    // Open the file
    std::ofstream dat(outFile_, std::ios::app | std::ios::binary);
    // Calculate the time in the right units and write it to the file
    double tt = t*tConv_;
    dat.write(reinterpret_cast<char *>(&tt), sizeof(tt));
    // Go through the file by sizes and output field at each point
    for(int kk = 0; kk < sz_[2]; ++kk)
    {
        for(int jj = 0; jj < sz_[1]; ++jj)
        {
            // Use transforms to do multiple points at onece
            std::fill_n(gridVals_.begin(), gridVals_.size(), 0.0);
            for(auto & field : fields_)
            {
                outputCollectFunction_(&field->outGrid()->point(0,                 jj,                   kk                   ), &field->outGrid()->point(0                 ,jj                   ,kk                   )+sz_[0], gridVals_.data(), convFactor_/2.0);
                outputCollectFunction_(&field->outGrid()->point(field->offSet(0),jj+field->offSet(1),kk+field->offSet(2)), &field->outGrid()->point(field->offSet(0),jj+field->offSet(1),kk+field->offSet(2))+sz_[0], gridVals_.data(), convFactor_/2.0);
            }
            // Write contiguously
            dat.write(reinterpret_cast<char *>(&gridVals_[0]), gridVals_.size()*sizeof(double));
        }
    }
    dat.close();
}
