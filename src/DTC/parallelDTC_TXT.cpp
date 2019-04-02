/** @file DTC/parallelDTC_TXT.cpp
 *  @brief Class that outputs FDTD field information into a text file
 *
 *  Uses FDTD grids to convert the base data into the form that should be outputted and prints
 *  it to a text file.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <src/DTC/parallelDTC_TXT.hpp>

parallelDetectorTXTReal::parallelDetectorTXTReal(std::vector< std::pair< real_pgrid_ptr, std::array<int,3> > > grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseReal(grid, SI, loc, sz, type, timeInterval, a, I0, dt),
    outFile_(out_name)
{
    // Construct output file stream
    outFileStream_ = std::make_shared<std::ofstream>();
    if(fields_[0]->master())
    {
        outFileStream_->open(outFile_);
        *outFileStream_ << "# time\tx\ty\tz\tfield" << std::endl;
    }
}
void parallelDetectorTXTReal::output(double t)
{
    // Import fields from outside processes
    for(auto & field :fields_)
        field->getField();
    if(fields_[0]->master())
    {
        double point = 0.0;
        // output time/location
        *outFileStream_ << std::setprecision(6) << t*tConv_ << "\t" << realSpaceLoc_[0] << "\t" << realSpaceLoc_[1] << "\t" << realSpaceLoc_[2];// << "\n";
        for(int jj = 0; jj < fields_[0]->outGrid()->y(); ++jj)
        {
            for(int kk = 0; kk < fields_[0]->outGrid()->z(); ++kk)
            {
                for(int ii = 0; ii < fields_[0]->outGrid()->x(); ++ii)
                {
                    // Calculate and output point for all fields
                    point = 0.0;
                    for(auto & field :fields_)
                    {
                        outputCollectFunction_(&field->outGrid()->point(ii                 ,jj                 ,kk                 ), &field->outGrid()->point(ii                 ,jj                 ,kk                 )+1, &point, convFactor_/2.0);
                        outputCollectFunction_(&field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2)), &field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2))+1, &point, convFactor_/2.0);
                    }
                    *outFileStream_ << "\t" << std::setw(24) << std::setprecision(18) << point;
                    // Format files properly
                }
            }
        }
        *outFileStream_ << '\n';
    }
}
parallelDetectorTXTCplx::parallelDetectorTXTCplx(std::vector< std::pair< cplx_pgrid_ptr, std::array<int,3> > > grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, std::string out_name, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseCplx(grid, SI, loc, sz, type, timeInterval, a, I0, dt),
    outFile_(out_name)
{
    // Construct output file stream
    outFileStream_ = std::make_shared<std::ofstream>();
    if(fields_[0]->master())
    {
        outFileStream_->open(outFile_);
        *outFileStream_ << "# time\tx\ty\tz\tfield" << std::endl;
    }
}
void parallelDetectorTXTCplx::output(double t)
{
    // Import fields from outside processes
    for(auto & field :fields_)
        field->getField();
    if(fields_[0]->master())
    {
        cplx point;
        // output time/location
        *outFileStream_ << t*tConv_ << "\t" << realSpaceLoc_[0] << "\t" << realSpaceLoc_[1] << "\t" << realSpaceLoc_[2];// << "\n";
        for(int jj = 0; jj < fields_[0]->outGrid()->y(); ++jj)
        {
            for(int kk = 0; kk < fields_[0]->outGrid()->z(); ++kk)
            {
                for(int ii = 0; ii < fields_[0]->outGrid()->x(); ++ii)
                {
                    // Calculate and output point for all fields
                    point = 0.0;
                    for(auto & field :fields_)
                    {
                        outputCollectFunction_(&field->outGrid()->point(ii                 ,jj                 ,kk                 ), &field->outGrid()->point(ii                 ,jj                 ,kk                 )+1, &point, convFactor_/2.0);
                        outputCollectFunction_(&field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2)), &field->outGrid()->point(ii+field->offSet(0),jj+field->offSet(1),kk+field->offSet(2))+1, &point, convFactor_/2.0);
                    }
                    *outFileStream_ << "\t" <<  std::real(point);
                }
            }
            // Format files properly
        }
        *outFileStream_ << "\n";
    }
}
