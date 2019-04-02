/** @file DTC/parallelDTC_BMP.cpp
 *  @brief Class that outputs FDTD field information into a bitmap file
 *
 *  Uses FDTD grids to convert the base data into the form that should be outputted and
 *  prints to a bitmap file and potentially prints the same data into a text file in
 *  either coordinate or matrix form.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include <src/DTC/parallelDTC_BMP.hpp>

parallelDetectorBMPReal::parallelDetectorBMPReal(std::vector< std::pair< real_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, bool outputAvg, double t_start, double t_end, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseReal(grid, SI, loc, sz, type, timeInterval, a, I0, dt),
    outputAvg_(outputAvg),
    t_start_(t_start),
    t_end_(t_end),
    outType_(txtType),
    dt_(dt),
    outFile_(out_name)
{
    if(outputAvg_)
        convFactor_ *= timeInterval;
    if(std::min_element(sz_.begin(), sz_.end(), std::less_equal<int>()) - sz_.begin() == 0 )
    {
        add_k_ = {{ 1, 0, 0 }};
        corInds_ = {{ 2, 1, 0 }};
        filePrefix_ = "/x_";
    }
    else if(std::min_element(sz_.begin(), sz_.end(), std::less_equal<int>()) - sz_.begin() == 1 )
    {
        add_k_ = {{ 0, 1, 0 }};
        corInds_ = {{ 0, 2, 1 }};
        filePrefix_ = "/y_";
    }
    else
    {
        add_k_ = {{ 0, 0, 1 }};
        corInds_ = {{ 1, 0, 2 }};
        filePrefix_ = "/z_";
    }
    // set the location, and reset the output file name
    loc_ = {{0,0,0}};
    outFile_ = outFile_.substr(0,outFile_.length()-4);
    // Based on what is being outputted set the outOpp_ function and PLOTTYPE
    if(fxn == GRIDOUTFXN::REAL)
    {
        outOpp_ = [](double in){return in;};
        funcComp_ = [](double a, double b)->bool{return std::real(a) <= std::real(b);};
        pType_  = PLOTTYPE::REAL;
    }
    else if(fxn == GRIDOUTFXN::IMAG)
    {
        throw std::logic_error("This is a real field, there is no imaginary part to return");
    }
    else if(fxn == GRIDOUTFXN::MAG)
    {
        // Even if detectors are not power detectors this type needs to collect power
        outputCollectFunction_ = pwrOutputFunction<double>;
        funcComp_ = [](double a, double b)->bool{return std::abs(a) <= std::abs(b);};
        outOpp_ = [](double in){return (in < 0) ? 0.0 : std::sqrt(in) ; };
        pType_  = PLOTTYPE::MAG;
    }
    else if(fxn == GRIDOUTFXN::POW)
    {
        // Even if detectors are not power detectors this type needs to collect power
        outputCollectFunction_ = pwrOutputFunction<double>;
        funcComp_ = [](double a, double b)->bool{return std::abs(a) <= std::abs(b);};
        outOpp_ = [](double in){return in;};
        pType_  = PLOTTYPE::POW;
    }
    else if(fxn == GRIDOUTFXN::LNPOW)
    {
        // Even if detectors are not power detectors this type needs to collect power
        outputCollectFunction_ = pwrOutputFunction<double>;
        funcComp_ = [](double a, double b)->bool{return std::abs(a) <= std::abs(b);};
        outOpp_ = [](double in){return (std::abs( in ) <= 0) ? -100 : std::log( std::abs( in) );};
        pType_  = PLOTTYPE::POW;
    }
    else
        throw std::logic_error("fxn in BMP detector is undefined in the enum.hpp file.");
    collectedGrid_ = std::make_shared<Grid<double>>( sz, std::get<0>(grid[0])->d() );
    for(int ii = 0; ii < grid.size(); ++ii)
        gridBMP_.push_back( std::make_shared<Grid<double>>(sz_, std::get<0>(grid[ii])->d() ) );
}
void parallelDetectorBMPReal::output(double t)
{
    if( t < t_start_ || t > t_end_)
        return;
    // Collect all the fields
    for(auto & field : fields_)
        field->getField();
    // if master output the grids to a file
    if(fields_[0]->master())
    {
        for(int ff = 0; ff < fields_.size(); ++ff)
        {
            for(int kk = 0; kk < sz_[2]; ++kk)
            {
                for(int jj = 0; jj < sz_[1]; ++jj)
                {
                    std::transform( &fields_[ff]->outGrid()->point(0, jj, kk), &fields_[ff]->outGrid()->point(0, jj, kk)+sz_[0], &fields_[ff]->outGrid()->point( fields_[ff]->offSet(0),jj+fields_[ff]->offSet(1),kk+fields_[ff]->offSet(2) ), &gridBMP_[ff]->point(0, jj, kk), [](double a, double b){return (a + b)/2.0;});
                }
            }
        }
        if(!outputAvg_)
            std::fill_n(collectedGrid_->data(), collectedGrid_->size(), 0.0);
        for(auto field : gridBMP_)
            outputCollectFunction_(field->data(), field->data()+field->size(), collectedGrid_->data(), convFactor_);
        if(!outputAvg_)
        {
            GridToBitMap (collectedGrid_, outFile_ + "t_" + std::to_string(t)  + filePrefix_, corInds_, add_k_, outOpp_, funcComp_);
            switch(outType_)
            {
                case GRIDOUTTYPE::BOX:
                    collectedGrid_->gridOutBox (outFile_ + "t_" + std::to_string(t) + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::LIST:
                    collectedGrid_->gridOutList(outFile_ + "t_" + std::to_string(t) + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::NONE:
                    break;
                default:
                    throw std::logic_error("Did you go and make a new output type without telling output_field in DetectorBMP.hpp? Well now you hurt its feelings and it is insisting on giving you an error, why don't you tell it about this new development and try again?");
                    break;
            }
        }
    }
}

void parallelDetectorBMPReal::toFile()
{
    if(fields_[0]->master())
    {
        if(outputAvg_)
        {
            GridToBitMap (collectedGrid_, outFile_ + filePrefix_, corInds_, add_k_, outOpp_, funcComp_);
            switch(outType_)
            {
                case GRIDOUTTYPE::BOX:
                    collectedGrid_->gridOutBox (outFile_ + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::LIST:
                    collectedGrid_->gridOutList(outFile_ + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::NONE:
                    break;
                default:
                    throw std::logic_error("Did you go and make a new output type without telling output_field in DetectorBMP.hpp? Well now you hurt its feelings and it is insisting on giving you an error, why don't you tell it about this new development and try again?");
                    break;
            }
        }
    }
}

parallelDetectorBMPCplx::parallelDetectorBMPCplx(std::vector< std::pair< cplx_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, bool outputAvg, double t_start, double t_end, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double timeInterval, double a, double I0, double dt) :
    parallelDetectorBaseCplx(grid, SI, loc, sz, type, timeInterval, a, I0, dt),
    outputAvg_(outputAvg),
    t_start_(t_start),
    t_end_(t_end),
    outType_(txtType),
    dt_(dt),
    outFile_(out_name)
{
    if(outputAvg_)
        convFactor_ *= timeInterval;
    if(std::min_element(sz_.begin(), sz_.end(), std::less_equal<int>()) - sz_.begin() == 0 )
    {
        add_k_ = {{ 1, 0, 0 }};
        corInds_ = {{ 2, 1, 0 }};
        filePrefix_ = "/x_";
    }
    else if(std::min_element(sz_.begin(), sz_.end(), std::less_equal<int>()) - sz_.begin() == 1 )
    {
        add_k_ = {{ 0, 1, 0 }};
        corInds_ = {{ 0, 2, 1 }};
        filePrefix_ = "/y_";
    }
    else
    {
        add_k_ = {{ 0, 0, 1 }};
        corInds_ = {{ 1, 0, 2 }};
        filePrefix_ = "/z_";
    }
    // set the location, and reset the output file name
    loc_ = {{0,0,0}};
    outFile_ = outFile_.substr(0,outFile_.length()-4);
    // Based on what is being outputted set the outOpp_ function and PLOTTYPE
    if(fxn == GRIDOUTFXN::REAL)
    {
        funcComp_ = [](cplx a, cplx b)->bool{return std::real(a) <= std::real(b);};
        outOpp_ = [](cplx in){return static_cast<double>(std::real(in));};
        pType_  = PLOTTYPE::REAL;
    }
    else if(fxn == GRIDOUTFXN::IMAG)
    {
        funcComp_ = [](cplx a, cplx b)->bool{return std::imag(a) <= std::imag(b);};
        outOpp_ = [](cplx in){return static_cast<double>(std::imag(in));};
        pType_  = PLOTTYPE::IMAG;
    }
    else if(fxn == GRIDOUTFXN::MAG)
    {
        // Even if detectors are not power detectors this type needs to collect power
        outputCollectFunction_ = pwrOutputFunction<cplx>;
        funcComp_ = [](cplx a, cplx b)->bool{return std::abs(a) <= std::abs(b);};
        outOpp_ = [](cplx in){return static_cast<double>( std::abs( std::sqrt(in) ) ); };
        pType_  = PLOTTYPE::MAG;
    }
    else if(fxn == GRIDOUTFXN::POW)
    {
        // Even if detectors are not power detectors this type needs to collect power
        outputCollectFunction_ = pwrOutputFunction<cplx>;
        funcComp_ = [](cplx a, cplx b)->bool{return std::abs(a) <= std::abs(b);};
        outOpp_ = [](cplx in){return static_cast<double>( std::abs(in) );};
        pType_  = PLOTTYPE::POW;
    }
    else if(fxn == GRIDOUTFXN::LNPOW)
    {
        // Even if detectors are not power detectors this type needs to collect power
        outputCollectFunction_ = pwrOutputFunction<cplx>;
        funcComp_ = [](cplx a, cplx b)->bool{return std::abs(a) <= std::abs(b);};
        outOpp_ = [](cplx in){return ( std::abs(in) <= 0.0 ) ? -100 : static_cast<double>( std::log( std::abs(in) ) ); };
        pType_  = PLOTTYPE::LNPOW;
    }
    else
        throw std::logic_error("fxn in BMP detector is undefined in the enum.hpp file.");

    collectedGrid_ = std::make_shared<Grid<cplx>>( sz, std::get<0>(grid[0])->d() );
    for(int ii = 0; ii < grid.size(); ++ii)
        gridBMP_.push_back( std::make_shared<Grid<cplx>>(sz_, std::get<0>(grid[ii])->d() ) );
}

void parallelDetectorBMPCplx::output(double t)
{
    if( t < t_start_ || t > t_end_)
        return;
    // Collect all the fields
    for(auto & field : fields_)
        field->getField();
    // if master output the grids to a file
    if(fields_[0]->master())
    {
        for(int ff = 0; ff < fields_.size(); ++ff)
            for(int kk = 0; kk < sz_[2]; ++kk)
                for(int jj = 0; jj < sz_[1]; ++jj)
                    std::transform( &fields_[ff]->outGrid()->point(0, jj, kk), &fields_[ff]->outGrid()->point(0, jj, kk)+sz_[0], &fields_[ff]->outGrid()->point( fields_[ff]->offSet(0),jj+fields_[ff]->offSet(1),kk+fields_[ff]->offSet(2) ), &gridBMP_[ff]->point(0, jj, kk), [](cplx a, cplx b){return (a + b)/2.0;});
        for(auto field : gridBMP_)
            outputCollectFunction_(field->data(), field->data()+field->size(), collectedGrid_->data(), convFactor_);
        if(!outputAvg_)
        {
            GridToBitMap (collectedGrid_, outFile_ + "t_" + std::to_string(t)  + filePrefix_, corInds_, add_k_, outOpp_, funcComp_);
            switch(outType_)
            {
                case GRIDOUTTYPE::BOX:
                    collectedGrid_->gridOutBox (outFile_ + "t_" + std::to_string(t) + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::LIST:
                    collectedGrid_->gridOutList(outFile_ + "t_" + std::to_string(t) + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::NONE:
                    break;
                default:
                    throw std::logic_error("Did you go and make a new output type without telling output_field in DetectorBMP.hpp? Well now you hurt its feelings and it is insisting on giving you an error, why don't you tell it about this new development and try again?");
                    break;
            }
        }
    }
}

void parallelDetectorBMPCplx::toFile()
{
    if(fields_[0]->master())
    {
        if(outputAvg_)
        {
            GridToBitMap (collectedGrid_, outFile_ + filePrefix_, corInds_, add_k_, outOpp_, funcComp_);
            switch(outType_)
            {
                case GRIDOUTTYPE::BOX:
                    collectedGrid_->gridOutBox (outFile_ + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::LIST:
                    collectedGrid_->gridOutList(outFile_ + ".txt", loc_, sz_, outOpp_);
                    break;
                case GRIDOUTTYPE::NONE:
                    break;
                default:
                    throw std::logic_error("Did you go and make a new output type without telling output_field in DetectorBMP.hpp? Well now you hurt its feelings and it is insisting on giving you an error, why don't you tell it about this new development and try again?");
                    break;
            }
        }
    }
}