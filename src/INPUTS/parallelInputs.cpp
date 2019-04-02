/** @file INPUTS/parallelInputs.cpp
 *  @brief Translates the input file into a structure that can be used by parallelFDTDField
 *
 *  Takes in a boost property tree and converts that information into a structure used by parallelFDTDField
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @author Joshua E. Szekely (jeszekely)
 *  @bug No known bugs.
 */

#include <INPUTS/parallelInputs.hpp>
parallelProgramInputs::parallelProgramInputs(boost::property_tree::ptree IP,std::string fn) :
    // Initialize the general computational cell parameters
    periodic_     (IP.get<bool>("CompCell.PBC", false) ),
    pol_          (string2pol(IP.get<std::string>("CompCell.pol") ) ),
    filename_     (fn),
    res_          (IP.get<int>("CompCell.res", -1) ),
    courant_      (IP.get<double>("CompCell.courant", 0.5) ),
    a_            (IP.get<double>("CompCell.a",1e-7) ),
    tMax_         (IP.get<double>("CompCell.tLim") ),
    I0_           (IP.get<double>("CompCell.I0", a_ * EPS0 * SPEED_OF_LIGHT) ),
    cplxFields_   (IP.get<bool>("CompCell.cplxFields", false) ),
    saveFreqField_(false),
    size_         ( as_ptArr<double>( IP, "CompCell.size") ),
    k_point_      ( as_ptArr<double>( IP, "CompCell.k-point", 0.0) ),
    d_            ( as_ptArr<double>( IP, "CompCell.stepSize", 1.0/res_) ),
    dt_           (IP.get<double>("CompCell.dt", courant_ / std::sqrt( 1.0/(d_[0]*d_[0]) + 1.0/(d_[1]*d_[1]) + 1.0/(d_[2]*d_[2]) ) ) ),
    // Initialize the PML parameters
    pmlSigOptRat_ ( IP.get<double>("PML.sigOptRat", 1.0) ),
    pmlKappaMax_  ( IP.get<double>("PML.kappaMax", 1.0) ),
    pmlAMax_      ( IP.get<double>("PML.aMax",0.00) ),
    pmlMa_        ( IP.get<double>("PML.ma", 1.0) ),
    pmlM_         ( IP.get<double>("PML.m", 3.0) ),
    // Get values of planes to take slices in
    inputMapSlicesX_(as_vector<double>(IP, "CompCell.InputMaps_x") ),
    inputMapSlicesY_(as_vector<double>(IP, "CompCell.InputMaps_y") ),
    inputMapSlicesZ_(as_vector<double>(IP, "CompCell.InputMaps_z") ),
    // Initialize the Source lists
    srcPol_             ( IP.get_child("SourceList").size(), POLARIZATION::EX),
    srcFxn_             ( IP.get_child("SourceList").size() ),
    srcLoc_             ( IP.get_child("SourceList").size() ),
    srcSz_              ( IP.get_child("SourceList").size() ),
    srcPhi_             ( IP.get_child("SourceList").size(), 0.0),
    srcTheta_           ( IP.get_child("SourceList").size(), 90.0),
    srcPulShape_        ( IP.get_child("SourceList").size() ),
    srcEmax_            ( IP.get_child("SourceList").size() ),
    srcEllipticalKratio_( IP.get_child("SourceList").size(), 0.0),
    srcPsi_             ( IP.get_child("SourceList").size(), 0.0),
    // Initialize the TFSF lists
    tfsfLoc_             ( IP.get_child("TFSF").size() ),
    tfsfSize_            ( IP.get_child("TFSF").size() ),
    tfsfM_               ( IP.get_child("TFSF").size() ),
    tfsfTheta_           ( IP.get_child("TFSF").size(), 90.0) ,
    tfsfPhi_             ( IP.get_child("TFSF").size(), 0.0) ,
    tfsfPsi_             ( IP.get_child("TFSF").size(), 0.0) ,
    tfsfPulFxn_          ( IP.get_child("TFSF").size() ),
    tfsfPulShape_        ( IP.get_child("TFSF").size() ),
    tfsfEmax_            ( IP.get_child("TFSF").size() ),
    tfsfCircPol_         ( IP.get_child("TFSF").size(), POLARIZATION::EX),
    tfsfEllipticalKratio_( IP.get_child("TFSF").size(), 0.0),
    tfsfPMLThick_        ( IP.get_child("TFSF").size(), 0),
    tfsfPMLAMax_         ( IP.get_child("TFSF").size(), 0),
    tfsfPMLM_            ( IP.get_child("TFSF").size(), 0),
    tfsfPMLMa_           ( IP.get_child("TFSF").size(), 0),

    // Initialize the Detector lists
    dtcSI_           ( IP.get_child("DetectorList").size(), false),
    dtcClass_        ( IP.get_child("DetectorList").size(), DTCCLASS::COUT),
    dtcLoc_          ( IP.get_child("DetectorList").size() ),
    dtcSz_           ( IP.get_child("DetectorList").size() ),
    dtcName_         ( IP.get_child("DetectorList").size() ),
    dtcType_         ( IP.get_child("DetectorList").size(), DTCTYPE::EX ),
    dtcTimeInt_      ( IP.get_child("DetectorList").size(), 0.0),
    dtcOutBMPFxnType_( IP.get_child("DetectorList").size(), GRIDOUTFXN::REAL),
    dtcOutBMPOutType_( IP.get_child("DetectorList").size(), GRIDOUTTYPE::NONE),
    dtcFreqList_     ( IP.get_child("DetectorList").size() ),
    dtcTStart_       ( IP.get_child("DetectorList").size(), 0.0),
    dtcTEnd_         ( IP.get_child("DetectorList").size(), 0.0),
    dtcOutputAvg_    ( IP.get_child("DetectorList").size(), false),
    dtcOutputMaps_    ( IP.get_child("DetectorList").size(), false),
    // Initialize the flux lists
    fluxXOff_              ( IP.get_child("FluxList").size(), 0),
    fluxYOff_              ( IP.get_child("FluxList").size(), 0),
    fluxTimeInt_           ( IP.get_child("FluxList").size(), 0),
    fluxLoc_               ( IP.get_child("FluxList").size() ),
    fluxSz_                ( IP.get_child("FluxList").size() ),
    fluxWeight_            ( IP.get_child("FluxList").size(), 0.0),
    fluxName_              ( IP.get_child("FluxList").size() ),
    fluxFreqList_          ( IP.get_child("FluxList").size() ),
    fluxSI_                ( IP.get_child("FluxList").size(), false),
    fluxCrossSec_          ( IP.get_child("FluxList").size(), false),
    fluxSave_              ( IP.get_child("FluxList").size(), false),
    fluxLoad_              ( IP.get_child("FluxList").size(), false),
    fluxIncdFieldsFilename_( IP.get_child("FluxList").size() )
{
    if(d_[0] < 0 || d_[1] < 0 || d_[2] < 0)
        throw std::logic_error("Please define a positive step size.");
    if(dt_ > 1.0/std::sqrt( std::accumulate(d_.begin(), d_.end(), 0.0, [](double a, double b){ return a + 1.0/(b*b); }  ) ) )
        throw std::logic_error("Time step is larger than the stable time step for the calculation.");
    // Convert PML thicnknesses to grid point values
    std::array<double,3> pmlThickness = as_ptArr<double>( IP, "PML.thickness");
    for(int ii = 0; ii < 3; ++ii)
    {
        if(pmlThickness[ii]*2 > size_[ii])
            throw std::logic_error("PML size is larger than the cell size, this will lead to infinte fields.");
        pmlThickness_[ii] = find_pt(pmlThickness[ii], d_[ii]);
    }
    // If using PBC and not normal k point use complex fields
    if(periodic_)
        for(int kk = 0; kk < k_point_.size(); kk++)
            if(k_point_[kk] != 0)
                cplxFields_= true;

    int ii = 0;
    for (auto& iter : IP.get_child("SourceList") )
    {
        // What field is the source acting on (if L or R then its circularly polarized)
        srcPol_[ii]      = string2pol(iter.second.get<std::string>("pol"));
        // Initialize the Pulse parameters
        boost::property_tree::ptree& PulseList = iter.second.get_child("PulseList");
        std::vector<std::vector<cplx>> pulFxn_( PulseList.size() );
        std::vector<PLSSHAPE> pulShape_(PulseList.size(), PLSSHAPE::GAUSSIAN);
        std::vector<double> pulEmax_(PulseList.size(), 0.0);
        int pp = 0;
        for(auto& pul : PulseList )
        {
            // get the pulse shape
            pulShape_[pp] = string2prof(pul.second.get<std::string>("profile"));
            // get the Maximum pulse values
            pulEmax_[pp]  = pul.second.get<double>("Field_Intensity",1.0) * a_ * EPS0 * SPEED_OF_LIGHT / I0_;
            std::vector<cplx> fxn;
            // Input all pulse function parameters
            switch (pulShape_[pp])
            {
                case PLSSHAPE::GAUSSIAN:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(1.0/pul.second.get<double>("fwidth"));
                    fxn.push_back(pul.second.get<double>("cutoff") * fxn[1]);
                    fxn.push_back(pul.second.get<double>("t_0", std::real(fxn[1]*fxn[2]) ) );
                    fxn[2] += fxn[3];
                break;
                case PLSSHAPE::BH:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(pul.second.get<double>("tau", 1.0/pul.second.get<double>("fwidth") ) );
                    fxn.push_back(pul.second.get<double>("t_0", std::real( fxn[1]*fxn[2] ) ) );
                    fxn.push_back(pul.second.get<double>("BH1", 0.35875));
                    fxn.push_back(pul.second.get<double>("BH2", 0.48829));
                    fxn.push_back(pul.second.get<double>("BH3", 0.14128));
                    fxn.push_back(pul.second.get<double>("BH4", 0.01168));
                break;
                case PLSSHAPE::RECT :
                    fxn.push_back(pul.second.get<double>("tau"));
                    fxn.push_back(pul.second.get<double>("t_0"));
                    fxn.push_back(pul.second.get<double>("n", 30) * 2.0);
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                break;
                case PLSSHAPE::CONTINUOUS:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                break;
                case PLSSHAPE::RAMP_CONT:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(pul.second.get<double>("ramp_val"));
                break;
                case PLSSHAPE::RICKER:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(pul.second.get<double>("fwidth"));
                    fxn.push_back(pul.second.get<double>("cutoff"));
                break;
                default:
                    throw std::logic_error("This pulse shape is undefined in source ");
                break;
            }
            // Add the pulse function list to the correct vector
            pulFxn_[pp] = fxn;
            ++pp;
        }
        // set the source function lists here
        srcPulShape_[ii] = pulShape_;
        srcEmax_[ii] = pulEmax_;
        srcFxn_[ii] = pulFxn_;
        // If the light is elliptical what is the ratio between the sizes?
        srcEllipticalKratio_[ii] = iter.second.get<double>("ellpiticalKRat", 1.0);
        // Get the size of the source in real space
        std::array<double,3> tempSz = as_ptArr<double>(iter.second, "size");
        // Convert real space size to grid points
        for(int cc = 0; cc < 3; ++cc )
            srcSz_[ii][cc] = find_pt(tempSz[cc], d_[cc]) + 1;
        // What is the angle of incidence of the source (azimuthal?)
        srcPhi_[ii] = iter.second.get<double>("phi", 90);
        if( (srcPol_[ii] == POLARIZATION::R || srcPol_[ii] == POLARIZATION::L) && isamin_(srcSz_[ii].size(), srcSz_[ii].data(), 1)-1 == 0 )
            srcPsi_[ii] = M_PI * iter.second.get<double>("psi", 0) / 180.0;
        else if( (srcPol_[ii] == POLARIZATION::R || srcPol_[ii] == POLARIZATION::L) && isamin_(srcSz_[ii].size(), srcSz_[ii].data(), 1)-1 == 1 )
            srcPsi_[ii] = M_PI * iter.second.get<double>("psi", 0) / 180.0;
        else if( (srcPol_[ii] == POLARIZATION::R || srcPol_[ii] == POLARIZATION::L) && isamin_(srcSz_[ii].size(), srcSz_[ii].data(), 1)-1 == 2 )
            srcPsi_[ii] = M_PI * iter.second.get<double>("psi", 0) / 180.0;
        int i = 0;
        // Get the location of the source in number of grid points
        for(auto& loc : as_ptArr<double>(iter.second, "loc") )
        {
            if(loc + tempSz[i]/2.0 > size_[i]/2.0 || loc - tempSz[i]/2.0 < -1.0*size_[i]/2.0)
                throw std::logic_error("The source is at least partially outside the FDTD Cell");
            // Is it normal source?
            if(srcPhi_[ii] == 90 || srcPhi_[ii] == 180 || srcPhi_[ii] == 270 || srcPhi_[ii] == 0 )
            {
                // Yes give bottom, left, back corner
                srcLoc_[ii][i] = find_pt(loc + size_[i]/2.0 - tempSz[i]/2.0, d_[i]);
            }
            else
            {
                // no give center point
                srcLoc_[ii][i] = find_pt(loc + size_[i]/2.0, d_[i]);
            }
            ++i;
        }
        ++ii;
    }

    ii = 0;
    for (auto& iter : IP.get_child("TFSF"))
    {
        int i = 0;
        // Get the size of the TFSF surface in grid points
        for(auto& sz : as_ptArr<double>(iter.second, "size") )
        {
            tfsfSize_[ii][i] = find_pt(sz, d_[i]) + 1;// - find_pt(sz, d_[i]) % 2;
            ++i;
        }
        if( (size_[2] == 0 && tfsfSize_[ii][0] != 1 && tfsfSize_[ii][1] != 1) || (size_[2] != 0.0 && !std::any_of(tfsfSize_[ii].begin(), tfsfSize_[ii].end(), [](int a){return a == 1;} ) ) )
        {
            for(auto& sz : tfsfSize_[ii] )
                sz -= (sz-1)%2;
        }
        // if( std::all_of(tfsfSize_[ii].begin(), tfsfSize_[ii].end(), [](int a){return a > 1; } ) )
        // {
        //     std::cout << "Warning setting all TFSF to be odd sized"
        //     for(auto& sz : tfsfSize_[ii])
        // }

        i = 0;
        // Get the location of bottom, left, and back corner of the TFSF surface in grid points
        for(auto& loc : as_ptArr<double>(iter.second, "loc") )
        {
            if(size_[i] != 0.0)
                tfsfLoc_[ii][i] = find_pt(loc + size_[i]/2.0, d_[i]) - (tfsfSize_[ii][i] - (tfsfSize_[ii][i] % 2) ) / 2 ;
            else
                tfsfLoc_[ii][i] = 0;
            ++i;
        }
        // Is the TFSF surface outside of the FDTD cell region?
        for(int tt = 0; tt < 3; ++tt)
        {
            if( size_[tt] > 0.0 && (tfsfLoc_[ii][tt] + tfsfSize_[ii][tt]/2 > find_pt(size_[tt], d_[tt]) || tfsfLoc_[ii][tt] < 0) )
            {
                if(tfsfLoc_[ii][tt] + tfsfSize_[ii][tt]/2.0 != 1)
                    throw std::logic_error("A TFSF surface is outside the FDTD cell.");
            }
        }
        // Try to define angles from the m vector, otherwise define m from angles
        try
        {
            if(iter.second.get_child("m").size() == 0)
                throw std::logic_error("m's not defined going to define them based on the phi and theta values");
            tfsfM_[ii] = as_ptArr<int>(iter.second, "m");
            if( (tfsfM_[ii][0] == 0 && tfsfM_[ii][1] == 0 ) || ( tfsfM_[ii][2] == 0 && tfsfM_[ii][1] == 0 ) || ( tfsfM_[ii][0] ==0 && tfsfM_[ii][2] == 0 ) )
            {
                int mInd = -1;
                if(tfsfM_[ii][0] != 0)
                    mInd = 0;
                if(tfsfM_[ii][1] != 0)
                    mInd = 1;
                if(tfsfM_[ii][2] != 0)
                    mInd = 2;
                if(std::abs(tfsfM_[ii][mInd]) != 1)
                {
                    std::cout << "Warning m_ currently has one non-zero elements whose absolute value is not equal to 1, setting the absolute value to 1." << std::endl;
                    tfsfM_[ii][mInd] /= std::abs(tfsfM_[ii][mInd]);
                }
            }

            tfsfPhi_[ii] = (tfsfM_[ii][0] == 0) ? ( (tfsfM_[ii][1] > 0) ? M_PI/2.0 : -1.0*M_PI/2.0) : std::atan(tfsfM_[ii][1]*d_[0]/(tfsfM_[ii][0]*d_[1]) );
            if(tfsfM_[ii][0] < 0)
                tfsfPhi_[ii] += M_PI;
            else if(tfsfM_[ii][1] < 0)
                tfsfPhi_[ii] += 2.0*M_PI;
            tfsfTheta_[ii]  = (tfsfM_[ii][2] == 0) ? M_PI/2.0 : std::atan( std::sqrt( std::pow(tfsfM_[ii][1]/d_[1], 2.0) + std::pow(tfsfM_[ii][0]/d_[0], 2.0) ) / (tfsfM_[ii][2]/d_[2]) );
            if(tfsfM_[ii][2] < 0)
                tfsfTheta_[ii] += M_PI;

            tfsfPsi_[ii] = M_PI*iter.second.get<double>("psi" , 90.0)/180.0;
        }
        catch(std::exception& e)
        {
            tfsfM_[ii] = {0,0,0};
            // Get the polar angle of the k vector of the TFSF pulse
            tfsfTheta_[ii] = M_PI*iter.second.get<double>("theta" , 90.0)/180.0;
            if(tfsfTheta_[ii] < 0 || tfsfTheta_[ii] > M_PI)
                throw std::logic_error("The theta value for the TFSF surfaces must be between 0 and 180 degrees.");
            // Get the azimuthal angle of the k vector of the TFSF pulse
            tfsfPhi_[ii] = M_PI*iter.second.get<double>("phi" , 90.0)/180.0;
            // Get the polarization angle of the TFSF surface
            tfsfPsi_[ii] = M_PI*iter.second.get<double>("psi" , 90.0)/180.0;

            if(size_[2] == 0.0)
            {
                std::cout << "WARNING: 2D calc detected Ensuring TFSF Psi and Theta angles are correct." << std::endl;
                tfsfTheta_[ii] = M_PI/2.0;
                if(pol_ == POLARIZATION::EZ || pol_ == POLARIZATION::HX || pol_ == POLARIZATION::HY)
                    tfsfPsi_[ii] = M_PI/2.0;
                else
                    tfsfPsi_[ii] = 0.0;
            }
            if( fmod(tfsfTheta_[ii], M_PI) == 0.0 )
                tfsfM_[ii][2] = tfsfTheta_[ii] == 0 ? 1 : -1;
            else if( fmod(tfsfPhi_[ii], M_PI) == 0.0 )
                tfsfM_[ii][0] = ( fmod(tfsfPhi_[ii], 2*M_PI) == 0.0 ) ? 1 : -1;
            else if( fmod(tfsfPhi_[ii], M_PI/2.0) == 0.0 )
                tfsfM_[ii][1] = ( fmod(tfsfPhi_[ii], 2*M_PI) == M_PI/2.0 ) ? 1 : -1;
        }
        if( std::all_of(tfsfM_[ii].begin(), tfsfM_[ii].end(), [](int mm){return mm == 0; } ) )
            throw std::logic_error("TFSF surface must have a direction of propgation.");
        // if( tfsfSize_[ii][0] == 1 && tfsfM_[ii][0] == 0 )
        //     throw std::logic_error("Propagation direction is in the plane of the TFSF surface, this can't produce correct results.");
        // if( tfsfSize_[ii][1] == 1 && tfsfM_[ii][1] == 0 )
        //     throw std::logic_error("Propagation direction is in the plane of the TFSF surface, this can't produce correct results.");
        // if( tfsfSize_[ii][2] == 1 && tfsfM_[ii][2] == 0 )
        //     throw std::logic_error("Propagation direction is in the plane of the TFSF surface, this can't produce correct results.");

        if(periodic_ && std::any_of(tfsfSize_[ii].begin(), tfsfSize_[ii].end(), [](int sz){return sz == 1; } ))
        {
            // if( std::all_of( k_point_.begin(), k_point_.end(), [](double kk){return kk == 0.0;} ) && ( k_point_[0] != tfsfM_[ii][0] || k_point_[1] != tfsfM_[ii][1] || k_point_[2] != tfsfM_[ii][2] ) )
            //     throw std::logic_error("TFSF surfaces are propagating in different directions with periodic boundary conditions, a single k-point vector is needed.");
            k_point_ = {{ static_cast<double>(tfsfM_[ii][0]), static_cast<double>(tfsfM_[ii][1]), static_cast<double>(tfsfM_[ii][2]) }};
            if( ( std::abs(k_point_[0]) > 0 && std::abs(k_point_[1]) > 0) || ( std::abs(k_point_[1]) > 0 && std::abs(k_point_[2]) > 0) || ( std::abs(k_point_[2]) > 0 && std::abs(k_point_[0]) > 0) )
                cplxFields_ = true;
        }
        // Initialize the pulse list
        boost::property_tree::ptree& PulseList = iter.second.get_child("PulseList");
        std::vector<std::vector<cplx>> pulFxn_(PulseList.size() );
        std::vector<PLSSHAPE> pulShape_(PulseList.size(), PLSSHAPE::GAUSSIAN);
        std::vector<double> pulEmax_(PulseList.size(), 0.0);
        int pp = 0;
        // Construct all the pulses
        for(auto & pul : PulseList )
        {
            // What is the profile of the pulse?
            pulShape_[pp] = string2prof(pul.second.get<std::string>("profile"));
            // What is the maximum value of the pulse
            pulEmax_[pp]  = pul.second.get<double>("Field_Intensity",1.0) * a_ * EPS0 * SPEED_OF_LIGHT / I0_;
            // Get the pulse function parameter
            std::vector<cplx> fxn;
            switch (pulShape_[pp])
            {
                case PLSSHAPE::GAUSSIAN:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(1.0/pul.second.get<double>("fwidth"));
                    fxn.push_back(pul.second.get<double>("cutoff") * fxn[1]);
                    fxn.push_back(pul.second.get<double>("t_0", std::real(fxn[1]*fxn[2]) ) );
                    fxn[2] += fxn[3];
                break;
                case PLSSHAPE::BH:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(pul.second.get<double>("tau", 1.0/pul.second.get<double>("fwidth") ) );
                    fxn.push_back(pul.second.get<double>("t_0", std::real( fxn[1]*fxn[2] ) ) );
                    fxn.push_back(pul.second.get<double>("BH1", 0.35875));
                    fxn.push_back(pul.second.get<double>("BH2", 0.48829));
                    fxn.push_back(pul.second.get<double>("BH3", 0.14128));
                    fxn.push_back(pul.second.get<double>("BH4", 0.01168));
                break;
                case PLSSHAPE::RECT :
                    fxn.push_back(pul.second.get<double>("tau"));
                    fxn.push_back(pul.second.get<double>("t_0"));
                    fxn.push_back(pul.second.get<double>("n", 30) * 2.0);
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                break;
                case PLSSHAPE::CONTINUOUS:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                break;
                case PLSSHAPE::RAMP_CONT:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(pul.second.get<double>("ramp_val"));
                break;
                case PLSSHAPE::RICKER:
                    fxn.push_back( cplx(0.0, -1.0 * pul.second.get<double>("fcen") * 2 * M_PI) );
                    fxn.push_back(pul.second.get<double>("fwidth"));
                    fxn.push_back(pul.second.get<double>("cutoff"));
                break;
                default:
                    throw std::logic_error("This pulse shape is undefined in source ");
                break;
            }
            // Add the function parameters to the list
            pulFxn_[pp] = fxn;
            ++pp;
        }
        // Give the parameters to the TFSF lists
        tfsfEmax_[ii] = pulEmax_;
        tfsfPulShape_[ii] = pulShape_;
        tfsfPulFxn_[ii] = pulFxn_;
        // Is the pulse circularly polarized, defaults to linear
        tfsfCircPol_[ii] = string2pol( iter.second.get<std::string>("circPol", "Ex") );
        // Is the light elliptical? Defaults to no, but if yes what is the ratio between the two axes
        tfsfEllipticalKratio_[ii] = iter.second.get<double>("ellpiticalKRat", 1.0);

        tfsfPMLThick_[ii] = iter.second.get<int>("pmlThick", 20);
        tfsfPMLAMax_[ii] = iter.second.get<double>("pmlAMax", 0.0);
        tfsfPMLM_[ii] = iter.second.get<double>("pmlM" , pmlM_);
        tfsfPMLMa_[ii] = iter.second.get<double>("pmlMa", pmlMa_);
    }

    int qq = 0;
    // Initialize unit vectors for vacuum baseline object
    std::array<std::array<double,3>,3> uVecs;
    for(ii = 0; ii < uVecs.size(); ++ii)
    {
        uVecs[ii] = {{ 0.0, 0.0, 0.0}};
        uVecs[ii][ii] = 1.0;
    }
    // set size of vacuum baseline object
    std::vector<double> szCell = {{ size_[0], size_[1], size_[2] }};
    // construct and add vacuum baseline object to objArr
    objArr_.push_back(std::make_shared<block>(1.0, 1.0, 0.0, std::vector<LorenzDipoleOscillator>(), false, szCell, std::array<double,3>({{0.0,0.0,0.0}}) , uVecs ) );
    // Start constructing all objects
    for (auto& iter : IP.get_child("ObjectList"))
    {
        // Construct the objects and make a shared_ptr to them
        std::shared_ptr<Obj> obj = ptreeToObject(iter);
        // What is the basis set for the object (Quantum Emitters)
        std::vector<std::array<int,2>> basis = {};
        boost::property_tree::ptree& basisSet = iter.second.get_child("Basis_Set");
        for(auto& iter2 : basisSet)
        {
            basis.push_back(std::array<int,2>{{iter2.second.get<int>("l"), iter2.second.get<int>("m")}});
        }
        // Number of grid pints in all directions of the entire cell
        int nx = find_pt(size_[0], d_[0]) + 1;
        int ny = find_pt(size_[1], d_[1]) + 1;
        int nz = find_pt(size_[2], d_[2]) + 1;
        // If a grid point is repeated in a qeLoc list then remove that loc from the list (objects over write each other)
        for(int q = 0; q < qeLoc_.size(); q++)
        {
            for(int ll = 0; ll < qeLoc_[q].size(); ll++)
            {
                std::array<double,3> loc = {static_cast<double>(qeLoc_[q][ll][0] - (nx-1)/2.0 - 1) * d_[0], static_cast<double>(qeLoc_[q][ll][1] - (ny-1)/2.0 - 1) * d_[1], static_cast<double>(qeLoc_[q][ll][2] - (nz-1)/2.0 - 1) * d_[2] };
                if(obj -> isObj(loc, *std::min_element( d_.begin(), d_.end() ), obj->geo() ) )
                {
                    qeLoc_[q].erase(qeLoc_[q].begin()+ll);
                    // used to ensure no locs are skipped
                    --ll;
                }
            }
        }
        // Conversion factor from eV to FDTD units
        double e_conv = ELEMENTARY_CHARGE * EPS0 * pow(SPEED_OF_LIGHT/I0_,2) / a_;
        // Is the object a quantum emitter if yes it will have a basis set size larger than 0?
        if(basis.size() > 0)
        {
            // add the basis to the quantum emitter list
            qeBasis_.push_back(basis);
            // get the molecular density
            qeDen_.push_back(iter.second.get<double>("mol_den", 1.0));
            // Create a list of for the energy level distributions
            boost::property_tree::ptree& Elevs = iter.second.get_child("Energy_Levels");
            int numElProportion = iter.second.get<int>("numElectorn", 1);
            for (auto& eLev : Elevs)
            {
                // Decribe the level
                EnergyLevelDiscriptor level;
                // Get the central energy of each distribution for the energy level in eV
                std::vector<double> e_cen = as_vector<double>(eLev.second, "E_cen");
                // Convert to FDTD units
                dscal_(e_cen.size(), e_conv, e_cen.data(), 1);
                // Get the relative weights of each distribution for the energy level
                std::vector<double> weight_vec = as_vector<double>(eLev.second, "weights", 1.0, e_cen.size());
                // Get the distribution type for each distribution
                level.EDist_ = string2dist(eLev.second.get<std::string>("distribution", "Delta_Fxn") );
                // Get the total number of states described by the total distribution for the energy level
                level.nstates_ = eLev.second.get<int>("nstates", 1);
                // Determine the highest and lowest energy level of the total distribution, either through hard coded values or by Band Width
                double topELevel = eLev.second.get<double>("highEdge", e_cen[0] / e_conv + eLev.second.get<double>("BandWith", 0.0)/2.0 ) * e_conv;
                double botELevel = eLev.second.get<double>("lowEdge", e_cen[0] / e_conv - eLev.second.get<double>("BandWith", 0.0)/2.0 ) * e_conv;
                double weightSum = 0.0;
                // Value of the number of different levels this distribution describes (degenerate levels)
                int numLevs = eLev.second.get<int>("levs_described", 1);
                // Construct the actual states for the distribution and the weights for each level(will be used to populate the initial ground state)
                if( ( level.EDist_ == DISTRIBUTION::DELTAFXN ) || ( level.nstates_ == 1) )
                {
                    // Delta function should be a single level
                    if(level.nstates_ > 1)
                        throw std::logic_error("The number of states is greater than one for a delta function distribution.");
                    level.weights_ = weight_vec;
                    level.energyStates_ = e_cen;
                    weightSum = 1.0;
                }
                else
                {
                    // get the equispaced energy levels
                    for(int nn = 0; nn < level.nstates_; ++nn)
                        level.energyStates_.push_back(botELevel + nn * (topELevel - botELevel)/(level.nstates_ - 1.0));
                    for(auto& Estate : level.energyStates_)
                    {
                        // Get the widths of each baseline width
                        std::vector<double> dist_width = as_vector<double>(eLev.second, "dist_width");
                        // get it in FDTD units
                        dscal_(dist_width.size(), e_conv, dist_width.data(), 1);
                        // Check all distribution vectors are the same size
                        if(dist_width.size() != e_cen.size() )
                            throw std::logic_error("The center frequency and distribution width is not the same size.");

                        double weight = 0.0;
                        // calculate the weight by summing each distributions contribution at that energy
                        if( level.EDist_ == DISTRIBUTION::GAUSSIAN )
                        {
                            for(int ii = 0; ii < dist_width.size(); ++ii)
                                weight += weight_vec[ii] * normDist(Estate, e_cen[ii], dist_width[ii]);
                        }
                        else if( level.EDist_ == DISTRIBUTION::LOG_NORMAL )
                        {
                            for(int ii = 0; ii < dist_width.size(); ++ii)
                                weight += weight_vec[ii] * logNormDist(Estate, e_cen[ii], dist_width[ii]);
                        }
                        else if(level.EDist_ == DISTRIBUTION::SKEW_NORMAL)
                        {
                            // Get the skew factor for the skew norm distribution
                            std::vector<double> skew_factor = as_vector<double>(eLev.second, "skew_factor");
                            // Check all distribution vectors are the same size
                            if(skew_factor.size() != e_cen.size())
                                throw std::logic_error("The center frequency and skew factor do not have the same size.");
                            // calculate the weight by summing each distributions contribution at that energy
                            for(int ii = 0; ii < dist_width.size(); ++ii)
                                weight += weight_vec[ii] * skewNormDist(Estate, e_cen[ii], dist_width[ii], skew_factor[ii]);
                        }
                        else if(level.EDist_ == DISTRIBUTION::GEN_SKEW_NORMAL)
                        {
                            // Get the skew factor for the skew norm distribution
                            std::vector<double> shape_factor = as_vector<double>(eLev.second, "shape_factor");
                            // Check all distribution vectors are the same size
                            if(shape_factor.size() != e_cen.size())
                                throw std::logic_error("The center frequency and shape factor do not have the same size.");
                            // calculate the weight by summing each distributions contribution at that energy
                            for(int ii = 0; ii < dist_width.size(); ++ii)
                                weight += weight_vec[ii] * genSkewNormDist(Estate, e_cen[ii], dist_width[ii], shape_factor[ii]);
                        }
                        else if(level.EDist_ == DISTRIBUTION::EXP_POW_DIST)
                        {
                            // Get the skew factor for the skew norm distribution
                            std::vector<double> shape_factor = as_vector<double>(eLev.second, "shape_factor");
                            // Check all distribution vectors are the same size
                            if(shape_factor.size() != e_cen.size())
                                throw std::logic_error("The center frequency and shape factor do not have the same size.");
                            // calculate the weight by summing each distributions contribution at that energy
                            for(int ii = 0; ii < dist_width.size(); ++ii)
                                weight += weight_vec[ii] * expPowDist(Estate, e_cen[ii], dist_width[ii], shape_factor[ii]);
                        }
                        else
                        {
                            throw std::logic_error("The distribution type is not defined for the construction of energy states and weights");
                        }
                        // add the weight to the vector
                        level.weights_.push_back(weight);
                        // add the weight to the toal sum
                        weightSum += level.weights_.back();
                    }
                    for(auto& weight : level.weights_)
                    {
                        weight /= weightSum / static_cast<double>(numElProportion);
                    }
                    std::ofstream outWeights;
                    outWeights.open("levelDist.dat");
                    for(int nn = 0; nn < level.nstates_; ++nn)
                    {
                        outWeights << level.energyStates_[nn] / (e_conv) << '\t' << level.weights_[nn] << '\n';
                    }
                    outWeights.close();
                }
                // Normalize the weights so the total sum is one
                // Add the energy level to the vector describing the qe levels
                level.levDescribed_ = numLevs;
                qeELevs_.push_back(level);
            }
            // get the transition dipole matrix values in FDTD units
            std::vector<double> couplings;
            for (auto& coup : iter.second.get_child("couplings"))
                couplings.push_back( coup.second.get_value<double>() * 1.0e-21 / pow(a_,2.0) / I0_ );
            qeCouplings_.push_back(couplings);
            // Get the relaxation matrix if it was added in by hand
            std::vector<std::vector<double>> gams ={};
            for(auto&  gvec : iter.second.get_child("gam"))
            {
                std::vector<double> gam = {};
                for(auto& gg : gvec.second.get_child("g"))
                    gam.push_back(gg.second.get_value<double>() * a_ / SPEED_OF_LIGHT);
                gams.push_back(gam);
            }
            qeGam_.push_back(gams);
            // Create a list of locations that the qe object contains
            std::vector<std::array<int,3>> locs;
            for(int zz = 0; zz < nz; ++zz)
            {
                for(int yy = 0; yy < ny; ++yy)
                {
                    for(int xx = 0; xx < nx; ++xx)
                    {
                        double x_pt = static_cast<double>(xx - (nx-1)/2.0) * d_[0] ;
                        double y_pt = static_cast<double>(yy - (ny-1)/2.0) * d_[1] ;
                        double z_pt = static_cast<double>(zz - (nz-1)/2.0) * d_[2] ;
                        std::array<double,3> loc ={x_pt, y_pt, z_pt};
                        if(obj->isObj(loc, *std::min_element(d_.begin(), d_.end() ), obj->geo() ) )
                        {
                            locs.push_back( {{xx, yy, zz}} );
                        }
                    }
                }
            }
            qeLoc_.push_back(locs);
            // Get the properties for the transitions if relaxation matrix is not given in the input file
            // Initialize all vectors
            std::vector<std::array<int,2>> transitionStates;
            std::vector<double> relaxRates;
            std::vector<double> dephaseRates;
            // take in all values into the right vector
            for(auto& transtionsRelax : iter.second.get_child("RelaxationOperators") )
            {
                transitionStates.push_back( {{ transtionsRelax.second.get<int>("state_i"), transtionsRelax.second.get<int>("state_f") }} ) ;
                relaxRates.push_back(transtionsRelax.second.get<double>("rate")  * a_ / SPEED_OF_LIGHT );
                dephaseRates.push_back(transtionsRelax.second.get<double>("dephasing_rate", 0.0) * a_ / SPEED_OF_LIGHT );
            }
            // Put all vectors in the right spots
            qeRelaxTransitonStates_.push_back(transitionStates);
            qeRelaxRates_.push_back(relaxRates);
            qeRelaxDephasingRate_.push_back(dephaseRates);
            // Output the QE polarizations?
            qeAccumP_.push_back(iter.second.get<bool>("output_pol", false) );
            // Add polarization output filename and create all directories
            if(qeAccumP_.back() )
            {
                qeOutPolFname_.push_back(iter.second.get<std::string>("output_pol_filename") );
                boost::filesystem::path p(qeOutPolFname_.back().c_str());
                boost::filesystem::create_directories(p.remove_filename());
            }
            else
                qeOutPolFname_.push_back("");
            // Output any of the qe's state's populations?
            std::vector<int> dtcLevs;
            for(auto& lev : iter.second.get_child("dtc_levs") )
                dtcLevs.push_back(lev.second.get_value<int>() );
            // Set the filenames for the outputing of levels
            std::vector<std::string> dtcPopOutFiles(dtcLevs.size(),"");
            qeDtcPopTimeInt_.push_back(iter.second.get<int>("levDTC_timeInt", 1) );
            for(int ll = 0; ll < dtcLevs.size(); ++ll)
                dtcPopOutFiles[ll] = iter.second.get<std::string>("dtc_pop_fname_base", "output_data/qe_") + std::to_string(qq) + "_level_" + std::to_string(dtcLevs[ll]) + ".dat";
            // Create directories for the population detectors
            if(dtcPopOutFiles.size() > 0)
            {
                boost::filesystem::path p(dtcPopOutFiles[0].c_str());
                boost::filesystem::create_directories(p.remove_filename());
            }
            // Add the population detectors to the right vectors
            qePopDtcLevs_.push_back(dtcLevs);
            qeDtcPopOutFile_.push_back(dtcPopOutFiles);
            // Add buffer to the base object because the polarizations will not line up directly on grid points
            obj->addMLBuff( static_cast<double>( *std::max_element( d_.begin(), d_.end() ) ) );
            ++qq;
        }
        // ADd the object to objArr
        objArr_.push_back(obj);
    }
    ii = 0;
    int jj = 0;
    int dd = 0;
    // Set up all of the detector values
    for (auto& iter : IP.get_child("DetectorList"))
    {
        dtcTStart_[dd] = iter.second.get<double>("t_start", 0.0);
        dtcTEnd_[dd] = iter.second.get<double>("t_end", tMax_);
        dtcOutputAvg_[dd] = iter.second.get<bool>("timeIntegrateMap", false);
        dtcOutputMaps_[dd] = iter.second.get<bool>("output_map", false);
        // Get the detector type
        dtcType_[dd] = string2out(iter.second.get<std::string>("type") );
        // Get the detector's file name
        std::string out_name = iter.second.get<std::string>("fname") + "_field_" + std::to_string(ii) + ".dat";
        // Create the directories for the detectors
        boost::filesystem::path p(out_name.c_str());
        boost::filesystem::create_directories(p.remove_filename());
        ++ii;
        // set the name to out_name
        dtcName_[dd] = out_name;
        // get the detector class
        dtcClass_[dd] = string2dtcclass(iter.second.get<std::string>("dtc_class" ,"cout"));
        // True if output should be in SI units
        dtcSI_[dd] = iter.second.get<bool>("SI", true);
        // Get the interval of output (output once every (value) time units)
        dtcTimeInt_[dd] = iter.second.get<double>("Time_Interval", dt_ );
        // For BMP detectors get what should be printed to the text file and how it should be printed
        dtcOutBMPFxnType_[dd] = string2GRIDOUTFXN(iter.second.get<std::string>("txt_dat_type", "real"));
        if( (dtcType_[dd] == DTCTYPE::EPOW || dtcType_[dd] == DTCTYPE::HPOW ) && ( dtcOutBMPFxnType_[dd] != GRIDOUTFXN::POW && dtcOutBMPFxnType_[dd] != GRIDOUTFXN::LNPOW && dtcOutBMPFxnType_[dd] != GRIDOUTFXN::MAG ) )
        {
            std::cout << "Converting output function for power detector to GRIDOUTFXN::POW" << std::endl;
            dtcOutBMPFxnType_[dd] = GRIDOUTFXN::POW;
        }
        dtcOutBMPOutType_[dd] = string2GRIDOUTTYPE(iter.second.get<std::string>("txt_format_type", "none"));

        // get the detector in real space values
        std::array<double,3> tempSz = as_ptArr<double>(iter.second, "size");
        // Convert to Grid points
        for(int cc = 0; cc < 3; ++cc )
            dtcSz_[dd][cc] = find_pt(tempSz[cc], d_[cc]) + 1;

        // Get loc in grid points
        int i = 0;
        for(auto& loc : as_ptArr<double>(iter.second, "loc") )
        {
            if(loc - tempSz[i]/2.0 < -1.0*size_[i]/2.0 || loc + tempSz[i]/2.0 > size_[i]/2.0 )
                throw std::logic_error("A detector is outside the FDTD cell.");
            dtcLoc_[dd][i] = find_pt(loc + size_[i]/2.0 - tempSz[i]/2.0, d_[i]);
            ++i;
        }
        // If freq detector get the freq list
        if(dtcClass_[dd] == DTCCLASS::FREQ)
        {
            // Either frequency or wavelength dependent input
            double fCen   = iter.second.get<double>("fcen",-1.0);
            double fWidth = iter.second.get<double>("fwidth",-1.0);
            double lamL   = iter.second.get<double>("lamL",-1.0);
            double lamR   = iter.second.get<double>("lamR",-1.0);
            int nFreq     = iter.second.get<int>("nfreq",-1);
            // if number of frequencies is less than 1 then the detector does not work; otherwise set up the freq list
            if(nFreq < 1)
                throw std::logic_error("The freq detector regions need to have a number of frequencies specified");
            else
                dtcFreqList_[dd] = std::vector<double>(nFreq, 0.0);
            // Check if frequency is defined; if it is use that
            if(fCen != -1.0 && fWidth != -1.0)
            {
                // Can't have conflicting freqLists
                if(lamL != -1.0 && lamR != -1.0)
                    throw std::logic_error("Both a freq and wavelength range is defined, please select one to define for the freq detector");
                // Frequency step size
                double dOmg = fWidth / static_cast<double>(nFreq-1);
                // Fill list
                for(int ii = 0; ii < nFreq; ++ii)
                    dtcFreqList_[dd][ii] = (fCen - fWidth/2.0 + ii*dOmg) * 2.0 * M_PI ;
            }
            else if(lamL != -1.0 && lamR != -1.0)
            {
                // Wavelength step size
                double dLam = (lamR - lamL) / static_cast<double>(nFreq - 1);
                // Fill list
                for(int ii = 0; ii < nFreq; ++ii)
                    dtcFreqList_[dd][ii] = 2.0 * M_PI / (lamL + ii*dLam);
            }
            else
                throw std::logic_error("All frequency detectors must either have fcen and fwidth defined or lamL and lamR defined");
        }
        ++dd;
    }
    dd = 0;
    for (auto& iter : IP.get_child("FluxList"))
    {
        // Flux's file name
        fluxName_[dd] = iter.second.get<std::string>("name");
        // Create the directories for the flux regions
        boost::filesystem::path p(iter.second.get<std::string>("name").c_str());
        boost::filesystem::create_directories(p.remove_filename());
        // Size of the region in real space
        std::array<double,3> tempSz = as_ptArr<double>(iter.second, "size");
        // Convert to grid poitns
        for(int cc = 0; cc < 3; ++cc )
            fluxSz_[dd][cc] = find_pt(tempSz[cc], d_[cc]) + 1;

        int i = 0;
        // Get the location of the flux region in grid points (lower, left, and back corner)
        for(auto& loc : as_ptArr<double>(iter.second, "loc") )
        {
            if(loc - tempSz[i]/2.0 < -1.0*size_[i] || loc + tempSz[i]/2.0 > size_[i] )
                throw std::logic_error("A flux region is outside the FDTD cell.");
            fluxLoc_[dd][i] = find_pt(loc + size_[i]/2.0 - tempSz[i]/2, d_[i]);
            ++i;
        }
        // Get the weight of the region (typically 1.0 or -1.0)
        fluxWeight_[dd] = iter.second.get<double>("weight", 1.0);
        // Get the time interval in terms of number of time steps (Time interval)/time step
        fluxTimeInt_[dd] = static_cast<int>( std::floor(iter.second.get<double>("Time_Interval", dt_) / (dt_) + 0.50) );
        // If the time interval is less than 0 time steps make it 1
        if(fluxTimeInt_[dd] <= 0)
        {
            fluxTimeInt_[dd]=1;
        }

        // Either frequency or wavelength dependent input
        double fCen   = iter.second.get<double>("fcen",-1.0);
        double fWidth = iter.second.get<double>("fwidth",-1.0);
        double lamL   = iter.second.get<double>("lamL",-1.0);
        double lamR   = iter.second.get<double>("lamR",-1.0);
        int nFreq     = iter.second.get<int>("nfreq",-1);
        // if number of frequencies is less than 1 then the detector does not work; otherwise set up the freq list
        if(nFreq <= 0)
            throw std::logic_error("The flux regions need to have a number of frequencies specified");
        else
            fluxFreqList_[dd] = std::vector<double>(nFreq, 0.0);
        // Check if frequency is defined; if it is use that
        if(fCen != -1.0 && fWidth != -1.0)
        {
            // Can't have conflicting freqLists
            if(lamL != -1.0 && lamR != -1.0)
                throw std::logic_error("Both a freq and wavelength range is defined, please select one to define");
            // Frequency step size
            double dOmg = fWidth / static_cast<double>(nFreq-1);
            // Fill list
            for(int ii = 0; ii < nFreq; ++ii)
                fluxFreqList_[dd][ii] = (fCen - fWidth/2.0 + ii*dOmg) * 2.0 * M_PI;
        }
        else if(lamL != -1.0 && lamR != -1.0)
        {
            // Wavelength step size
            double dLam = (lamR - lamL) / static_cast<double>(nFreq - 1);
            // Fill list
            for(int ii = 0; ii < nFreq; ++ii)
                fluxFreqList_[dd][ii] = 2.0 * M_PI / (lamL + ii*dLam);
        }
        else
            throw std::logic_error("All fluxes must either have fcen and fwidth defined or lamL and lamR defined");
        // True if outputting in SI units
        fluxSI_[dd] = iter.second.get<bool>("SI", false);
        // true if outputting as a cross-section instead of relative fraction
        fluxCrossSec_[dd] = iter.second.get<bool>("cross_sec", false);

        // File names of incident fields if inputting incident fields
        fluxIncdFieldsFilename_[dd] = iter.second.get<std::string>("incd_fileds", "");
        // True if fields need to be saved
        fluxSave_[dd] = iter.second.get<bool>("save", false);
        // True if fields need to be loaded in at the start
        fluxLoad_[dd] = iter.second.get<bool>("load", false);
        // Make sure the incident field files exist if they are needed
        if(fluxLoad_[dd] && fluxIncdFieldsFilename_[dd] == "")
            throw std::logic_error("Trying to load in file without a valid path, in the " + std::to_string(dd) +" flux detector");

        dd++;
    }

}

POLARIZATION parallelProgramInputs::string2pol(std::string p)
{
    if(p.compare("Ex") == 0)
        return POLARIZATION::EX;
    else if(p.compare("Ey") == 0)
        return POLARIZATION::EY;
    else if(p.compare("Ez") == 0)
        return POLARIZATION::EZ;
    else if(p.compare("Hx") == 0)
        return POLARIZATION::HX;
    else if(p.compare("Hy") == 0)
        return POLARIZATION::HY;
    else if(p.compare("Hz") == 0)
        return POLARIZATION::HZ;
    else if(p.compare("L") == 0)
        return POLARIZATION::L;
    else if(p.compare("R") == 0)
        return POLARIZATION::R;
    else
        throw std::logic_error("POLARIZATION undefined");

}
GRIDOUTFXN parallelProgramInputs::string2GRIDOUTFXN (std::string f)
{
    if(f.compare("real") == 0)
        return GRIDOUTFXN::REAL;
    else if(f.compare("imag") == 0)
        return GRIDOUTFXN::IMAG;
    else if(f.compare("magnitude") == 0)
        return GRIDOUTFXN::MAG;
    else if(f.compare("power") == 0)
        return GRIDOUTFXN::POW;
    else if(f.compare("ln_power") == 0)
        return GRIDOUTFXN::LNPOW;
    else
        throw std::logic_error( f + " is not a valid GRIDOUTFXN type");
}
GRIDOUTTYPE parallelProgramInputs::string2GRIDOUTTYPE (std::string t)
{
    if(t.compare("box") == 0)
        return GRIDOUTTYPE::BOX;
    else if(t.compare("list") == 0)
        return GRIDOUTTYPE::LIST;
    else if(t.compare("none") == 0)
        return GRIDOUTTYPE::NONE;
    else
        throw std::logic_error( t + " is not a valid GRIDOUTTYPE type");
}
DIRECTION parallelProgramInputs::string2dir(std::string dir)
{
    if((dir.compare("x") == 0) || (dir.compare("X") == 0))
        return DIRECTION::X;
    else if((dir.compare("y") == 0) || (dir.compare("Y") == 0))
        return DIRECTION::Y;
    else if ((dir.compare("z") == 0) || (dir.compare("Z") == 0))
        return DIRECTION::Z;
    else
        throw std::logic_error("A direction in the input file is undefined.");

}

DISTRIBUTION parallelProgramInputs::string2dist(std::string dist)
{
    if( (dist.compare("Gaussian") == 0) || (dist.compare("gaussian") == 0))
        return DISTRIBUTION::GAUSSIAN;
    else if( (dist.compare("delta_fxn") == 0) || (dist.compare("Delta_fxn") == 0) || (dist.compare("Delta_Fxn") == 0) || (dist.compare("delta_Fxn") == 0) || (dist.compare("DeltaFxn") == 0))
        return DISTRIBUTION::DELTAFXN;
    else if( (dist.compare("skew_normal") == 0) || (dist.compare("Skew_Normal") == 0) || (dist.compare("Skew_normal") == 0) || (dist.compare("skew_Normal") == 0) )
        return DISTRIBUTION::SKEW_NORMAL;
    else if( (dist.compare("log_normal") == 0) || (dist.compare("Log_Normal") == 0) || (dist.compare("Log_normal") == 0) || (dist.compare("log_Normal") == 0) )
        return DISTRIBUTION::LOG_NORMAL;
    else if( (dist.compare("exp_pow") == 0) || (dist.compare("Exp_Pow") == 0) || (dist.compare("Exp_pow") == 0) || (dist.compare("exp_Pow") == 0) )
        return DISTRIBUTION::EXP_POW_DIST;
    else if( (dist.compare("gen_skew_normal") == 0) || (dist.compare("Gen_Skew_Normal") == 0) || (dist.compare("Gen_Skew_normal") == 0) || (dist.compare("gen_skew_Normal") == 0) )
        return DISTRIBUTION::GEN_SKEW_NORMAL;
    else
        throw std::logic_error("The distribution type " + dist + " is not defined, please define it for me.");

}
SHAPE parallelProgramInputs::string2shape(std::string s)
{
    if(s.compare("sphere") == 0)
        return SHAPE::SPHERE;
    else if(s.compare("hemisphere") == 0)
        return SHAPE::HEMISPHERE;
    else if (s.compare("block") == 0)
        return SHAPE::BLOCK;
    else if (s.compare("triangle_prism") == 0)
        return SHAPE::TRIANGLE_PRISM;
    else if (s.compare("trapezoid_prism") == 0)
        return SHAPE::TRAPEZOIDAL_PRISM;
    else if (s.compare("ters_tip") == 0)
        return SHAPE::TERS_TIP;
    else if (s.compare("ellipsoid") == 0)
        return SHAPE::ELLIPSOID;
    else if(s.compare("hemiellipsoid") == 0)
        return SHAPE::HEMIELLIPSOID;
    else if (s.compare("cylinder") == 0)
        return SHAPE::CYLINDER;
    else if (s.compare("cone") == 0)
        return SHAPE::CONE;
    else if (s.compare("tetrahedron") == 0)
        return SHAPE::TETRAHEDRON;
    else if (s.compare("paraboloid") == 0)
        return SHAPE::PARABOLOID;
    else if (s.compare("torus") == 0)
        return SHAPE::TORUS;
    else
        throw std::logic_error("Shape undefined");
}

DTCTYPE parallelProgramInputs::string2out(std::string t)
{
    if(t.compare("Ex") == 0)
        return DTCTYPE::EX;
    else if(t.compare("Ey") == 0)
        return DTCTYPE::EY;
    else if(t.compare("Ez") == 0)
        return DTCTYPE::EZ;
    else if(t.compare("Hx") == 0)
        return DTCTYPE::HX;
    else if(t.compare("Hy") == 0)
        return DTCTYPE::HY;
    else if(t.compare("Hz") == 0)
        return DTCTYPE::HZ;
    else if(t.compare("Dx") == 0)
        return DTCTYPE::DX;
    else if(t.compare("Dy") == 0)
        return DTCTYPE::DY;
    else if(t.compare("Dz") == 0)
        return DTCTYPE::DZ;
    else if(t.compare("Bx") == 0)
        return DTCTYPE::BX;
    else if(t.compare("By") == 0)
        return DTCTYPE::BY;
    else if(t.compare("Bz") == 0)
        return DTCTYPE::BZ;
    else if(t.compare("Px") == 0)
        return DTCTYPE::PX;
    else if(t.compare("Py") == 0)
        return DTCTYPE::PY;
    else if(t.compare("Pz") == 0)
        return DTCTYPE::PZ;
    else if(t.compare("Mx") == 0)
        return DTCTYPE::MX;
    else if(t.compare("My") == 0)
        return DTCTYPE::MY;
    else if(t.compare("Mz") == 0)
        return DTCTYPE::MZ;
    else if(t.compare("ChiPx") == 0)
        return DTCTYPE::CHIPX;
    else if(t.compare("ChiPy") == 0)
        return DTCTYPE::CHIPY;
    else if(t.compare("ChiPz") == 0)
        return DTCTYPE::CHIPZ;
    else if(t.compare("ChiMx") == 0)
        return DTCTYPE::CHIMX;
    else if(t.compare("ChiMy") == 0)
        return DTCTYPE::CHIMY;
    else if(t.compare("ChiMz") == 0)
        return DTCTYPE::CHIMZ;
    else if(t.compare("E_pow") == 0)
        return DTCTYPE::EPOW;
    else if(t.compare("H_pow") == 0)
        return DTCTYPE::HPOW;
    else
        throw std::logic_error("DTCTYPE (DetectorList.type) from input file not defined");
}

DTCCLASS parallelProgramInputs::string2dtcclass(std::string c)
{
    if(c.compare("bin") == 0)
        return DTCCLASS::BIN;
    else if(c.compare("bmp") == 0)
        return DTCCLASS::BMP;
    else if(c.compare("txt") == 0)
        return DTCCLASS::TXT;
    else if(c.compare("cout") == 0)
        return DTCCLASS::COUT;
    else if(c.compare("freq") == 0)
        return DTCCLASS::FREQ;
    else
        throw std::logic_error("DTCCLASS (DetectorList.class) for input file is not defined");
}

PLSSHAPE parallelProgramInputs::string2prof(std::string p)
{
    if(p.compare("gaussian") == 0)
        return PLSSHAPE::GAUSSIAN;
    else if(p.compare("BH") == 0)
        return PLSSHAPE::BH;
    else if(p.compare("rectangle") == 0)
        return PLSSHAPE::RECT;
    else if(p.compare("continuous") == 0)
        return PLSSHAPE::CONTINUOUS;
    else if(p.compare("ricker") == 0)
        return PLSSHAPE::RICKER;
    else if(p.compare("ramped_cont") == 0)
        return PLSSHAPE::RAMP_CONT;
    else
        throw std::logic_error("Pulse shape undefined");
}

MAT_DIP_ORIENTAITON parallelProgramInputs::string2dipor(std::string dipOr)
{
    if(dipOr.compare("isotropic") == 0)
        return MAT_DIP_ORIENTAITON::ISOTROPIC;
    else if(dipOr.compare("normal") == 0)
        return MAT_DIP_ORIENTAITON::REL_TO_NORM;
    else if(dipOr.compare("tangent") == 0)
        return MAT_DIP_ORIENTAITON::REL_TO_NORM;
    else if(dipOr.compare("lat_tangent") == 0)
        return MAT_DIP_ORIENTAITON::LAT_TAN;
    else if(dipOr.compare("long_tangent") == 0)
        return MAT_DIP_ORIENTAITON::LONG_TAN;
    else if(dipOr.compare("rel_norm") == 0)
        return MAT_DIP_ORIENTAITON::REL_TO_NORM;
    else if(dipOr.compare("unidirectional") == 0)
        return MAT_DIP_ORIENTAITON::UNIDIRECTIONAL;
    else
        throw std::logic_error("The dipole orientation style is undefined");
}


std::tuple<double,double,double, std::vector<LorenzDipoleOscillator> >  parallelProgramInputs::getMater(std::string mat)
{
    if(mat.compare("Au") == 0 || mat.compare("au") == 0 || mat.compare("AU") == 0)
    {
        return std::make_tuple( 1.0+1.0e-14, 1.0, 0.0, getMetal(AU_MAT_) );
    }
    else if(mat.compare("Ag") == 0 || mat.compare("ag") == 0 || mat.compare("AG") == 0)
    {
        return std::make_tuple( 1.0+1.1e-14, 1.0, 0.0, getMetal(AG_MAT_) );
    }
    else if(mat.compare("Al") == 0 || mat.compare("al") == 0 || mat.compare("AL") == 0)
    {
        return std::make_tuple( 1.0+1.2e-14, 1.0, 0.0, getMetal(AL_MAT_) );
    }
    else if(mat.compare("Cu") == 0 || mat.compare("cu") == 0 || mat.compare("CU") == 0)
    {
        return std::make_tuple( 1.0+1.3e-14, 1.0, 0.0, getMetal(CU_MAT_) );
    }
    else if(mat.compare("Be") == 0 || mat.compare("be") == 0 || mat.compare("BE") == 0)
    {
        return std::make_tuple( 1.0+1.4e-14, 1.0, 0.0, getMetal(BE_MAT_) );
    }
    else if(mat.compare("Cr") == 0 || mat.compare("cr") == 0 || mat.compare("CR") == 0)
    {
        return std::make_tuple( 1.0+1.5e-14, 1.0, 0.0, getMetal(CR_MAT_) );
    }
    else if(mat.compare("Ni") == 0 || mat.compare("ni") == 0 || mat.compare("NI") == 0)
    {
        return std::make_tuple( 1.0+1.6e-14, 1.0, 0.0, getMetal(NI_MAT_) );
    }
    else if(mat.compare("Pt") == 0 || mat.compare("pt") == 0 || mat.compare("PT") == 0)
    {
        return std::make_tuple( 1.0+1.7e-14, 1.0, 0.0, getMetal(PT_MAT_) );
    }
    else if(mat.compare("Pd") == 0 || mat.compare("pd") == 0 || mat.compare("PD") == 0)
    {
        return std::make_tuple( 1.0+1.8e-14, 1.0, 0.0, getMetal(PD_MAT_) );
    }
    else if(mat.compare("Ti") == 0 || mat.compare("ti") == 0 || mat.compare("TI") == 0)
    {
        return std::make_tuple( 1.0+1.9e-14, 1.0, 0.0, getMetal(TI_MAT_) );
    }
    else if(mat.compare("W") == 0 || mat.compare("w") == 0)
    {
        return std::make_tuple( 1.0+2.0e-14, 1.0, 0.0, getMetal(W_MAT_) );
    }
    else if(mat.compare("TiO2") == 0)
    {
        return std::make_tuple( 6.20001, 1.0, 0.0, std::vector<LorenzDipoleOscillator>() );
    }
    else if(mat.compare("SiO2") == 0)
    {
        return std::make_tuple( 2.1025, 1.0, 0.0, std::vector<LorenzDipoleOscillator>() );
    }
    else if(mat.compare("CdSe") == 0)
    {
        return std::make_tuple( 6.20, 1.0, 0.0, std::vector<LorenzDipoleOscillator>() );
    }
    else if(mat.compare("CdSeQD") == 0)
    {
        return std::make_tuple( 6.20, 1.0, 0.0, getMetal(CDSE_QD_MAT_) );
    }
    else if(mat.compare("PbS") == 0)
    {
        return std::make_tuple( 17.20, 1.0, 0.0, std::vector<LorenzDipoleOscillator>() );
    }
    else if(mat.compare("PbSQD") == 0)
    {
        return std::make_tuple( 17.20, 1.0, 0.0, getMetal(PBS_QD_MAT_) );
    }
    else if(mat.compare("vac") == 0 || mat.compare("Vac") == 0 || mat.compare("VAC") == 0)
    {
        return std::make_tuple( 1.0, 1.0, 0.0, std::vector<LorenzDipoleOscillator>() );
    }
    else if(mat.compare("JM_Ag") == 0 || mat.compare("JM_ag") == 0 || mat.compare("JM_AG") == 0)
    {
        return std::make_tuple( 1.256, 1.0, 0.0, getMetalJM(JM_AG_MAT_) );
    }
    else if(mat.compare("JM_Au") == 0 || mat.compare("JM_au") == 0 || mat.compare("JM_AU") == 0)
    {
        return std::make_tuple( 5.513, 1.0, 0.0, getMetalJM(JM_AU_MAT_) );
    }
    else if(mat.compare("JM_Al") == 0 || mat.compare("JM_al") == 0 || mat.compare("JM_AL") == 0)
    {
        return std::make_tuple( 1.000+2.1e-14, 1.0, 0.0, getMetalJM(JM_AL_MAT_) );
    }
    else if(mat.compare("JM_Ga") == 0 || mat.compare("JM_ga") == 0 || mat.compare("JM_GA") == 0)
    {
        return std::make_tuple( 1.000+2.2e-14, 1.0, 0.0, getMetalJM(JM_GA_MAT_) );
    }
    else if(mat.compare("JM_Sn") == 0 || mat.compare("JM_sn") == 0 || mat.compare("JM_SN") == 0)
    {
        return std::make_tuple( 1.203, 1.0, 0.0, getMetalJM(JM_SN_MAT_) );
    }
    else if(mat.compare("JM_Tl") == 0 || mat.compare("JM_tl") == 0 || mat.compare("JM_TL") == 0)
    {
        return std::make_tuple( 1.456, 1.0, 0.0, getMetalJM(JM_TL_MAT_) );
    }
    else if(mat.compare("JM_Pb") == 0 || mat.compare("JM_pb") == 0 || mat.compare("JM_PB") == 0)
    {
        return std::make_tuple( 1.000+2.3e-14, 1.0, 0.0, getMetalJM(JM_PB_MAT_) );
    }
    else if(mat.compare("JM_Bi") == 0 || mat.compare("JM_bi") == 0 || mat.compare("JM_BI") == 0)
    {
        return std::make_tuple( 3.702, 1.0, 0.0, getMetalJM(JM_BI_MAT_) );
    }
    else
    {
        throw std::logic_error("The material name " + mat + " is not a predefined material. Please give me all the parameters or select from one of our predefined materials");
    }
}

inline double parallelProgramInputs::ev2FDTD(double eV)
{
    return eV / 4.135666e-15 * a_ / SPEED_OF_LIGHT; // Energy -> freq (SI) -> freq(FDTD)
}

std::vector<LorenzDipoleOscillator> parallelProgramInputs::getMetal(const std::vector<double> params)
{
    std::vector<LorenzDipoleOscillator> metalOscillators;

    double wp = ev2FDTD(params[0]);
    for(int ii = 0; ii < (params.size()-1)/3; ii++)
    {
        LorenzDipoleOscillator osc;
        osc.dipOrE_ = MAT_DIP_ORIENTAITON::ISOTROPIC;
        osc.dipOrM_ = MAT_DIP_ORIENTAITON::ISOTROPIC;
        osc.sigM_ = 0.0;
        osc.tau_ = 0.0;
        osc.uVecDipE_ = {1.0, 1.0, 1.0};
        osc.uVecDipM_ = {1.0, 1.0, 1.0};
        osc.normCompWeightE_ = 1.0;
        osc.tangentLatCompWeightE_ = 1.0;
        osc.tangentLongCompWeightE_ = 1.0;

        osc.normCompWeightM_ = 1.0;
        osc.tangentLatCompWeightM_ = 1.0;
        osc.tangentLongCompWeightM_ = 1.0;

        double f   = params[3*ii+1];
        double GAM = ev2FDTD(params[3*ii+2]);
        double OMG = ev2FDTD(params[3*ii+3]);
        if(OMG == 0.0)
            OMG = ev2FDTD(1.0e-20);

        osc.sigP_ = f*pow(wp/OMG,2);
        osc.gam_ = GAM*M_PI;
        osc.omg_ = OMG*2.0*M_PI;
        metalOscillators.push_back(osc);
    }
    return metalOscillators;
}

std::vector<LorenzDipoleOscillator> parallelProgramInputs::getMetalJM(const std::vector<double> params)
{
    double wp = ev2FDTD(params[0]);
    std::vector<LorenzDipoleOscillator> metalOscillators;
    for(int ii = 0; ii < (params.size()-1)/3; ii++)
    {
        LorenzDipoleOscillator osc;
        osc.dipOrE_ = MAT_DIP_ORIENTAITON::ISOTROPIC;
        osc.dipOrM_ = MAT_DIP_ORIENTAITON::ISOTROPIC;
        osc.sigM_ = 0.0;
        osc.tau_ = 0.0;
        osc.uVecDipE_ = {1.0, 1.0, 1.0};
        osc.uVecDipM_ = {1.0, 1.0, 1.0};
        osc.normCompWeightE_ = 1.0;
        osc.tangentLatCompWeightE_ = 1.0;
        osc.tangentLongCompWeightE_ = 1.0;

        osc.normCompWeightM_ = 1.0;
        osc.tangentLatCompWeightM_ = 1.0;
        osc.tangentLongCompWeightM_ = 1.0;

        double f   = params[3*ii+1];
        double GAM = ev2FDTD(params[3*ii+2]);
        double OMG = ev2FDTD(params[3*ii+3]);
        bool drude = false;
        if(OMG == 0.0)
        {
            drude = true;
            OMG = ev2FDTD(1.0e-20);
        }

        osc.sigP_ = drude ? std::pow(wp / OMG, 2.0) : f;
        osc.gam_ =  drude ? GAM*M_PI : 2.0*GAM*M_PI;
        osc.omg_ = OMG*2.0*M_PI;
        // std::cout << osc.sigP_ << '\t' << osc.gam_ << '\t' << osc.omg_ << std::endl;
        metalOscillators.push_back(osc);
    }
    return metalOscillators;
}

std::shared_ptr<Obj> parallelProgramInputs::ptreeToObject(boost::property_tree::ptree::value_type &iter)
{
    //  Get information common amongst all objects
    std::shared_ptr<Obj> out;
    double ang;

    std::array<double,3> loc = as_ptArr<double>(iter.second, "loc");

    // Parse the material information
    std::string material = iter.second.get<std::string>("material");

    double eps_infty = 0.0;
    double mu_infty = 0.0;
    double tellegen = 0.0;

    std::vector<LorenzDipoleOscillator> lorPols;

    if(material.compare("custom") == 0)
    {
        eps_infty = iter.second.get<double>("eps",1.00);
        mu_infty = iter.second.get<double>("mu", 1.0);
        tellegen = iter.second.get<double>("tellegen", 0.0);

        boost::property_tree::ptree& pols = iter.second.get_child("pols");
        for(auto& iter2 : pols)
        {
            LorenzDipoleOscillator osc;
            bool useTanIso = iter2.second.get<bool>("tanIso", false);
            osc.dipOrE_ = string2dipor(iter2.second.get<std::string>("dipOrE", "isotropic") ) ;
            if(useTanIso && (osc.dipOrE_ == MAT_DIP_ORIENTAITON::ISOTROPIC || osc.dipOrE_ == MAT_DIP_ORIENTAITON::UNIDIRECTIONAL) )
                throw std::logic_error("Object can't be isotropic in tangential directions and unidirectional or isotropic");
            osc.dipOrM_ = string2dipor(iter2.second.get<std::string>("dipOrM", "isotropic") ) ;
            if(useTanIso && (osc.dipOrM_ == MAT_DIP_ORIENTAITON::ISOTROPIC || osc.dipOrM_ == MAT_DIP_ORIENTAITON::UNIDIRECTIONAL) )
                throw std::logic_error("Object can't be isotropic in tangential directions and unidirectional or isotropic");
            osc.molec_ = iter2.second.get<bool>("molecular_trans", false );
            osc.gam_  = iter2.second.get<double>("gamma")*M_PI;
            if(!osc.molec_)
            {
                osc.omg_  = iter2.second.get<double>("omega")*2*M_PI;
                osc.sigP_ = iter2.second.get<double>("sigma_p", 0.0 );
                osc.sigM_ = iter2.second.get<double>("sigma_m", 0.0 );
                osc.tau_  = iter2.second.get<double>("tau", 0.0 );
            }
            else
            {
                osc.omg_  = std::sqrt( std::pow(iter2.second.get<double>("omega")*2*M_PI, 2.0) + std::pow(osc.gam_*M_PI, 2.0) ) ;
                double sigPreFact = 8.0*M_PI/(HBAR) * iter2.second.get<double>("molDen", 0.0);
                double omg0SI = iter2.second.get<double>("omega") * 2 * M_PI * SPEED_OF_LIGHT / a_;
                double omgNSI = osc.omg_ * SPEED_OF_LIGHT / a_;
                // Convert from Debye to SI
                double muE = iter2.second.get<double>("dipMoment_E", 0.0 ) * 1e-21 / SPEED_OF_LIGHT;
                if(osc.dipOrE_ == MAT_DIP_ORIENTAITON::ISOTROPIC)
                    muE /= std::sqrt(3.0);
                osc.sigP_ = sigPreFact/EPS0 * std::pow(muE, 2.0) * omg0SI / std::pow(omgNSI, 2.0);
                // Convert from Bohr Magnetron to SI
                double muM = iter2.second.get<double>("dipMoment_M", 0.0 ) * 9.27400999457e-24;
                if(osc.dipOrM_ == MAT_DIP_ORIENTAITON::ISOTROPIC)
                    muM /= std::sqrt(3.0);
                osc.sigM_ =  sigPreFact / (4.0e-7*M_PI) * std::pow(muM,2.0) * omg0SI / std::pow(omgNSI, 2.0);

                double relAngleMuM_E = std::cos( iter2.second.get<double>("relAnlgeMus", 0.0 ) * M_PI / 180.0 );
                if(std::abs( relAngleMuM_E ) < 1e-14 )
                    relAngleMuM_E = 0.0;
                osc.tau_ = relAngleMuM_E * sigPreFact / sqrt(EPS0*4.0e-7*M_PI) * muE * muM / std::pow(omgNSI, 2.0)  * SPEED_OF_LIGHT / a_;
            }

            double polAngRelE = iter2.second.get<double>("polAngRelNormE", 45.0);
            double azAngRelE  = iter2.second.get<double>("azAngRelNormE", 45.0);

            double polAngRelM = iter2.second.get<double>("polAngRelNormM", 45.0);
            double azAngRelM  = iter2.second.get<double>("azAngRelNormM", 45.0);

            if( iter2.second.get<std::string>("dipOrE", "isotropic").compare("normal") == 0)
            {
                polAngRelE = 0.0;
            }
            else if (iter2.second.get<std::string>("dipOrE", "isotropic").compare("tangent") == 0 )
            {
                polAngRelE = 90.0;
            }
            else if( osc.dipOrE_ == MAT_DIP_ORIENTAITON::LONG_TAN )
            {
                polAngRelE = 90.0;
                azAngRelE  = 0.0;
            }
            else if( osc.dipOrE_ == MAT_DIP_ORIENTAITON::LAT_TAN )
            {
                polAngRelE = 90.0;
                azAngRelE  = 90.0;
            }
            if(useTanIso && ( ( static_cast<int>(azAngRelE) % 45) != 0 || ( static_cast<int>(azAngRelE) % 90) == 0 ) )
                throw std::logic_error("isotropic tangent is true, but azimuthal angle of electric dipole is not 45 degrees");
            osc.normCompWeightE_        = std::cos( polAngRelE * M_PI / 180.0 );
            osc.tangentLatCompWeightE_  = std::sin( polAngRelE * M_PI / 180.0 ) * std::sin( azAngRelE * M_PI / 180.0 );
            osc.tangentLongCompWeightE_ = std::sin( polAngRelE * M_PI / 180.0 ) * std::cos( azAngRelE * M_PI / 180.0 );

            if( iter2.second.get<std::string>("dipOrM", "isotropic").compare("normal") == 0)
            {
                polAngRelM = 0.0;
            }
            else if (iter2.second.get<std::string>("dipOrM", "isotropic").compare("tangent") == 0 )
            {
                polAngRelM = 90.0;
            }
            else if( osc.dipOrE_ == MAT_DIP_ORIENTAITON::LONG_TAN )
            {
                polAngRelM = 90.0;
                azAngRelM  = 0.0;
            }
            else if( osc.dipOrE_ == MAT_DIP_ORIENTAITON::LAT_TAN )
            {
                polAngRelM = 90.0;
                azAngRelM  = 90.0;
            }
            if(useTanIso && ( ( static_cast<int>(azAngRelM) % 45) != 0 || ( static_cast<int>(azAngRelM) % 90) == 0 ) )
                throw std::logic_error("isotropic tangent is true, but azimuthal angle of electric dipole is not 45 degrees");

            osc.normCompWeightM_        = std::cos( polAngRelM * M_PI / 180.0 );
            osc.tangentLatCompWeightM_  = std::sin( polAngRelM * M_PI / 180.0 ) * std::sin( azAngRelM * M_PI / 180.0 );
            osc.tangentLongCompWeightM_ = std::sin( polAngRelM * M_PI / 180.0 ) * std::cos( azAngRelM * M_PI / 180.0 );

            if(osc.dipOrE_ == MAT_DIP_ORIENTAITON::UNIDIRECTIONAL)
            {
                if(osc.sigP_ > 0.0 || std::abs(osc.tau_) > 0.0)
                {
                    osc.uVecDipE_ = as_ptArr<double>(iter2.second, "dirDipE");
                    normalize(osc.uVecDipE_);
                    std::cout << osc.uVecDipE_[0] << '\t' << osc.uVecDipE_[1] << '\t' << osc.uVecDipE_[2] << std::endl;
                }
                else
                    osc.uVecDipE_ = {{ 0.0, 0.0, 0.0 }};
            }
            else
            {
                osc.uVecDipE_ = {{1.0, 1.0, 1.0}};
            }
            if(osc.dipOrM_ == MAT_DIP_ORIENTAITON::UNIDIRECTIONAL)
            {
                if(osc.sigM_ > 0.0 || std::abs(osc.tau_) > 0.0)
                {
                    osc.uVecDipM_ = as_ptArr<double>(iter2.second, "dirDipM");
                    normalize(osc.uVecDipM_);
                }
                else
                    osc.uVecDipM_ = {{ 0.0, 0.0, 0.0 }};
            }
            else
            {
                osc.uVecDipM_ = {{1.0, 1.0, 1.0}};
            }
            if(useTanIso)
            {
                LorenzDipoleOscillator oscTanLong(osc);
                LorenzDipoleOscillator oscTanLat(osc);
                oscTanLong.tangentLatCompWeightM_ = 0.0;
                oscTanLong.tangentLatCompWeightE_ = 0.0;
                oscTanLat.tangentLongCompWeightM_ = 0.0;
                oscTanLat.tangentLongCompWeightE_ = 0.0;

                oscTanLong.normCompWeightM_ /= std::sqrt(2.0);
                oscTanLong.normCompWeightE_ /= std::sqrt(2.0);
                oscTanLat.normCompWeightM_ /= std::sqrt(2.0);
                oscTanLat.normCompWeightE_ /= std::sqrt(2.0);

                lorPols.push_back(oscTanLat);
                lorPols.push_back(oscTanLong);
            }
            else
                lorPols.push_back(osc);
        }
    }
    else
    {
        std::tie(eps_infty, mu_infty, tellegen, lorPols) = getMater(material);
    }
    std::vector<double> geo_param;
    //Make unit vectors from user inputs
    std::array<std::array<double,3>,3> unitVecs;
    boost::property_tree::ptree& uvecs = iter.second.get_child("unit_vectors");
    int cc = 0;
    for(auto& iter2 : uvecs)
    {
        unitVecs[cc] = as_ptArr<double>(iter2.second, "uvec");
        ++cc;
    }
    //If user inputs are not there use angles
    if(uvecs.size() == 0)
    {
        double orTheta = M_PI/2.0 - iter.second.get<double>("orTheta", 90.0) * M_PI / 180.0;
        double orPhi   = -1.0*iter.second.get<double>("orPhi", 0.0) * M_PI / 180.0;
        if(size_[2] == 0)
        {
            unitVecs[0] = {{ std::cos(orPhi), -1.0*std::sin(orPhi), 0}};
            unitVecs[1] = {{ std::sin(orPhi),      std::cos(orPhi), 0}};
            unitVecs[2] = {{        0.0,             0.0, 0}};
        }
        else
        {
            // First rotate by theta about y axis then phi around z axis
            unitVecs[0] = {{      std::cos(orTheta)*std::cos(orPhi), -1.0*std::cos(orTheta)*std::sin(orPhi), std::sin(orTheta) }};
            unitVecs[1] = {{                        std::sin(orPhi),                        std::cos(orPhi), 0                 }};
            unitVecs[2] = {{ -1.0*std::sin(orTheta)*std::cos(orPhi),      std::sin(orTheta)*std::sin(orPhi), std::cos(orTheta) }};
        }
    }
    bool ML = iter.second.get_child("Basis_Set").size() > 0 ?  true : false;
    // Construct based on shape parameters
    SHAPE s = string2shape(iter.second.get<std::string>("shape"));
    if( s == SHAPE::BLOCK)
    {
        geo_param = as_vector<double>(iter.second, "size");
        geo_param.push_back(iter.second.get<double>("rad_curve", 0.0) );
        if(geo_param.back() == 0.0)
            return std::make_shared<block>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
        else
            return std::make_shared<rounded_block>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::ELLIPSOID)
    {
        geo_param = as_vector<double>(iter.second, "size");
        std::array<double,3> axisCutNeg = as_ptArr<double>(iter.second, "cut_neg", -1.0);
        std::array<double,3> axisCutPos = as_ptArr<double>(iter.second, "cut_pos",  1.0);
        std::array<double,3> axisCutGlobal = as_ptArr<double>(iter.second, "cut_global",  1.0);
        for(int nn = 0; nn < 3; ++nn)
        {
            if( std::abs(axisCutNeg[nn]) > 1.0)
                throw std::logic_error("abs(ellipsoid cutoff) must be <= 1.0");
            if( std::abs(axisCutPos[nn]) > 1.0)
                throw std::logic_error("abs(ellipsoid cutoff) must be <= 1.0");
            if( std::abs(axisCutGlobal[nn]) > 1.0)
                throw std::logic_error("abs(ellipsoid cutoff) must be <= 1.0");
        }
        return std::make_shared<ellipsoid>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, axisCutNeg, axisCutPos, axisCutGlobal, loc, unitVecs);
    }
    else if( s == SHAPE::HEMIELLIPSOID)
    {
        geo_param = as_vector<double>(iter.second, "size");
        return std::make_shared<hemiellipsoid>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::SPHERE)
    {
        geo_param = { iter.second.get<double>("radius",0.0) };
        return std::make_shared<sphere>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::HEMISPHERE)
    {
        geo_param = { iter.second.get<double>("radius",0.0) };
        return std::make_shared<hemisphere>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::CYLINDER)
    {
        geo_param = { iter.second.get<double>("radius",0.0), iter.second.get<double>("length",0.0) };
        return std::make_shared<cylinder>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::CONE)
    {
        double rad1 =  iter.second.get<double>("radSmall",0.0);
        double rad2 =  iter.second.get<double>("radLarge",0.0);
        if(rad2 <= rad1)
            throw std::logic_error("radLarge for a cone must be greater than radSmall.");
        double len  =  iter.second.get<double>("height",0.0);
        double raiseAng = std::atan( (rad2 - rad1) / len);
        double height = rad2 * std::tan(raiseAng);
        geo_param = { rad2, len, height };
        return std::make_shared<cone>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    if( s == SHAPE::TRAPEZOIDAL_PRISM)
    {
        double base1 = iter.second.get<double>("base1", 0.0);
        double base2 = iter.second.get<double>("base2", 0.0);
        double base = std::max(base1, base2);
        double height = iter.second.get<double>("height", 0.0);
        double heightTri = base * 2.0*height/(base1-base2);
        std::array<std::array<double, 2>,3> vertLocs;
        vertLocs[0] = {0, 0.5*heightTri};
        vertLocs[1] = {0.5*base, -0.5*heightTri};
        vertLocs[2] = {-0.5*base, -0.5*heightTri};

        geo_param = {{ std::abs(height/heightTri), 1.0, 1.0, iter.second.get<double>("length", 0.0) }};
        return std::make_shared<tri_prism>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, vertLocs, loc, unitVecs);
    }
    else if( s == SHAPE::TRIANGLE_PRISM)
    {
        try
        {
            boost::property_tree::ptree& vLocs = iter.second.get_child("vertLocs");
            std::array<std::array<double,2>,3> vertLocs;
            geo_param = as_vector(iter.second, "barry_center_cutoff", 1.0, 3);
            geo_param.push_back(iter.second.get<double>("length", 0.0));

            int cc = 0;
            for(auto& iter2 : vLocs)
            {
                if(cc > 2)
                    throw std::logic_error("A triangle has exactly 3 verticies.");
                vertLocs[cc] = as_ptArr2<double>(iter2.second, "vert");
                ++cc;
            }
            if(cc != 3)
                throw std::logic_error("A triangle has exactly 3 verticies.");
            return std::make_shared<tri_prism>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, vertLocs, loc, unitVecs);
        }
        catch(std::exception& e)
        {
            geo_param = {{ iter.second.get<double>("base", 0.0), iter.second.get<double>("height", 0.0), 1.0, iter.second.get<double>("length", 0.0) }};
            return std::make_shared<tri_prism>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
        }
    }
    else if( s == SHAPE::TERS_TIP)
    {
        geo_param = {{ 2.0*iter.second.get<double>("rad_curve", 0.0), iter.second.get<double>("base", 0.0),  iter.second.get<double>("length", 0.0) }};
        return std::make_shared<ters_tip>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::PARABOLOID)
    {
        geo_param = {{ iter.second.get<double>("a", 1.0/(2.0*iter.second.get<double>("rad_curve", 0.0) ) ) , iter.second.get<double>("b", 1.0/(2.0*iter.second.get<double>("rad_curve", 0.0) ) ), iter.second.get<double>("length", 0.0) }};
        return std::make_shared<paraboloid>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::TORUS)
    {
        geo_param = {{ iter.second.get<double>("rad_Ring", 0.0), iter.second.get<double>("rad_CrossSec", 0.0), iter.second.get<double>("phiMin", 0.0)*M_PI/180.0, iter.second.get<double>("phiMax", 360.0)*M_PI/180.0 }};
        return std::make_shared<torus>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
    }
    else if( s == SHAPE::TETRAHEDRON )
    {
        try
        {
            boost::property_tree::ptree& vLocs = iter.second.get_child("vertLocs");
            std::array<std::array<double,3>,4> vertLocs;
            geo_param = as_vector(iter.second, "barry_center_cutoff", 1.0, 4);
            int cc = 0;
            for(auto& iter2 : vLocs)
            {
                if(cc > 3)
                    throw std::logic_error("A tetrahedron has exactly 4 verticies.");
                vertLocs[cc] = as_ptArr<double>(iter2.second, "vert");
                ++cc;
            }
            if(cc != 4)
                throw std::logic_error("A tetrahedron has exactly 4 verticies.");
            std::cout << "WARNING: Resetting location and unit vectors since the tetrahedron is defined by real space verticies." << std::endl;
            loc = {{0.0, 0.0, 0.0}};
            unitVecs[0] = {1,0,0};
            unitVecs[1] = {0,1,0};
            unitVecs[2] = {0,0,1};
            return std::make_shared<tetrahedron>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, vertLocs, loc, unitVecs);
        }
        catch(std::exception& e)
        {
            geo_param = {{ iter.second.get<double>("sideLen", 2.0/std::sqrt(3.0)*iter.second.get<double>("perpBiSect",-1.0) ) }};
            if(geo_param[0] == -1)
                throw std::logic_error("please give a side length or prepindicular bisector length for the tetrahedron." );
            geo_param.push_back( iter.second.get<double>( "height", std::sqrt(2.0/3.0)*geo_param[0] ) );
            return std::make_shared<tetrahedron>(eps_infty, mu_infty, tellegen, lorPols, ML, geo_param, loc, unitVecs);
        }
    }
    else
    {
        throw std::logic_error("A shape in the ObjectList is not defined in the code");
        return nullptr;
    }
}

void stripComments(std::string& filename)
{
    //Open input and output file
    std::string newfn = "stripped_" + filename;
    std::fstream inputfile;
    inputfile.open(filename);
    std::ofstream inputcopy;
    inputcopy.open(newfn);

    //search for '//', delete everything following, print remainder to new file
    std::string line;
    int found, found2;
    while (getline(inputfile,line))
    {
        found  = line.find('/');
        found2 = line.find('/', found+1);
        if (found != line.npos && found2 == found+1)
            inputcopy << line.erase(found, line.length()) << std::endl;
        else
            inputcopy << line << std::endl;
    }
    inputcopy.close();
    //update filename;
    filename = newfn;
}

