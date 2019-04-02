/** @file INPUTS/parallelInputs.cpp
 *  @brief Translates the input file into a structure that can be used by parallelFDTDField
 *
 *  Takes in a boost property tree and converts that information into a structure used by parallelFDTDField
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @author Joshua E. Szekely (jeszekely)
 *  @bug No known bugs.
 */

#ifndef PRALLEL_FDTD_INPUTS
#define PRALLEL_FDTD_INPUTS

#include <src/OBJECTS/Obj.hpp>
#include <src/UTIL/ml_consts.hpp>
#include <src/UTIL/dielectric_params.hpp>
#include <src/UTIL/utilityFxns.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <UTIL/ML_Dist_Fxn.hpp>
#include <iterator>

struct EnergyLevelDiscriptor
{
    DISTRIBUTION EDist_; //!< Distribution type for the energy level
    int nstates_; //!< number of states in the level
    int levDescribed_; //!< how many levels are described by this distribution
    std::vector<double> energyStates_; //!< energy of all the states in the level
    std::vector<double> weights_; //!< weights for the levels
};

/**
 * @brief input parameters class for the parallel FDTD class
 * @details sores all the values necessary to convert the input files into a fdtd grid
 */
class parallelProgramInputs
{
public:
    bool periodic_; //!< if true use PBC
    POLARIZATION pol_; //!< polarization of the grid (EX, EY, HZ = TE; HX, HY, EZ = TM)

    std::string filename_; //!< filename of the input file
    int res_; //!< number of grid points per unit length

    double courant_; //!< Courant factor of the cell
    double a_; //!< the unit length of the calculations
    double tMax_; //!< the max time of the calculation
    double I0_; //!< the unit current of the cell

    bool cplxFields_; //!< if true use complex fields
    bool saveFreqField_; //!< if true save the flux fields

    std::array<int,3> pmlThickness_; //!< thickness of the PMLs in all directions
    std::array<double,3> k_point_; //!< k_point vector of the light
    std::array<double,3> size_; //!< size of the cell in units of the unit length
    std::array<double,3> d_; //!< size of the cell in units of the unit length

    double dt_; //!< time step of the algorithm
    double pmlSigOptRat_; //!< Ratio of SigMax to SigOpt
    double pmlKappaMax_; //!< max a value for CPML
    double pmlAMax_; //!< max a value for CPML
    double pmlMa_; //!< scaling factor of a for the CPML
    double pmlM_; //!< scaling factor for the CPML

    std::vector<double> inputMapSlicesX_; //!< list of slices in the YZ plane
    std::vector<double> inputMapSlicesY_; //!< list of slices in the XZ plane
    std::vector<double> inputMapSlicesZ_; //!< list of slices in the XY plane

    std::vector<POLARIZATION> srcPol_; //!<polarization of all sources
    std::vector<std::vector<std::vector<cplx>>> srcFxn_; //!< pulse function parameters for the sources
    std::vector<std::array<int,3>> srcLoc_; //!<location of all sources
    std::vector<std::array<int,3>> srcSz_; //!<sizes of all all sources
    std::vector<double> srcPhi_; //!< angle of incidence for the source detector
    std::vector<double> srcTheta_; //!< angle of incidence for the source detector
    std::vector<std::vector<PLSSHAPE>> srcPulShape_; //!< pulse shape of all sources
    std::vector<std::vector<double>> srcEmax_; //!< max E incd field of all sources
    std::vector<double> srcEllipticalKratio_; //!< ratio between the long and short axis for current source surfaces
    std::vector<double> srcPsi_; //!< angle of polarization for circular and elliptical light

    std::vector<std::array<int,3>> tfsfLoc_;//!< location of the tfsf lower left point
    std::vector<std::array<int,3>> tfsfSize_; //!< size of the TFSF region
    std::vector<std::array<int,3>> tfsfM_; //!< vector describing the m value for tfsf surfaces
    std::vector<double> tfsfTheta_; //!< Phi angle of the TFSF surface
    std::vector<double> tfsfPhi_; //!< Phi angle of the TFSF surface
    std::vector<double> tfsfPsi_; //!< Phi angle of the TFSF surface
    std::vector<std::vector<std::vector<cplx>>> tfsfPulFxn_; //!< pulse function parameters for the sources
    std::vector<std::vector<PLSSHAPE>> tfsfPulShape_; //!< pulse shape of all sources
    std::vector<std::vector<double>> tfsfEmax_; //!< max E incd field of all sources
    std::vector<POLARIZATION> tfsfCircPol_; //!< sets the TFSF surface to be L or R polarized
    std::vector<double> tfsfEllipticalKratio_; //!< ratio between the long and short axis for TFSF surfaces
    std::vector<int> tfsfPMLThick_; //!< thickness of the incd PML's in number of grid points
    std::vector<double> tfsfPMLAMax_; //!< max a value for CPML
    std::vector<double> tfsfPMLM_; //!< scaling factor for the CPML
    std::vector<double> tfsfPMLMa_; //!< scaling factor of a for the CPML

    std::vector<std::shared_ptr<Obj>> objArr_; //!< array of all objects in the grid

    std::vector<std::vector<std::array<int,2>>> qeBasis_; //!< vector describing the basis functions for each state
    std::vector<EnergyLevelDiscriptor> qeELevs_; //!< vector storing the level descriptor for each level
    std::vector<std::vector<double>> qeCouplings_; //!< dipole coupling matrix for all systems
    std::vector<std::vector<std::array<int,3>>> qeLoc_; //!< location of all QE points
    std::vector<std::vector<std::vector<double>>> qeGam_; //!< relaxation matrix for all QEs
    std::vector<double> qeDen_; //!< molecular density of all QEs
    std::vector<std::vector<std::array<int,2>>> qeRelaxTransitonStates_; //!< Transitions for Lindblad Operator
    std::vector<std::vector<double>> qeRelaxRates_; //!< Rate of relaxation for Linblad operators
    std::vector<std::vector<double>> qeRelaxDephasingRate_; //!< Rate of dephasing between both states
    std::vector<bool> qeAccumP_; //!< bool to determine if output Polarization vector at every time
    std::vector<std::string> qeOutPolFname_; //!< filename for output
    std::vector<int> qeDtcPopTimeInt_; //!< Time interval for level dtc
    std::vector<std::vector<int>> qePopDtcLevs_; //!< vector storing the levels outputted at each step
    std::vector<std::vector<std::string>> qeDtcPopOutFile_; //!< filename for the pop output

    std::vector<int> qeContinuumStates_; //!< vector containing all of the Continuum states
    std::vector<double> qeContinuumNa_; //!< vector containing all of the Continuum band gap energy
    std::vector<double> qeContinuumOmgGap_; //!< vector containing all of the Continuum band gap energy
    std::vector<double> qeContinuum_dOmg_; //!< vector containing all of the Continuum inter-continuum state energy separation
    std::vector<double> qeContinuumMu_; //!< vector containing all of the Continuum transition dipole moments
    std::vector<double> qeContinuumGam1_; //!< vector containing all of the Continuum relaxation to the ground state (rationalness)
    std::vector<double> qeContinuumGamk_; //!< vector containing all of the Continuum relaxation to the conduction band edge (raditionless)
    std::vector<double> qeContinuumGamP_; //!< vector containing all of the Continuum relaxation to the conduction band edge (raditionless)
    std::vector<std::vector<std::array<int,2>>> qeContinuumLocs_; //!< vector containing all of the Continuum locations

    std::vector<DTCCLASS> dtcClass_; //!< describer of output file type for all detectors
    std::vector<bool> dtcSI_; //!< if true use SI units
    std::vector<std::array<int,3>> dtcLoc_; //!< location of all detectors
    std::vector<std::array<int,3>> dtcSz_; //!< sizes of all detectors
    std::vector<std::string> dtcName_; //!< file name for all detectors
    std::vector<DTCTYPE> dtcType_; //!< whether the dtc stores the Ex, Ey, Ez, Hx, Hy, Hz fields or the H or E power
    std::vector<double> dtcTimeInt_; //!< time interval for all detectors
    std::vector<GRIDOUTFXN> dtcOutBMPFxnType_; //!< what function should bmp converter use
    std::vector<GRIDOUTTYPE> dtcOutBMPOutType_; //!< how to output the values for the detector ina text file
    std::vector<std::vector<double>> dtcFreqList_; //!< center frequency
    std::vector<double> dtcTStart_; //!< time to start collecting fields
    std::vector<double> dtcTEnd_; //!< time to end collecting fields
    std::vector<bool> dtcOutputAvg_; //!< True if only outputting time_integrated fields
    std::vector<bool> dtcOutputMaps_; //!< True if only outputting time_integrated fields

    std::vector<int> fluxXOff_; //!< the x location offset of the fields
    std::vector<int> fluxYOff_; //!< the y location offset of the fields
    std::vector<int> fluxTimeInt_; //!< time interval used to intake the fields
    std::vector<std::array<int,3>> fluxLoc_; //!< Location of the lower left corner of the flux surface
    std::vector<std::array<int,3>> fluxSz_; //!< size of the flux surface
    std::vector<double> fluxWeight_; //!< weight of the flux surface
    std::vector<std::string> fluxName_; //!< file name for the output file
    std::vector<std::vector<double>> fluxFreqList_; //!<  center frequency
    std::vector<bool> fluxSI_; //!<  use SI units
    std::vector<bool> fluxCrossSec_; //!< calculate the cross-section?
    std::vector<bool> fluxSave_; //!< save the fields?
    std::vector<bool> fluxLoad_; //!< load the fields?
    std::vector<std::string> fluxIncdFieldsFilename_; //!< incident file names

    /**
     * @brief      Constructs the input parameter object
     *
     * @param[in]  IP    boost property tree generated from the input json file
     * @param[in]  fn    The filename of the input file
     */
    parallelProgramInputs(boost::property_tree::ptree IP,std::string fn);

    /**
     * @brief      Gets the dielectric parameters for a material.
     *
     * @param[in]  mat   String identifier of the material
     *
     * @return     The dielectric parameters.
     */
    std::vector<double> getDielectricParams(std::string mat);

    /**
     * @brief      converts a string to GRIDOUTFXN
     *
     * @param[in]  f     String identifier to a GRIDOUTFXN
     *
     * @return     GRIDOUTFXN from that input string
     */
    GRIDOUTFXN string2GRIDOUTFXN (std::string f);

    /**
     * @brief      converts a string to GRIDOUTTYPE
     *
     * @param[in]  t     String identifier to a GRIDOUTTYPE
     *
     * @return     GRIDOUTTYPE from that input string
     */
    GRIDOUTTYPE string2GRIDOUTTYPE (std::string t);

    /**
     * @brief      converts a string to POLARIZATION
     *
     * @param[in]  p     String identifier to a POLARIZATION
     *
     * @return     POLARIZATION from that input string
     */
    POLARIZATION string2pol(std::string p);

    /**
     * @brief      converts a string to SHAPE
     *
     * @param[in]  s     String identifier to a SHAPE
     *
     * @return     SHAPE from that input string
     */
    SHAPE string2shape(std::string s);

    /**
     * @brief      converts a string to DTCTYPE
     *
     * @param[in]  t     String identifier to a DTCTYPE
     *
     * @return     DTCTYPE from that input string
     */
    DTCTYPE string2out(std::string t);

    /**
     * @brief      converts a string to DTCCLASS
     *
     * @param[in]  c     String identifier to a DTCCLASS
     *
     * @return     DTCCLASS from that input string
     */
    DTCCLASS string2dtcclass(std::string c);

    /**
     * @brief      converts a string to PLSSHAPE
     *
     * @param[in]  p     String identifier to a PLSSHAPE
     *
     * @return     PLSSHAPE from that input string
     */
    PLSSHAPE string2prof(std::string p);

    /**
     * @brief      converts a string to DIRECTION
     *
     * @param[in]  dir   String identifier to a DIRECTION
     *
     * @return     DIRECTION from that input string
     */
    DIRECTION string2dir(std::string dir);

    /**
     * @brief      converts a string to DISTRIBUTION
     *
     * @param[in]  f     String identifier to a DISTRIBUTION
     *
     * @return     DISTRIBUTION from that input string
     */
    DISTRIBUTION string2dist(std::string dist);

    /**
     * @brief      Converts a string to MAT_DIP_ORIENTAITON
     *
     * @param[in]  dipOr  String identifier for the MAT_DIP_ORIENTAITON
     *
     * @return     The MAT_DIP_ORIENTAITON from the input string
     */
    MAT_DIP_ORIENTAITON string2dipor(std::string dipOr);

    /**
     * @brief      Gets the material parameters for a given material
     *
     * @param[in]  mat   String identifying the material
     *
     * @return     The material parameters
     */
    std::tuple<double,double,double, std::vector<LorenzDipoleOscillator> > getMater(std::string mat);

    /**
     * @brief      Converts the metal parameters from eV based units to FDTD based units
     *
     * @param[in]  params  The parameters for the metallic material in eV
     *
     * @return     The parameters for the metallic material in FDTD units
     */
    std::vector<LorenzDipoleOscillator> getMetal(std::vector<double> params);

    std::vector<LorenzDipoleOscillator> getMetalJM(std::vector<double> params);

    /**
     * @brief      Converts eV units to FDTD units
     *
     * @param[in]  eV    The value in eV
     *
     * @return     The value in FDTD frequency units
     */
    double ev2FDTD(double eV);

    /**
     * @brief      Converts the point from real space to grid points
     *
     * @param[in]  pt    Real space value
     *
     * @return     Corresponding grid point value
     */
    inline int find_pt(double pt, double d) {return int( floor(pt/d + 0.5) );}

    /**
     * @brief      Accessor function to tMax_
     *
     * @return     Maximum time of the simulation
     */
    inline double tMax() {return tMax_;}

    /**
     * @brief      Constructs an object from the object list child tree
     *
     * @param      iter  boost::ptree child corresponding to the object
     *
     * @return     shared_ptr to the object
     */
    std::shared_ptr<Obj> ptreeToObject(boost::property_tree::ptree::value_type &iter);
};
/**
 * @brief      strips comments from the input file
 *
 * @param      filename  The filename of the file to strip
 */
void stripComments(std::string& filename);


/**
 * @brief      boost json to std::vector<T>
 *
 * @param[in]  pt          property tree
 * @param[in]  key         property tree key
 *
 * @tparam     T           double, int
 *
 * @return     json input as a std::vector<T>
 */
template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const &pt, boost::property_tree::ptree::key_type const &key)
{
    std::vector<T> r;
    for (auto& item : pt.get_child(key))
        r.push_back(item.second.get_value<T>());
    return r;
}

/**
 * @brief      boost json to std::vector<T>
 *
 * @param[in]  pt          property tree
 * @param[in]  key         property tree key
 * @param[in]  defaultVal  The default value
 * @param[in]  szVec       The size of the vector
 *
 * @tparam     T           double, int
 *
 * @return     json input as a std::vector<T>
 */
template <typename T>
std::vector<T> as_vector(boost::property_tree::ptree const &pt, boost::property_tree::ptree::key_type const &key, T defaultVal, int szVec)
{
    std::vector<T> r;
    try
    {
        for (auto& item : pt.get_child(key))
            r.push_back(item.second.get_value<T>());
    }
    catch(std::exception& e)
    {
        r = std::vector<T>(szVec, defaultVal);
    }
    return r;
}


/**
 * @brief      boost json to std::array<T,3>
 *
 * @param[in]  pt          property tree
 * @param[in]  key         property tree key
 * @param[in]  defaultVal  The default value
 *
 * @tparam     T           double, int
 *
 * @return     json input as a std::array<T,3>
 */
template <typename T>
std::array<T,3> as_ptArr(boost::property_tree::ptree const &pt, boost::property_tree::ptree::key_type const &key, T defaultVal=0)
{
    std::array<T, 3> r = {0,0,0};
    try
    {
        int ii = 0;
        for (auto& item : pt.get_child(key))
        {
            r[ii] = item.second.get_value<T>();
            ++ii;
        }
    }
    catch(std::exception& e)
    {
        r = {{ defaultVal, defaultVal, defaultVal}};
    }
    return r;
}

/**
 * @brief      boost json to std::array<T,3>
 *
 * @param[in]  pt          property tree
 * @param[in]  key         property tree key
 * @param[in]  defaultVal  The default value
 *
 * @tparam     T           double, int
 *
 * @return     json input as a std::array<T,3>
 */
template <typename T>
std::array<T,2> as_ptArr2(boost::property_tree::ptree const &pt, boost::property_tree::ptree::key_type const &key, T defaultVal=0)
{
    std::array<T, 2> r = {0,0};
    try
    {
        int ii = 0;
        for (auto& item : pt.get_child(key))
        {
            r[ii] = item.second.get_value<T>();
            ++ii;
        }
    }
    catch(std::exception& e)
    {
        r = {{ defaultVal, defaultVal}};
    }
    return r;
}
#endif