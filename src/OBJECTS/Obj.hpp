/** @file OBJECTS/Obj.cpp
 *  @brief Class that stores all object/material information
 *
 *  A class that determines if a grid point is inside the object, stores the material parameters, and
 *  finds the gradient to determine the surface normal vector.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_OBJECT
#define FDTD_OBJECT

#include <math.h>
#include <UTIL/typedefs.hpp>
#include <src/UTIL/utilityFxns.hpp>

struct LorenzDipoleOscillator
{
    bool molec_; //!< True if molecular material
    MAT_DIP_ORIENTAITON dipOrE_; //!< Descriptor of how the dipoles are oriented
    MAT_DIP_ORIENTAITON dipOrM_; //!< Descriptor of how the dipoles are oriented
    double omg_; //!< Center frequency of the oscillator
    double gam_; //!< Damping factor (line width) of the oscillator
    double sigP_; //!< Strength of the electric dipole moment of the oscillator
    double sigM_; //!< Strength of the magnetic dipole moment of the oscillator
    double tau_; //!< Characteristic chiral time scale of the oscillator (based on $\mu \otimes m$)

    double normCompWeightE_; //!< Weight of the normal component of the oscillator's response
    double tangentLatCompWeightE_; //!< Weight of the normal component of the oscillator's response
    double tangentLongCompWeightE_; //!< Weight of the normal component of the oscillator's response

    double normCompWeightM_; //!< Weight of the normal component of the oscillator's response
    double tangentLatCompWeightM_; //!< Weight of the normal component of the oscillator's response
    double tangentLongCompWeightM_; //!< Weight of the normal component of the oscillator's response

    std::array<double,3> uVecDipE_; //!< Direction of the electric dipole moment and chiral response (used if dipOr_ is UNIDIRECTIONAL)
    std::array<double,3> uVecDipM_; //!< Direction of the magnetic dipole moment (used if dipOr_ is UNIDIRECTIONAL)
};

class Obj
{
protected:
    bool ML_; //!< True if object represents ML
    bool useOrientedDipols_; //!< True is one of the pols is not ISOTROPIC

    double eps_infty_; //!< high frequency dielectric constant
    double mu_infty_; //!< high frequency permeability of the material
    double tellegen_; //!< the tellegen parameter of the material (0 for reciprocal material, not used at the moment in the code but could be in the future)

    std::array<std::array<double,3>,3> unitVec_; //!< vectors describing object's coordinate sys relative to main grids

    std::vector<MAT_DIP_ORIENTAITON> dipOr_; //!< Vector storing the dipole orientation types for all electric dipoles
    std::vector<MAT_DIP_ORIENTAITON> magDipOr_; //!< Vector storing the dipole orientation types for all magnetic dipoles
    std::vector<MAT_DIP_ORIENTAITON> chiEDipOr_; //!< Vector storing the dipole orientation types for all magnetic dipoles
    std::vector<MAT_DIP_ORIENTAITON> chiMDipOr_; //!< Vector storing the dipole orientation types for all magnetic dipoles

    std::vector<double> dipNormCompE_; //!< The normal component of the electric dipole moment
    std::vector<double> dipTanLatCompE_; //!< The latitudinal component of the electric dipole moment
    std::vector<double> dipTanLongCompE_; //!< The longitudinal component of the electric dipole moment

    std::vector<double> dipNormCompM_; //!< The normal component of the magnetic dipole moment
    std::vector<double> dipTanLatCompM_; //!< The latitudinal component of the magnetic dipole moment
    std::vector<double> dipTanLongCompM_; //!< The longitudinal component of the magnetic dipole moment

    std::vector<double> dipNormCompChiE_; //!< The normal component of the chiral electric dipole moment
    std::vector<double> dipTanLatCompChiE_; //!< The latitudinal component of the chiral electric dipole moment
    std::vector<double> dipTanLongCompChiE_; //!< The longitudinal component of the chiral electric dipole moment

    std::vector<double> dipNormCompChiM_; //!< The normal component of the chiral magnetic dipole moment
    std::vector<double> dipTanLatCompChiM_; //!< The latitudinal component of the chiral magnetic dipole moment
    std::vector<double> dipTanLongCompChiM_; //!< The longitudinal component of the chiral magnetic dipole moment

    std::vector<double> geoParam_; //!< parameters describing the geometry of the objects
    std::vector<double> geoParamML_; //!< parameters describing the geometry of the objects

    std::vector<double> alpha_; //!<Lorentz model pole parameter (see Taflove 2005 book ch 9)
    std::vector<double> xi_; //!<Lorentz model pole parameter (see Taflove 2005 book ch 9)
    std::vector<double> gamma_; //!<Lorentz model pole parameter (see Taflove 2005 book ch 9)

    std::vector<double> magAlpha_; //!<Lorentz model pole parameter for magnetic materials (equivlant model for normal dispersive material)
    std::vector<double> magXi_; //!<Lorentz model pole parameter for magnetic materials (equivlant model for normal dispersive material)
    std::vector<double> magGamma_; //!<Lorentz model pole parameter for magnetic materials (equivlant model for normal dispersive material)

    std::vector<double> chiAlpha_; //!<Lorentz model pole parameter for chiral materials (fourier transfrom of Condon params in Lindell 1994)
    std::vector<double> chiXi_; //!<Lorentz model pole parameter for chiral materials (fourier transfrom of Condon params in Lindell 1994)
    std::vector<double> chiGamma_; //!<Lorentz model pole parameter for chiral materials (fourier transfrom of Condon params in Lindell 1994)
    std::vector<double> chiGammaPrev_; //!<Lorentz model pole parameter for chiral materials for previous timestep (fourier transfrom of Condon params in Lindell 1994)

    std::vector<std::array<double,3>> dipE_; //!< Vector of arrays describing the direction of the electric dipole for each pol
    std::vector<std::array<double,3>> dipM_; //!< Vector of arrays describing the direction of the magnetic dipole for each pol

    std::vector<std::array<double,3>> dipChiE_; //!< Vector of arrays describing the direction of the electric dipole for each chiral pol
    std::vector<std::array<double,3>> dipChiM_; //!< Vector of arrays describing the direction of the magnetic dipole for each chiral pol

    std::vector<LorenzDipoleOscillator> pols_; //!< parameters for storing the material polarization properties of the system

    std::array<double,3> location_; //!< location of the center point of the object
    std::array<double,9> coordTransform_; //!< Coordinate Transform Matrix
    std::array<double,9> invCoordTransform_; //!< The inverse matrix of the Coordinate Transform Matrix
public:

    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { radius of curvature of the tip, the tip length }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    Obj(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     Object to be copied
     */
    Obj(const Obj &o);
    /**
     * @brief Calculates the dispersion parameters from physical ones
     * @details Show equations here
     *
     * @param dt time step
     */
    void setUpConsts (double dt);

    /**
     * @brief Checks if the material of the object is vacuum returns True if it is
     * @return True if object Material is vacuum false otherwise
     */
    inline bool isVac(){ return !( gamma().size() > 0 || epsInfty() != 1.0 || chiGamma().size() > 0 || tellegen() != 0.0 || magGamma().size() > 0 || muInfty() != 1.0 || ML() ); }

    /**
     * @brief      Accessor function to geoParam_
     *
     * @return the shape of the object
     */
    inline std::vector<double>& geo() {return geoParam_;}

    inline std::vector<double>& geoML() {return geoParamML_;}
    /**
     * @brief      Accessor function to location_
     *
     * @return the location of the object
     */
    inline std::array<double,3>& loc() {return location_;}
    /**
     * @brief      Accessor function to pols_
     *
     * @return the material polarization parameters of the object
     */
    inline std::vector<LorenzDipoleOscillator>& pols() {return pols_;}
    /**
     * @brief      Accessor function to alpha_
     *
     * @return the alpha value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& alpha() {return alpha_;}
    /**
     * @brief      Accessor function to xi_
     *
     * @return the xi value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& xi() {return xi_;}
    /**
     * @brief      Accessor function to gamma_
     *
     * @return the gamma value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& gamma() {return gamma_;}

    /**
     * @brief      Accessor function to magAlpha_
     *
     * @return the alpha value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& magAlpha() {return magAlpha_;}
    /**
     * @brief      Accessor function to magXi_
     *
     * @return the xi value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& magXi() {return magXi_;}
    /**
     * @brief      Accessor function to magGamma_
     *
     * @return the gamma value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& magGamma() {return magGamma_;}

    /**
     * @brief      Accessor function to chiAlpha_
     *
     * @return the alpha value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& chiAlpha() {return chiAlpha_;}
    /**
     * @brief      Accessor function to chiXi_
     *
     * @return the xi value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& chiXi() {return chiXi_;}
    /**
     * @brief      Accessor function to chiGamma_
     *
     * @return the gamma value for each Lorenz-Drude Pole in the material of the object
     */
    inline std::vector<double>& chiGamma() {return chiGamma_;}

    /**
     * @brief      Accessor function to chiGammaPrev_
     *
     * @return the gamma value for each Lorenz-Drude Pole in the material of the object for the previous time step
     */
    inline std::vector<double>& chiGammaPrev() {return chiGammaPrev_;}

    /**
     * @brief      Accessor function to eps_infty_
     *
     * @return     return the $\eps_{\infty}$
     */
    inline double epsInfty(){return eps_infty_;}

    /**
     * @brief      Accessor function to mu_infty_
     *
     * @return     returns $\mu_{\infty}$
     */
    inline double muInfty(){return mu_infty_;}
    /**
     * @brief      Accessor function to tellegen_
     *
     * @return     return the $\eps_{\infty}$
     */
    inline double tellegen(){return tellegen_;}

    /**
     * @return     the unit vectors of the object's coordinate system
     */
    inline std::array<std::array<double,3>,3> unitVec() {return unitVec_;}

    /**
     * @brief      Accessor function to ML_
     *
     * @return     True if object is a quantum emitter
     */
    inline bool ML(){return ML_;}

    /**
     * @brief      Accessor function to useOrientedDipols_
     *
     * @return     True if object has oriented dipoles
     */
    inline bool useOrdDip() {return useOrientedDipols_;}

    /**
     * @brief      Find the orientation of a particular electric dipole oscillator
     *
     * @param[in]  pp    oscillator index
     *
     * @return     How the dipoles of the structure are oriented for the oscillator number pp, if pp is larger than the size of the vector, return ISOTROPIC
     */
    inline MAT_DIP_ORIENTAITON dipOr(int pp){ if ( pp < dipOr_.size() )  return dipOr_[pp]; else  return MAT_DIP_ORIENTAITON::ISOTROPIC; }

    /**
     * @brief      Find the orientation of a particular magnetic dipole oscillator
     *
     * @param[in]  pp    oscillator index
     *
     * @return     How the dipoles of the structure are oriented for the oscillator number pp, if pp is larger than the size of the vector, return ISOTROPIC
     */
    inline MAT_DIP_ORIENTAITON magDipOr(int pp){ if ( pp  < magDipOr_.size() )  return magDipOr_[pp]; else  return MAT_DIP_ORIENTAITON::ISOTROPIC; }

    /**
     * @brief      Find the orientation of a particular chiral electric dipole oscillator
     *
     * @param[in]  pp    oscillator index
     *
     * @return     How the dipoles of the structure are oriented for the oscillator number pp, if pp is larger than the size of the vector, return ISOTROPIC
     */
    inline MAT_DIP_ORIENTAITON chiEDipOr(int pp){ if ( pp  < chiEDipOr_.size() )  return chiEDipOr_[pp]; else  return MAT_DIP_ORIENTAITON::ISOTROPIC; }

    /**
     * @brief      Find the orientation of a particular chiral electric dipole oscillator
     *
     * @param[in]  pp    oscillator index
     *
     * @return     How the dipoles of the structure are oriented for the oscillator number pp, if pp is larger than the size of the vector, return ISOTROPIC
     */
    inline MAT_DIP_ORIENTAITON chiMDipOr(int pp){ if ( pp  < chiEDipOr_.size() )  return chiMDipOr_[pp]; else  return MAT_DIP_ORIENTAITON::ISOTROPIC; }

    /**
     * @brief      Returns an array describing the unit vector of the electric dipole of oscillator pp
     *
     * @param[in]  pp    oscillator index
     *
     * @return     if pp < the size return the dipole at that point, otherwise return 0.0, 0.0, 0.0
     */
    inline std::array<double,3> dipE(int pp) { if ( pp < dipE_.size() )  return dipE_[pp]; else  return std::array<double,3>( {0.0, 0.0, 0.0} ); }

    /**
     * @brief      Returns an array describing the unit vector of the magnetic dipole of oscillator pp
     *
     * @param[in]  pp    oscillator index
     *
     * @return     if pp < the size return the dipole at that point, otherwise return 0.0, 0.0, 0.0
     */
    inline std::array<double,3> dipM(int pp) { if ( pp < dipM_.size() )  return dipM_[pp]; else  return std::array<double,3>( {0.0, 0.0, 0.0} ); }

    /**
     * @brief      Returns an array describing the unit vector of the electric dipole of oscillator of the chiral transition pp
     *
     * @param[in]  pp    oscillator index
     *
     * @return     if pp < the size return the dipole at that point, otherwise return 0.0, 0.0, 0.0
     */
    inline std::array<double,3> dipChiE(int pp) { if ( pp < dipChiE_.size() )  return dipChiE_[pp]; else  return std::array<double,3>( {0.0, 0.0, 0.0} ); }

    /**
     * @brief      Returns an array describing the unit vector of the magnetic dipole of oscillator of the chiral transition pp
     *
     * @param[in]  pp    oscillator index
     *
     * @return     if pp < the size return the dipole at that point, otherwise return 0.0, 0.0, 0.0
     */
    inline std::array<double,3> dipChiM(int pp) { if ( pp < dipChiM_.size() )  return dipChiM_[pp]; else  return std::array<double,3>( {0.0, 0.0, 0.0} ); }

    /**
     * @brief      Accessor function to dipNormCompE_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the normal component of the ppth electric dipole of the material, 0 if pp > size of the vector
     */
    inline double dipNormCompE(int pp) { if (pp < dipNormCompE_.size() ) return dipNormCompE_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to dipTanLatCompE_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the normal component of the ppth electric dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLatCompE(int pp) { if (pp < dipTanLatCompE_.size() ) return dipTanLatCompE_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to dipTanLongCompE_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the latitudinal component of the ppth electric dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLongCompE(int pp) { if (pp < dipTanLongCompE_.size() ) return dipTanLongCompE_[pp]; else return 0.0; }

    /**
     * @brief      Accessor function to dipNormCompM_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
longitudinal *
     * @return     The value of the normal component of the ppth electric dipole of the material, 0 if pp > size of the vector
     */
    inline double dipNormCompM(int pp) { if (pp < dipNormCompM_.size() ) return dipNormCompM_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to dipTanLatCompM_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the latitudinal component of the ppth magnetic dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLatCompM(int pp) { if (pp < dipTanLatCompM_.size() ) return dipTanLatCompM_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to dipTanLongCompM_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the longitudinal component of the ppth magnetic dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLongCompM(int pp) { if (pp < dipTanLongCompM_.size() ) return dipTanLongCompM_[pp]; else return 0.0; }

    /**
     * @brief      Accessor function to dipNormCompChiE_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the normal component of the ppth magnetic dipole of the material, 0 if pp > size of the vector
     */
    inline double dipNormCompChiE(int pp) { if (pp < dipNormCompChiE_.size() ) return dipNormCompChiE_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to return dipTanLatCompChiE_[
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the latitudinal component of the ppth chiral electric dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLatCompChiE(int pp) { if (pp < dipTanLatCompChiE_.size() ) return dipTanLatCompChiE_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to return dipTanLongCompChiE_[
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the longitudinal component of the ppth chiral electric dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLongCompChiE(int pp) { if (pp < dipTanLongCompChiE_.size() ) return dipTanLongCompChiE_[pp]; else return 0.0; }

    /**
     * @brief      Accessor function to dipNormCompChiM_[pp]
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the normal component of the ppth chiral electric dipole of the material, 0 if pp > size of the vector
     */
    inline double dipNormCompChiM(int pp) { if (pp < dipNormCompChiM_.size() ) return dipNormCompChiM_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to return dipTanLatCompChiM_[
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the latitudinal component of the ppth chiral magnetic dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLatCompChiM(int pp) { if (pp < dipTanLatCompChiM_.size() ) return dipTanLatCompChiM_[pp]; else return 0.0; }
    /**
     * @brief      Accessor function to return dipTanLongCompChiM_[
     *
     * @param[in]  pp    Index to the dipole moment to be accessed
     *
     * @return     The value of the longitudinal component of the ppth chiral magnetic dipole of the material, 0 if pp > size of the vector
     */
    inline double dipTanLongCompChiM(int pp) { if (pp < dipTanLongCompChiM_.size() ) return dipTanLongCompChiM_[pp]; else return 0.0; }

    /**
     * @brief      Converts a point from object space coordinates to real space coordinates
     *
     * @param[in]  pt    The point in object space
     *
     * @return     The point in real space
     */
    std::array<double, 3> ObjectSpace2RealSpace(const std::array<double,3>& pt);

    /**
     * @brief      Converts a point from real space coordinates to object space coordinates
     *
     * @param[in]  pt    The point in real space
     *
     * @return     The point in object space
     */
    std::array<double, 3> RealSpace2ObjectSpace(const std::array<double,3>& pt);
    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    virtual bool isObj(std::array<double,3> v, double dx, std::vector<double> geo) = 0;

    /**
     * @brief      Returns the distance between two points
     *
     * @param[in]  pt1   The point 1
     * @param[in]  pt2   The point 2
     *
     * @return     The distance between the points
     */
    double dist(std::array<double,3> pt1, std::array<double,3> pt2);

    /**
     * @return     the shape of the object
     */
    virtual SHAPE shape() = 0;

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    virtual void addMLBuff(double dx) = 0;

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    virtual std::array<double,3> findGradient(std::array<double,3> pt) = 0;
};

class sphere : public Obj
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { radius of sphere }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    sphere(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     sphere to be copied
     */
    sphere(const sphere &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (SPHERE)
     *
     * @return     SHAPE::SPHERE
     */
    SHAPE shape() {return SHAPE::SPHERE;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class hemisphere : public Obj
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { radius of hemi-sphere }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    hemisphere(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     hemisphere to be copied
     */
    hemisphere(const hemisphere &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (HEMISPHERE)
     *
     * @return     SHAPE::HEMISPHERE
     */
    SHAPE shape() {return SHAPE::HEMISPHERE;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class block : public Obj
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { length of edge in direction of uvec1, length of edge in direction of uvec2, length of edge in direction of uvec3 }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    block(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     Block to be copied
     */
    block(const block &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (BLOCK)
     *
     * @return     SHAPE::BLOCK
     */
    SHAPE shape() {return SHAPE::BLOCK;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; geoParamML_[2]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);

};
class rounded_block : public Obj
{
protected:
    std::array<std::array<double,3>,8> curveCens_; //!< centers of curvature for all the corners
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { length of edge in direction of uvec1, length of edge in direction of uvec2, length of edge in direction of uvec3, radius of curvature of the corners }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    rounded_block(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     rounded_block to be copied
     */
    rounded_block(const rounded_block &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (ROUNDED_BLOCK)
     *
     * @return     SHAPE::ROUNDED_BLOCK
     */
    SHAPE shape() {return SHAPE::ROUNDED_BLOCK;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; geoParamML_[2]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};
class ellipsoid : public Obj
{
private:
    std::array<double,3> axisCutNeg_; //!< Array storing the cutoff plane in the negative {axis1, axis2, axis3} directions
    std::array<double,3> axisCutPos_; //!< Array storing the cutoff plane in the positive {axis1, axis2, axis3} directions
    std::array<double,3> axisCutGlobal_; //!< Array storing the global cutoff plane {axis1, axis2, axis3} directions
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater         Vector containing all electric field dispersive material parameters
     * @param[in]  magMater      Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater      Vector containing all chiral material parameters
     * @param[in]  ML            True if object is a quantum emitter
     * @param[in]  dipOr         Describes how the dipoles of the material are oriented
     * @param[in]  geo           Vector Containing { length of axis in direction of uvec1, length of axis in direction of uvec2, length of axis in direction of uvec3 }
     * @param[in]  axisCutNeg    Array storing the cutoff plane in the negative {axis1, axis2, axis3} directions
     * @param[in]  axisCutPos    Array storing the cutoff plane in the positive {axis1, axis2, axis3} directions
     * @param[in]  axisCutGlobal Array storing the global cutoff plane {axis1, axis2, axis3} directions
     * @param[in]  loc           The location of the center of the object
     * @param[in]  unitVec       An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    ellipsoid(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> axisCutNeg, std::array<double,3> axisCutPos, std::array<double,3> axisCutGlobal, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     ellipsoid to be copied
     */
    ellipsoid(const ellipsoid &o);
    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (ELLIPSOID)
     *
     * @return     SHAPE::ELLIPSOID
     */
    SHAPE shape() {return SHAPE::ELLIPSOID;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; geoParamML_[2]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class hemiellipsoid : public Obj
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { length of axis in direction of uvec1, length of axis in direction of uvec2, length of axis in direction of uvec3
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    hemiellipsoid(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     hemiellipsoid to be copied
     */
    hemiellipsoid(const hemiellipsoid &o);
    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (HEMIELLIPSOID)
     *
     * @return     SHAPE::HEMIELLIPSOID
     */
    SHAPE shape() {return SHAPE::HEMIELLIPSOID;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; geoParamML_[2]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};
class cone : public Obj
{
public:

    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { radius of base, length of the truncated cone, height of cone if not truncated }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    cone(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     Cone to be copied
     */
    cone(const cone &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (CONE)
     *
     * @return     SHAPE::CONE
     */
    SHAPE shape() {return SHAPE::CONE;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) { geoParamML_[1]+=3.0*d;  }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class cylinder : public Obj
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { radius of the cylinder, length of the cylinder }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    cylinder(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     Cylinder to be copied
     */
    cylinder(const cylinder &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (CYLINDER)
     *
     * @return     SHAPE::CYLINDER
     */
    SHAPE shape() {return SHAPE::CYLINDER;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class tri_prism : public Obj
{
protected:
    double d0_; //!< base determinnant of the vertix locations
    std::vector<int> invVertIpiv_;  //!< pivot table for the inverse vertex matrix
    std::vector<double> invVertMat_;  //!< inverse of vertex Matrix
    std::array<std::array<double, 2>, 3> vertLocs_; //!< array storing the vertex locations
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { base of the triangle base, height of the triangle base, truncation ratio, length of the prism }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    tri_prism(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { base of the triangle base, height of the triangle base, truncation ratio, length of the prism }
     * @param[in]  vertLocs  Vector containing vertix point locations
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    tri_prism(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<std::array<double, 2>, 3> vertLocs, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     isosceles triangular prism to be copied
     */
    tri_prism(const tri_prism &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (TRIANGLE_PRISM)
     *
     * @return     SHAPE::TRIANGLE_PRISM
     */
    SHAPE shape() {return SHAPE::TRIANGLE_PRISM;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; geoParamML_[2]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);

    // double getVertDet(std::vector<double> dMat);
};

class ters_tip : public Obj
{
protected:
    std::array<double,3> radCen_; //!< location of the center point of the circle
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { radius of curvature of the tip, radius of the tip's upper base, the tip length }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    ters_tip(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     TERS Tip to be copied
     */
    ters_tip(const ters_tip &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (TERS_TIP)
     *
     * @return     SHAPE::TERS_TIP
     */
    inline SHAPE shape() {return SHAPE::TERS_TIP;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; geoParamML_[2]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class paraboloid : public Obj
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { a in the elliptical paraboloid equation, b in the elliptical paraboloid equation, the tip length }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    paraboloid(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     Parabolic TERS Tip to be copied
     */
    paraboloid(const paraboloid &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (PARABOLOID)
     *
     * @return     SHAPE::PARABOLOID
     */
    inline SHAPE shape() {return SHAPE::PARABOLOID;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class torus : public Obj
{
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { Radius of the ring of the Torus, radius of the circular cross section, minimum azimuthal angle of the arc, maximum azimuthal angle of the arc }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    torus(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     Parabolic TERS Tip to be copied
     */
    torus(const torus &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (TORUS)
     *
     * @return     SHAPE::TORUS
     */
    inline SHAPE shape() {return SHAPE::TORUS;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {geoParamML_[0]+=3.0*d; geoParamML_[1]+=3.0*d; }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);
};

class tetrahedron : public Obj
{
protected:
    double d0_; //!< base determinant of the vertex locations
    std::vector<int> invVertIpiv_; //!< pivot table for the inverse vertex matrix
    std::vector<double> invVertMat_; //!< inverse of vertex Matrix
    std::array<std::array<double, 3>, 4> vertLocs_; //!< array storing the vertex locations
public:
    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { side length of the tetraheadron and the height (will be used to set truncation) }
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    tetrahedron(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Constructor
     *
     * @param[in]  mater     Vector containing all electric field dispersive material parameters
     * @param[in]  magMater  Vector containing all magnetic field dispersive material parameters
     * @param[in]  chiMater  Vector containing all chiral material parameters
     * @param[in]  ML        True if object is a quantum emitter
     * @param[in]  dipOr     Describes how the dipoles of the material are oriented
     * @param[in]  geo       Vector Containing { Truncation points in barycenteric coordinates for each vertix}
     * @param[in]  vertLocs  Vector containing vertix point locations
     * @param[in]  loc       The location of the center of the object
     * @param[in]  unitVec   An array of vectors describing the coordinate transform to make the objects orientation in the grids along the x,y,z axises
     */
    tetrahedron(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<std::array<double, 3>, 4> vertLocs, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec);

    /**
     * @brief      Copy Constructor
     *
     * @param[in]  o     isosceles triangular prism to be copied
     */
    tetrahedron(const tetrahedron &o);

    /**
     * @brief      Determines if a point is inside the object
     *
     * @param[in]  v     real space point
     * @param[in]  dx    grid spacing
     *
     * @return     True if point is in the object, False otherwise.
     */
    bool isObj(std::array<double,3> v, double dx, std::vector<double> geo);

    /**
     * @brief      Returns the shape of the object (TETRAHEDRON)
     *
     * @return     SHAPE::TETRAHEDRON
     */
    SHAPE shape() {return SHAPE::TETRAHEDRON;}

    /**
     * @brief      Adds buffer space for ML polariztions
     *
     * @param[in]  dx    grid spacing
     */
    inline void addMLBuff(double d) {for(auto& vert : vertLocs_) dscal_(4, ( std::sqrt( std::accumulate(vert.begin(), vert.end(), 0.0, vecMagAdd<double>()) ) + 3.0*d)/std::sqrt( std::accumulate(vert.begin(), vert.end(), 0.0, vecMagAdd<double>()) ), vert.begin(), 1); }

    /**
     * @brief      Finds the gradient of the object at a specified point in real space
     *
     * @param[in]  pt    The point where the gradient should be calculated
     *
     * @return     The Gradient of the object
     */
    std::array<double,3> findGradient(std::array<double,3> pt);

    // double getVertDet(std::vector<double> dMat);
};
#endif

