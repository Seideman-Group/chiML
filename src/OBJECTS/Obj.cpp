/** @file OBJECTS/Obj.cpp
 *  @brief Class that stores all object/material information
 *
 *  A class that determines if a grid point is inside the object, stores the material parameters, and
 *  finds the gradient to determine the surface normal vector.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#include "Obj.hpp"

std::array<double, 3> Obj::RealSpace2ObjectSpace(const std::array<double,3>& pt)
{
    std::array<double,3> v_trans;
    std::array<double,3> v_cen;
    // Move origin to object's center
    std::transform(pt.begin(), pt.end(), location_.begin(), v_cen.begin(), std::minus<double>() );
    // Convert x, y, z coordinates to coordinates along the object's unit axis
    dgemv_('T', v_cen.size(), v_cen.size(), 1.0, coordTransform_.data(), v_cen.size(), v_cen.data(), 1, 0.0, v_trans.data(), 1 );
    return v_trans;
}

std::array<double, 3> Obj::ObjectSpace2RealSpace(const std::array<double,3>& pt)
{
    std::array<double,3> realSpacePt;
    dgemv_('T', pt.size(), pt.size(), 1.0, invCoordTransform_.data(), pt.size(), pt.data(), 1, 0.0, realSpacePt.data(), 1 );
    // Normalize the gradient vector
    if(std::accumulate(realSpacePt.begin(), realSpacePt.end(), 0.0, vecMagAdd<double>() ) < 1e-20)
        realSpacePt = {{ 0.0, 0.0, 0.0 }};
    else
        normalize(realSpacePt);
    return realSpacePt;
}

Obj::Obj(const Obj &o) :
    ML_(o.ML_),
    useOrientedDipols_(o.useOrientedDipols_),
    eps_infty_(o.eps_infty_),
    mu_infty_(o.mu_infty_),
    tellegen_(o.tellegen_),
    unitVec_(o.unitVec_),
    dipOr_(o.dipOr_),
    magDipOr_(o.magDipOr_),
    geoParam_(o.geoParam_),
    geoParamML_(o.geoParamML_),
    location_(o.location_),
    alpha_(o.alpha_),
    xi_(o.xi_),
    gamma_(o.gamma_),
    magAlpha_(o.magAlpha_),
    magXi_(o.magXi_),
    magGamma_(o.magGamma_),
    chiAlpha_(o.chiAlpha_),
    chiXi_(o.chiXi_),
    chiGamma_(o.chiGamma_),
    chiGammaPrev_(o.chiGammaPrev_),
    dipE_(o.dipE_),
    dipM_(o.dipM_),
    pols_(o.pols_)
{}

Obj::Obj(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    ML_(ML),
    useOrientedDipols_(false),
    eps_infty_(eps_infty),
    mu_infty_(mu_infty),
    tellegen_(tellegen),
    unitVec_(unitVec),
    geoParam_(geo),
    geoParamML_(geo),
    location_(loc),
    pols_(pols)
{
    for(int ii = 0; ii < 3; ++ii)
    {
        for(int jj = 0; jj < 3; ++jj)
        {
            std::array<double,3>cartCoord = {{ 0, 0, 0 }};
            cartCoord[jj] = 1.0;
            coordTransform_[(ii)*3+jj] =  ddot_(unitVec[ii].size(), unitVec[ii].data(), 1, cartCoord.data(), 1) / ( std::sqrt( std::accumulate(unitVec[ii].begin(),unitVec[ii].end(),0.0, vecMagAdd<double>() ) ) );
        }
    }
    std::copy_n(coordTransform_.begin(), coordTransform_.size(), invCoordTransform_.begin());
    std::vector<int>invIpiv(invCoordTransform_.size(), 0.0);
    std::vector<double>work(invCoordTransform_.size(), -1) ;
    int info;
    dgetrf_(3, 3, invCoordTransform_.data(), 3, invIpiv.data(), &info);
    if(info < 0)
        throw std::logic_error("The " + std::to_string(-1.0*info) + "th value of the coordTransform_ matrix is an illegal value.");
    else if(info > 0)
        throw std::logic_error("WARNING: The " + std::to_string(info) + "th diagonal element of u is 0. The invCoordTransform_ can't be calculated!");
    dgetri_(3, invCoordTransform_.data(), 3, invIpiv.data(), work.data(), work.size(),  &info);
    if(info < 0)
        throw std::logic_error("The " + std::to_string(-1.0*info) + "th value of the coordTransform_ matrix is an illegal value.");
    else if(info > 0)
        throw std::logic_error("WARNING: The " + std::to_string(info) + "th diagonal element of u is 0. The invCoordTransform_ can't be calculated!");

    // for(int ii = 0; ii < 3; ++ii)
    //     for(int jj = 0; jj < 3; ++jj)
    //         invCoordTransform_[(jj)*3+ii] = coordTransform_[ ( (ii+1)%3 )*3 + ( (jj+1)%3 ) ]*coordTransform_[ ( (ii+2)%3 )*3 + ( (jj+2)%3 ) ] - coordTransform_[ ( (ii+2)%3 )*3 + ( (jj+1)%3 ) ]*coordTransform_[ ( (ii+1)%3 )*3 + ( (jj+2)%3 ) ];

    // double detCoord = coordTransform_[0]*invCoordTransform_[0] +  coordTransform_[3]*invCoordTransform_[1] + coordTransform_[6]*invCoordTransform_[2];
    // if(std::abs(detCoord) < 1e-15)
    //     throw std::logic_error("Determinant of the coordTransform_ matrix is 0, inverse is undefined.");
    // dscal_(invCoordTransform_.size(), 1.0/detCoord, invCoordTransform_.data(), 1);
}

sphere::sphere(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}
hemisphere::hemisphere(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}
cylinder::cylinder(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}
cone::cone(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}
ellipsoid::ellipsoid(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> axisCutNeg, std::array<double,3> axisCutPos, std::array<double,3> axisCutGlobal, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec),
    axisCutNeg_(axisCutNeg),
    axisCutPos_(axisCutPos),
    axisCutGlobal_(axisCutGlobal)
{}
hemiellipsoid::hemiellipsoid(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}
block::block(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}
rounded_block::rounded_block(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{
    double radCurv = geoParam_[geoParam_.size()-1];
    curveCens_[0] = {     geoParam_[0]/2.0 - radCurv,      geoParam_[1]/2.0 - radCurv,      geoParam_[2]/2.0 - radCurv};
    curveCens_[1] = {-1.0*geoParam_[0]/2.0 + radCurv,      geoParam_[1]/2.0 - radCurv,      geoParam_[2]/2.0 - radCurv};
    curveCens_[2] = {     geoParam_[0]/2.0 - radCurv, -1.0*geoParam_[1]/2.0 + radCurv,      geoParam_[2]/2.0 - radCurv};
    curveCens_[3] = {-1.0*geoParam_[0]/2.0 + radCurv, -1.0*geoParam_[1]/2.0 + radCurv,      geoParam_[2]/2.0 - radCurv};
    curveCens_[4] = {     geoParam_[0]/2.0 - radCurv,      geoParam_[1]/2.0 - radCurv, -1.0*geoParam_[2]/2.0 + radCurv};
    curveCens_[5] = {-1.0*geoParam_[0]/2.0 + radCurv,      geoParam_[1]/2.0 - radCurv, -1.0*geoParam_[2]/2.0 + radCurv};
    curveCens_[6] = {     geoParam_[0]/2.0 - radCurv, -1.0*geoParam_[1]/2.0 + radCurv, -1.0*geoParam_[2]/2.0 + radCurv};
    curveCens_[7] = {-1.0*geoParam_[0]/2.0 + radCurv, -1.0*geoParam_[1]/2.0 + radCurv, -1.0*geoParam_[2]/2.0 + radCurv};
}

tri_prism::tri_prism(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec),
    invVertIpiv_(3, 0),
    invVertMat_(9,1.0)
{
    vertLocs_[0] = {{  0.5*geoParam_[0], -0.5*geoParam_[1] }};
    vertLocs_[1] = {{ -0.5*geoParam_[0], -0.5*geoParam_[1] }};
    vertLocs_[2] = {{  0               ,  0.5*geoParam_[1] }};
    // For scaling recenter the object space geometry to the centroid
    std::array<double, 3> centroidLoc = {0.0,0.0,0.0};
    for(int ii = 0; ii < 3; ++ii)
        std::transform(vertLocs_[ii].begin(), vertLocs_[ii].end(), centroidLoc.begin(), centroidLoc.begin(), [](double vert, double cent){return cent + vert/3.0;});
    std::transform(location_.begin(), location_.end(), centroidLoc.begin(), location_.begin(), std::plus<double>() );
    for(int ii = 0; ii < 3; ++ii)
    {
        std::transform(vertLocs_[ii].begin(), vertLocs_[ii].end(), centroidLoc.begin(), vertLocs_[ii].begin(), std::minus<double>() );
        dcopy_(2, vertLocs_[ii].begin(), 1, &invVertMat_[ii], 3);
    }
    int info;
    dgetrf_(3, 3, invVertMat_.data(), 3, invVertIpiv_.data(), &info);
    d0_ = getLUFactMatDet(invVertMat_, invVertIpiv_, 3);
    if(d0_ == 0)
        throw std::logic_error("The triangular base is a line.");
    std::vector<double>work(9,0.0);
    dgetri_(3, invVertMat_.data(), 3, invVertIpiv_.data(), work.data(), work.size(),  &info);

    geoParam_ = {{ 1.0, 1.0, 1.0, geoParam_[3] }};
}

tri_prism::tri_prism(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<std::array<double, 2>, 3> vertLocs, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec),
    invVertIpiv_(3, 0),
    invVertMat_(9,1.0),
    vertLocs_(vertLocs)
{
    // For scaling recenter the object space geometry to the centroid
    std::array<double, 3> centroidLoc = {0.0,0.0,0.0};
    for(int ii = 0; ii < 3; ++ii)
        std::transform(vertLocs_[ii].begin(), vertLocs_[ii].end(), centroidLoc.begin(), centroidLoc.begin(), [](double vert, double cent){return cent + vert/3.0;});
    std::transform(location_.begin(), location_.end(), centroidLoc.begin(), location_.begin(), std::plus<double>() );
    for(int ii = 0; ii < 3; ++ii)
    {
        std::transform(vertLocs_[ii].begin(), vertLocs_[ii].end(), centroidLoc.begin(), vertLocs_[ii].begin(), std::minus<double>() );
        dcopy_(2, vertLocs_[ii].begin(), 1, &invVertMat_[ii], 3);
    }
    int info;
    dgetrf_(3, 3, invVertMat_.data(), 3, invVertIpiv_.data(), &info);
    d0_ = getLUFactMatDet(invVertMat_, invVertIpiv_, 3);
    if(d0_ == 0)
        throw std::logic_error("The triangular base is a line.");
    std::vector<double>work(9,0.0);
    dgetri_(3, invVertMat_.data(), 3, invVertIpiv_.data(), work.data(), work.size(),  &info);

}

ters_tip::ters_tip(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{
    radCen_ = {{ 0.0, 0.0, -1.0*geo[2]/2.0 + geo[0]/2.0 }};
}
paraboloid::paraboloid(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}

torus::torus(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec)
{}

tetrahedron::tetrahedron(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec),
    invVertIpiv_(4, 0),
    invVertMat_(16,1.0)
{
    std::vector<double> vert = {{ std::sqrt(8.0/9.0), 0, -1.0/3.0 }};
    dgemv_('T', vert.size(), vert.size(), 1.0, coordTransform_.data(), vert.size(), vert.data(), 1, 0.0, vertLocs_[0].data(), 1 );

    vert = {{-1.0*std::sqrt(2.0/9.0), std::sqrt(2.0/3.0), -1.0/3.0 }};
    dgemv_('T', vert.size(), vert.size(), 1.0, coordTransform_.data(), vert.size(), vert.data(), 1, 0.0, vertLocs_[1].data(), 1 );

    vert = {{-1.0*std::sqrt(2.0/9.0), -1.0*std::sqrt(2.0/3.0), -1.0/3.0 }};
    dgemv_('T', vert.size(), vert.size(), 1.0, coordTransform_.data(), vert.size(), vert.data(), 1, 0.0, vertLocs_[2].data(), 1 );

    vert = {{ 0, 0, 1}};
    dgemv_('T', vert.size(), vert.size(), 1.0, coordTransform_.data(), vert.size(), vert.data(), 1, 0.0, vertLocs_[3].data(), 1 );

    for(int ii = 0; ii < 4; ++ii)
    {
        dscal_(3, geoParam_[0]*std::sqrt(3.0/8.0), vertLocs_[ii].begin(), 1);
        daxpy_(3, 1.0, location_.begin(), 1, vertLocs_[ii].begin(), 1);
    }
    for(int ii = 0; ii < 4; ++ii)
        dcopy_(3, vertLocs_[ii].begin(), 1, &invVertMat_[ii], 4);
    int info;
    dgetrf_(4, 4, invVertMat_.data(), 4, invVertIpiv_.data(), &info);
    d0_ = getLUFactMatDet(invVertMat_, invVertIpiv_, 4);
    if(d0_ == 0)
        throw std::logic_error("The tetrahedron is planer.");
    std::vector<double>work(16,0.0);
    dgetri_(4, invVertMat_.data(), 4, invVertIpiv_.data(), work.data(), work.size(),  &info);

    double heightRat = ( geoParam_[1]/( geoParam_[0]*sqrt(2.0/3.0) )*(5.0/4.0) ) - 1.0/4.0;
    std::array<double,3>v4(location_);
    daxpy_(3, heightRat, vertLocs_[3].begin(), 1, v4.data(), 1);
    std::array<double,4> v4Bary = cart2bary<double,4>(invVertMat_, v4, d0_);
    geoParam_ = {{ 1.0, 1.0, 1.0, v4Bary[3] }};
}

tetrahedron::tetrahedron(double eps_infty, double mu_infty, double tellegen, std::vector<LorenzDipoleOscillator> pols, bool ML, std::vector<double> geo, std::array<std::array<double, 3>, 4> vertLocs, std::array<double,3> loc, std::array<std::array<double,3>,3> unitVec) :
    Obj(eps_infty, mu_infty, tellegen, pols, ML, geo, loc, unitVec),
    invVertIpiv_(4, 0),
    invVertMat_(16,1.0),
    vertLocs_(vertLocs)
{
    for(int ii = 0; ii < 4; ++ii)
        dcopy_(3, vertLocs_[ii].begin(), 1, &invVertMat_[ii], 4);
    int info;
    dgetrf_(4, 4, invVertMat_.data(), 4, invVertIpiv_.data(), &info);
    d0_ = getLUFactMatDet(invVertMat_, invVertIpiv_, 4);
    if(d0_ == 0)
        throw std::logic_error("The tetrahedron is planer.");
    std::vector<double>work(16,0.0);
    dgetri_(4, invVertMat_.data(), 4, invVertIpiv_.data(), work.data(), work.size(),  &info);

    if(d0_ == 0)
        throw std::logic_error("The tetrahedron is planer.");
}

sphere::sphere(const sphere &o) : Obj(o)
{}
hemisphere::hemisphere(const hemisphere &o) : Obj(o)
{}
cone::cone(const cone &o) : Obj(o)
{}
cylinder::cylinder(const cylinder &o) : Obj(o)
{}
block::block(const block &o) : Obj(o)
{}
rounded_block::rounded_block(const rounded_block &o) : Obj(o), curveCens_(o.curveCens_)
{}
ellipsoid::ellipsoid(const ellipsoid &o) : Obj(o), axisCutNeg_(o.axisCutNeg_), axisCutPos_(o.axisCutPos_), axisCutGlobal_(o.axisCutGlobal_)
{}
hemiellipsoid::hemiellipsoid(const hemiellipsoid &o) : Obj(o)
{}
tri_prism::tri_prism(const tri_prism &o) : Obj(o)
{}
ters_tip::ters_tip(const ters_tip &o) : Obj(o), radCen_(o.radCen_)
{}
paraboloid::paraboloid(const paraboloid &o) : Obj(o)
{}
torus::torus(const torus &o) : Obj(o)
{}

void Obj::setUpConsts (double dt)
{
    // Converts Lorentzian style functions into constants that can be used for time updates based on Taflove Chapter 9
    for(auto& pol : pols_)
    {
        if(pol.dipOrE_ != MAT_DIP_ORIENTAITON::ISOTROPIC || pol.dipOrM_ != MAT_DIP_ORIENTAITON::ISOTROPIC)
            useOrientedDipols_ = true;

        double sigP = pol.sigP_;
        double sigM = pol.sigM_;
        double tau  = pol.tau_;
        double gam  = pol.gam_;
        double omg  = pol.omg_;

        if(std::abs( sigP ) != 0.0)
        {
            dipOr_.push_back(pol.dipOrE_);
            dipE_.push_back(pol.uVecDipE_);
            dipNormCompE_.push_back(pol.normCompWeightE_);
            dipTanLatCompE_.push_back(pol.tangentLatCompWeightE_);
            dipTanLongCompE_.push_back(pol.tangentLongCompWeightE_);
            alpha_.push_back( ( (2-pow(omg*dt,2.0))    / (1+gam*dt) ) );
               xi_.push_back( ( (gam*dt -1)            / (1+gam*dt) ) );
            gamma_.push_back( ( (sigP*pow(omg*dt,2.0)) / (1+gam*dt) ) );
        }
        if(std::abs( sigM ) != 0.0)
        {
            magDipOr_.push_back(pol.dipOrM_);
            dipM_.push_back(pol.uVecDipM_);
            dipNormCompM_.push_back(pol.normCompWeightM_);
            dipTanLatCompM_.push_back(pol.tangentLatCompWeightM_);
            dipTanLongCompM_.push_back(pol.tangentLongCompWeightM_);
            magAlpha_.push_back( ( (2-pow(omg*dt,2.0))   / (1+gam*dt) ) );
               magXi_.push_back( ( (gam*dt -1)           / (1+gam*dt) ) );
            magGamma_.push_back( ( (sigM*pow(omg*dt,2.0)) / (1+gam*dt) ) );
        }
        if(std::abs( tau ) != 0.0)
        {
            chiEDipOr_.push_back(pol.dipOrE_);
            chiMDipOr_.push_back(pol.dipOrM_);
            dipChiE_.push_back(pol.uVecDipE_);
            dipChiM_.push_back(pol.uVecDipM_);

            dipNormCompChiE_.push_back(pol.normCompWeightE_);
            dipTanLatCompChiE_.push_back(pol.tangentLatCompWeightE_);
            dipTanLongCompChiE_.push_back(pol.tangentLongCompWeightE_);

            dipNormCompChiM_.push_back(pol.normCompWeightM_);
            dipTanLatCompChiM_.push_back(pol.tangentLatCompWeightM_);
            dipTanLongCompChiM_.push_back(pol.tangentLongCompWeightM_);

            chiAlpha_.push_back( ( (2-pow(omg*dt,2.0))   / (1+gam*dt) ) );
               chiXi_.push_back( ( (gam*dt -1)           / (1+gam*dt) ) );
                chiGamma_.push_back( ( -1.0/dt ) * ( (tau*pow(omg*dt,2.0) ) / ( (1+gam*dt) ) ) ); //!< 1/dt for gamma is due to the time derivative needed for chiral interactions
            chiGammaPrev_.push_back( (  1.0/dt ) * ( (tau*pow(omg*dt,2.0) ) / ( (1+gam*dt) ) ) ); //!< 1/dt for gamma is due to the time derivative needed for chiral interactions
                // chiGamma_.push_back( ( pol.molec_ ? gam/2.0 - 1.0/dt : -1.0/dt ) * ( (tau*pow(omg*dt,2.0) ) / ( (1+gam*dt) ) ) ); //!< 1/dt for gamma is due to the time derivative needed for chiral interactions
            // chiGammaPrev_.push_back( ( pol.molec_ ? gam/2.0 + 1.0/dt :  1.0/dt ) * ( (tau*pow(omg*dt,2.0) ) / ( (1+gam*dt) ) ) ); //!< 1/dt for gamma is due to the time derivative needed for chiral interactions

        }
    }
    alpha_.reserve( alpha_.size() );
    xi_.reserve( xi_.size() );
    gamma_.reserve( gamma_.size() );

    magAlpha_.reserve( magAlpha_.size() );
    magXi_.reserve( magXi_.size() );
    magGamma_.reserve( magGamma_.size() );

    chiAlpha_.reserve( chiAlpha_.size() );
    chiXi_.reserve( chiXi_.size() );
    chiGamma_.reserve( chiGamma_.size() );
    chiGammaPrev_.reserve( chiGammaPrev_.size() );
}

bool sphere::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    if(dist(v,location_) > geo[0] + dx/1.0e6)
    {
        return false;
    }
    return true;
}

bool hemisphere::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    if(dist(v,location_) > geo[0] + dx/1.0e6)
    {
        // Move origin to object's center
        return false;
    }
    // Convert x, y, z coordinates to coordinates along the object's unit axis
    std::array<double,3> v_trans;
    // Do geo checks based on the the object oriented coordinates
    std::array<double,3> v_cen;
    std::transform(v.begin(), v.end(), location_.begin(), v_cen.begin(), std::minus<double>() );
    dgemv_('T', v_cen.size(), v_cen.size(), 1.0, coordTransform_.data(), v_cen.size(), v_cen.data(), 1, 0.0, v_trans.data(), 1 );
    if(v_trans[0] > dx/1e6)
        return false;
    return true;
}

bool block::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates
    for(int ii = 0; ii < v.size(); ii++)
    {
        if((v_trans[ii] > geo[ii]/2.0 + dx/1.0e6) || (v_trans[ii] < -1.0*geo[ii]/2.0 - dx/1.0e6))
            return false;
    }
    return true;
}
bool rounded_block::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates
    double radCurv = geo.back();
    for(int ii = 0; ii < v.size(); ii++)
    {
        if((v_trans[ii] > geo[ii]/2.0 + dx/1.0e6) || (v_trans[ii] < -1.0*geo[ii]/2.0 - dx/1.0e6))
            return false;
    }
    if( ( (v_trans[0] > geo[0]/2.0 - radCurv + dx/1.0e6) || (v_trans[0] < -1.0*geo[0]/2.0 + radCurv - dx/1.0e6) ) ||
        ( (v_trans[1] > geo[1]/2.0 - radCurv + dx/1.0e6) || (v_trans[1] < -1.0*geo[1]/2.0 + radCurv - dx/1.0e6) ) ||
        ( (v_trans[2] > geo[2]/2.0 - radCurv + dx/1.0e6) || (v_trans[2] < -1.0*geo[2]/2.0 + radCurv - dx/1.0e6) ) )
    {
        for(int cc = 0; cc < curveCens_.size(); cc++)
            if( ( std::sqrt( std::pow(v_trans[0] - curveCens_[cc][0], 2.0) + std::pow(v_trans[1] - curveCens_[cc][1], 2.0) ) < radCurv + dx/1.0e6 ) || ( std::sqrt( std::pow(v_trans[2] - curveCens_[cc][2], 2.0) + std::pow(v_trans[1] - curveCens_[cc][1], 2.0) ) < radCurv + dx/1.0e6 ) || ( std::sqrt( std::pow(v_trans[0] - curveCens_[cc][0], 2.0) + std::pow(v_trans[2] - curveCens_[cc][2], 2.0) ) < radCurv + dx/1.0e6 )  )
                return true;
        if( (v_trans[0] < geo[0]/2.0 - radCurv + dx/1.0e6) && (v_trans[0] > -1.0*geo[0]/2.0 + radCurv - dx/1.0e6) && (v_trans[1] < geo[1]/2.0 - radCurv + dx/1.0e6) && (v_trans[1] > -1.0*geo[1]/2.0 + radCurv - dx/1.0e6) ||
            (v_trans[2] < geo[2]/2.0 - radCurv + dx/1.0e6) && (v_trans[2] > -1.0*geo[2]/2.0 + radCurv - dx/1.0e6) && (v_trans[1] < geo[1]/2.0 - radCurv + dx/1.0e6) && (v_trans[1] > -1.0*geo[1]/2.0 + radCurv - dx/1.0e6) ||
            (v_trans[0] < geo[0]/2.0 - radCurv + dx/1.0e6) && (v_trans[0] > -1.0*geo[0]/2.0 + radCurv - dx/1.0e6) && (v_trans[2] < geo[2]/2.0 - radCurv + dx/1.0e6) && (v_trans[2] > -1.0*geo[2]/2.0 + radCurv - dx/1.0e6) )
            return true;
        return false;
    }
    return true;
}

bool ellipsoid::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates
    double ptSum = 0.0;
    for( int ii = 0; ii < ( geo[2] != 0.0 ? v_trans.size() : 2 ); ++ii)
        ptSum += std::pow( (v_trans[ii]-dx/1.0e6) / (geo[ii]/2.0), 2.0 );

    if(1.0 < ptSum )
        return false;
    for(int ii = 0; ii < 3; ++ii)
    {
        if( (2.0*v_trans[ii]/geo[ii] < axisCutNeg_[ii]) || (2.0*v_trans[ii]/geo[ii] > axisCutPos_[ii]) )
            return false;
        if( axisCutGlobal_[ii] >= 0.0 && (v_trans[ii] + geo[ii]/2.0) / geo[ii] >= axisCutGlobal_[ii])
            return false;
        else if( axisCutGlobal_[ii] <= 0.0 && (v_trans[ii] - geo[ii]/2.0) / geo[ii] <= axisCutGlobal_[ii] )
            return false;
    }
    return true;
}

bool hemiellipsoid::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates
    if(v_trans[0] > dx/1e6)
        return false;
    double ptSum = 0.0;
    for( int ii = 0; ii < ( geo[2] != 0.0 ? v_trans.size() : 2 ); ++ii)
        ptSum += std::pow( (v_trans[ii]-dx/1.0e6) / (geo[ii]/2.0), 2.0 );
    if(1.0 < ptSum )
        return false;
    return true;
}

bool tri_prism::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // If prism point is outside prism's length return false
    if( (v_trans[2] < -1.0*geo[3]/2.0 - dx*1e-6) || (v_trans[2] > geo[3]/2.0 + dx*1e-6) )
        return false;
    // Is point within the triangle's cross-section?
    std::array<double,2> v_transTriFace = {v_trans[0], v_trans[1]};
    std::array<double,3> baryCenCoords = cart2bary<double,3>(invVertMat_, v_transTriFace, d0_);
    // If Barycenteric coordinates are < 0 or > geoParam then they must be outside the cross-section
    for(int ii = 0; ii < 3; ++ii)
        if(baryCenCoords[ii] < -1e-13 || baryCenCoords[ii] > geo[ii]+1e-13)
            return false;
    // Sum barycenteric coords must = 1
    if(std::abs( 1 - std::accumulate( baryCenCoords.begin(), baryCenCoords.end(), 0.0 ) ) > 1e-14 )
    {
        std::cout << std::accumulate( baryCenCoords.begin(), baryCenCoords.end(), 0.0 ) << '\t' << d0_ << '\t' << v[0] << '\t' << v[1] << '\t' << v[2] << std::endl;
        throw std::logic_error("The sum of the boundary checks for a triangular base does not equal the determinant of the verticies, despite passing the checks. This is an error.");
    }
    return true;
}

bool tetrahedron::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double, 4> baryCenCoords = cart2bary<double,4>(invVertMat_, v, d0_);
    // If Barycenteric coordinates are < 0 or > geoParam then they must be outside the cross-section
    for(int ii = 0; ii < 4; ++ii)
        if(baryCenCoords[ii] < -1e-6*dx|| (baryCenCoords[ii] + dx*1e-6) > geo[ii])
            return false;

    // Sum barycenteric coords must = 1
    if(std::abs( 1 - std::accumulate( baryCenCoords.begin(), baryCenCoords.end(), 0.0 ) ) > 1e-14 )
    {
        std::cout << std::accumulate( baryCenCoords.begin(), baryCenCoords.end(), 0.0 ) << '\t' << d0_ << '\t' << v[0] << '\t' << v[1] << '\t' << v[2] << std::endl;
        throw std::logic_error("The sum of the boundary checks for a tetrahedron does not equal the determinant of the verticies, despite passing the checks. This is an error.");
    }
    return true;
}

bool cone::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates
    if( (v_trans[1] < -1.0*geo[1]/2.0 - dx*1e-6) || (v_trans[1] > geo[1]/2.0 + dx*1e-6) )
        return false;
    else if( dist(std::array<double,3>( {{ v_trans[0], 0.0, v_trans[2] }} ), std::array<double,3>({{0,0,0}}) ) > std::pow(geo[0]/geo[2] * ( v_trans[2] - geo[1]/2.0 ), 2.0) )
        return false;
    return true;
}

bool cylinder::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates

    if( (v_trans[1] < -1.0*geo[1]/2.0 - dx*1e-6) || (v_trans[1] > geo[1]/2.0 + dx*1e-6) )
        return false;
    else if( dist(std::array<double,3>( {{ v_trans[0], 0.0, v_trans[2] }} ), std::array<double,3>({{0,0,0}}) ) > geo[0] + 1.0e-6*dx )
        return false;
    return true;
}

bool ters_tip::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates
    if( dist(v_trans, radCen_) < geo[0]/2.0 )
        return true;

    if( (v_trans[2] < geo[0] - geo[2]/2.0 - dx*1e-6) || (v_trans[2] > geo[1]/2.0 + dx*1e-6) )
        return false;
    else if( ( std::pow( v_trans[0], 2.0 ) + std::pow( v_trans[1], 2.0 ) ) * std::pow( cos(geo[0]), 2.0 ) - std::pow( v_trans[2] * sin(geo[0]), 2.0)  > dx*1e-6 )
        return false;

    return true;
}

bool paraboloid::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    // Do geo checks based on the the object oriented coordinates
    v_trans[1] += geo[2]/2.0;
    if(v_trans[1] > geo[2])
        return false;
    else if( std::pow(v_trans[0]/geo[0], 2.0) + std::pow(v_trans[2]/geo[1], 2.0)  > v_trans[1] )
        return false;
    return true;
}

bool torus::isObj(std::array<double,3> v, double dx, std::vector<double> geo)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(v);
    if( std::pow( std::sqrt( std::pow(v_trans[0], 2.0) + std::pow(v_trans[1], 2.0) ) - geo[0] , 2.0) > std::pow(geo[1],2.0) - std::pow(v_trans[2],2.0) )
        return false;
    double phiPt = v_trans[0] !=0 ? std::atan(v_trans[1]/v_trans[0]) : v_trans[1]/std::abs(v_trans[1]) * M_PI / 2.0;
    if(v_trans[0] < 0)
        phiPt += M_PI;
    else if(v_trans[1] < 0)
        phiPt += 2.0*M_PI;
    if(phiPt < geo[2] || phiPt > geo[3])
        return false;
    return true;
}

double Obj::dist(std::array<double,3> pt1, std::array<double,3> pt2)
{
    double sum = 0;
    for(int cc = 0; cc < pt1.size(); cc ++)
        sum += std::pow((pt1[cc]-pt2[cc]),2);
    return sqrt(sum);
}

std::array<double, 3> sphere::findGradient(std::array<double,3> pt)
{
    std::array<double,3> grad;
    // Move origin to object's center
    std::transform(pt.begin(), pt.end(), location_.begin(), grad.begin(), std::minus<double>() );
    // Find the magnitude of the gradient vector (grad in this case since the radial and gradient vector are parallel)
    if(std::accumulate(grad.begin(), grad.end(), 0.0, vecMagAdd<double>() ) < 1e-20)
        grad = { 0.0, 0.0, 0.0 };
    else
        normalize(grad);
    return grad;
}

std::array<double, 3> hemisphere::findGradient(std::array<double,3> pt)
{
    std::array<double,3> grad;
    // Move origin to object's center
    std::transform(pt.begin(), pt.end(), location_.begin(), grad.begin(), std::minus<double>() );
    // Find the magnitude of the gradient vector (grad in this case since the radial and gradient vector are parallel)
    if(std::accumulate(grad.begin(), grad.end(), 0.0, vecMagAdd<double>() ) < 1e-20)
        grad = { 0.0, 0.0, 0.0 };
    else
        normalize(grad);
    return grad;
}

std::array<double, 3> ellipsoid::findGradient(std::array<double,3> pt)
{
    std::array<double,3> grad = RealSpace2ObjectSpace(pt);
    // Find gradient by modifying the v_trans
    std::transform(grad.begin(), grad.end(), geoParam_.begin(), grad.begin(), [](double g, double geo){return g/std::pow(geo,2.0); } );
    // Convert gradient back to real space
    return ObjectSpace2RealSpace(grad);
}

std::array<double, 3> hemiellipsoid::findGradient(std::array<double,3> pt)
{
    std::array<double,3> grad = RealSpace2ObjectSpace(pt);
    // Find gradient by modifying the v_trans
    std::transform(grad.begin(), grad.end(), geoParam_.begin(), grad.begin(), [](double g, double geo){return g/std::pow(geo,2.0); } );
    // Convert gradient back to real space
    return ObjectSpace2RealSpace(grad);

}

std::array<double, 3> cone::findGradient(std::array<double,3> pt)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(pt);
    // Find gradient by modifying the v_trans
    double z_prime = v_trans[2] + geoParam_[1]/2.0;
    double c = geoParam_[0]/geoParam_[2];
    double z0 = z_prime * (1.0 + std::sqrt( (std::pow(v_trans[0],2.0) + std::pow(v_trans[1],2.0) ) / std::pow(c,2.0) ) );
    std::array<double,3> grad = {{ 2.0*v_trans[0], 2.0*v_trans[1], -2.0*std::pow(c,2.0) * ( z_prime - z0 ) }};
    if(std::abs(v_trans[2]) == std::abs(z0/2.0) )
        grad = {{0, 0, -2.0*v_trans[2]/geoParam_[1] }};
    // Convert gradient back to real space
    return ObjectSpace2RealSpace(grad);
}

std::array<double, 3> cylinder::findGradient(std::array<double,3> pt)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(pt);
    double r   = std::sqrt( std::pow(v_trans[0], 2.0) + std::pow(v_trans[2], 2.0) );
    double len = geoParam_[1] * (r / geoParam_[0] );
    std::array<double,3> grad = {{ v_trans[0], 0.0, v_trans[2] }};
    if(std::abs(v_trans[1]) >= len/2.0 )
        grad = {{ 0.0, v_trans[1]/std::abs(v_trans[1]), 0.0 }};
    return ObjectSpace2RealSpace(grad);
}

std::array<double, 3> block::findGradient(std::array<double,3> pt)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(pt);
    std::array<double,3> grad = {0.0,0.0,0.0};
    std::array<double,3> ptRat;
    std::transform(v_trans.begin(), v_trans.end(), geoParam_.begin(), ptRat.begin(), std::divides<double>() );
    int ptRatMax = idamax_(ptRat.size(), ptRat.data(), 1) - 1;
    for(int ii = 0; ii < 3; ++ii)
        if(ptRat[ii] == ptRat[ptRatMax])
            grad[ ii ] = v_trans[ii] >=0 ? 1.0 : -1.0;
    return ObjectSpace2RealSpace(grad);
}

std::array<double, 3> rounded_block::findGradient(std::array<double,3> pt)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(pt);
    std::array<double,3> grad = {0.0,0.0,0.0};
    std::array<double,3> ptRat;
    std::transform(v_trans.begin(), v_trans.end(), geoParam_.begin(), ptRat.begin(), std::divides<double>() );
    int ptRatMax = idamax_(ptRat.size(), ptRat.data(), 1) - 1;
    double t = ptRat[ptRatMax];
    bool atCorner = true;
    for(int ii = 0; ii < 3; ++ii)
        if(v_trans[ii] - t * (geoParam_[ii]-geoParam_[3]) <= 0)
            atCorner = false;
    if(atCorner)
    {
        std::array<double,3> absPt;
        std::transform(pt.begin(), pt.end(), absPt.begin(), [](double pt){return std::abs(pt);});
        double cenPtDot = ddot_(3, absPt.data(), 1, curveCens_[0].data(), 1);
        double cenMag = std::accumulate(curveCens_[0].begin(), curveCens_[0].end(), 0.0, vecMagAdd<double>() );
        double ptMag = std::accumulate(absPt.begin(), absPt.end(), 0.0, vecMagAdd<double>());
        t = 1.0/ ( cenMag-std::pow(geoParam_[3],2.0) ) * ( std::pow(cenPtDot,2.0) - std::sqrt(std::pow(cenPtDot, 2.0) - ptMag*(cenMag-std::pow(geoParam_[3],2.0) ) ) );
        std::transform(absPt.begin(), absPt.end(), curveCens_[0].begin(), grad.begin(), [=](double p, double cen){return p - t*cen;} );
        std::transform(pt.begin(), pt.end(), grad.begin(), grad.begin(), [](double p, double grad){return grad*p/std::abs(p); } );
    }
    else
    {
        for(int ii = 0; ii < 3; ++ii)
            if(ptRat[ii] == ptRat[ptRatMax])
                grad[ ii ] = v_trans[ii] >=0 ? 1.0 : -1.0;
    }
    return ObjectSpace2RealSpace(grad);
}

std::array<double, 3> tri_prism::findGradient(std::array<double,3> pt)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(pt);
    std::array<double,2> v_transTriFace = {v_trans[0], v_trans[1]};
    std::array<double,3> grad = {0.0,0.0,0.0};

    double scalRat = std::abs(2*v_trans[2]/geoParam_[3]);
    if(scalRat > 0)
    {
        std::array<std::array<double, 2>, 3> scaledVertLocs_(vertLocs_);
        std::vector<double> invScalVertMat(9,1.0);
        for(int ii = 0; ii < 3; ++ii)
        {
            dscal_(2, scalRat, scaledVertLocs_[ii].begin(), 1);
            dcopy_(2, scaledVertLocs_[ii].begin(), 1, &invScalVertMat[ii], 3);
        }
        int info;
        std::vector<int>ivip(3, 0);
        dgetrf_(3, 3, invScalVertMat.data(), 3, ivip.data(), &info);
        double d0Scaled = getLUFactMatDet(invScalVertMat, ivip, 3);
        std::vector<double>work(9,0.0);
        dgetri_(3, invScalVertMat.data(), 3, ivip.data(), work.data(), work.size(),  &info);
        std::array<double, 3> baryScaled = cart2bary<double,3>(invScalVertMat, v_transTriFace, d0Scaled);
        bool inTri = true;
        bool onEdge = false;
        for(int ii = 0; ii < 3; ++ii)
        {
            if( baryScaled[ii] < -1e-15 || baryScaled[ii] > geoParam_[ii]+1e-15 )
                inTri = false;
            if( baryScaled[ii] == 0 || baryScaled[ii] == geoParam_[ii] )
                onEdge = true;
        }
        // If the point is in the scaled triangle's cross section then it is on the cap face
        if(inTri)
        {
            grad = { 0.0, 0.0, (v_trans[2] >= 0 ? 1.0 : -1.0 ) } ;
            // If it's not on the edge then no need to calculate which triangle face is closest
            if(!onEdge)
                return ObjectSpace2RealSpace(grad);
        }
    }
    // Calculate the barycenteric coordinates
    std::array<double, 3> baryCenCoords = cart2bary<double,3>(invVertMat_, v_transTriFace, d0_);
    // Calculate the ratio from the base surface and truncated surace
    std::array<double, 3> baseRat;
    std::array<double, 3> truncRat;
    for(int ii = 0; ii < 3; ++ii)
    {
        baseRat[ii] = baryCenCoords[ii] / geoParam_[ii];
        truncRat[ii] = ( geoParam_[ii] - baryCenCoords[ii] ) / geoParam_[ii];
    }
    // Find the min distance to the base and truncated faces
    int indMinBaseRat = idamin_(baseRat.size(), baseRat.begin(), 1 );
    int indMinTruncRat = idamin_(truncRat.size(), truncRat.begin(), 1 );
    // Determine which of the two is closest and get it's distance
    int closestFace = ( truncRat[indMinTruncRat-1] < baseRat[indMinBaseRat-1] ) ? -1*indMinTruncRat : indMinBaseRat;
    double dist2ClosestFace = (closestFace < 0) ? truncRat[indMinTruncRat-1] : baseRat[indMinBaseRat-1];

    // Collect all faces equidistant to the closest face
    std::vector<int> equiDistFaces;
    for(int ii = 0; ii < 3; ++ii)
    {
        // If both the truncated face and base face are equidistant give presidence to the base
        if(baseRat[ii] == dist2ClosestFace)
            equiDistFaces.push_back(ii+1);
        if(truncRat[ii] == dist2ClosestFace)
            equiDistFaces.push_back(-1*ii-1);
    }
    std::vector<std::vector<int>> activeVerts(equiDistFaces.size());
    // Average over all faces
    for(int ff = 0; ff < equiDistFaces.size(); ++ff)
    {
        // Find the vertices needed to calculate the surface normal
        std::vector<int> activeVerts;
        for(int ii = 0; ii < 3; ++ii)
        {
            if( ii != std::abs(equiDistFaces[ff])-1 )
                activeVerts.push_back(ii);
        }
        // Calculate the gradient for that face and add it to grad
        std::array<double, 3> tempGrad;
        double xSign = ( vertLocs_[ activeVerts[1] ][0]+vertLocs_[ activeVerts[0] ][0] )/2.0 - ( vertLocs_[ activeVerts[1] ][0]+vertLocs_[ activeVerts[0] ][0]+vertLocs_[closestFace][0] )/3.0;
        xSign /= -1.0*std::abs(xSign) * equiDistFaces[ff] / std::abs(equiDistFaces[ff]);
        double slopePerp = -1.0*(vertLocs_[ activeVerts[1] ][0]-vertLocs_[ activeVerts[0] ][0])/(vertLocs_[ activeVerts[1] ][1]-vertLocs_[ activeVerts[0] ][1]);
        if( std::abs(vertLocs_[ activeVerts[1] ][1]-vertLocs_[ activeVerts[0] ][1]) < 1e-14)
            tempGrad = { 0.0, xSign, 0.0 } ;
        else
            tempGrad = { xSign, xSign*slopePerp, 0.0 } ;
        normalize(tempGrad);
        std::transform(tempGrad.begin(), tempGrad.end(), grad.begin(), grad.begin(), std::plus<double>() );
    }
    return ObjectSpace2RealSpace(grad);
}

std::array<double, 3> tetrahedron::findGradient(std::array<double,3> pt)
{
    // Convert pt to barycentric coodrinates
    std::array<double, 4> baryCenCoords = cart2bary<double,4>(invVertMat_, pt, d0_);
     // Get the ratios from of face distance to the geo_params
    std::array<double, 4> truncRat;
    std::array<double, 4> baseRat;
    for(int ii = 0; ii < 4; ++ii)
    {
        baseRat[ii] = baryCenCoords[ii] / geoParam_[ii];
        truncRat[ii] = ( geoParam_[ii] - baryCenCoords[ii] ) / geoParam_[ii];
    }
    // Get the minimum distance to a base and truncated surface
    int indMinBaseDist = idamin_(4, baseRat.begin(), 1);
    int indMinTruncDist = idamin_(4, truncRat.begin(), 1 );
    // Which is closer, and what is that distance
    int closestFace = ( truncRat[indMinTruncDist-1] < baseRat[indMinBaseDist-1] ) ? -1*indMinTruncDist : indMinBaseDist;
    double dist2ClosestFace = (closestFace < 0) ? truncRat[indMinTruncDist-1] : baseRat[indMinBaseDist-1];
    // Find all faces equidistant to the closest face
    std::vector<int> equiDistFaces;
    for(int ii = 0; ii < 4; ++ii)
    {
        // If both the truncated face and base face are equidistant give presidence to the base
        if(baseRat[ii] == dist2ClosestFace)
            equiDistFaces.push_back(ii+1);
        if(truncRat[ii] == dist2ClosestFace)
            equiDistFaces.push_back(-1*ii-1);
    }
    std::array<double,3> grad = {0.0, 0.0, 0.0};
    // Loop over all faces and add average their gradients
    for(int ff = 0; ff < equiDistFaces.size(); ++ff)
    {
        std::vector<int> activeVerts;
        // Find the vertices needed to calculate the surface normal in the correct order (https://math.stackexchange.com/questions/183030/given-a-tetrahedron-how-to-find-the-outward-surface-normals-for-each-side)
        for(int ii = 0; ii < 4; ++ii)
        {
            int index = ( static_cast<int>( std::pow( -1, std::abs(equiDistFaces[ff])-1 ) * ii) % 4 ) * static_cast<int>(equiDistFaces[ff] / std::abs(equiDistFaces[ff]) );
            index += (index < 0) ? 4 : 0;

            if( index != std::abs(equiDistFaces[ff])-1 )
                activeVerts.push_back(index);
        }
        // Calculate the gradient for that face and add it to grad
        std::array<double,3> v1;
        std::array<double,3> v2;
        std::transform(vertLocs_[activeVerts[1]].begin(), vertLocs_[activeVerts[1]].end(), vertLocs_[activeVerts[0]].begin(), v1.begin(), std::minus<double>());
        std::transform(vertLocs_[activeVerts[2]].begin(), vertLocs_[activeVerts[2]].end(), vertLocs_[activeVerts[1]].begin(), v2.begin(), std::minus<double>());
        std::array<double, 3> tempGrad = {{ v1[1]*v2[2]-v1[2]*v2[1], v1[2]*v2[0]-v1[0]*v2[2], v1[0]*v2[1]-v1[1]*v2[0] }};
        normalize(tempGrad);
        std::transform(tempGrad.begin(), tempGrad.end(), grad.begin(), grad.begin(), std::plus<double>() );
    }
    // If 0 return 0; otherwise normalize and return
    if( std::abs( std::accumulate(grad.begin(), grad.end(), 0.0, vecMagAdd<double>() ) ) < 1.0e-20 )
        grad ={0.0, 0.0, 0.0};
    else
        normalize(grad);
    return grad;
}

std::array<double, 3> ters_tip::findGradient(std::array<double,3> pt)
{
    throw std::logic_error("Gradient is not defined for this object yet defined");
    return std::array<double,3>({{0.0,0.0,0.0}});
}

std::array<double, 3> paraboloid::findGradient(std::array<double,3> pt)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(pt);
    v_trans[1] += geoParam_[2]/2.0;
    std::array<double,3> grad = {2*v_trans[0]/std::pow(geoParam_[0],2.0), -1.0, 2.0*v_trans[2]/std::pow(geoParam_[2],2.0) };
    normalize(grad);
    return grad;
}

std::array<double, 3> torus::findGradient(std::array<double,3> pt)
{
    std::array<double,3> v_trans = RealSpace2ObjectSpace(pt);
    double radRing = std::sqrt( std::pow(v_trans[0], 2.0) + std::pow(v_trans[1], 2.0) );
    double radRingTerm = (radRing - geoParam_[0]) / radRing;
    std::array<double,3> grad = { 2.0*v_trans[0] * radRingTerm, 2.0*v_trans[1] * radRingTerm, 2.0*v_trans[2]};
    normalize(grad);
    return grad;
}
