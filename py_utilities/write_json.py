import json
import numpy as np
import math

def write_sphere(radius, loc, mat, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Creates a sphere for the the input file

    @param      radius    radius of sphere
    @param      loc       location of center point
    @param      mat       material keyword for json file
    @param      uvecs     list of unit vectors for the three main axes (use instead of angles)
    @param      pols      Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps       high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu        high frequency permeability of material (use if mat = 'custom')
    @param      tellegen  tellegen parameter of the material (use if mat = 'custom')

    @return     Dictionary describing the sphere
    """
    return {'shape' : 'sphere', 'material' : mat, 'loc' : loc, 'radius' : round(radius,7), 'unit_vectors' : [], "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_hemisphere(radius, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes a hemisphere.

    @param      radius    radius of sphere
    @param      loc       location of center point
    @param      mat       material keyword for json file
    @param      orPhi     azimuthal angle of the main axis
    @param      orTheta   polar angle of the main axis
    @param      uvecs     list of unit vectors for the three main axes (use instead of angles)
    @param      pols      Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps       high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu        high frequency permeability of material (use if mat = 'custom')
    @param      tellegen  tellegen parameter of the material (use if mat = 'custom')

    @return     Dictionary describing the hemisphere
    """
    return {'shape' : 'hemisphere', 'material' : mat, 'loc' : loc, 'radius' : round(radius,7), "orPhi" : orPhi, "orTheta" : orTheta, 'unit_vectors' : [], "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_block(sz, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes a block.

    @param      sz        [size u_vec 1, size uvec 2, size uvec 3]
    @param      loc       location of center point
    @param      mat       material keyword for json file
    @param      orPhi     azimuthal angle of the main axis
    @param      orTheta   polar angle of the main axis
    @param      uvecs     list of unit vectors for the three main axes (use instead of angles)
    @param      pols      Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps       high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu        high frequency permeability of material (use if mat = 'custom')
    @param      tellegen  tellegen parameter of the material (use if mat = 'custom')

    @return     A Dictionary describing the block
    """
    return {'shape' : 'block', 'material' : mat, 'loc' : loc, 'size' : sz, "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_ellipsoid(sz, loc, mat, cut_neg=[-1, -1, -1], cut_pos=[1, 1, 1], cut_global=[1,1,1], orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes an ellipsoid.

    @param      sz          [size u_vec 1, size uvec 2, size uvec 3]
    @param      loc         location of center point
    @param      mat         material keyword for json file
    @param      cut_neg     List of cutoffs for the positive side of the ellipsoids  in all directions
    @param      cut_pos     List of cutoffs for the negative side of the ellipsoids  in all directions
    @param      cut_global  List of cutoffs for the side of the defined globally (negative to go negative)  in all directions
    @param      orPhi       azimuthal angle of the main axis
    @param      orTheta     polar angle of the main axis
    @param      uvecs       list of unit vectors for the three main axes (use instead of angles)
    @param      pols        Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps         high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu          high frequency permeability of material (use if mat = 'custom')
    @param      tellegen    tellegen parameter of the material (use if mat = 'custom')

    @return     definition of an ellipsoid
    """
    return {'shape' : 'ellipsoid', 'material' : mat, 'loc' : loc, 'size' : sz, 'cut_neg' : cut_neg, 'cut_pos' : cut_pos, 'cut_global': cut_global, "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_hemiellipsoid(sz, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes a hemiellipsoid.

    @param      sz        [size u_vec 1, size uvec 2, size uvec 3]
    @param      loc       location of center point
    @param      mat       material keyword for json file
    @param      orPhi     azimuthal angle of the main axis
    @param      orTheta   polar angle of the main axis
    @param      uvecs     list of unit vectors for the three main axes (use instead of angles)
    @param      pols      Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps       high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu        high frequency permeability of material (use if mat = 'custom')
    @param      tellegen  tellegen parameter of the material (use if mat = 'custom')

    @return     { description_of_the_return_value }
    """
    return {'shape' : 'hemiellipsoid', 'material' : mat, 'loc' : loc, 'size' : sz, "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_rounded_box(sz, rad_curve, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes a rounded box.

    @param      sz         [size u_vec 1, size uvec 2, size uvec 3]
    @param      rad_curve  radius of curvature of the corners
    @param      loc        location of center point
    @param      mat        material keyword for json file
    @param      orPhi      azimuthal angle of the main axis
    @param      orTheta    polar angle of the main axis
    @param      uvecs      list of unit vectors for the three main axes (use instead of angles)
    @param      pols       Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps        high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu         high frequency permeability of material (use if mat = 'custom')
    @param      tellegen   tellegen parameter of the material (use if mat = 'custom')

    @return     { description_of_the_return_value }
    """
    return {'shape' : 'block', 'material' : mat, 'loc' : loc, 'size' : sz, "rad_curve": round(rad_curve,7), "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_triangle(base, height, length, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes a triangle.

    @param      base      base of the triangle
    @param      height    height of the triangle
    @param      length    The length of the prism
    @param      loc       location of center point
    @param      mat       material keyword for json file
    @param      orPhi     azimuthal angle of the main axis
    @param      orTheta   polar angle of the main axis
    @param      uvecs     list of unit vectors for the three main axes (use instead of angles)
    @param      pols      Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps       high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu        high frequency permeability of material (use if mat = 'custom')
    @param      tellegen  tellegen parameter of the material (use if mat = 'custom')

    @return     { description_of_the_return_value }
    """
    return {'shape' : 'triangle_prism', 'material' : mat, 'loc' : loc, "base" : round(base, 7), "height" : round(height,7), "length" : round(length,7), "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_trapezoid(base1, base2, height, length, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes a trapezoid.

    @param      base1     bottom base of the trapezoid
    @param      base2     top base of the trapezoid
    @param      height    height of the triangle
    @param      length    length of the prism
    @param      loc       location of center point
    @param      mat       material keyword for json file
    @param      orPhi     azimuthal angle of the main axis
    @param      orTheta   polar angle of the main axis
    @param      uvecs     list of unit vectors for the three main axes (use instead of angles)
    @param      pols      Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps       high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu        high frequency permeability of material (use if mat = 'custom')
    @param      tellegen  tellegen parameter of the material (use if mat = 'custom')

    @return     { description_of_the_return_value }
    """
    return {'shape' : 'trapezoid_prism', 'loc' : loc, 'material' : mat, "base1" : round(base1, 7), "base2" : round(base2, 7), "height" : round(height,7), "length" : round(length,7), "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}

def write_tetrahedron(loc, mat, sideLen=-1.0, perpBiSec=-1.0, height=0.0, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
    """
    @brief      Writes a tetrahedron.

    @param      loc        The location of the tetraheadrons barycenter
    @param      mat        The material
    @param      sideLen    The side length of a regular tetraheadron
    @param      perpBiSec  The perpindicular bisector of a tetraheadron's face
    @param      height     The height
    @param      orPhi      The azimuthal angle of the main axis
    @param      orTheta    The polar angle of the main axis
    @param      uvecs      list of unit vectors for the three main axes (use instead of angles)
    @param      pols       Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
    @param      eps        high frequency dielectric constant of material (use if mat = 'custom')
    @param      mu         high frequency permeability of material (use if mat = 'custom')
    @param      tellegen   tellegen parameter of the material (use if mat = 'custom')

    @return     { description_of_the_return_value }
    """
    return {'shape' : 'tetrahedron', 'material' : mat, "loc" : loc, "height" : round(height,7), "sideLen" : round(sideLen,7), "perpBiSec":round(perpBiSec,7), "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), 'Basis_Set' : []}


def write_cone(rad1, rad2, height, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
        # # rad1     : top radius of the cone
        # # rad2     : bottom radius of the cone
        # # height   : height of the triangle
        # # loc      : location of center point
        # # mat      : material keyword for json file
        # # orPhi    : azimuthal angle of the main axis
        # # orTheta  : polar angle of the main axis
        # # uvecs    : list of unit vectors for the three main axes (use instead of angles)
        # # pols     : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps      : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu       : high frequency permeability of material (use if mat = 'custom')
        # tellegen : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'cone', 'material' : mat, "eps":round(eps, 7), "radius1" : round(rad1, 7), "radius2" : round(rad2, 7), "height" : round(height,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, 'Basis_Set' : []}

def write_cylinder(radius, length, loc, mat, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0):
        # # radius   : radius of the cylinder
        # # length   : length of the cylinder
        # # loc      : location of center point
        # # mat      : material keyword for json file
        # # orPhi    : azimuthal angle of the main axis
        # # orTheta  : polar angle of the main axis
        # # uvecs    : list of unit vectors for the three main axes (use instead of angles)
        # # pols     : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps      : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu       : high frequency permeability of material (use if mat = 'custom')
        # tellegen : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'cylinder', 'material' : mat, "eps":round(eps, 7), "radius" : round(radius, 7), "length" : round(length,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, 'Basis_Set' : []}

def write_ters_tip(base, len, rad_curv, loc, mat, orPhi=0.0, orTheta=90.0, pols=[], eps=1.0):
        # # len      : length of the TERS Tip
        # rad_curv : radius of curvature of the tip
        # # base     : radius of the base
        # # loc      : location of center point
        # # mat      : material keyword for json file
        # # orPhi    : azimuthal angle of the main axis
        # # orTheta  : polar angle of the main axis
        # # uvecs    : list of unit vectors for the three main axes (use instead of angles)
        # # pols     : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps      : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu       : high frequency permeability of material (use if mat = 'custom')
        # tellegen : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'ters_tip', 'material' : mat, "eps":round(eps, 7), "rad_curve" : round(rad_curv, 7), "base": round(base, 7), "length" : round(length,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, 'Basis_Set' : []}

def write_parabolic_ters_tip(len, rad_curv, loc, mat, orPhi=0.0, orTheta=90.0, pols=[], eps=1.0):
        # # len      : length of the TERS Tip
        # rad_curv : radius of curvature of the tip
        # # loc      : location of center point
        # # mat      : material keyword for json file
        # # orPhi    : azimuthal angle of the main axis
        # # orTheta  : polar angle of the main axis
        # # uvecs    : list of unit vectors for the three main axes (use instead of angles)
        # # pols     : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps      : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu       : high frequency permeability of material (use if mat = 'custom')
        # tellegen : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'parabolic_ters_tip', 'material' : mat, "eps":round(eps, 7), "rad_curve" : round(rad_curv, 7), "length" : round(length,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, 'Basis_Set' : []}

def write_ml_sphere(radius, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, numelec=1, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # # radius       : radius of sphere
        # # loc          : location of center point
        # # mat          : material keyword for json file
        # # basis_set    : list of spherical harmonic basis set factors
        # # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # # couplings    : transition dipole moments for each transition
        # # relaxOps     : list of relaxation operator descriptors
        # # dtc_levs     : list of levels to keep track of
        # # orPhi        : azimuthal angle of the main axis
        # # orTheta      : polar angle of the main axis
        # # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu           : high frequency permeability of material (use if mat = 'custom')
        # # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'sphere', 'material' : mat, 'loc' : loc, 'radius' : round(radius,7), 'unit_vectors' : [], "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol, "numElectorn": numelec }

def write_ml_block(sz, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs=[], dtc_lev_time_int=1, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # # sz           : [size u_vec 1, size uvec 2, size uvec 3]
        # # loc          : location of center point
        # # mat          : material keyword for json file
        # # basis_set    : list of spherical harmonic basis set factors
        # # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # # couplings    : transition dipole moments for each transition
        # # relaxOps     : list of relaxation operator descriptors
        # # dtc_levs     : list of levels to keep track of
        # # orPhi        : azimuthal angle of the main axis
        # # orTheta      : polar angle of the main axis
        # # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu           : high frequency permeability of material (use if mat = 'custom')
        # # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'block', 'material' : mat, 'loc' : loc, 'size' : sz, "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, "levDTC_timeInt" : dtc_lev_time_int, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_ellipse(sz, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, cut_neg=[-1, -1, -1], cut_pos=[1, 1, 1], orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # # sz           : [size u_vec 1, size uvec 2, size uvec 3]
        # # loc          : location of center point
        # # mat          : material keyword for json file
        # # basis_set    : list of spherical harmonic basis set factors
        # # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # # couplings    : transition dipole moments for each transition
        # # relaxOps     : list of relaxation operator descriptors
        # # dtc_levs     : list of levels to keep track of
        # # cut_neg      : array of cutoff planes for ellipsoids in the negative direction range(0, -1)
        # # cut_neg      : array of cutoff planes for ellipsoids in the positive direction range(0, 1)
        # # orPhi        : azimuthal angle of the main axis
        # # orTheta      : polar angle of the main axis
        # # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu           : high frequency permeability of material (use if mat = 'custom')
        # # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'ellipse', 'material' : mat, 'loc' : loc, 'size' : sz, 'cut_neg' : cut_neg, 'cut_pos' : cut_pos, "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_rounded_box(sz, rad_curve, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # # sz           : [size u_vec 1, size uvec 2, size uvec 3]
        # # rad_curve    : radius of curvature of the corners
        # # loc          : location of center point
        # # mat          : material keyword for json file
        # # basis_set    : list of spherical harmonic basis set factors
        # # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # # couplings    : transition dipole moments for each transition
        # # relaxOps     : list of relaxation operator descriptors
        # # dtc_levs     : list of levels to keep track of
        # # orPhi        : azimuthal angle of the main axis
        # # orTheta      : polar angle of the main axis
        # # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu           : high frequency permeability of material (use if mat = 'custom')
        # # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'block', 'material' : mat, 'loc' : loc, 'size' : sz, "rad_curve": round(rad_curve,7), "orPhi" : orPhi, "orTheta" : orTheta, "unit_vectors" : uvecs, "pols": pols, "eps":round(eps, 7), "mu":round(mu, 7), "tellegen":round(tellegen, 7), "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_triangle(base, height, length, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # # base         : base of the triangle
        # # height       : height of the triangle
        # # loc          : location of center point
        # # mat          : material keyword for json file
        # # basis_set    : list of spherical harmonic basis set factors
        # # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # # couplings    : transition dipole moments for each transition
        # # relaxOps     : list of relaxation operator descriptors
        # # dtc_levs     : list of levels to keep track of
        # # orPhi        : azimuthal angle of the main axis
        # # orTheta      : polar angle of the main axis
        # # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # # mu           : high frequency permeability of material (use if mat = 'custom')
        # # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'triangle_prism', 'material' : mat, "eps":round(eps, 7), "mu":round(mu,7), "tellegen" : tellegen, "pols" : pols, "base" : round(base, 7), "height" : round(height,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_trapezoid(base1, base2, height, length, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # base1        : bottom base of the trapezoid
        # base2        : top base of the trapezoid
        # height       : height of the triangle
        # length       : length of the prism
        # loc          : location of center point
        # mat          : material keyword for json file
        # basis_set    : list of spherical harmonic basis set factors
        # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # couplings    : transition dipole moments for each transition
        # relaxOps     : list of relaxation operator descriptors
        # dtc_levs     : list of levels to keep track of
        # orPhi        : azimuthal angle of the main axis
        # orTheta      : polar angle of the main axis
        # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # mu           : high frequency permeability of material (use if mat = 'custom')
        # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'trapezoid_prism', 'material' : mat, "eps":round(eps, 7), "mu":round(mu,7), "tellegen" : tellegen, "pols" : pols, "base1" : round(base1, 7), "base2" : round(base2, 7), "height" : round(height,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_cone(rad1, rad2, height, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # rad1         : top radius of the cone
        # rad2         : bottom radius of the cone
        # height       : height of the triangle
        # loc          : location of center point
        # mat          : material keyword for json file
        # basis_set    : list of spherical harmonic basis set factors
        # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # couplings    : transition dipole moments for each transition
        # relaxOps     : list of relaxation operator descriptors
        # dtc_levs     : list of levels to keep track of
        # orPhi        : azimuthal angle of the main axis
        # orTheta      : polar angle of the main axis
        # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # mu           : high frequency permeability of material (use if mat = 'custom')
        # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'cone', 'material' : mat, "eps":round(eps, 7), "mu":round(mu,7), "tellegen" : tellegen, "pols" : pols, "radius1" : round(rad1, 7), "radius2" : round(rad2, 7), "height" : round(height,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_cylinder(radius, length, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, orPhi=0.0, orTheta=90.0, uvecs=[], pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # radius       : radius of the cylinder
        # length       : length of the cylinder
        # loc          : location of center point
        # mat          : material keyword for json file
        # basis_set    : list of spherical harmonic basis set factors
        # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # couplings    : transition dipole moments for each transition
        # relaxOps     : list of relaxation operator descriptors
        # dtc_levs     : list of levels to keep track of
        # orPhi        : azimuthal angle of the main axis
        # orTheta      : polar angle of the main axis
        # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # mu           : high frequency permeability of material (use if mat = 'custom')
        # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'cylinder', 'material' : mat, "eps":round(eps, 7), "mu":round(mu,7), "tellegen" : tellegen, "pols" : pols, "radius" : round(radius, 7), "length" : round(length,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_ters_tip(base, len, rad_curv, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, orPhi=0.0, orTheta=90.0, pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # len          : length of the TERS Tip
        # rad_curv     : radius of curvature of the tip
        # base         : radius of the base
        # loc          : location of center point
        # mat          : material keyword for json file
        # basis_set    : list of spherical harmonic basis set factors
        # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # couplings    : transition dipole moments for each transition
        # relaxOps     : list of relaxation operator descriptors
        # dtc_levs     : list of levels to keep track of
        # orPhi        : azimuthal angle of the main axis
        # orTheta      : polar angle of the main axis
        # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # mu           : high frequency permeability of material (use if mat = 'custom')
        # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'ters_tip', 'material' : mat, "eps":round(eps, 7), "mu" : round(mu,7), "tellegen" : (tellegen, 7), "rad_curve" : round(rad_curv, 7), "base": round(base, 7), "length" : round(length,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_ml_parabolic_ters_tip(len, rad_curv, loc, mat, basis_set, mol_den, E_state_list, couplings, gam, relaxOps, dtc_levs, orPhi=0.0, orTheta=90.0, pols=[], eps=1.0, mu=1.0, tellegen=0.0, output_pol=False, output_pol_filename=""):
        # len          : length of the TERS Tip
        # rad_curv     : radius of curvature of the tip
        # loc          : location of center point
        # mat          : material keyword for json file
        # basis_set    : list of spherical harmonic basis set factors
        # mol_den      : molecular number density of the system
        # E_state_list : list of Energy level distribution parameters
        # couplings    : transition dipole moments for each transition
        # relaxOps     : list of relaxation operator descriptors
        # dtc_levs     : list of levels to keep track of
        # orPhi        : azimuthal angle of the main axis
        # orTheta      : polar angle of the main axis
        # uvecs        : list of unit vectors for the three main axes (use instead of angles)
        # pols         : Lorentz-Drude parameter list for polarizations [[sigma, gamma, omega]] (use if mat = 'custom')
        # eps          : high frequency dielectric constant of material (use if mat = 'custom')
        # mu           : high frequency permeability of material (use if mat = 'custom')
        # tellegen     : tellegen parameter of the material (use if mat = 'custom')
    return {'shape' : 'parabolic_ters_tip', 'material' : mat, "eps":round(eps, 7), "mu" : round(mu,7), "tellegen" : (tellegen, 7), "rad_curve" : round(rad_curv, 7), "length" : round(length,7), "length" : round(length,7), "orPhi" : round(orPhi,7), "orTheta" : round(orTheta,7) , 'loc' : loc, "RelaxationOperators": relaxOps, "dtc_levs": dtc_levs, 'Basis_Set' : [write_basis(basis[0],basis[1]) for basis in basis_set], "Energy_Levels":E_state_list, "couplings" : couplings, "gam":[write_gam(g) for g in gam], "mol_den": round(mol_den,  8), "output_pol_filename":output_pol_filename, "output_pol" : output_pol }

def write_elev(distribution, e_cen, nstates=1, weights=[1.0], dist_width=[0.0], shape_factor=[0.0], highEdge=0.0, lowEdge=0.0, levs=1) :
    if(distribution == "Gaussian"):
        return {"distribution" : distribution, "E_cen" : e_cen, "weights" : weights, "nstates" : nstates, "dist_width" : dist_width, "highEdge":highEdge, "lowEdge":lowEdge, "levs_described" : levs }
    elif(distribution == "delta_fxn"):
        return {"distribution" : distribution, "E_cen" : e_cen, "weights" : weights, "nstates" : nstates, "levs_described" : levs }
    elif(distribution == "skew_normal"):
        return {"distribution" : distribution, "E_cen" : e_cen, "weights" : weights, "nstates" : nstates, "dist_width" : dist_width, "skew_factor" : shape_factor, "highEdge":highEdge, "lowEdge":lowEdge, "levs_described" : levs }
    elif(distribution == "log_normal"):
        return {"distribution" : distribution, "E_cen" : e_cen, "weights" : weights, "nstates" : nstates, "dist_width" : dist_width, "highEdge":highEdge, "lowEdge":lowEdge, "levs_described" : levs }
    elif(distribution == "exp_pow"):
        return {"distribution" : distribution, "E_cen" : e_cen, "weights" : weights, "nstates" : nstates, "dist_width" : dist_width, "shape_factor" : shape_factor, "highEdge":highEdge, "lowEdge":lowEdge, "levs_described" : levs }
    elif(distribution == "gen_skew_normal"):
        return {"distribution" : distribution, "E_cen" : e_cen, "weights" : weights, "nstates" : nstates, "dist_width" : dist_width, "shape_factor" : shape_factor, "highEdge":highEdge, "lowEdge":lowEdge, "levs_described" : levs }
    else:
        raise ValueError("Distribution for energy states not defined")

def write_relaxation(state_i, state_f,  rate, del_omg, radiative, dephasing_rate):
    return {"state_i": state_i, "state_f": state_f, "rate": rate, "del_omg": del_omg, "radiative": radiative, "dephasing_rate" : dephasing_rate}

def write_tfsf(sz, loc, pulseList, theta=90.0, phi=0.0, psi=0.0, m=[], circPol="Ex", ellpiticalKRat=1.0, pmlThick=20, pmlAMax=0.00, pmlM=3.0, pmlMa=1.0):
        # phi     : azimuthal angle of the k vector
        # theta   : polar angle of the k vector
        # psi     : angle describing light's polarization
        # sz      : [size_x, size_y]
        # loc     : location of center point
        pulseList: list of all pulse
    return {"loc":loc, "size":sz, "theta" : round(theta, 8), "phi":round(phi,  8), "psi" : round(psi,8), "m" :m, "PulseList" : pulseList, "circPol" : circPol,  "ellpiticalKRat" : ellpiticalKRat, "pmlThick" : pmlThick, "pmlAMax" : pmlAMax, "pmlM" : pmlM, "pmlMa" : pmlMa}

def write_tfsf_m_def(sz, loc, pulseList, m=[1,0,0], psi=0.0, circPol="Ex", ellpiticalKRat=1.0, pmlThick=20, pmlAMax=0.00, pmlM=3.0, pmlMa=1.0):
    return {"loc":loc, "size":sz, "m" : m, "psi" : round(psi,8), "m" :m, "PulseList" : pulseList, "circPol" : circPol,  "ellpiticalKRat" : ellpiticalKRat, "pmlThick" : pmlThick, "pmlAMax" : pmlAMax, "pmlM" : pmlM, "pmlMa" : pmlMa}

def write_normal_src(loc, sz, pulseList, pol):
        # sz       : [size_x, size_y]
        # loc      : location of center point
        pulseList: list of all pulse
        # pol      : polarization of the light
    return {"PulseList" : pulseList, "size": sz, "pol":pol , "loc_y":loc, "loc_x":round(loc[0], 7)}

def write_oblique_src(loc, len, phi, pulseList, pol):
        # phi     : angle of incident light (90 for positive y, 0 for positive x)
        # sz      : [size_x, size_y]
        # loc     : location of center point
        pulseList: list of all pulse
        # pol     : polarization of the light
    if(phi % 90.0 == 0):
        raise ValueError('Making an oblique source with a normal incidence angle, please use write_normal_source for this')
    return {"PulseList" : pulseList, "phi":round(phi, 7), "len":round(len, 7), "pol":pol , "loc":loc}

def write_pulse(profile, Field_Intensity=1.0, fcen=0.0, fwidth=0.0, cutoff=0.0, t_0=0.0, tau=0.0, n=0.0, ramp_val=0.0, BH1=0.35875, BH2=0.48829, BH3=0.14128, BH4=0.01168 ):
        # profile : function type of pulse
        # fcen    : pulse center frequency
        # fwidth  : frequency width
        # cutoff  : cutoff value for pulse
        # tstart  : Not actually used
        # ramp_val: slope of pulse ramp up for cw sources
    if(profile == "gaussian"):
        return {"profile" : profile, "Field_Intensity":Field_Intensity, "fcen": fcen, "fwidth" : fwidth, "t_0" : t_0, "cutoff" : cutoff }
    elif(profile == "BH"):
        return {"profile" : profile, "Field_Intensity":Field_Intensity, "fcen": fcen, "fwidth" : 1.0/tau, "tau" : tau, "t_0" : t_0, "BH1" : BH1, "BH2" : BH2, "BH3" : BH3, "BH4" : BH4}
    elif(profile == "rectangle"):
        return {"profile" : profile, "Field_Intensity":Field_Intensity, "fcen": fcen, "tau" : tau, "t_0" : t_0, "n" : n }
    elif(profile == "continuous"):
        return {"profile" : profile, "Field_Intensity":Field_Intensity, "fcen": fcen }
    elif(profile=="ramped_cont"):
        return {"profile" : profile, "Field_Intensity":Field_Intensity, "fcen": fcen, "ramp_val" : ramp_val }
    elif(profile=="ricker"):
        return {"profile" : profile, "Field_Intensity":Field_Intensity, "fcen": fcen, "fwidth" : fwidth, "cutoff" : cutoff }
    else:
        raise ValueError('Pulse is not defined.')

def write_flux(name, loc, sz, save, load, weight, fcen, fwidth, nfreq, SI, cross_sec):
        # name     : filename of the flux detector
        # sz       : [size_x, size_y]
        # loc      : location of center point
        direction: Direction of flux
        # refl     : true if refelection flux
        # weight   : -1.0 if flux is for oppisite direction
        # fcen     : pulse center frequency
        # fwidth   : frequency width
        # nfreq    : number of frequencies to recotd
        # SI       : true if you want to use SI units
        cross_sec: true if flux is used for cross section calculation
    return {"name":name, "save":save, "load":load, "loc":loc, "size":sz, "SI":SI, "fcen":round(fcen,  8), "fwidth":round(fwidth,  8), "nfreq":nfreq,"weight":weight, "cross_sec":cross_sec}

def write_flux_lam_def(name, loc, sz, save, load, weight, lamL, lamR, nfreq, SI, cross_sec):
        # name     : filename of the flux detector
        # sz       : [size_x, size_y]
        # loc      : location of center point
        # direction: Direction of flux
        # refl     : true if refelection flux
        # weight   : -1.0 if flux is for oppisite direction
        # lamL     : left wavelength edge
        # lamR     : right wavelength edge
        # nfreq    : number of frequencies to recotd
        # SI       : true if you want to use SI units
        cross_sec: true if flux is used for cross section calculation
    return {"name":name, "save":save, "load":load,  "loc":loc, "size":sz, "SI":SI, "lamL":round(lamL,  8), "lamR":round(lamR,  8), "nfreq":nfreq,"weight":weight, "cross_sec":cross_sec}

def write_dtc(loc, sz, SI, fname, typ, dtc_class, time_int, txt_dat_type, txt_format_type, integrateMap=False, tStart=0.0, tEnd=1e8):
        # sz             : [size_x, size_y]
        # loc            : location of center point
        # SI             : true if you want to use SI units
        # typ            : type of detecor (Ex,Ey,Ez,Hx,Hy,Hz, E_pow,H_pow,Px,Py,Pz)
        # dtc_class      : class of detector (bin,bmp,txt,cout)
        # time_int       : time interval of calculation
        # txt_dat_type   : what is stored in the text data file for bmp files
        # txt_format_type: shape of data in bmp txt files (box or line)
    return  {"loc":loc, "size":sz, "SI":SI, "dtc_class":dtc_class, "fname":fname, "type":typ, "txt_dat_type":txt_dat_type, "txt_format_type":txt_format_type, "Time_Interval":round(time_int,  8), "timeIntegrateMap":integrateMap, "t_start" : tStart, "t_end" : tEnd}

def write_dtc_freq_lam(loc, sz, SI, fname, typ, dtc_class, time_int, nLam, lamL, lamR, txt_dat_type, txt_format_type, output_map=False):
    return  {"loc":loc, "size":sz, "SI":SI, "dtc_class":dtc_class, "fname":fname, "nfreq":nLam, "lamL":round(lamL,  8), "lamR":round(lamR,  8), "type":typ, "txt_dat_type":txt_dat_type, "txt_format_type":txt_format_type, "Time_Interval":round(time_int,  8), "output_map": output_map}

def write_dtc_freq_omg(loc, sz, SI, fname, typ, dtc_class, time_int, txt_dat_type, nfreq, fcen, fwidth, txt_format_type, output_map=False):
    return  {"loc":loc, "size":sz, "SI":SI, "dtc_class":dtc_class, "fname":fname, "nfreq":nfreq, "fcen":round(fcen,  8), "fwidth":round(fwidth,  8), "type":typ, "txt_dat_type":txt_dat_type, "txt_format_type":txt_format_type, "Time_Interval":round(time_int,  8), "output_map": output_map}

def write_pml(aMax, ma, m, thickness, sigmaMax=1.0, kappaMax=1.0):
        # aMax       :  maximum a value for cpml(use 0.25)
        # ma         : ma poly scaling exponent(use 1.0)
        # m          : general poly scaling exponent(use 3.0)
        # thickness  : Thickness of pmls in the all directions
    return {"aMax":round(aMax,  8), "ma":round(ma,  8), "m":round(m,  8), "thickness": thickness, "sigOptRat" : sigmaMax, "kappaMax" : kappaMax}

def write_comp_cell(procs, sz, res, courant, tLim, PBC, pol, E_max, a, mapInputs_x=[], mapInputs_y=[], mapInputs_z=[], cplxfields=False ):
        # procs  : number of procs for the calculation
        # sz     : [size_x,size_y,size_z]
        # res    : resolution
        # courant: courant factor
        # tLim   : max time of caclualtions
        # PBC    : true if use PBC
        # pol    : source polarization (sets TE/TM)
        # E_max  : max value for E_incd
        # a      : unit length
    return {"InputMaps_x" : mapInputs_x, "InputMaps_y" : mapInputs_y, "InputMaps_z" : mapInputs_z, "procs":procs, "size":sz, "res":res, "courant":courant, "tLim":tLim, "PBC":PBC, "pol":pol, "E_max":E_max, "a":a, "cplxFields": cplxfields}

def write_comp_cell_d_def(procs, sz, gridSpace, dt, tLim, PBC, pol, E_max, a, mapInputs_x=[], mapInputs_y=[], mapInputs_z=[] ):
    return {"InputMaps_x" : mapInputs_x, "InputMaps_y" : mapInputs_y, "InputMaps_z" : mapInputs_z, "procs":procs, "size":sz, "stepSize":gridSpace, "dt":dt, "tLim":tLim, "PBC":PBC, "pol":pol, "E_max":E_max, "a":a}

def write_basis(l, m):
    return {"l":l, "m":m}

def write_gam(gam):
    return {"g" : [round( g, 7) for g in gam]}

def write_pol(dipOrE, dipOrM, sigma_p, sigma_m, tau, gamma, omega, dipOrAngE=0.0, dipOrAngM=0.0, dipE=[0.0,0.0,0.0], dipM=[0.0, 0.0, 0.0]):
    return {"dipOrE":dipOrE, "dipOrM": dipOrM, "polAngRelNormE": dipOrAngE, "polAngRelNormE": dipOrAngM, "sigma_p" : sigma_p, "sigma_m" : sigma_m, "tau" : tau, "gamma" : gamma, "omega" : omega, "dirDipE" : dipE, "dirDipM" : dipM}

def write_molec_pol(dipOrE, dipOrM, molDen, muE, muM, tau, gamma, omega, dipOrPolAngE=0.0, dipOrPolAngM=0.0, dipOrAzAngE=0.0, dipOrAzAngM=0.0, dipE=[0.0,0.0,0.0], dipM=[0.0, 0.0, 0.0], tanIso=False):
    return {"dipOrE":dipOrE, "dipOrM": dipOrM, "polAngRelNormE": dipOrPolAngE, "polAngRelNormM": dipOrPolAngM, "azAngRelNormE": dipOrAzAngE, "azAngRelNormM": dipOrAzAngM, "molecular_trans":True, "tanIso":tanIso, "molDen" : molDen, "dipMoment_E":muE, "dipMoment_M" : muM, "tau" : tau, "gamma" : gamma, "omega" : omega, "dirDipE" : dipE, "dirDipM" : dipM}
