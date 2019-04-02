/** @file UTIL/utlityFxn.hpp
 *  @brief Common functions used for transforms
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs
 */

#ifndef PARALLEL_FDTD_LAM_FXN_UTIL
#define PARALLEL_FDTD_LAM_FXN_UTIL
#include <algorithm>

/**
 * @brief      {Calculates x+y^2 (used with std::accumulate to get magnitude of a vector)
 *
 * @tparam     T     type of data structure
 * @param[in]  x     value of current sum
 * @param[in]  y     value of what you want to add
 *
 * @return     x_y^2
 */
template <class T> struct vecMagAdd {
    T operator() (const T& x, const T& y) const {return x+y*y;}
    typedef T first_argument_type;
    typedef T second_argument_type;
    typedef T result_type;
};

/**
 * @brief      {Calculates x*y/2 (used with std::transform to average out field values)
 *
 * @tparam     T1    type of x
 * @tparam     T2    type of y
 * @param[in]  x     object x
 * @param[in]  y     object y
 *
 * @return     x*y/2
 */
template <class T1, class T2> struct multAvg {
    T2 operator() (const T1& x, const T2& y) const {return x*y/2.0;}
    typedef T1 first_argument_type;
    typedef T2 second_argument_type;
    typedef T2 result_type;
};

/**
 * @brief      Normalizes vector v
 *
 * @param      v     the vector to be normalized
 *
 * @tparam     T     Type of the vector v
 */
template <typename T> void normalize(std::vector<T>& v)
{
    T norm = std::sqrt( std::accumulate(v.begin(), v.end(), 0.0, vecMagAdd<T>() ) );
    std::transform(v.begin(), v.end(), v.begin(), [&](T g){return g/norm;});
    return;
}

/**
 * @brief      Normalizes std::array<T, SIZE v
 *
 * @param      v     the array to be normalized
 *
 * @tparam     T     Type of the the data structures in  v
 * @tparam     SIZE  Size of array  v
 */
template <typename T, std::size_t SIZE > void normalize(std::array<T,SIZE>& v)
{
    T norm = std::sqrt( std::accumulate(v.begin(), v.end(), 0.0, vecMagAdd<T>() ) );
    std::transform(v.begin(), v.end(), v.begin(), [&](T g){return g/norm;});
    return;
}

/**
 * @brief      LU factorizes using getrf and calculates the determent of a matrix mat
 *
 * @param[in]  mat    The matrix
 * @param[in]  order  The order of the matrix
 *
 * @tparam     T      Type of the values in the matrix
 *
 * @return     The matrix determinant.
 */
template <typename T> T getMatDet(std::vector<T> mat, int order)
{
    if( order > std::sqrt( mat.size() ) )
        throw std::logic_error("The size of the vector storing the matrix is smaller than the stated order^2. The determinant can't be calculated");
    std::vector<int>ipiv(order, 0.0);
    int info;
    dgetrf_(order, order, mat.data(), order, ipiv.data(), &info);
    double det = 1.0;
    for(int jj = 1; jj <=order; ++jj)
        det*= (ipiv[jj-1] != jj) ? -1.0*mat[(jj-1)*(order+1)] : mat[(jj-1)*(order+1)];
    return det;
}

/**
 * @brief      Calculates the determinant of an LU factorized matrix mat (LU factorization done by getrf)
 *
 * @param[in]  mat    The LU factorized matrix
 * @param[in]  ipiv   The pivot table from getrf+
 * @param[in]  order  The order of the matrix
 *
 * @tparam     T      type of data in materix
 *
 * @return     The determinant.
 */
template <typename T> T getLUFactMatDet(const std::vector<T>& mat, const std::vector<int>& ipiv, int order)
{
    if( order > std::sqrt( mat.size() ) )
        throw std::logic_error("The size of the vector storing the matrix is smaller than the stated order^2. The determinant can't be calculated");
    double det = 1.0;
    for(int jj = 1; jj <=order; ++jj)
        det*= (ipiv[jj-1] != jj) ? -1.0*mat[(jj-1)*(order+1)] : mat[(jj-1)*(order+1)];
    return det;
}

/**
 * @brief      Converts Cartesian coordinate array to barycentric coordinates
 *
 * @param[in]  invVertMat  The inverse of the vertex matrix
 * @param[in]  pt          The point in Cartesian
 * @param[in]  d0          The determinant of the vertex matrix
 *
 * @tparam     T           type of data in point
 * @tparam     SIZE        size of the array
 *
 * @return     pt in barycentric coordinates with respect to the simplex defined by invVertMat
 */
template <typename T, std::size_t SIZE> std::array<T,SIZE> cart2bary(const std::vector<T>& invVertMat, const std::array<T, SIZE-1> pt, double d0)
{
    if( SIZE > std::sqrt( invVertMat.size() ) )
        throw std::logic_error("The size of the vector storing the matrix is smaller than the stated order^2. The determinants to calculate the barycenteric coordinates can't be calculated");
    std::array<T,SIZE> baryCen;
    std::array<T,SIZE> ptPad;
    std::copy_n(pt.data(), SIZE-1, ptPad.data());
    ptPad[SIZE-1] = 1.0;
    dgemv_('T', SIZE, SIZE, 1.0, invVertMat.data(), SIZE, ptPad.data(), 1, 0.0, baryCen.data(), 1);
    return baryCen;
}


#endif