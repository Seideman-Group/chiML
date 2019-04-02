/** @file GRID/Grid.hpp
 *  @brief Storage and accessing class for the FDTD component fields
 *
 *  Class that acts as the data storage for the FDTD grids on one process
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_GRID
#define FDTD_GRID

#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include <complex>
#include <assert.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <numeric>

template <typename T> class Grid
{
protected:
    static unsigned int memSize; //!< total memory of the grid
    std::array<int,3> n_vec_; //!< the number of grid points in all directions
    std::array<double,3>  d_; //!< the grid point spacing in all directions
    std::unique_ptr<T[]> vals_; //!< the field values in the grid

public:
    /**
     * @brief      Constructs a grid
     *
     * @param[in]  n_vec  grid size in terms of number of grid points in all directions
     * @param[in]  d      the gird spacing in all directions
     */
    Grid(std::array<int,3> n_vec, std::array<double,3> d) :
        n_vec_(n_vec),
        d_(d),
        vals_(std::unique_ptr<T[]>(new T[std::accumulate (n_vec.begin(), n_vec.end(), 1, std::multiplies<int>())]))
    {
        std::fill_n(vals_.get(), std::accumulate(n_vec_.begin(), n_vec_.end(), 1, std::multiplies<int>()), T(0.0));
        memSize += sizeof(T)*size();
    }

    /**
     * @brief      Copies a grid into a new grid object
     *
     * @param[in]  o     Grid to be copied
     */
    Grid(const Grid& o) :
        n_vec_(o.n_vec_),
        d_(o.d_),
        vals_(std::unique_ptr<T[]>(new T[std::accumulate (o.n_vec_.begin(), o.n_vec_.end(), 1, std::multiplies<int>())]))
    {
       std::copy_n(o.vals_.get(), std::accumulate (n_vec_.begin(), n_vec_.end(), 1, std::multiplies<int>()), vals_.get());
       memSize += sizeof(T)*size();
    }

    /**
     * @brief      Moves a grid into a new grid object
     *
     * @param[in]  o  Grid to be moved
     */
    Grid(Grid&& o) :
        n_vec_(o.n_vec_),
        d_(o.d_),
        vals_(std::move(o.vals_))
    {
        o.n_vec_ = std::array<int,3>(3,0);
    }

    /**
     * @brief Grid destructor
     */
    ~Grid(){memSize -= sizeof(T)*size();}

    /**
     * @brief  Gets the total size of the grid's vals array
     *
     * @return the size of the grid
     */
    inline int size() const { return std::accumulate(n_vec_.begin(), n_vec_.end(), 1, std::multiplies<int>());}

    /**
     * @brief  Accessor function to vals_ data
     *
     * @return the data stored in the grid
     */
    T* data() {return vals_.get();}

    /**
     * @brief  Accessors function to vals_ data
     *
     * @return a const form of the data
     */
    const T* data() const { return vals_.get(); }

    /**
     * @brief  Accessor function to n_vec_[0]
     *
     * @return the number of grid points in the x direction
     */
    inline int x() const {return n_vec_[0];}

    /**
     * @brief  Accessor function to n_vec_[1]
     *
     * @return the number of grid points in the y direction
     */
    inline int y() const {return n_vec_[1];}

    /**
     * @brief  Accessor function to n_vec_[2]
     *
     * @return the number of grid points in the z direction
     */
    inline int z() const {return n_vec_[2];}

    /**
     * @brief  Accessor function to n_vec_
     *
     * @return the number of grid points in all directions
     */
    inline std::array<int,3> n_vec(){return n_vec_;}

    /**
     * @brief  Accessor function to n_vec_[ii]
     *
     * @return the number of grid points in the iith directions
     */
    inline int  n_vec(int ii){return n_vec_[ii];}

    /**
     * @brief  Accessor function to d_[0]
     *
     * @return the grid spacing in the x direction
     */
    inline double dx() const {return d_[0];}

    /**
     * @brief  Accessor function to d_[1]
     *
     * @return the grid spacing in the y direction
     */
    inline double dy() const {return d_[1];}

    /**
     * @brief  Accessor function to d_[2]
     *
     * @return the grid spacing in the z direction
     */
    inline double dz() const {return d_[2];}

    /**
     * @brief  Accessor function to d_
     *
     * @return the grid spacing in all directions
     */
    inline std::array<double,3> d(){return d_;}

    /**
     * @brief      Zeros the grid
     */
    inline void zero() { std::fill_n(vals_.get(), std::accumulate(n_vec_.begin(), n_vec_.end(), 1, std::multiplies<int>()), T(0.0)); }

    /**
     * @brief      returns the value at a point x_val,y_val,0
     *
     * @param[in]  x_val  gird coordinate in the x direction
     * @param[in]  y_val  grid coordinate in the y direction
     *
     * @return     reference to the data point at (x,y,0)
     */
    T& point(const int ind)
    {
        return vals_[ind];
    }

    /**
     * @brief      returns the value at a point x_val,y_val,0
     *
     * @param[in]  x_val  gird coordinate in the x direction
     * @param[in]  y_val  grid coordinate in the y direction
     *
     * @return     reference to the data point at (x,y,0)
     */
    const T& point(const int ind) const{ return vals_[ind];}

    /**
     * @brief      Gets the ind value of the point x, y, z
     *
     * @param[in]  x     x coordinate
     * @param[in]  y     y coordinate
     * @param[in]  z     z coordinate
     *
     * @return     The index of point (x, y, z).
     */
    inline int getInd(int x, int y, int z=0)
    {
        assert(0 <= x && x < n_vec_[0] && 0 <= y && y < n_vec_[1] && 0 <= z && z < n_vec_[2] );
        return x + n_vec_[0] * ( z + y*n_vec_[2] );
    }
    /**
     * @brief      Gets the ind.
     *
     * @param      ptArr  The point to get the index
     *
     * @return     The  of point (ptArr[0], ptArr[1], ptArr[2]).
     */
    inline int getInd(std::array<int,3>& pt)
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] && 0 <= pt[2] && pt[2] < n_vec_[2] );
        return pt[0] + n_vec_[0] * ( pt[2] + pt[1]*n_vec_[2] );
    }
    /**
     * @brief      Gets the ind.
     *
     * @param      ptArr  The point to get the index
     *
     * @return     The  of point (ptArr[0], ptArr[1]).
     */
    inline int getInd(std::array<int,2>& pt)
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] );
        return pt[0] + n_vec_[0] * ( pt[1] );
    }

    /**
     * @brief      returns the value at a point x_val,y_val,z_val
     *
     * @param[in]  x_val  gird coordinate in the x direction
     * @param[in]  y_val  grid coordinate in the y direction
     * @param[in]  z_val  grid coordinate in the z direction
     *
     * @return     reference to the data point at (x,y,z)
     */
    T& point(const int x_val, const int y_val, const int z_val=0)
    {
        assert(0 <= x_val && x_val < n_vec_[0] && 0 <= y_val && y_val < n_vec_[1] && 0 <= z_val && z_val < n_vec_[2] );
        return vals_[( y_val*n_vec_[2] + z_val ) * n_vec_[0] + x_val];
    }

    /**
     * @brief      returns the value at a point x_val,y_val,z_val
     *
     * @param[in]  x_val  gird coordinate in the x direction
     * @param[in]  y_val  grid coordinate in the y direction
     * @param[in]  z_val  grid coordinate in the z direction
     *
     * @return     reference to the data point at (x,y,z)
     */
    const T& point(const int x_val, const int y_val, const int z_val=0) const
    {
        assert(0 <= x_val && x_val < n_vec_[0] && 0 <= y_val && y_val < n_vec_[1] && 0 <= z_val && z_val < n_vec_[2] );
        return vals_[( y_val*n_vec_[2] + z_val ) * n_vec_[0] + x_val];
    }

    /**
     * @brief      returns the value at a point x_val,y_val,z_val
     *
     * @param[in]  x_val  gird coordinate in the x direction
     * @param[in]  y_val  grid coordinate in the y direction
     * @param[in]  z_val  grid coordinate in the z direction
     *
     * @return     reference to the data point at (x,y,z)
     */
    T& operator()(const int x_val, const int y_val, const int z_val=0)
    {
        assert(0 <= x_val && x_val < n_vec_[0] && 0 <= y_val && y_val < n_vec_[1] && 0 <= z_val && z_val < n_vec_[2] );
        return vals_[( y_val*n_vec_[2] + z_val ) * n_vec_[0] + x_val];
    }

    /**
     * @brief      returns the value at a point x_val,y_val,z_val
     *
     * @param[in]  x_val  gird coordinate in the x direction
     * @param[in]  y_val  grid coordinate in the y direction
     * @param[in]  z_val  grid coordinate in the z direction
     *
     * @return     reference to the data point at (x,y,z)
     */
    const T& operator()(const int x_val, const int y_val, const int z_val=0) const
    {
        assert(0 <= x_val && x_val < n_vec_[0] && 0 <= y_val && y_val < n_vec_[1] && 0 <= z_val && z_val < n_vec_[2] );
        return vals_[( y_val*n_vec_[2] + z_val ) * n_vec_[0] + x_val];
    }

        /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline T& point(std::array<int,3>& pt)
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] && 0 <= pt[2] && pt[2] < n_vec_[2] );
        return vals_[ ( pt[1]*n_vec_[2] + pt[2] ) * n_vec_[0] + pt[0] ];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline const T& point(std::array<int,3>& pt) const  {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] && 0 <= pt[2] && pt[2] < n_vec_[2] );
        return vals_[ ( pt[1]*n_vec_[2] + pt[2] ) *n_vec_[0] + pt[0]];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline T* point_ptr(std::array<int,3>& pt)
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] && 0 <= pt[2] && pt[2] < n_vec_[2] );
        return &vals_[ ( pt[1]*n_vec_[2] + pt[2] ) *n_vec_[0] + pt[0]];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline const T* point_ptr(std::array<int,3>& pt) const
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] && 0 <= pt[2] && pt[2] < n_vec_[2] );
        return &vals_[ ( pt[1]*n_vec_[2] + pt[2] ) *n_vec_[0] + pt[0]];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline T& operator()(std::array<int,3>& pt) { return vals_[ ( pt[1]*n_vec_[2] + pt[2] ) *n_vec_[0] + pt[0]];; }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline const T& operator()(std::array<int,3>& pt) const { return vals_[ ( pt[1]*n_vec_[2] + pt[2] ) *n_vec_[0] + pt[0]]; }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline T& point(std::array<int,2>& pt)
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] );
        return vals_[ ( pt[1] ) * n_vec_[0] + pt[0] ];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline const T& point(std::array<int,2>& pt) const  {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] );
        return vals_[ ( pt[1] ) *n_vec_[0] + pt[0]];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline T* point_ptr(std::array<int,2>& pt)
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] );
        return &vals_[ ( pt[1] ) *n_vec_[0] + pt[0]];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline const T* point_ptr(std::array<int,2>& pt) const
    {
        assert(0 <= pt[0] && pt[0] < n_vec_[0] && 0 <= pt[1] && pt[1] < n_vec_[1] );
        return &vals_[ ( pt[1] ) *n_vec_[0] + pt[0]];
    }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline T& operator()(std::array<int,2>& pt) { return vals_[ ( pt[1] ) *n_vec_[0] + pt[0]];; }

    /**
     * @brief      returns the value at a point pt
     *
     * @param[in]  pt    The point
     *
     * @return     The value at point pt
     */
    inline const T& operator()(std::array<int,2>& pt) const { return vals_[ ( pt[1] ) *n_vec_[0] + pt[0]]; }

    /**
     * @brief      Gets the XZ plane at global point y=j.
     *
     * @param[in]  j     global value of y to take the plane at
     *
     * @return     The XZ plane at global point y=j.
     */
    std::vector<T> getPlaneXZ(const int j)
    {
        std::vector<T> planeXZ(n_vec_[0]*n_vec_[2], 0.0);
        std::copy_n(&point(0,j,0), n_vec_[0]*n_vec_[2], planeXZ.data());
        return planeXZ;
    }

    /**
     * @brief      Gets the XY plane at global point z=k.
     *
     * @param[in]  k     global value of z to take the plane at
     *
     * @return     The XZ plane at global point z=k.
     */
    std::vector<T> getPlaneXY(const int k)
    {
        std::vector<T> planeXY(n_vec_[0]*n_vec_[1], 0.0);
        for(int yy = 0; yy < n_vec_[1]; ++yy)
            std::copy_n( &point(0, yy, k), n_vec_[0], &planeXY[n_vec_[0]*yy] );
        return planeXY;
    }

    /**
     * @brief      Gets the YZ plane at global point x=i.
     *
     * @param[in]  i     global value of x to take the plane at
     *
     * @return     The YZ plane at global point x=i.
     */
    std::vector<T> getPlaneYZ(const int i)
    {
        std::vector<T> planeYZ(n_vec_[1]*n_vec_[2], 0.0);
        for(int yy = 0; yy < n_vec_[1]; ++yy)
            for(int zz = 0; zz < n_vec_[2]; ++zz)
                planeYZ[n_vec_[2]*yy + zz] = point(i, yy, zz);
        return planeYZ;
    }

    /**
     * @brief      Prints out a field snapshot in the area specified by lower, left, back corner loc and size of sz, of the field values in a grid, in a box format separate XY planes separated by a line
     *
     * @param[in]  filename  output file name
     * @param[in]  loc       location of the lower left corner of the field area desired
     * @param[in]  sz        size in grid points of the region you want to print out
     * @param[in]  fxn       function defining what to print out from the field, real, imaginary, magnitude intensity
     */
    void gridOutBox(std::string filename, std::array<int,3> loc, std::array<int,3> sz, std::function<double(T)> fxn)
    {
        std::ofstream outFile;
        outFile.open(filename,std::ios_base::out);
        for(int kk = loc[2]; kk < loc[2]+sz[2]; ++kk)
        {
            for(int jj = loc[1]; jj < loc[1]+sz[1]; ++jj)
            {
                for(int ii = loc[0]; ii < loc[0]+sz[0]; ++ii)
                {
                    outFile << fxn(point(ii,jj,kk)) << "\t";
                }
                outFile << "\n";
            }
            outFile << "\n";
        }
    }

    /**
     * @brief      Prints out a field snapshot of the field values in a grid, in a box format separate XY planes separated by a line
     *
     * @param[in]  filename  output file name
     * @param[in]  fxn       function defining what to print out from the field, real, imaginary, magnitude intensity
     */
    void gridOutBox(std::string filename, std::function<double(T)> fxn)
    {
        std::ofstream outFile;
        outFile.open(filename,std::ios_base::out);
        for(int kk = 0; kk < n_vec_[2]; ++kk)
        {
            for(int jj = 0; jj < n_vec_[1]; ++jj)
            {
                for(int ii = 0; ii < n_vec_[0]; ++ii)
                {
                    outFile << fxn(point(ii,jj,kk)) << "\t";
                }
                outFile << "\n";
            }
            outFile << "\n";
        }
    }

    /**
     * @brief      Prints out a field snapshot in the area specified by lower, left, back corner loc and size of sz, of the field values in a grid, in a coordinate list format
     *
     * @param[in]  filename  output file name
     * @param[in]  loc       location of the lower left corner of the field area desired
     * @param[in]  sz        size in grid points of the region you want to print out
     * @param[in]  fxn       function defining what to print out from the field, real, imaginary, magnitude intensity
     */
    void gridOutList(std::string filename, std::array<int,3> loc, std::array<int,3> sz, std::function<double(T)> fxn)
    {
        std::ofstream outFile;
        outFile.open(filename,std::ios_base::out);
        outFile << "#x\ty\tz\toutput" << std::endl;
        for(int ii = loc[0]; ii < loc[0]+sz[0]; ++ii)
        {
            for(int jj = loc[1]; jj < loc[1]+sz[1]; ++jj)
            {
                for(int kk = loc[2]; kk < loc[2]+sz[2]; ++kk)
                {
                    outFile << ii << "\t" << jj << "\t" << kk << "\t" << fxn( point(ii,jj,kk) ) <<  "\n";
                }
            }
        }
    }

    /**
     * @brief      Prints out a field snapshot in the area specified by lower, left, back corner loc and size of sz, of the field values in a grid, in a coordinate list format
     *
     * @param[in]  filename  output file name
     * @param[in]  fxn       function defining what to print out from the field, real, imaginary, magnitude intensity
     */
    void gridOutList(std::string filename, std::function<double(T)> fxn)
    {
        std::ofstream outFile;
        outFile.open(filename,std::ios_base::out);
        outFile << "#x\ty\tz\toutput" << std::endl;
        for(int ii = 0; ii < n_vec_[0]; ++ii)
        {
            for(int jj = 0; jj < n_vec_[1]; ++jj)
            {
                for(int kk = 0; kk < n_vec_[2]; ++kk)
                {
                    outFile << ii << "\t" << jj << "\t" << kk << "\t" << fxn( point(ii,jj,kk) ) << "\n";
                }
            }
        }
    }

    /**
     * @brief      Prints out a field snapshot in the area specified by lower, left, back corner loc and size of sz, of the field values in a grid, in a box format separate XY planes separated by a line
     *
     * @param[in]  filename     output file name
     * @param[in]  otherFields  A vector storing other field grid pointers to need to be added to the output
     * @param[in]  loc          location of the lower left corner of the field area desired
     * @param[in]  sz           size in grid points of the region you want to print out
     * @param[in]  fxn          function defining what to print out from the field, real, imaginary, magnitude intensity
     */
    void gridOutBox(std::string filename, std::vector<std::shared_ptr<Grid<T>>> otherFields, std::array<int,3> loc, std::array<int,3> sz, std::function<double(T)> fxn)
    {
        std::ofstream outFile;
        outFile.open(filename,std::ios_base::out);
        T pt = 0.0;
        for(int kk = loc[2]; kk < loc[2]+sz[2]; ++kk)
        {
            for(int jj = loc[1]; jj < loc[1]+sz[1]; ++jj)
            {
                for(int ii = loc[0]; ii < loc[0]+sz[0]; ++ii)
                {
                    pt = fxn(point(ii,jj,kk));
                    for(auto& grid : otherFields)
                        pt += fxn(grid->point(ii,jj,kk));
                    outFile << pt << "\t";
                }
                outFile << "\n";
            }
           outFile << "\n";
        }
    }

    /**
     * @brief      Prints out a field snapshot in the area specified by lower, left, back corner loc and size of sz, of the field values in a grid, in a coordinate list format
     *
     * @param[in]  filename     output file name
     * @param[in]  otherFields  A vector storing other field grid pointers to need to be added to the output
     * @param[in]  loc          location of the lower left corner of the field area desired
     * @param[in]  sz           size in grid points of the region you want to print out
     * @param[in]  fxn          function defining what to print out from the field, real, imaginary, magnitude intensity
     */
    void gridOutList(std::string filename, std::vector<std::shared_ptr<Grid<T>>> otherFields, std::array<int,3> loc, std::array<int,3> sz, std::function<double(T)> fxn)
    {
        std::ofstream outFile;
        outFile.open(filename,std::ios_base::out);
        outFile << "#x\ty\tz\toutput" << std::endl;
        double pt = 0.0;
        for(int ii = loc[0]; ii < loc[0]+sz[0]; ++ii)
        {
            for(int jj = loc[1]; jj < loc[1]+sz[1]; ++jj)
            {
                for(int kk = loc[2]; kk < loc[2]+sz[2]; ++kk)
                {
                    pt = fxn(point(ii,jj,kk));
                    for(auto& grid : otherFields)
                        pt += fxn(grid->point(ii,jj,kk));
                    outFile << ii << "\t" << jj << "\t" << kk << "\t" << pt << "\n";
                }
                outFile << "\n";
            }
        }
    }
};

template <typename T> unsigned int Grid<T>::memSize = 0;

#endif