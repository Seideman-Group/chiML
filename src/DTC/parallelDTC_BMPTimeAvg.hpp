/** @file DTC/parallelDTC_BMP.hpp
 *  @brief Class that outputs FDTD field information into a bitmap file
 *
 *  Uses FDTD grids to convert the base data into the form that should be outputted and
 *  prints to a bitmap file and potentially prints the same data into a text file in
 *  either coordinate or matrix form.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */
#ifndef FDTD_pARALLELDETECTOR_BMP
#define FDTD_pARALLELDETECTOR_BMP

#include <src/DTC/parallelDTC.hpp>
#include <src/DTC/toBitMap.hpp>

class parallelDetectorBMPReal : public parallelDetectorBaseReal
{
protected:
    using parallelDetectorBaseReal::timeInterval_; //!< The stride of the time (How often should the detector print?)
    using parallelDetectorBaseReal::tConv_; //!< conversion factor for t to get it in the correct units
    using parallelDetectorBaseReal::convFactor_; //!< Conversion factor for the type of output (to SI units from FDTD)
    using parallelDetectorBaseReal::sz_; //!< the location of the detector's lower left corner
    using parallelDetectorBaseReal::loc_; //!< the size in grid points of the detector
    using parallelDetectorBaseReal::realSpaceLoc_; //!< Location of lower, left, back corner in real spaceZ
    using parallelDetectorBaseReal::fields_; //!< A vector of shared pointers to each of the grids associated with the detector
    using parallelDetectorBaseReal::outputCollectFunction_; //!< function to take grids and output to file in the correct manner

    int t_start_; //!< Start collecting and averaging fields
    int t_end_; //!< Finish collecting and averaging the fields

    std::array<int,3> corInds_; //!< Array storing the indexes of the {blas direction, loop in same file directions, loop and output to separate file directions} for outputting (all either 0,1,2)
    std::array<int,3> add_k_; //!< vector to add k in (moves to new slice)
    std::string filePrefix_; //!< prefix for each image file

    GRIDOUTTYPE outType_; //!< What of the field to output (real comp, imag comp, magnitude, power)
    std::function<bool(double, double)> funcComp_; //!< function used to compare values
    std::function<double(double)> outOpp_; //!< function assocated with outType_
    double dt_; //!< step of the simulation
    PLOTTYPE pType_; //!< plotting output type associated with outType
    std::string outFile_; //!< output file name
    std::vector<real_grid_ptr> gridBMP_; //!< shared_ptr to grid that will store the bitmap images
    real_grid_ptr collectedGrid_; //!< shared_ptr to the a grid storing all the collected average values

public:

    /**
     * @brief      Constructs a detector that outputs to a bmp file for real fields
     *
     * @param[in]  grid          a vector of pointers to output grids
     * @param[in]  SI            bool to determine if SI units are used
     * @param[in]  loc           location of lower left corner of the dtc
     * @param[in]  sz            size in grid points for the dtc
     * @param[in]  out_name      output file name
     * @param[in]  fxn           what to output into bmp file
     * @param[in]  txtType       The format of the associated text file
     * @param[in]  type          The type: output type of dtc: fields or power
     * @param[in]  timeInterval  Time interval for how often to record data
     * @param[in]  a             unit length
     * @param[in]  I0            unit current
     * @param[in]  dt            time step
     */
    parallelDetectorBMPReal(std::vector< std::pair< real_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, int t_start, int t_end std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double timeInterval, double a, double I0, double dt);
    /**
     * @brief Outputs to a bmp and potentially text file
     *
     * @param[in] current simulation time
     */
    void output(double t);

    void toFile();
    /**
     * @brief returns the output file name
     */
    inline std::string outfile() {return outFile_;}
};

class parallelDetectorBMPCplx : public parallelDetectorBaseCplx
{
protected:
    using parallelDetectorBaseCplx::timeInterval_; //!< The stride of the time (How often should the detector print?)
    using parallelDetectorBaseCplx::tConv_; //!< conversion factor for t to get it in the correct units
    using parallelDetectorBaseCplx::convFactor_; //!< Conversion factor for the type of output (to SI units from FDTD)
    using parallelDetectorBaseCplx::sz_; //!< the location of the detector's lower left corner
    using parallelDetectorBaseCplx::loc_; //!< the size in grid points of the detector
    using parallelDetectorBaseCplx::realSpaceLoc_; //!< Location of lower, left, back corner in real spaceZ
    using parallelDetectorBaseCplx::fields_; //!< A vector of shared pointers to each of the grids associated with the detector
    using parallelDetectorBaseCplx::outputCollectFunction_; //!< function to take grids and output to file in the correct manner

    int t_start_; //!< Start collecting and averaging fields
    int t_end_; //!< Finish collecting and averaging the fields

    std::array<int,3> corInds_;
    std::array<int,3> add_k_;
    std::string filePrefix_;

    GRIDOUTTYPE outType_; //!< What of the field to output (real comp, imag comp, magnitude, power)
    std::function<double(cplx)> outOpp_; //!< function assocated with outType_
    std::function<bool(cplx, cplx)> funcComp_; //!< function used to compare values
    double dt_; //!< step of the simulation
    PLOTTYPE pType_; //!< plotting output type associated with outType
    std::string outFile_; //!< output file name
    std::vector<cplx_grid_ptr> gridBMP_; //!< shared_ptr to grid that will store the bitmap images
    cplx_grid_ptr collectedGrid_; //!< shared_ptr to the a grid storing all the collected average values

public:
    /**
     * @brief      Constructs a detector that outputs to a bmp file for real fields
     *
     * @param[in]  grid          a vector of pointers to output grids
     * @param[in]  SI            bool to determine if SI units are used
     * @param[in]  loc           location of lower left corner of the dtc
     * @param[in]  sz            size in grid points for the dtc
     * @param[in]  out_name      output file name
     * @param[in]  fxn           what to output into bmp file
     * @param[in]  txtType       The format of the associated text file
     * @param[in]  type          The type: output type of dtc: fields or power
     * @param[in]  timeInterval  Time interval for how often to record data
     * @param[in]  a             unit length
     * @param[in]  I0            unit current
     * @param[in]  dt            time step
     */
    parallelDetectorBMPCplx(std::vector< std::pair< cplx_pgrid_ptr, std::array<int,3> > >  grid, bool SI, std::array<int,3> loc, std::array<int,3> sz, int t_start, int t_end, std::string out_name, GRIDOUTFXN fxn, GRIDOUTTYPE txtType, DTCTYPE type, double timeInterval, double a, double I0, double dt);
    /**
     * @brief Outputs to a bmp and potentially text file
     *
     * @param[in] current simulation time
     */
    void output(double t);

    void toFile();
    /**
    /**
     * @brief returns the output file name
     */
    std::string outfile() {return outFile_;}
};
#endif
