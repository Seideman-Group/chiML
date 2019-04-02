/** @file DTC/parallelDTC.hpp
 *  @brief Parent Class for all detectors
 *
 *  This contains the prototypes for the field detector classes (defines the parent class)
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_PARALLELDETECTOR
#define FDTD_PARALLELDETECTOR

#include <DTC/parallelDTCOutputFxn.hpp>
#include <DTC/parallelStorageDTC.hpp>

template <typename T> class parallelDetectorBase
{
protected:
    DTCTYPE type_; //!< The type of the detector: EX,EY,EZ,HX,HY,HZ,EPWR,HPWR
    int timeInterval_; //!< The stride of the time (How often should the detector print?)
    double tConv_; //!< conversion factor for t to get it in the correct units
    double convFactor_; //!< Conversion factor for the type of output (to SI units from FDTD)
    std::array<int,3> loc_; //!< the location of the detector's lower left corner
    std::array<int,3> sz_; //!< the size in grid points of the detector
    std::array<double,3> realSpaceLoc_; //!< Location of lower, left, back corner in real spaceZ

    std::vector<std::shared_ptr<parallelStorageDTC<T>>> fields_; //!< A vector of shared pointers to each of the grids associated with the detector
    std::function<void(T* gridInBegin, T* gridInEnd, T* valsOut, double convFactor)> outputCollectFunction_; //!< function to take grids and output to file in the correct manner
public:

    /**
     * @brief      Constructs a base detector
     *
     * @param[in]  grids         The fields that need to be output
     * @param[in]  SI            True if using SI units
     * @param[in]  loc           The location in grid point position
     * @param[in]  sz            The size in number of grid points
     * @param[in]  type          The type: output type of dtc: fields or power
     * @param[in]  timeInterval  The time interval
     * @param[in]  a             unit length
     * @param[in]  I0            unit current
     * @param[in]  dt            time step
     */
    parallelDetectorBase(std::vector<std::pair< std::shared_ptr<parallelGrid<T>>, std::array<int,3> > > gridsAndOffSets, bool SI, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, double timeInterval, double a, double I0, double dt)  :
        type_(type),
        timeInterval_(static_cast<int>(std::floor(timeInterval/dt) ) ),
        tConv_(1.0),
        convFactor_(1.0),
        loc_(loc),
        sz_(sz)
    {
        realSpaceLoc_ = {0,0,0};
        if(timeInterval_ == 0)
            throw std::logic_error("The time step of a detector is less than the main grid or set to 0.");
        std::shared_ptr<parallelGrid<T>> grid0 = std::get<0>(gridsAndOffSets[0]);
        if(static_cast<double>(timeInterval_) * dt != timeInterval && grid0->gridComm()->rank() == 0)
            std::cout << "Time step changed to " << std::to_string(timeInterval_*dt) << std::endl;
        // Find location of detector in real space
        for(int ii = 0; ii < 3; ++ii)
        {
            realSpaceLoc_[ii] = grid0->d()[ii] * (loc_[ii] - (grid0->n_vec(ii) - grid0->gridComm()->npArr()[ii]*2 - grid0->n_vec(ii)%2)/2);
        }
        if(std::get<0>(gridsAndOffSets[0])->z() == 1 )
            realSpaceLoc_[2] == 0.0;
        // Conversions to SI
        if(SI)
        {
            tConv_ *= a / SPEED_OF_LIGHT;
            dscal_(realSpaceLoc_.size(), a, realSpaceLoc_.data(), 1);
            // Base for both Magnetic and Electric fields
            convFactor_ = (I0/a);
            // Do additional conversions for the given types
            if(type == DTCTYPE::EX || type == DTCTYPE::EY || type == DTCTYPE::EZ)
                convFactor_ /= EPS0*SPEED_OF_LIGHT;
            else if(type == DTCTYPE::PX || type == DTCTYPE::PY || type == DTCTYPE::PZ)
                convFactor_ /= EPS0*SPEED_OF_LIGHT;
            else if(type == DTCTYPE::MX || type == DTCTYPE::MY || type == DTCTYPE::MZ)
                convFactor_ /= EPS0*std::pow(SPEED_OF_LIGHT,2.0);
            for(auto& rr : realSpaceLoc_)
                rr *= a;
        }
        // Set unit conversions and output functions for the fields
        if(type == DTCTYPE::EPOW || type == DTCTYPE::HPOW)
        {
            // Power the conversion factor is the square of the field one
            convFactor_ *= convFactor_;
            outputCollectFunction_ = pwrOutputFunction<T>;
        }
        else
        {
            outputCollectFunction_ = fieldOutputFunction<T>;
        }
    }
    /**
     * @return timeInterval_
     */
    inline int &timeInt() {return timeInterval_;}

    /**
     * @return location of dtc lower left corner
     */
    inline std::array<int,3> &loc() {return fields_[0]->loc();}

    /**
     * @return size of dtc lower left corner
     */
    inline std::array<int,3> &sz() {return fields_[0]->sz();}

    virtual void toFile()
    {
        return;
    }
    /**
     * @brief Output the fields
     * @details At time t output the fields
     *
     * @param t Time of the output.
     */
    virtual void output(double t) = 0;
};

class parallelDetectorBaseReal : public parallelDetectorBase<double>
{
public:
    /**
     * @brief      Constructs a base detector
     *
     * @param[in]  grids         The fields that need to be output
     * @param[in]  SI            True if using SI units
     * @param[in]  loc           The location in grid point position
     * @param[in]  sz            The size in number of grid points
     * @param[in]  type          The type: output type of dtc: fields or power
     * @param[in]  timeInterval  The time interval
     * @param[in]  a             unit length
     * @param[in]  I0            unit current
     * @param[in]  dt            time step
     */
    parallelDetectorBaseReal(std::vector<std::pair< real_pgrid_ptr, std::array<int,3> > > gridsAndOffSets, bool SI, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, double timeInterval, double a, double I0, double dt);
    /**
     * @brief      outputs the data to the field
     *
     * @param[in]  t     current simulation time
     */
    virtual void output(double t) = 0;
};

/**
 * Complex version of FluxDTC see base class for more descriptions
 */
class parallelDetectorBaseCplx : public parallelDetectorBase<cplx>
{
public:
    /**
     * @brief      Constructs a base detector
     *
     * @param[in]  grids         The fields that need to be output
     * @param[in]  SI            True if using SI units
     * @param[in]  loc           The location in grid point position
     * @param[in]  sz            The size in number of grid points
     * @param[in]  type          The type: output type of dtc: fields or power
     * @param[in]  timeInterval  The time interval
     * @param[in]  a             unit length
     * @param[in]  I0            unit current
     * @param[in]  dt            time step
     */
    parallelDetectorBaseCplx(std::vector<std::pair< cplx_pgrid_ptr, std::array<int,3> > > gridsAndOffSets, bool SI, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, double timeInterval, double a, double I0, double dt);
    /**
     * @brief      outputs the data to the field
     *
     * @param[in]  t     current simulation time
     */
    virtual void output(double t) = 0;
};

#endif
