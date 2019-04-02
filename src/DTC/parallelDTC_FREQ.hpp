/** @file DTC/parallelDTC_FREQ.hpp
 *  @brief Class that stores and outputs Fourier transformed field information
 *
 *  Uses FDTD grids to Fourier transform the FDTD fields on the fly and output the final
 *  results into a text file.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */

#ifndef FDTD_PARALLEL_DTC_FREQ
#define FDTD_PARALLEL_DTC_FREQ

#include <DTC/parallelStorageFreqDTC.hpp>
#include <DTC/parallelDTCOutputFxn.hpp>


template <typename T> class parallelDetectorFREQ_Base
{
public:

    struct masterImportDat
    {
        int addIndex_; //!< index of the first point on the surface
        int slaveProc_; //!< rank of the slave process this describes
        int sz_; //!< total size of the spatial coordinates of the detector
        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & addIndex_;
            ar & slaveProc_;
            ar & sz_;
        }
    };

    typedef std::shared_ptr<parallelGrid<T>> pgrid_ptr;
protected:

    std::shared_ptr<mpiInterface> gridComm_; //!< MPI Interface that handles all Communication
    bool pow_; //!< True if outputting a power
    bool outputMaps_; //!< True if output to a map
    DTCTYPE type_; //!< type of dtector
    int masterProc_; //!< rank of the master process

    int t_step_; //!< current time step
    int timeInt_; //!< Time interval
    int nfreq_; //!< number of frequencies
    int addIndex_; //!< index of the first point on the surface

    double dt_; //!< time step
    double freqConv_; //!< conversion factor for the frequency
    double convFactor_; //!< conversion factor for fields
    double dOmg_; //!< step size for angular frequency (-1 if using wavelength)
    double dLam_; //!< step size for wavelength (-1 if using freq)

    std::vector<std::shared_ptr<parallelStorageFreqDTC<T>> > gridsIn_; //!< struct for inputting fields
    std::vector<std::shared_ptr<masterImportDat>> outFInfo_; //!< strcut for master to put slave info in right place

    std::array<int,3> loc_; //!< location of the lower, left, back corner of the detector in grid points
    std::array<int,3> sz_; //!< size of the detector in grid points

    std::array<double,3> d_; //!< grid spacing in all directions

    std::vector<cplx> fftFact_; //!< vector storing the values of exp(-i $\omg$ t) at each time step
    std::vector<cplx> incIn_; //!< incident field input

    std::string fname_; //!< output file name

    std::vector<cplx_grid_ptr> freqFields_; //!< fields being used in detector
    std::vector<double> freqList_; //!< list of all frequencies

    std::function<cplx(int, cplx*, int, cplx*, int, double)> toOutFile_; //!< function to output correct values to the output file
    std::function<cplx(cplx)> getIncdField_; //!< converts real fields by returning real(cplx)
public:

    /**
     * @brief      Constructs a frequency detector based on frequencies
     *
     * @param[in]  name       filename of the detector
     * @param[in]  grids      The grids used for the detector
     * @param[in]  loc        The location of the lower, left, back corner of the detector region
     * @param[in]  sz         size of the detector region
     * @param[in]  type       The type: output type of dtc
     * @param[in]  timeInt    Time interval for how often to record data
     * @param[in]  freqList   The frequency list
     * @param[in]  outputMaps True if output to a map
     * @param[in]  d          grid spacing in all directions
     * @param[in]  dt         time step of the calculation
     * @param[in]  SI         store data in SI units
     * @param[in]  I0         unit current
     * @param[in]  a          unit length
     */
    parallelDetectorFREQ_Base(std::string name, std::vector< std::pair<pgrid_ptr, std::array<int,3> > > grids, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, int timeInt, bool outputMaps, std::vector<double> freqList, std::array<double,3> d, double dt, bool SI, double I0, double a) :
        gridComm_  (std::get<0>(grids[0])->gridComm() ),
        pow_       (false),
        outputMaps_(outputMaps),
        type_      (type),
        masterProc_(std::get<0>(grids[0])->getLocsProc_no_boundaries(loc[0], loc[1], loc[2]) ),
        t_step_    (0),
        timeInt_   (timeInt),
        nfreq_     (freqList.size()),
        dt_        (dt),
        d_         (d),
        freqConv_  (1.0),
        convFactor_(1.0),
        dOmg_      ( (freqList.size() > 1 && freqList[1]-freqList[0] == freqList[2]-freqList[1] ) ? freqList[1]-freqList[0] : 0.0 ),
        dLam_      ( (freqList.size() > 1 && freqList[1]-freqList[0] == freqList[2]-freqList[1] ) ? 0.0 : 1.0/freqList[1]-1.0/freqList[0]  ),
        loc_       (loc),
        sz_        (sz),
        fftFact_   (nfreq_,0.0),
        incIn_     (std::max(sz[0],sz[1]),0.0),
        fname_     (name),
        freqList_  (freqList)
    {
        // Do unit conversions if SI is true
        if(SI)
        {
            convFactor_ = (I0/a);
            if(type == DTCTYPE::EX || type == DTCTYPE::EY || type == DTCTYPE::EZ)
                convFactor_ /= EPS0*SPEED_OF_LIGHT;
            else if(type == DTCTYPE::PX || type == DTCTYPE::PY || type == DTCTYPE::PZ)
                convFactor_ /= EPS0*SPEED_OF_LIGHT;
            else if(type == DTCTYPE::MX || type == DTCTYPE::MY || type == DTCTYPE::MZ)
                convFactor_ /= EPS0*std::pow(SPEED_OF_LIGHT,2.0);
            freqConv_ = SPEED_OF_LIGHT / a;
        }
        // Set output function type
        if(type == DTCTYPE::EPOW || type == DTCTYPE::HPOW )
        {
            convFactor_ = std::pow(convFactor_, 2.0);
            pow_ = true;
            toOutFile_ = pwrOutFreqFunction;
        }
        else
        {
            toOutFile_ = fieldOutFreqFunction;
        }
        freqConv_ /= (M_PI*2.0);
        // Generate the data structure
        genOutStruct( std::get<0>(grids[0]) );
    }

    /**
     * @brief      Generates the list of parameters for each of the salve process and the master
     *
     * @param[in]  grids  vector containing all the grids that need to be outputted
     */
    void genOutStruct( pgrid_ptr grid0 )
    {
        masterImportDat toMaster;
        toMaster.slaveProc_ = -1;
        // FDTD grid split up into y lamella see if the location is in the correct area
        if(loc_[1] >= grid0->procLoc(1) && loc_[1] < grid0->procLoc(1) + grid0->local_y() -2) //!< Is the process in the process row that contains the lower boundary of the detector region?
        {
            toMaster.addIndex_ = 0;
        }
        else if(loc_[1] < grid0->procLoc(1) && loc_[1] + sz_[1] > grid0->procLoc(1)) //!< Is the process in a process row that the detector region covers?
        {
            toMaster.addIndex_ = (grid0->procLoc(1) - loc_[1]) * sz_[0]*sz_[2];
        }
        else
        {
            toMaster.addIndex_ = -1;
        }
        // If the detector is in the process
        if(toMaster.addIndex_ != -1)
        {
            int sz_x = sz_[0];
            int sz_y = -1;
            int sz_z = sz_[2];
            if( (sz_[1] + loc_[1] > grid0->procLoc(1) + grid0->local_y() - 2)) // Does the detector go through the end of the process' grid?
            {
                if(loc_[1] >= grid0->procLoc(1))
                    sz_y = grid0->local_y() + grid0->procLoc(1) - loc_[1] - 1;
                else
                    sz_y = grid0->local_y() - 2;
            }
            else
            {
                if(loc_[1] < grid0->procLoc(1))
                    sz_y = loc_[1] + sz_[1] - grid0->procLoc(1) - 1;
                else
                    sz_y = sz_[1];
            }
            toMaster.sz_ = sz_x*sz_y*sz_z;
            toMaster.slaveProc_ = gridComm_->rank();
        }
        if(gridComm_->rank() == masterProc_)
        {
            addIndex_ =  toMaster.addIndex_;
            std::vector<masterImportDat> allProcs;
            mpi::gather(*gridComm_, toMaster, allProcs, masterProc_);
            for(auto & proc : allProcs)
                if(proc.slaveProc_ != -1 && proc.slaveProc_ != masterProc_)
                    outFInfo_.push_back(std::make_shared<masterImportDat>(proc) );
        }
        else
        {
            mpi::gather(*gridComm_, toMaster, masterProc_);
        }
    }

    /**
     * @brief      Accessor function for fname_
     *
     * @return fname_
     */
    inline std::string &fname(){return fname_;}

    /**
     * @brief      Accessor function for loc_
     *
     * @return location of the flux dtector
     */
    inline std::vector<int> loc() {return loc_;}
    /**
     * @brief      Accessor function for sz_
     *
     * @return size of flux detector
     */
    inline std::vector<int> sz() {return sz_;}

    /**
     * @brief      Accessor function for timeInt_
     *
     * @return     The time interval
     */
    inline int & timeInt(){return timeInt_;}

    /**
     * @brief      Accessor function for pow_
     *
     * @return     True if outputting a power frequency
     */
    inline bool pow() {return pow_;}

    /**
     * @brief      Accessor function for outputMaps_
     *
     * @return     True if outputting fields as maps
     */
    inline bool outputMaps() {return outputMaps_;}

    /**
     * @brief      Accessor function to type_
     *
     * @return     type_
     */
    inline DTCTYPE type() {return type_;}

    /**
     * @brief take in the field information
     * @details takes in the electromagnetic filed information at each time step
     *
     * @param[in]  current simulation time
     *
     */
    void output(double& tt)
    {
        std::transform(freqList_.begin(), freqList_.end(), fftFact_.begin(), [&tt](double freq){return std::exp(cplx(0.0,-1.0*tt*freq) ); } );
        for(auto& field : gridsIn_)
            field->fieldIn(fftFact_.data() );
        ++t_step_;
    }

    /**
     * @brief calculates the flux at the end of the calculation
     * @details uses the stored field information to Fourier transform the fields
     */
    void collectFreqFields()
    {
        for(int vv = 0; vv < gridsIn_.size(); vv++)
        {
            // Copy grids into the right poistion
            if(gridComm_->rank() == masterProc_)
            {
                gridsIn_[vv]->toOutGrid();
                zcopy_(gridsIn_[vv]->outGrid()->size(), gridsIn_[vv]->outGrid()->data(), 1, &freqFields_[vv]->point(0, addIndex_), 1);
                for(auto& getFields : outFInfo_)
                {
                    // recv all other process's data and put it in the grid
                    std::vector<cplx> temp_store(nfreq_ * getFields->sz_,0.0);
                    gridComm_->recv(getFields->slaveProc_, gridComm_->cantorTagGen(getFields->slaveProc_, gridComm_->rank(), 1, 0), temp_store);
                    zcopy_(temp_store.size(), temp_store.data(), 1, &freqFields_[vv]->point(0, getFields->addIndex_), 1);
                }
            }
            else if(gridsIn_[vv]->outGrid())
            {
                // If not master proc but has some of the grid send data to master
                gridsIn_[vv]->toOutGrid();
                std::vector<cplx> to_send(gridsIn_[vv]->outGrid()->size(),0.0);
                zcopy_(to_send.size(), gridsIn_[vv]->outGrid()->data(), 1, to_send.data(), 1);
                gridComm_->send(masterProc_, gridComm_->cantorTagGen(gridComm_->rank(), masterProc_, 1, 0), to_send);
            }
        }
    }

    /**
     * @brief      does a numerical integration via the Simpson's rule
     *
     * @param      vec   The vector of all values to be integrated over
     * @param      d     the step size
     *
     * @return     The approximated integral value
     */
    cplx simps(std::vector<cplx>& vec, double& d)
    {
        cplx result(0.0,0.0);
        if(vec.size() % 2 == 0)
        {
            for(int ii = 0; ii < (vec.size()-2)/2; ii ++)
                result += d/6.0 * (vec[ii*2] + 4.0*vec[ii*2+1] + vec[(ii+1)*2]);

            for(int ii = 1; ii < (vec.size())/2; ii ++)
                result += d/6.0 * (vec[ii*2-1] + 4.0*vec[ii*2] + vec[ii*2+1]);

            result += d/4.0*(vec[0]+vec[1] + vec[vec.size()-1] + vec[vec.size()-2]);
        }
        else
        {
            for(int ii = 0; ii < (vec.size()-1)/2; ii ++)
                result += d/3.0 * (vec[ii*2] + 4.0*vec[ii*2+1] + vec[(ii+1)*2]);
        }
        return result;
    }

    std::vector<cplx_grid_ptr> fieldTranspose()
    {
        std::vector<cplx_grid_ptr> transFields = {};
        // find the total number of grid points collected
        int szFreq = std::accumulate( sz_.begin(), sz_.end(), 1, std::multiplies<int>() );
        for(auto& grid : freqFields_)
        {
            // Construct a field transpose to the collection grid
            transFields.push_back(std::make_shared<Grid<cplx>> ( std::array<int,3>( {{ szFreq, nfreq_, 1}}) , std::array<double,3>({{d_[0], dOmg_, 1}}) ) );
            // Take transpose the grid there is an mkl extension to do this, but I am neglecting since it is not in all bals implementations have it
            for(int yy = 0; yy < grid->y(); ++yy)
                zcopy_(grid->x(), &grid->point(0,yy,0), 1, &transFields.back()->point(yy,0,0), transFields.back()->x() );
        }
        return transFields;
    }

    /**
     * @brief      Takes the collected EM fields and outputs the Fourier transform
     */
    void toFile()
    {
        // Collect all frequency grids
        collectFreqFields();
        // if not master return
        if(gridComm_->rank() != masterProc_)
            return;
        int szFreq = std::accumulate( sz_.begin(), sz_.end(), 1, std::multiplies<int>() );
        std::vector<cplx_grid_ptr> transFields = fieldTranspose();

        std::ofstream f;
        f.open(fname_ );
        cplx freq(0.0,0.0);
        std::vector<cplx> pwr(szFreq, 0.0);
        f << "#" << std::setw(16) << "freq";
        for(int nn = 1; nn < transFields.size(); ++nn)
            f << "\tabs(field " << nn << ")\treal(field " << nn << ")\timag(field " << nn << ")";
        if(transFields.size() > 1)
            f << "\tabs(total)\treal(total)\timag(total)";
        f << std::endl;
        for(int ii=0; ii < nfreq_; ii++)
        {
            f << std::setw(16) <<  std::setprecision(12) << freqConv_ * freqList_[ii];
            // Calculate the Fourier transformed fields
            freq = 0;
            for(auto & grid : transFields)
            {
                cplx pt = toOutFile_(szFreq, &grid->point(0,ii), pwr.size(), pwr.data(), t_step_, convFactor_/2.0);
                f << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(pt) << "\t" << std::real(pt) << "\t" << std::imag(pt);
                freq += pt;
            }
            if(transFields.size() > 1)
                f << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(freq) << "\t" << std::real(freq) << "\t" << std::imag(freq) << std::endl;
            else
                f << "\n";
        }
        f.close();
    }
    /**
     * @brief   Takes the collected EM fields including the incident fields and outputs the Fourier transform
     *
     * @param[in]  incd  The incident fields
     * @param[in]  dt    time step of simulation
     */
    void toFile(std::vector<std::vector<cplx>> & incd, double dt)
    {
        // Collect all frequency grids
        collectFreqFields();
        // if not master return
        gridComm_->barrier();
        if(gridComm_->rank() != masterProc_)
            return;
        int szFreq = std::accumulate( sz_.begin(), sz_.end(), 1, std::multiplies<int>() );
        std::vector<cplx_grid_ptr> transFields = fieldTranspose();

        std::ofstream f;
        f.open(fname_ );
        f << "#" << std::setw(16) << "freq\tabs(incd)\treal(incd)\timag(incd)";
        for(int nn = 1; nn < transFields.size(); ++nn)
            f << "\tabs(field " << nn << ")\treal(field " << nn << ")\timag(field " << nn << ")";
        if(transFields.size() > 1)
            f << "\tabs(total)\treal(total)\timag(total)";
        f << std::endl;
        cplx freq(0.0,0.0);
        std::vector<cplx> pwr(szFreq, 0.0);
        for(int ii=0; ii < nfreq_; ii++)
        {
            // Calculate the incident field value
            cplx incd_field(0.0, 0.0);
            f << std::setw(16) <<  std::setprecision(12) << freqConv_ * freqList_[ii];
            for(auto& field : incd)
            {
                cplx pt(0.0,0.0);
                for(int tt = 0; tt < field.size(); tt+=2)
                    pt += getIncdField_(0.5*field[tt]) * std::exp(cplx(0.0,-1.0*freqList_[ii]*static_cast<double>(tt/2)*(dt/static_cast<double>(timeInt_))));
                for(int tt = 1; tt < field.size(); tt+=2)
                    pt += getIncdField_(0.5*field[tt]) * std::exp(cplx(0.0,-1.0*freqList_[ii]*static_cast<double>(tt/2)*(dt/static_cast<double>(timeInt_))));

                pt /= static_cast<double>(t_step_*timeInt_);
                if(pow_)
                    pt *= std::conj(pt);
                pt *= convFactor_/2.0;

                f << "\t" << std::setw(16) <<  std::setprecision(12) << std::real(pt) << "\t" << std::setw(16) <<  std::setprecision(12) << std::imag(pt) << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(pt);
                incd_field += pt;
            }
            if(incd.size() > 2)
                f << "\t" << std::setw(16) <<  std::setprecision(12) << std::real(incd_field) << "\t" << std::setw(16) <<  std::setprecision(12) << std::imag(incd_field) << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(incd_field);
            //Calculate the actual field value
            freq = 0;
            for(auto & grid : transFields)
            {
                cplx pt = toOutFile_(szFreq, &grid->point(0,ii), pwr.size(), pwr.data(), t_step_, convFactor_/2.0);
                f << "\t" << std::setw(16) <<  std::setprecision(12) << std::real(pt) << "\t" << std::setw(16) <<  std::setprecision(12) << std::imag(pt) << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(pt);
                freq += pt;
            }
            if(transFields.size() > 1)
                f << '\t' << std::setw(16) <<  std::setprecision(12) << std::real(freq) << "\t" << std::setw(16) <<  std::setprecision(12) << std::imag(freq) << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(freq) << std::endl;
            else
                f << "\n" ;
        }
        f.close();
    }

    /**
     * @brief      Takes the collected EM fields and outputs the Fourier transform
     */
    void toMap()
    {
        // Collect all frequency grids
        collectFreqFields();
        // if not master return
        if(gridComm_->rank() != masterProc_)
            return;
        std::vector<cplx_grid_ptr> transFields = fieldTranspose();

        for(int ii=0; ii < nfreq_; ii++)
        {
            std::ofstream f;
            std::string fileName = fname_ + "." + std::to_string(freqConv_ * freqList_[ii]);
            f.open(fileName );
            cplx freq(0.0,0.0);
            std::vector<cplx> pwr(1, 0.0);
            f << "#" << std::setw(16) << "\tx\ty\tz" ;
            for(int nn = 1; nn < transFields.size(); ++nn)
                f << "\tabs(field " << nn << ")\treal(field " << nn << ")\timag(field " << nn << ")";
            if(transFields.size() > 1)
                f << "\tabs(total)\treal(total)\timag(total)";
            f << std::endl;
            // Calculate the Fourier transformed fields
            freq = 0;
            for(int yy = 0; yy < sz_[1]; ++yy)
            {
                for(int zz = 0; zz < sz_[2]; ++zz)
                {
                    for(int xx = 0; xx < sz_[0]; ++xx)
                    {
                        int jj = xx + zz*sz_[0] + yy*sz_[0]*sz_[2];
                        f << std::setw(16) << xx << '\t' << yy << '\t' << zz;
                        for(auto & grid : transFields)
                        {
                            cplx pt = toOutFile_(1, &grid->point(jj,ii), pwr.size(), pwr.data(), t_step_, convFactor_/2.0);
                            f << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(pt) << "\t" << std::real(pt) << "\t" << std::imag(pt);
                            freq += pt;
                        }
                    }
                }
            }
            if(transFields.size() > 1)
                f << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(freq) << "\t" << std::real(freq) << "\t" << std::imag(freq) << std::endl;
            else
                f << "\n";
            f.close();
        }
    }
    /**
     * @brief   Takes the collected EM fields including the incident fields and outputs the Fourier transform
     *
     * @param[in]  incd  The incident fields
     * @param[in]  dt    time step of simulation
     */
    void toMap(std::vector<std::vector<cplx>> & incd, double dt)
    {
        // Collect all frequency grids
        collectFreqFields();
        // if not master return
        if(gridComm_->rank() != masterProc_)
            return;
        std::vector<cplx_grid_ptr> transFields = fieldTranspose();

        for(int ii=0; ii < nfreq_; ii++)
        {
            std::vector<cplx> pwr(1, 0.0);
            cplx freq(0.0,0.0);
            std::string fileName = fname_ + "." + std::to_string(freqConv_ * freqList_[ii]);
            std::ofstream f;
            f.open(fileName );
            f << "#" << std::setw(16) << "x\ty\tz\tabs(incd)\treal(incd)\timag(incd)";
            for(int nn = 1; nn < transFields.size(); ++nn)
                f << "\tabs(field " << nn << ")\treal(field " << nn << ")\timag(field " << nn << ")";
            if(transFields.size() > 1)
                f << "\tabs(total)\treal(total)\timag(total)";
            f << std::endl;
            // Calculate the incident field value
            cplx incd_field(0.0, 0.0);
            std::vector<double> outIncdToFile(0, 0.0);
            for(auto& field : incd)
            {
                cplx pt(0.0,0.0);
                for(int tt = 0; tt < field.size(); tt+=2)
                    pt += getIncdField_(0.5*field[tt]) * std::exp(cplx(0.0,-1.0*freqList_[ii]*static_cast<double>(tt/2)*(dt/static_cast<double>(timeInt_))));
                for(int tt = 1; tt < field.size(); tt+=2)
                    pt += getIncdField_(0.5*field[tt]) * std::exp(cplx(0.0,-1.0*freqList_[ii]*static_cast<double>(tt/2)*(dt/static_cast<double>(timeInt_))));

                pt /= static_cast<double>(t_step_*timeInt_);
                if(pow_)
                    pt *= std::conj(pt);
                pt *= convFactor_/2.0;

                outIncdToFile.push_back(std::real(pt));
                outIncdToFile.push_back(std::imag(pt));
                outIncdToFile.push_back(std::abs(pt));
                incd_field += pt;
            }
            if(incd.size() > 2)
            {
                outIncdToFile.push_back(std::real(incd_field));
                outIncdToFile.push_back(std::imag(incd_field));
                outIncdToFile.push_back(std::abs(incd_field));
            }
            for(int yy = 0; yy < sz_[1]; ++yy)
            {
                for(int zz = 0; zz < sz_[2]; ++zz)
                {
                    for(int xx = 0; xx < sz_[0]; ++xx)
                    {
                        f << std::setw(16) << xx << '\t' << yy << '\t' << zz;
                        for(auto& val : outIncdToFile)
                            f << "\t" << std::setw(16) <<  std::setprecision(12) << val;

                        //Calculate the actual field value
                        int jj = xx + zz*sz_[0] + yy*sz_[0]*sz_[2];
                        freq = 0;
                        for(auto & grid : transFields)
                        {
                            cplx pt = toOutFile_(1, &grid->point(jj,ii), pwr.size(), pwr.data(), t_step_, convFactor_/2.0);
                            f << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(pt) << "\t" << std::real(pt) << "\t" << std::imag(pt);
                            freq += pt;
                        }
                        if(transFields.size() > 1)
                            f << '\t' << std::setw(16) <<  std::setprecision(12) << std::real(freq) << "\t" << std::setw(16) <<  std::setprecision(12) << std::imag(freq) << "\t" << std::setw(16) <<  std::setprecision(12) << std::abs(freq) << std::endl;
                        else
                            f << "\n" ;
                    }
                }
            }
            f.close();
        }
    }
};

class parallelDetectorFREQReal : public parallelDetectorFREQ_Base<double>
{
public:
    /**
     * @brief      Constructs a frequency detector based on frequencies
     *
     * @param[in]  name      filename of the detector
     * @param[in]  grids     The grids used for the detector
     * @param[in]  loc       The location of the lower, left, back corner of the detector region
     * @param[in]  sz        size of the detector region
     * @param[in]  type      The type: output type of dtc
     * @param[in]  timeInt   Time interval for how often to record data
     * @param[in]  freqList  The frequency list
     * @param[in]  d         grid spacing in all directions
     * @param[in]  dt        time step of the calculation
     * @param[in]  SI        store data in SI units
     * @param[in]  I0        unit current
     * @param[in]  a         unit length
     */
    parallelDetectorFREQReal(std::string name, std::vector< std::pair<pgrid_ptr, std::array<int,3> > > grids, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, int timeInt, bool outputMaps, std::vector<double> freqList, std::array<double,3> d, double dt, bool SI, double I0, double a);
};

class parallelDetectorFREQCplx : public parallelDetectorFREQ_Base<cplx>
{
public:
    /**
     * @brief      Constructs a frequency detector based on frequencies
     *
     * @param[in]  name      filename of the detector
     * @param[in]  grids     The grids used for the detector
     * @param[in]  loc       The location of the lower, left, back corner of the detector region
     * @param[in]  sz        size of the detector region
     * @param[in]  type      The type: output type of dtc
     * @param[in]  timeInt   Time interval for how often to record data
     * @param[in]  freqList  The frequency list
     * @param[in]  d         grid spacing in all directions
     * @param[in]  dt        time step of the calculation
     * @param[in]  SI        store data in SI units
     * @param[in]  I0        unit current
     * @param[in]  a         unit length
     */
    parallelDetectorFREQCplx(std::string name, std::vector< std::pair<pgrid_ptr, std::array<int,3> > > grids, std::array<int,3> loc, std::array<int,3> sz, DTCTYPE type, int timeInt, bool outputMaps, std::vector<double> freqList, std::array<double,3> d, double dt, bool SI, double I0, double a);
};

#endif