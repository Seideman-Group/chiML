// #include <iostream>
// #include "FDTDField.hpp"
// #include "FDTDFieldTE.hpp"
// #include "FDTDFieldTM.hpp"

#include <FDTD_MANAGER/parallelFDTDField.hpp>
// #include <iomanip>

namespace mpi = boost::mpi;

int main(int argc, char const *argv[])
{
    // Initialize the boost mpi environment and communicator
    mpi::environment env;
    std::shared_ptr<mpiInterface> gridComm = std::make_shared<mpiInterface>();
    std::clock_t start;
    std::string filename;
    double duration= 0.0;
    if (argc < 2)
    {
        std::cout << "Provide an input json file" << std::endl;
        exit(1);
    }
    else
    {
        // Take in the file name and strip out all comments for the parser
        filename = argv[1];
        if(gridComm->rank() == 0)
            stripComments(filename);
        else
            filename = "stripped_" + filename;
    }
    gridComm->barrier();

    if(gridComm->rank() == 0)
        std::cout << "Reading input file " << argv[1] << "..." << std::endl;
    //construct the parser and pass it to the inputs
    boost::property_tree::ptree propTree;
    boost::property_tree::json_parser::read_json(filename,propTree);
    parallelProgramInputs IP(propTree, filename);
    gridComm->barrier();
    if(gridComm->rank() == 0)
         boost::filesystem::remove(filename) ;
    if(gridComm->rank() == 0)
        std::cout << "I TOOK ALL THE INPUT PARAMETERS" << std::endl;

    double maxT = IP.tMax();
    if(!IP.cplxFields_)
    {
        parallelFDTDFieldReal FF(IP, gridComm);
        if(gridComm->rank() == 0)
            std::cout << "made with real fields" << std::endl;

        int nSteps = int(std::ceil( IP.tMax_ / (IP.dt_) ) );
        start = std::clock();
        for(int tt = 0; tt < nSteps; ++tt )
            FF.step();

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        for(int ii = 0; ii < gridComm->size(); ii ++)
        {
            gridComm->barrier();
            if(gridComm->rank() == ii)
                std::cout << gridComm->rank() << "\t" << duration<<std::endl;
        }
        // Output the final flux for all flux objects, Polarization terms are because of units for continuous boxes defined by the z component (TE off grid is E fields, TM H fields)
        for(auto & flux : FF.fluxArr() )
        {
            // flux->getFlux(FF.EIncd(), FF.HIncd(), FF.H_mnIncd());
            flux->getFlux(FF.ExIncd(), FF.EyIncd(), FF.EzIncd(), FF.HxIncd(), FF.HyIncd(), FF.HzIncd(), true);
        }

        // Power outputs need incident fields for normalization
        for(auto & dtc : FF.dtcFreqArr())
        {
            try
            {
                std::vector<std::vector<cplx>> incdFields;
                if(dtc->type() == DTCTYPE::HPOW)
                    incdFields = { FF.HxIncd(), FF.HyIncd(), FF.HzIncd() };
                else if(dtc->type() == DTCTYPE::EPOW)
                    incdFields = { FF.ExIncd(), FF.EyIncd(), FF.EzIncd() };
                else if(dtc->type() == DTCTYPE::EX || dtc->type() == DTCTYPE::PX)
                    incdFields = { FF.ExIncd() };
                else if(dtc->type() == DTCTYPE::EY || dtc->type() == DTCTYPE::PY)
                    incdFields = { FF.EyIncd() };
                else if(dtc->type() == DTCTYPE::EZ || dtc->type() == DTCTYPE::PZ)
                    incdFields = { FF.EzIncd() };
                else if(dtc->type() == DTCTYPE::HX || dtc->type() == DTCTYPE::MX)
                    incdFields = { FF.HxIncd() };
                else if(dtc->type() == DTCTYPE::HY || dtc->type() == DTCTYPE::MY)
                    incdFields = { FF.HyIncd() };
                else if(dtc->type() == DTCTYPE::HZ || dtc->type() == DTCTYPE::MZ)
                    incdFields = { FF.HzIncd() };
                if(dtc->outputMaps())
                    dtc->toMap(incdFields, FF.dt());
                else
                    dtc->toFile(incdFields, FF.dt());
            }
            catch(std::exception& e)
            {
                if(dtc->outputMaps())
                    dtc->toMap();
                else
                    dtc->toFile();
            }
        }

        for(auto& dtc : FF.dtcArr())
            dtc->toFile();

        for(auto& qe : FF.qeArr())
        {
            if( qe->pAccuulate() )
                qe->outputPol();
            for(auto& dtcPop: qe->dtcPopArr() )
                dtcPop->toFile();
        }

        // output scaling information
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        for(int ii = 0; ii < gridComm->size(); ii ++)
        {
            gridComm->barrier();
            if(gridComm->rank() == ii)
                std::cout << gridComm->rank() << "\t" << duration<<std::endl;
        }
    }
    else
    {
        parallelFDTDFieldCplx FF(IP, gridComm);
        if(gridComm->rank() == 0)
            std::cout << "made with complex fields" << std::endl;

        int nSteps = int(std::ceil( IP.tMax_ / (IP.dt_) ) );
        start = std::clock();
        for(int tt = 0; tt < nSteps; ++tt )
            FF.step();

        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        for(int ii = 0; ii < gridComm->size(); ii ++)
        {
            gridComm->barrier();
            if(gridComm->rank() == ii)
                std::cout << gridComm->rank() << "\t" << duration<<std::endl;
        }
        // Output the final flux for all flux objects, Polarization terms are because of units for continuous boxes defined by the z component (TE off grid is E fields, TM H fields)
        for(auto & flux : FF.fluxArr() )
        {
            flux->getFlux(FF.ExIncd(), FF.EyIncd(), FF.EzIncd(), FF.HxIncd(), FF.HyIncd(), FF.HzIncd(), true);
        }
        // Power outputs need incident fields for normalization
        for(auto & dtc : FF.dtcFreqArr())
        {
            try
            {
                std::vector<std::vector<cplx>> incdFields;
                if(dtc->type() == DTCTYPE::HPOW)
                    incdFields = { FF.HxIncd(), FF.HyIncd(), FF.HzIncd() };
                else if(dtc->type() == DTCTYPE::EPOW)
                    incdFields = { FF.ExIncd(), FF.EyIncd(), FF.EzIncd() };
                else if(dtc->type() == DTCTYPE::EX || dtc->type() == DTCTYPE::PX)
                    incdFields = { FF.ExIncd() };
                else if(dtc->type() == DTCTYPE::EY || dtc->type() == DTCTYPE::PY)
                    incdFields = { FF.EyIncd() };
                else if(dtc->type() == DTCTYPE::EZ || dtc->type() == DTCTYPE::PZ)
                    incdFields = { FF.EzIncd() };
                else if(dtc->type() == DTCTYPE::HX || dtc->type() == DTCTYPE::MX)
                    incdFields = { FF.HxIncd() };
                else if(dtc->type() == DTCTYPE::HY || dtc->type() == DTCTYPE::MY)
                    incdFields = { FF.HyIncd() };
                else if(dtc->type() == DTCTYPE::HZ || dtc->type() == DTCTYPE::MZ)
                    incdFields = { FF.HzIncd() };
                if(dtc->outputMaps())
                    dtc->toMap(incdFields, FF.dt());
                else
                    dtc->toFile(incdFields, FF.dt());
            }
            catch(std::exception& e)
            {
                if(dtc->outputMaps())
                    dtc->toMap();
                else
                    dtc->toFile();
            }
        }
        for(auto& dtc : FF.dtcArr())
            dtc->toFile();

        for(auto& qe : FF.qeArr())
        {
            if( qe->pAccuulate() )
                qe->outputPol();
            for(auto& dtcPop: qe->dtcPopArr() )
                dtcPop->toFile();
        }
        // output scaling information
        duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
        for(int ii = 0; ii < gridComm->size(); ii ++)
        {
            gridComm->barrier();
            if(gridComm->rank() == ii)
                std::cout << gridComm->rank() << "\t" << duration<<std::endl;
        }
    }
    return 0;
}
