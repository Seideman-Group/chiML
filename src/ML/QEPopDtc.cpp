/** @file ML/QEPopDtc.cpp
 *  @brief Class used to output the average population in a quantum emitter state
 *
 *  Class that collects the quantum emitter density matrix for all processes, averages them
 *  and outputs it to a text file.
 *
 *  @author Thomas A. Purcell (tpurcell90)
 *  @bug No known bugs.
 */
#include "QEPopDtc.hpp"

QEPopDtc::QEPopDtc(std::shared_ptr<mpiInterface> gridComm, int level, int nlevel, int npoints, std::vector<double> eWeights, std::string outFile, int timeInt, int nt, double dt):
    gridComm_(gridComm),
    level_(level),
    nlevel_(nlevel),
    npoints_(npoints),
    t_step_(0),
    timeInt_(timeInt),
    tcur_(0.0),
    dt_(dt),
    curPop_(0.0),
    eWeights_(eWeights),
    outFile_(outFile)
{
    allPop_.reserve(nt);
}

void QEPopDtc::accumPop()
{
    if(t_step_ % timeInt_ == 0)
        allPop_.push_back(curPop_/static_cast<double>(npoints_) );
    curPop_ = 0.0;
    tcur_ += dt_;
    ++t_step_;
}

void QEPopDtc::toFile()
{
    int gatherProc = 0;
    if(gridComm_->rank() == gatherProc)
    {
        std::vector<std::vector<cplx>> allPops;
        // mpi::gather(*gridComm_, totPz_, allPz, outputPAccuProc_);
        mpi::gather(*gridComm_, allPop_, allPops, gatherProc);
        std::ofstream outFileStream;
        outFileStream.open(outFile_);
        for(int tt = 0; tt < allPop_.size(); ++tt)
        {
            cplx totalPop = 0.0;
            for(int pp = 0; pp < allPops.size(); ++pp)
                totalPop += allPops[pp][tt];
            outFileStream << std::setw(9) << std::setprecision(9) << tt*timeInt_*dt_ << "\t" << std::setw(16) << std::setprecision(16) << std::real(totalPop) << "\t" << std::setw(16) << std::setprecision(16) << std::imag(totalPop) << "\t" << std::setw(16) << std::setprecision(16) << std::abs(totalPop) << std::endl;
        }
        outFileStream.close();
    }
    else
    {
        mpi::gather(*gridComm_, allPop_, 0);
    }
    // std::cout << outFile_ << std::endl;
}
