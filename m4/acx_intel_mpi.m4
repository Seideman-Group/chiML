AC_DEFUN([ACX_INTEL_MPI], [
  AC_ARG_VAR(MPICC,[MPI C compiler command])
  AC_CHECK_PROGS(MPICC, mpiicc, $CC)
  acx_mpi_save_CC="$CC"
  CC="$MPICC"
  AC_SUBST(MPICC)

  AC_ARG_VAR(MPICXX,[MPI C++ compiler command])
  AC_CHECK_PROGS(MPICXX, mpiicpc, $CXX)
  acx_mpi_save_CXX="$CXX"
  CXX="$MPICXX"
  AC_SUBST(MPICXX)
])
