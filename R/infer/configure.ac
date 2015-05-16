# Configure.ac to produce configure file by autoconf
#
# Author: Chris Jewell <c.jewell@lancaster.ac.uk> (c) 2015
# License: GPL(>=3.0)
#
# With help from gputools automake structure.

AC_INIT([DESCRIPTION],[VERSION],[c.jewell@lancaster.ac.uk])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_MACRO_DIR([m4])
AC_CONFIG_AUX_DIR([tools])
AM_INIT_AUTOMAKE

# Get C++ compiler
AC_PROG_CXX
AC_PROG_CC
AC_LANG([C++])

# Host OS
AC_CANONICAL_HOST

# Don't touch user flags!
USR_CPPFLAGS=${CPPFLAGS}
USR_LDFLAGS=${LDFLAGS}

##################
# R requirements #
##################
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
  echo "could not determine R_HOME"
  exit 1
fi
CXX=`"${R_HOME}/bin/R" CMD config CXX`
CPP=`"${R_HOME}/bin/R" CMD config CPP`
R_CXXFLAGS=`"${R_HOME}/bin/R" CMD config CXXFLAGS`
R_CXXPICFLAGS=`"${R_HOME}/bin/R" CMD config CXXPICFLAGS`
R_CPPFLAGS=`"${R_HOME}/bin/R" CMD config --cppflags`
R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`


######################
# CUDA related stuff #
######################
SAVED_LDFLAGS=${LDFLAGS}

AX_CHECK_CUDA

CUDA_VERSION=`${NVCC} --version | awk -F, 'END{print $3}'`
AC_MSG_NOTICE([Using CUDA version ${CUDA_VERSION}])

LDFLAGS=${CUDA_LDFLAGS}
AC_CHECK_LIB([cublas],[cublasGetVersion],,[AC_MSG_ERROR([libcublas not found])])
AC_CHECK_LIB([curand],[curandCreateGenerator],,[AC_MSG_ERROR([libcurand not found])])
AC_CHECK_LIB([cusparse],[cusparseCreate],,[AC_MSG_ERROR([libcusparse not found])])
LDFLAGS="${CUDA_LDFLAGS} ${SAVED_LDFLAGS}"

# CUDPP
AC_CHECK_HEADER([cudpp.h],,[AC_MSG_ERROR([cudpp.h not found])])
AC_CHECK_LIB([cudpp],[cudppReduce],,[AC_MSG_ERROR([libcudpp not found])])

# Cache and reset LIBS
CUDA_LIBS=${LIBS}
LIBS=""
LDFLAGS=${USR_LDFLAGS}


#########
# Boost #
#########
AC_CHECK_HEADER([boost/numeric/ublas/matrix_sparse.hpp],,[AC_MSG_ERROR([Boost headers not found])])

#######
# GSL #
#######
AC_CHECK_HEADER([gsl/gsl_randist.h],,[AC_MSG_ERROR([GSL headers not found.])])
AC_CHECK_LIB([m],[cos],,[AC_MSG_ERROR([libm not found])])
AC_CHECK_LIB([gslcblas],[cblas_dgemm],,[AC_MSG_ERROR([libgslcblas not found. Either set LDFLAGS or install GSL])])
AC_CHECK_LIB([gsl],[gsl_blas_dgemm],,[AC_MSG_ERROR([libgsl not found.  Either set LDFLAGS or install GSL])])

# Cache and reset LIBS
GSL_LIBS=${LIBS}
LIBS=""

# HDF5
AC_CHECK_HEADER([H5Cpp.h],,[AC_MSG_ERROR([H5Cpp.h not found])])
AC_CHECK_HEADER([H5PacketTable.h],,[AC_MSG_ERROR([H5Packetable.h not found])])
AC_CHECK_LIB([hdf5],[H5get_libversion],,[AC_MSG_ERROR([libhdf5 not found])])
AC_CHECK_LIB([hdf5_hl],[H5PTopen],,[AC_MSG_ERROR([libhdf5_hl not found])])

# Following C++ test from
# http://nerdland.net/2009/07/detecting-c-libraries-with-autotools/
AC_MSG_CHECKING([for PacketTable in -lhdf5_hl_cpp])
AC_LANG(C++)
SAVED_LDFLAGS=${LDFLAGS}
LDFLAGS="${LDFLAGS} -lhdf5_hl_cpp -lhdf5_cpp"
AC_LINK_IFELSE(
	[AC_LANG_PROGRAM([#include <H5Cpp.h>]
	                 [#include <H5PacketTable.h>],
			 [PacketTable p])],
	[LIBS="-lhdf5_cpp -lhdf5_hl_cpp ${LIBS}"]
	[HAVE_HDF5_CPP=1]
	[HAVE_HDF5_HL_CPP=1]
	[AC_MSG_RESULT([yes])],
	[AC_MSG_RESULT([no])]
	[AC_MSG_ERROR([HDF5 C++ high level interface not found])])
LDFLAGS=${SAVED_LDFLAGS}

HDF5_LIBS=${LIBS}
LIBS=""

LDFLAGS=${USR_LDFLAGS}

# Rcpp library requirements - not required since R 3.1
RCPP_PKG_LIBS=`${R_HOME}/bin/Rscript -e "Rcpp:::LdFlags()"`
RCPP_PKG_INCLUDE=`${R_HOME}/bin/Rscript -e "Rcpp:::CxxFlags()"`

# Libraries
PKG_CPPFLAGS="${CPPFLAGS} ${CUDA_CPPFLAGS} ${R_CPPFLAGS} ${RCPP_PKG_INCLUDE} -DNDEBUG"
PKG_CXXFLAGS="${PKG_CXXFLAGS} ${R_CXXFLAGS} ${R_CXXPICFLAGS}"
PKG_CXXFLAGS=`sed 's/,/\\\\,/g' <<< ${PKG_CXXFLAGS}` # Escape commas to stop them being turned into spaces by nvcc when passed to GCC
AC_MSG_NOTICE([CXXFLAGS: $PKG_CXXFLAGS])
PKG_LDFLAGS="${CUDA_LDFLAGS} ${LDFLAGS} ${R_LIB}"

# Rpath -- break LDFLAGS into tokens, get unique list,
#   find tokens beginning with -L, strip the -L and
#   add to PKG_RPATH
PKG_RPATH="-Wl"
for i in `tr " " "\n" <<< ${PKG_LDFLAGS} | sort -u`; do
    path=`sed -ne '/^\-L/p' <<< ${i} | sed 's/^\-L//'`
    if test ${path}; then
      PKG_RPATH="${PKG_RPATH},-rpath,${path}"
    fi
done
AC_MSG_NOTICE([RPath: ${PKG_RPATH}])


AC_SUBST([RCPP_PKG_LIBS])
AC_SUBST([CUDA_LIBS])
AC_SUBST([GSL_LIBS])
AC_SUBST([HDF5_LIBS])
AC_SUBST([PKG_CPPFLAGS])
AC_SUBST([PKG_CXXFLAGS])
AC_SUBST([PKG_LDFLAGS])
AC_SUBST([PKG_RPATH])
AC_SUBST([NVCC])
AC_SUBST([NVCC_OPTS])

# Generate Makefiles
AC_CONFIG_FILES([Makefile src/Makefile src/interface/Makefile src/data/Makefile src/sim/Makefile src/mcmc/Makefile src/Framework/Makefile])
AC_OUTPUT
