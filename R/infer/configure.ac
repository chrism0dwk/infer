# Configure.ac to produce configure file by autoconf
#
# Author: Chris Jewell <chris.jewell@warwick.ac.uk>
# License: GPL(>=2.0)
#

AC_INIT([DESCRIPTION],[VERSION],[chris.jewell@warwick.ac.uk])

# Config dir
AC_CONFIG_SRCDIR([src])

# Get C++ compiler
AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_CC
AC_LANG([C++])

CC=`${R_HOME}/bin/R CMD config CPP`
CXXFLAGS=`${R_HOME}/bin/R CMD config CXXFLAGS`
CFLAGS=`${R_HOME}/bin/R CMD config CFLAGS`
FLIBS=`${R_HOME}/bin/R CMD config FLIBS`


# Check headers
AC_CHECK_HEADERS([math.h])
AC_CHECK_HEADERS([H5Cpp.h],,[AC_MSG_ERROR([HDF5 library headers not found])])

# Check and add Rcpp
RCPP_PKG_LIBS=`${R_HOME}/bin/Rscript -e "Rcpp:::LdFlags()"`

AC_SUBST([OPENMP_CXXFLAGS])
AC_SUBST([RCPP_PKG_LIBS])

# Generate Makefiles
AC_OUTPUT(src/Makevars)
