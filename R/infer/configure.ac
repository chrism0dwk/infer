# Configure.ac to produce configure file by autoconf
#
# Author: Chris Jewell <c.p.jewell@massey.ac.nz> (c) 2013
# License: GPL(>=3.0)
#
# With help from gputools automake structure.

AC_INIT([DESCRIPTION],[VERSION],[c.p.jewell@massey.ac.nz])
AC_CONFIG_SRCDIR([src])
AC_CONFIG_AUX_DIR([tools])
LT_INIT
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE

# Get C++ compiler
AC_PROG_CXXCPP
AC_PROG_CXX
AC_PROG_CC
AC_LANG([C++])


AC_MSG_NOTICE([checking for critical tools used by configure])

# Compiler directories
AC_ARG_WITH([compiler-bindir],
        AC_HELP_STRING([--with-compiler-bindir=compiler_bindir],
                [specify the directory containing gcc, e.g. /usr/bin]),
        [COMPILER_BINDIR="$withval"])

AC_ARG_WITH([cuda-home],
        AC_HELP_STRING([--with-cuda-home=cuda_home],
                [specify the root of your cuda install, e.g. /usr/local/cuda]),
        [CUDA_HOME="$withval"])

AC_ARG_WITH([cudpp-home],
	AC_HELP_STRING([--with-cudpp-home=cudpp_home],
		[specify the root of your cudpp install, e.g. /usr/local/cudpp]),
	[CUDPP_HOME="$withval"])

AC_ARG_WITH([r-include],
        AC_HELP_STRING([--with-r-include=r_include],
                [specify the full path to your R header files, e.g. /usr/share/R/include]),
        [R_INCLUDE="$withval"])

AC_ARG_WITH([r-lib],
        AC_HELP_STRING([--with-r-lib=r-lib],
                [specify the full path to your R shared libraries, e.g. /usr/lib64/R/lib]),
        [R_LIB="$withval"])

AC_ARG_WITH([boost-home],
	AC_HELP_STRING([--with-boost-home=boost_home],
		[specify the root of your boost install, e.g. /usr/local/boost]),
	[BOOST_HOME="$withval"])

AC_ARG_WITH([gsl-home],
	AC_HELP_STRING([--with-gsl-home],
		[specify the root of your GSL install, e.g. /usr/local/gsl]),
	[GSL_HOME="$withval"])

AC_ARG_WITH([hdf5-home],
	AC_HELP_STRING([--with-hdf5-home],
		[specify the root of your HDF5 install, e.g. /usr/local/hdf5]),
	[HDF5_HOME="$withval"])

# Check directory locations
AC_MSG_CHECKING("environment variable COMPILER_BINDIR")
if test -z "${COMPILER_BINDIR}"; then
        AC_MSG_RESULT("COMPILER_BINDIR not set; using user's default path")
else
        AC_MSG_RESULT("using COMPILER_BINDIR=${COMPILER_BINDIR}")
        AC_CHECK_FILE([${COMPILER_BINDIR}/gcc], [HAS_GCC="yes"])
        if test -z "${HAS_GCC}"; then
                AC_MSG_ERROR("cannot find gcc; check that ${COMPILER_BINDIR} is the correct location; using user's default path")
                MISC_PARAMS=""
        else
                AC_MSG_RESULT("gcc found in ${COMPILER_BINDIR}")
                MISC_PARAMS="--compiler-bindir ${COMPILER_BINDIR}"
        fi
fi

AC_MSG_CHECKING("environment variable CUDA_HOME")
if test -z "${CUDA_HOME}"; then
        AC_MSG_RESULT("CUDA_HOME not set; using default /usr/local/cuda")
        CUDA_HOME="/usr/local/cuda"
else
        AC_MSG_RESULT("using CUDA_HOME=${CUDA_HOME}")
fi

AC_MSG_CHECKING("environment variable CUDPP_HOME")
if test -z "${CUDPP_HOME}"; then
   	AC_MSG_RESULT("CUDPP_HOME not set; using default /usr/local/cudpp")
	CUDPP_HOME="/usr/local/cudpp"
else
	AC_MSG_RESULT("using CUDPP_HOME=${CUDPP_HOME}")
fi

AC_CHECK_FILE([${CUDA_HOME}/bin/nvcc], [HAS_NVCC="yes"])
if test -z "${HAS_NVCC}"; then
        AC_MSG_ERROR("cannot find nvcc; check CUDA install")
else
        AC_MSG_RESULT("nvcc found")
fi

AC_MSG_CHECKING("whether this is the 64 bit linux version of CUDA")
AC_CHECK_FILE([${CUDA_HOME}/lib64/libcublas.so], [BIT64="yes"])
if test -z "${BIT64}"; then
        AC_MSG_RESULT("no -- using ${CUDA_HOME}/lib for CUDA libs")
        CUDA_LIB_DIR="/lib"
else
        AC_MSG_RESULT("yes -- using ${CUDA_HOME}/lib64 for CUDA libs")
        CUDA_LIB_DIR="/lib64"
fi

AC_MSG_CHECKING("environment variable BOOST_HOME")
if test -z "${BOOST_HOME}"; then
   	AC_MSG_RESULT("BOOST_HOME not set; using default /usr")
	BOOST_HOME="/usr"
else
	AC_MSG_RESULT("using BOOST_HOME=${BOOST_HOME}")
fi

AC_MSG_CHECKING("environment variable GSL_HOME")
if test -z "${GSL_HOME}"; then
   	AC_MSG_RESULT("GSL_HOME not set; using default /usr")
	GSL_HOME="/usr"
else
	AC_MSG_RESULT("using GSL_HOME=${GSL_HOME}")
fi

AC_MSG_CHECKING("environment variable HDF5_HOME")
if test -z "${HDF5_HOME}"; then
   	AC_MSG_RESULT("HDF5_HOME not set; using default /usr")
	HDF5_HOME="/usr"
else
	AC_MSG_RESULT("using HDF5_HOME=${HDF5_HOME}")
fi

: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
        AC_MSG_ERROR("could not determine R_HOME")
else
        CPICFLAGS="`"${R_HOME}/bin/R" CMD config CPICFLAGS` -g"
###	CPICFLAGS="${CPICFLAGS} `${R_HOME}/bin/R CMD config CXXFLAGS`"
        if test -z "${R_INCLUDE}"; then
                R_INCLUDE=`"${R_HOME}/bin/R" CMD config --cppflags`
        else
                R_INCLUDE="-I${R_INCLUDE}"
        fi
        if test -z "${R_LIB}"; then
                R_LIB=`"${R_HOME}/bin/R" CMD config --ldflags`
        else
                R_LIB="-L${R_LIB} -lR"
        fi
        AC_MSG_RESULT("using ${R_INCLUDE} for R header files")
        AC_MSG_RESULT("using ${R_LIB} for R shared libraries")
fi


# Set platform-dependent constants
DEVICEOPTS="-O3 -use_fast_math"

AC_MSG_NOTICE([determining host platform])
arch=`uname -m`
osrelease=`uname -r`
osversion=`uname -v`

RPATHFLAG=""
AC_CANONICAL_HOST
case "${host_os}" in
        *darwin*)
                R_LIB=""
		RPATHFLAG="-rpath,${CUDA_HOME}${CUDA_LIB_DIR}"
		R_LIB=""
                AC_PATH_PROG([SW_VERS],[sw_vers],[no],[$PATH])
                if test "$arch" = "i386"; then
                        $SW_VERS -productVersion | grep '^10.6' > /dev/null 2>&1
                        if test "x$?" = "x0" ; then
                                DEVICEOPTS="-m64"
                        fi
                elif test "$arch" = "x86_64"; then
                        DEVICEOPTS="-m64"
                else
                        AC_MSG_WARN([Unrecognised Darwin Architecture])
                fi
        ;;
	*linux*)
		RPATHFLAG="-Wl,-rpath=${CUDA_HOME}${CUDA_LIB_DIR}:${CUDPP_HOME}/lib"
	;;
esac


# Check headers
AC_CHECK_HEADERS([math.h])
AC_CHECK_HEADERS([H5Cpp.h],,[AC_MSG_ERROR([HDF5 library headers not found])])

# Assemble library requirements
RCPP_PKG_LIBS=`${R_HOME}/bin/Rscript -e "Rcpp:::LdFlags()"`
RCPP_PKG_INCLUDE=`${R_HOME}/bin/Rscript -e "Rcpp:::CxxFlags()"`
HDF5_LIBS="-lhdf5 -lhdf5_cpp -lhdf5_hl -lhdf5_hl_cpp"

AC_SUBST([RCPP_PKG_LIBS])
AC_SUBST([RCPP_PKG_INCLUDE])
AC_SUBST([HDF5_LIBS])
AC_SUBST([LIBS])
AC_SUBST([CUDA_HOME])
AC_SUBST([CUDA_LIB_DIR])
AC_SUBST([CUDPP_HOME])
AC_SUBST([CUDPP_LIBS])
AC_SUBST([DEVICEOPTS])
AC_SUBST([BOOST_HOME])
AC_SUBST([GSL_HOME])
AC_SUBST([HDF5_HOME])
AC_SUBST([R_INCLUDE])
AC_SUBST([R_LIB])
AC_SUBST([RPATHFLAG])
AC_SUBST([CPICFLAGS])
# Generate Makefiles
AC_CONFIG_FILES([src/Makefile src/interface/Makefile src/data/Makefile src/sim/Makefile src/mcmc/Makefile src/Framework/Makefile])
AC_OUTPUT
