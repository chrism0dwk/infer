#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

VERSION = "0.5-1alpha"

AC_PREREQ([2.50])
AC_INIT(brandenburg, 0.5-1alpha, Chris Jewell <chris.jewell@warwick.ac.uk>)

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE
AM_PROG_LIBTOOL

AC_LANG([C++])

# Checks for MPI compiler -- this requires MPI for the entire package (only the MCMC alg actually requires it)
AC_CHECK_TOOLS([CXX], [mpic++ mpicxx mpiCC sxmpic++ hcp mpxlC_r mpxlC mpixlcxx_r mpixlcxx mpg++ mpc++ mpCC cmpic++ mpiFCC CCicpc pgCC pathCC sxc++ xlC_r xlC bgxlC_r bgxlC openCC sunCC crayCC g++ c++ gpp aCC CC cxx cc++ cl.exe FCC KCC RCC])
AX_PROG_CXX_MPI( , ,
				[AC_MSG_ERROR([MPI compiler not found])])
				
# Check which math library to use
AC_ARG_WITH(mathlib, [AS_HELP_STRING([--with-mathlib],
					 [the maths library to use. Default: -lm])
					 ],,[with_mathlib=-lm])

MATHLIB=$with_mathlib

# Checks for Boost Base
AX_BOOST_BASE([1.47.0], , [AC_MSG_ERROR([Boost library not found])])
AX_BOOST_SERIALIZATION

CPPFLAGS_SAVED=$CPPFLAGS
CPPFLAGS="$CPPFLAGS $BOOST_CPPFLAGS"
LDFLAGS_SAVED=$LDFLAGS
LDFLAGS="$BOOST_LDFLAGS $LDFLAGS"
AC_CHECK_LIB([boost_mpi], [MPI_Allreduce],
			 [BOOST_MPI_LIB="-lboost_mpi"],
			 [AC_MSG_ERROR([Boost::mpi not found, BOOST_LDFLAGS $BOOST_LDFLAGS])],[-lboost_serialization])
CPPFLAGS="$CPPFLAGS_SAVED"
LDFLAGS="$LDFLAGS_SAVED"



# GSL
AC_ARG_WITH(gsl, [AS_HELP_STRING([--with-gsl],
					 [path to GSL root.])
					 ],,)

GSL_LDFLAGS="-L$with_gsl/lib"
GSL_CPPFLAGS="-I$with_gsl/include"

CPPFLAGS_SAVED=$CPPFLAGS
LDFLAGS_SAVED=$LDFLAGS				
CPPFLAGS="$CPPFLAGS $GSL_CPPFLAGS"
LDFLAGS="$LDFLAGS $GSL_LDFLAGS"
# Checks for header files.
AC_CHECK_HEADER([gsl/gsl_version.h], ,
		[AC_MSG_ERROR([GNU Scientific Library headers not found.  You may need to supply -I<path to gsl headers> to CXXFLAGS.])])

# Checks for libraries.
AC_CHECK_LIB(gsl, gsl_ran_gaussian,
	     [GSL_LIBS="$GSL_LDFLAGS -lgsl -lgslcblas"],
	     [AC_MSG_ERROR([GNU Scientific Library not found.  You may need to supply -L<path to GSL libs> to LDFLAGS.])],
	     [-lgslcblas])

CPPFLAGS="$CPPFLAGS_SAVED"
LDFLAGS="$LDFLAGS_SAVED"

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# Set up MPI build
#AX_MPI
#CC="$MPICC"
#CXX="$MPICXX"

# Substitutions
AC_SUBST([GSL_CPPFLAGS])
AC_SUBST([GSL_LIBS])
AC_SUBST([MATHLIB])
AC_SUBST([BOOST_MPI_LIB])

AC_OUTPUT(Makefile src/Makefile src/Framework/Makefile src/data/Makefile src/diags/Makefile src/mcmc/Makefile src/sim/Makefile src/unitTests/Makefile)
