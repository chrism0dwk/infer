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
AC_MSG_CHECKING([mpi])
AX_PROG_CXX_MPI( , ,
				[AC_MSG_ERROR([MPI compiler not found])])
				
# Check which math library to use
AC_ARG_WITH(mathlib, [AS_HELP_STRING([--with-mathlib],
					 [the maths library to use. Default: -lm])
					 ],,[with_mathlib=-lm])

MATHLIB=$with_mathlib

# Checks for Boost Base
AX_BOOST_BASE([1.47.0], , [AC_MSG_ERROR([Boost library not found])])

# Checks for header files.
AC_CHECK_HEADER([gsl/gsl_version.h], ,
		[AC_MSG_ERROR([GNU Scientific Library headers not found.  You may need to supply -I<path to gsl headers> to CXXFLAGS.])])


# Checks for libraries.
AC_CHECK_LIB(gsl, gsl_ran_gaussian,
	     [GSL_LIBS="-lgsl -lgslcblas"],
	     [AC_MSG_ERROR([GNU Scientific Library not found.  You may need to supply -L<path to GSL libs> to LDFLAGS.])],
	     [-lgsl -lgslcblas])



# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# Set up MPI build
#AX_MPI
#CC="$MPICC"
#CXX="$MPICXX"

# Substitutions
AC_SUBST([OPENMP_CXXFLAGS])
AC_SUBST([GSL_LIBS])
AC_SUBST([MATHLIB])
AC_OUTPUT(Makefile src/Makefile src/Framework/Makefile src/data/Makefile src/diags/Makefile src/mcmc/Makefile src/sim/Makefile src/unitTests/Makefile)
