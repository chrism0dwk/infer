#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

#AC_PREREQ([2.61])
AC_INIT(brandenburg, 0.1, Chris Jewell <chris.jewell@warwick.ac.uk>)

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE(brandenburg, 0.1)

# Checks for programs.
AC_PROG_CXXCPP
AC_PROG_CXX

AC_LANG([C++])
AM_PROG_LIBTOOL



# Checks for libraries.
AC_CHECK_LIB(gsl, gsl_ran_gaussian,
	     [GSL_LIBS="${LIBS} -lgsl -lgslcblas"],
	     [AC_MSG_ERROR([GNU Scientific Library not found.  You may need to supply -L<path to GSL libs> to LDFLAGS.])],
	     [-lgsl -lgslcblas])


# Checks for header files.
AC_CHECK_HEADER([math.h])

AC_CHECK_HEADER([gsl/gsl_version.h], ,
		[AC_MSG_ERROR([GNU Scientific Library headers not found.  You may need to supply -I<path to gsl headers> to CXXFLAGS.])])

# Checks for typedefs, structures, and compiler characteristics.

# Checks for library functions.

# Set up MPI build
AX_MPI
CC="$MPICC"
CXX="$MPICXX"

# Substitutions
AC_SUBST([OPENMP_CXXFLAGS])
AC_SUBST([GSL_LIBS])

AC_OUTPUT(Makefile src/Makefile src/Framework/Makefile src/data/Makefile src/mcmc/Makefile src/unitTests/Makefile)
