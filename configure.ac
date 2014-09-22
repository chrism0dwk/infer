#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

#AC_PREREQ([2.61])
AC_INIT(infer-ausei, 4.0, Chris Jewell <chrism0dwk@gmail.com>)

AC_CONFIG_HEADERS([config.h])
AC_CONFIG_MACRO_DIR([m4])
AM_INIT_AUTOMAKE(infer-ausei, 4.0)

# Checks for programs.
AC_PROG_CXXCPP
AC_PROG_CXX

AC_LANG([C++])
AM_PROG_LIBTOOL


# Checks for header files.
AC_CHECK_HEADER([math.h])

# Check for GSL availability
AX_PATH_GSL([1.15], ,
		    [AC_MSG_ERROR([GNU Scientific Library >= 1.15 required])])

# Boost 1.51.0
AX_BOOST_BASE([1.51.0], ,
 			[AC_MSG_ERROR([Boost >= 1.51.0 required])])
AX_BOOST_PROGRAM_OPTIONS
AX_BOOST_SERIALIZATION

# Set up MPI build
AX_MPI
CC="$MPICC"
CXX="$MPICXX"
AX_BOOST_MPI


AC_OUTPUT(Makefile src/Makefile src/Framework/Makefile src/data/Makefile src/mcmc/Makefile src/unitTests/Makefile src/sim/Makefile)
