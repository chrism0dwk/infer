AM_CPPFLAGS = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data -I$(top_srcdir)/src/mcmc -I$(top_srcdir)/src/sim @GSL_CFLAGS@ @BOOST_CPPFLAGS@

METASOURCES = AUTO
bin_PROGRAMS = auseiMcmc

auseiMcmc_SOURCES = auseiMcmc.cpp
auseiMcmc_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/mcmc/libmcmc.la @BOOST_LDFLAGS@ @BOOST_MPI_LIB@ @BOOST_SERIALIZATION_LIB@ @BOOST_PROGRAM_OPTIONS_LIB@ @GSL_LIBS@
