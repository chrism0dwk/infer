AM_CPPFLAGS = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data -I$(top_srcdir)/src/sim @GSL_CFLAGS@ @BOOST_CPPFLAGS@

METASOURCES = AUTO
noinst_LTLIBRARIES = libmcmc.la

libmcmc_la_SOURCES = Mcmc.cpp MCMCUpdater.cpp
noinst_HEADERS = Mcmc.hpp MCMCUpdater.hpp StochasticNode.hpp

libmcmc_la_LIBADD = $(top_builddir)/src/Framework/librandom.la