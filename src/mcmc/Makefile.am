INCLUDES = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data
METASOURCES = AUTO
noinst_LTLIBRARIES = libmcmc.la
libmcmc_la_SOURCES = Mcmc.cpp
noinst_HEADERS = Mcmc.hpp
libmcmc_la_LIBADD = $(top_builddir)/src/Framework/librandom.la -lacml