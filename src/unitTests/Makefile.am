INCLUDES = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data -I$(top_srcdir)/src/mcmc -I$(top_srcdir)/src/sim -I$(top_srcdir)/src/diags
METASOURCES = AUTO
bin_PROGRAMS = fmdMcmc fmdSim testLikelihood testPosteriorReader fmdOneStepAhead

noinst_HEADERS = fmdModel.hpp

fmdMcmc_SOURCES = fmdMcmc.cpp
fmdMcmc_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/mcmc/libmcmc.la -lboost_mpi -lboost_serialization

fmdSim_SOURCES = fmdSim.cpp fmdModel.cpp
fmdSim_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la -lboost_program_options

testLikelihood_SOURCES = testLikelihood.cpp
testLikelihood_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/mcmc/libmcmc.la -lboost_mpi -lboost_serialization

testPosteriorReader_SOURCES = testPosteriorReader.cpp
testPosteriorReader_LDADD = $(top_builddir)/src/data/libPosteriorReader.la $(top_builddir)/src/Framework/libstlstrtok.la

fmdOneStepAhead_SOURCES = fmdOneStepAhead.cpp fmdModel.cpp
fmdOneStepAhead_LDADD = $(top_builddir)/src/data/libPosteriorReader.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la -lgslcblas -lgsl -lboost_program_options