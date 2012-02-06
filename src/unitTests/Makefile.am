INCLUDES = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data -I$(top_srcdir)/src/mcmc -I$(top_srcdir)/src/sim -I$(top_srcdir)/src/diags
AM_CPPFLAGS = @BOOST_CPPFLAGS@ @GSL_CPPFLAGS@
METASOURCES = AUTO
bin_PROGRAMS = fmdMcmc fmdSim testLikelihood testPosteriorReader fmdOneStepAhead fmdSellkeDiagnostic

noinst_HEADERS = fmdModel.hpp

fmdMcmc_SOURCES = fmdMcmc.cpp MatLikelihood.cpp
fmdMcmc_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/mcmc/libmcmc.la @GSL_LIBS@ @BOOST_LDFLAGS@ @BOOST_MPI_LIB@ @BOOST_SERIALIZATION_LIB@ @MATHLIB@

fmdSim_SOURCES = fmdSim.cpp fmdModel.cpp
fmdSim_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la @BOOST_LDFLAGS@ -lboost_program_options @GSL_LIBS@ @MATHLIB@

testLikelihood_SOURCES = testLikelihood.cpp
testLikelihood_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/mcmc/libmcmc.la @BOOST_LDFLAGS@ @BOOST_MPI_LIB@ @BOOST_SERIALIZATION_LIB@ @GSL_LIBS@ @MATHLIB@

testPosteriorReader_SOURCES = testPosteriorReader.cpp
testPosteriorReader_LDADD = $(top_builddir)/src/data/libPosteriorReader.la $(top_builddir)/src/Framework/libstlstrtok.la

fmdOneStepAhead_SOURCES = fmdOneStepAhead.cpp fmdModel.cpp
fmdOneStepAhead_LDADD = $(top_builddir)/src/data/libPosteriorReader.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la @GSL_LIBS@ @BOOST_LDFLAGS@ -lboost_program_options @MATHLIB@

fmdSellkeDiagnostic_SOURCES = fmdSellkeDiagnostic.cpp fmdModel.cpp
fmdSellkeDiagnostic_LDADD = $(top_builddir)/src/data/libPosteriorReader.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la @GSL_LIBS@ @BOOST_LDFLAGS@ -lboost_program_options @MATHLIB@
