INCLUDES = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data
AM_CPPFLAGS = @BOOST_CPPFLAGS@ @GSL_CPPFLAGS@
METASOURCES = AUTO

.cu.o:
	nvcc @NVCXXFLAGS@ --compiler-options "$(CXXFLAGS)" $(INCLUDES) $(CPPFLAGS) -c -o $@ $<
	
noinst_HEADERS = Mcmc.hpp MCMCUpdater.hpp McmcLikelihood.hpp GpuLikelihood.hpp
noinst_LTLIBRARIES = libgpulikelihood.la libmcmc.la

libmcmc_la_SOURCES = Mcmc.cpp MCMCUpdater.cpp McmcLikelihood.cpp
libmcmc_la_LIBADD = $(top_builddir)/src/Framework/librandom.la

libgpulikelihood_la_SOURCES = GpuLikelihoodgpu.cu GpuLikelihood.cpp 
libgpulikelihood_la_LIBADD = $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la @CUDA_LDFLAGS@ @BOOST_LDFLAGS@ @BOOST_SERIALIZATION_LIB@ @GSL_LIBS@ @MATHLIB@
