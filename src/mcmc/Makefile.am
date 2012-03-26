INCLUDES = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data
AM_CPPFLAGS = @BOOST_CPPFLAGS@ @GSL_CPPFLAGS@
METASOURCES = AUTO

NVCC=nvcc

.cu.o:
	$(NVCC) @NVCXXFLAGS@ --compiler-options "$(CXXFLAGS)" $(INCLUDES) $(CPPFLAGS) -c -o $@ $<

.cu.lo:
	$(top_srcdir)/cudalt.py $@ $(NVCC) @NVCXXFLAGS@ --compiler-options \"$(CXXFLAGS)\" $(INCLUDES) $(CPPFLAGS) -c $<
        
        
noinst_HEADERS = Mcmc.hpp MCMCUpdater.hpp McmcLikelihood.hpp GpuLikelihood.hpp
noinst_LTLIBRARIES = libgpulikelihood.la libmcmc.la

libmcmc_la_SOURCES = Mcmc.cpp MCMCUpdater.cpp McmcLikelihood.cpp
libmcmc_la_LIBADD = $(top_builddir)/src/Framework/librandom.la

libgpulikelihood_la_SOURCES = GpuLikelihood.cpp GpuLikelihood_gpu.cu
libgpulikelihood_la_LIBADD = $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la @CUDA_LDFLAGS@ @BOOST_LDFLAGS@ @BOOST_SERIALIZATION_LIB@ @GSL_LIBS@ @MATHLIB@
nodist_EXTRA_libgpulikelihood_la_SOURCES = dummy.cpp 