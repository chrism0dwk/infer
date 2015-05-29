SUBDIRS = interface mcmc sim Framework data

AM_CPPFLAGS = -I. -I@CUDA_HOME@/include @R_INCLUDE@ @RCPP_PKG_INCLUDE@

CUEXT = cu 
EXT = cpp

MCMCOBJS = mcmc/GpuLikelihood.o mcmc/GpuLikelihood_gpu.o mcmc/Mcmc.o mcmc/McmcFactory.o mcmc/MCMCUpdater.o mcmc/McmcLikelihood.o mcmc/PosteriorWriter.o mcmc/PosteriorHDF5Reader.o mcmc/PosteriorHDF5Writer.o mcmc/GpuUtils.o

FRAMEWORKOBJS = Framework/Random.o Framework/stlStrTok.o

IFACEOBJS = interface/posterior.o interface/theileriaMcmc.o interface/RData.o

SIMOBJS = sim/simulate.o

DATAOBJS = data/Data.o

OBJS = $(IFACEOBJS) $(FRAMEWORKOBJS) $(SIMOBJS) $(MCMCOBJS) $(DATAOBJS) 

LIBS =  @HDF5_LIBS@ @CUDA_LIBS@ @GSL_LIBS@

TARGETS = nztheileria.so


all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(CXX) @PKG_CXXFLAGS@ @PKG_CPPFLAGS@ -shared $(OBJS) @PKG_LDFLAGS@ $(LIBS) @PKG_RPATH@ -o $@

clean:
	rm -rf *o

.PHONY: all clean
