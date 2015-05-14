SUBDIRS = interface mcmc sim Framework data
#data Framework mcmc sim
AM_CPPFLAGS = -I. -I@CUDA_HOME@/include @R_INCLUDE@ @RCPP_PKG_INCLUDE@

CUEXT = cu 
EXT = cpp

MCMCOBJS =  mcmc/SmpSirMcmc.o mcmc/GpuLikelihood.o mcmc/GpuLikelihood_gpu.o mcmc/Likelihood.o mcmc/Mcmc.o mcmc/McmcFactory.o mcmc/MCMCUpdater.o mcmc/McmcLikelihood.o mcmc/PosteriorWriter.o mcmc/PosteriorHDF5Reader.o mcmc/PosteriorHDF5Writer.o

FRAMEWORKOBJS = Framework/Random.o Framework/SpatMetaPop.o Framework/SmpIndividual.o Framework/stlStrTok.o

IFACEOBJS = interface/posterior.o interface/SmpIface.o interface/SpSINRMcmc.o interface/RData.o

SIMOBJS = sim/SmpSirSim.o

DATAOBJS = data/Data.o

OBJS = $(IFACEOBJS) $(FRAMEWORKOBJS) $(SIMOBJS) $(MCMCOBJS) $(DATAOBJS) 

LIBS =  @HDF5_LIBS@ @CUDA_LIBS@ @GSL_LIBS@

TARGETS = infer.so


all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(CXX) @PKG_CXXFLAGS@ @PKG_CPPFLAGS@ -shared $(OBJS) @PKG_LDFLAGS@ $(LIBS) -o $@

clean:
	rm -rf *o

.PHONY: all clean
