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


#linker options
LD_PARAMS = @RPATHFLAG@
NVLD_PARAMS = @DEVICEOPTS@ -Xlinker @RPATHFLAG@

CUDPP_LIBS = -L@CUDPP_HOME@/lib -lcudpp
CUDA_LIBS = -L@CUDA_HOME@@CUDA_LIB_DIR@ -lcudart -lcublas -lcurand -lcusparse
GSL_LIBS = -L@GSL_HOME@/lib -lgsl -lgslcblas
HDF5_LIBS= -L@HDF5_HOME@/lib -lhdf5 -lhdf5_cpp -lhdf5_hl -lhdf5_hl_cpp
LIBS =  @R_LIB@ @LIBS@ @RCPP_PKG_LIBS@ $(HDF5_LIBS) $(CUDA_LIBS) $(CUDPP_LIBS) $(GSL_LIBS)

TARGETS = infer.so


all: $(TARGETS) 

$(TARGETS): $(OBJS)
	$(CXX) -shared $(OBJS) $(LIBS) $(LD_PARAMS) -o $@

clean:
	rm -rf *o

.PHONY: all clean
