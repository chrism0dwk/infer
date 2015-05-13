INCLUDES = -I. -I../data -I../Framework -I@CUDA_HOME@/include @R_INCLUDE@ @RCPP_PKG_INCLUDE@ -I@CUDPP_HOME@/include -I@BOOST_HOME@/include -I@GSL_HOME@/include -I@HDF5_HOME@/include

NVCC = @CUDA_HOME@/bin/nvcc -gencode arch=compute_11,code=sm_11 -gencode arch=compute_12,code=sm_12 -gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35

PARAMS = @CPICFLAGS@ -fPIC @AM_CXXFLAGS@
NVPARAMS = @DEVICEOPTS@ -Xcompiler "@CPICFLAGS@ -fPIC @AM_CXXFLAGS@"

OBJS = SmpSirMcmc.o Likelihood.o GpuLikelihood.o Mcmc.o McmcFactory.o MCMCUpdater.o McmcLikelihood.o PosteriorHDF5Writer.o PosteriorHDF5Reader.o PosteriorWriter.o 


all: $(OBJS) GpuLikelihood_gpu.o

$(OBJS): %.o: %.cpp
	$(CXX) -c $(INCLUDES) $(PARAMS) $^ -o $@

GpuLikelihood_gpu.o: %.o: %.cu
	$(NVCC) -c $(INCLUDES) $(NVPARAMS) -o $@ $^

clean:
	rm -rf *o

.PHONY: all clean
