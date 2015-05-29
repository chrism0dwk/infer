AM_CPPFLAGS = -I. -I../data -I../interface -I../Framework @PKG_CPPFLAGS@

NVCC = @NVCC@

NVCC_CXXFLAGS = @NVCCFLAGS@ -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 --use_fast_math -O3 -Xcompiler "@PKG_CXXFLAGS@"

OBJS = GpuLikelihood.o Mcmc.o McmcFactory.o MCMCUpdater.o McmcLikelihood.o PosteriorHDF5Writer.o PosteriorHDF5Reader.o PosteriorWriter.o GpuUtils.o


all: $(OBJS) GpuLikelihood_gpu.o

$(OBJS): %.o: %.cpp
	$(CXX) -c @PKG_CXXFLAGS@ $(AM_CPPFLAGS) $^ -o $@

GpuLikelihood_gpu.o: %.o: %.cu
	$(NVCC) -c $(AM_CPPFLAGS) $(NVCC_CXXFLAGS) -o $@ $^

clean:
	rm -rf *o

.PHONY: all clean
