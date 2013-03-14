INCLUDES = -I. -I../data -I../Framework -I../sim -I../mcmc -I@CUDA_HOME@/include @R_INCLUDE@ @RCPP_PKG_INCLUDE@ -I@CUDPP_HOME@/include -I@BOOST_HOME@/include -I@GSL_HOME@/include -I@HDF5_HOME@/include

EXT = cpp

OBJS = posterior.o SmpIface.o SpSINRMcmc.o

#compiler/preprocessor options
PARAMS = @CPICFLAGS@

#linker options
LD_PARAMS = @RPATHFLAG@

all: $(OBJS) 

$(OBJS): %.o: %.$(EXT)
	$(CXX) -c $(INCLUDES) $(PARAMS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
