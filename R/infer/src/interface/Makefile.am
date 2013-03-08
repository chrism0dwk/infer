INCLUDES = -I. -I${top_srcdir}/src/Framework -I${top_srcdir}/src/mcmc -I${top_srcdir}/src/sim -I@CUDA_HOME@/include @R_INCLUDE@ @RCPP_PKG_INCLUDE@

EXT = cpp

OBJS = posterior.o SmpIface.o

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
