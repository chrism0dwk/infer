INCLUDES = -I. -I@CUDA_HOME@/include @R_INCLUDE@ @RCPP_PKG_INCLUDE@

CUEXT = cu 
EXT = cpp

OBJS = SpatMetaPop.o SmpIndividual.o stlStrTok.o Random.o

#compiler/preprocessor options
PARAMS = @CPICFLAGS@ -fPIC

#linker options
LD_PARAMS = @RPATHFLAG@

all: $(OBJS) 

$(OBJS): %.o: %.$(EXT)
	$(CXX) -c $(INCLUDES) $(PARAMS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
