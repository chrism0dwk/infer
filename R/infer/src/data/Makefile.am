AM_CPPFLAGS = -I. -I../Framework @R_INCLUDE@ @RCPP_PKG_INCLUDE@ -I@BOOST_HOME@/include -I@GSL_HOME@/include

PARAMS = @CPICFLAGS@ -fPIC

OBJS = Data.o

all: $(OBJS)

$(OBJS): %.o: %.cpp
	$(CXX) -c $(AM_CPPFLAGS) $(PARAMS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
