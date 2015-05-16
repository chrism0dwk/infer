AM_CPPFLAGS = -I. -I../data -I../Framework -I../sim -I../mcmc @PKG_CPPFLAGS@

EXT = cpp

OBJS = posterior.o SmpIface.o SpSINRMcmc.o RData.o

all: $(OBJS) 

$(OBJS): %.o: %.$(EXT)
	$(CXX) -c @PKG_CXXFLAGS@ $(AM_CPPFLAGS) $(AM_CXXFLAGS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
