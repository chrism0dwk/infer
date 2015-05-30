AM_CPPFLAGS = -I. -I../Framework -I../mcmc @PKG_CPPFLAGS@

CUEXT = cu 
EXT = cpp

OBJS = simulate.o

all: $(OBJS) 

$(OBJS): %.o: %.$(EXT)
	$(CXX) -c @PKG_CXXFLAGS@ $(AM_CPPFLAGS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
