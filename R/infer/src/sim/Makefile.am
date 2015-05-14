AM_CPPFLAGS = -I. -I../Framework @PKG_CPPFLAGS@

CUEXT = cu 
EXT = cpp

OBJS = SmpSirSim.o

all: $(OBJS) 

$(OBJS): %.o: %.$(EXT)
	$(CXX) -c @PKG_CXXFLAGS@ $(AM_CPPFLAGS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
