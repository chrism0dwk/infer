AM_CPPFLAGS = -I. @PKG_CPPFLAGS@

CUEXT = cu 
EXT = cpp

OBJS = Random.o stlStrTok.o

all: $(OBJS) 

$(OBJS): %.o: %.$(EXT)
	$(CXX) -c $(PKG_CXXFLAGS) $(PKG_CPPFLAGS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
