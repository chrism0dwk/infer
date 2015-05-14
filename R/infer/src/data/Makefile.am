AM_CPPFLAGS = -I. -I../Framework @PKG_CPPFLAGS@

OBJS = Data.o

all: $(OBJS)

$(OBJS): %.o: %.cpp
	$(CXX) -c @PKG_CXXFLAGS@ $(AM_CPPFLAGS) $^ -o $@

clean:
	rm -rf *o

.PHONY: all clean
