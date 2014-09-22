if DEBUG
AM_CXXFLAGS = -g3 -O0 -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data -I$(top_srcdir)/src/sim
else
AM_CXXFLAGS = -O3 -fomit-frame-pointer -DNDEBUG -DBOOST_MPI_HOMOGENEOUS
endif

SUBDIRS=sim data Framework mcmc unitTests