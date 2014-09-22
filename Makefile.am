if DEBUG
AM_CXXFLAGS = -g3 -O0
else
AM_CXXFLAGS = -O3 -fomit-frame-pointer -DNDE456BUG -DBOOST_MPI_HOMOGENEOUS
endif

SUBDIRS = src

DISTCHECK_CONFIGURE_FLAGS="CXXFLAGS=-O3 -fomit-frame-pointer -DNDEBUG -DBOOST_MPI_HOMOGENEOUS"

exampledir = $(datarootdir)/examples/@PACKAGE@
example_DATA = auseiConfig.xml

docdir = $(datarootdir)/doc/@PACKAGE@
doc_DATA = README
