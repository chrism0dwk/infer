SUBDIRS = src

DISTCHECK_CONFIGURE_FLAGS="CXXFLAGS=-O3 -fomit-frame-pointer -DNDEBUG -DBOOST_MPI_HOMOGENEOUS"

exampledir = $(pkgdatadir)/examples
example_DATA = $(top_srcdir)/auseiConfig.xml

docdir = $(datadir)/doc/@PACKAGE@
doc_DATA = $(top_srcdir)/README
