SUBDIRS = src

DISTCHECK_CONFIGURE_FLAGS="CXXFLAGS=-O3 -fomit-frame-pointer -DNDEBUG -DBOOST_MPI_HOMOGENEOUS"

exampledir = $(pkgdatadir)/examples
example_DATA = $(top_srcdir)/auseiConfig.xml
EXTRA_DIST = $(top_srcdir)/auseiConfig.xml
dist-hook:
	chmod u+w $(distdir)/README
	sed -i 's/%PACKAGE%/@PACKAGE@/g' $(distdir)/README
	chmod u-w $(distdir)/README

docdir = $(datadir)/doc/@PACKAGE@
doc_DATA = $(top_srcdir)/README
