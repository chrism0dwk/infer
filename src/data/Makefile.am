AM_CPPFLAGS = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/sim @GSL_CFLAGS@ @BOOST_CPPFLAGS@
METASOURCES = AUTO
noinst_LTLIBRARIES = libdata.la
libdata_la_SOURCES = Data.cpp
noinst_HEADERS = Data.hpp
libdata_la_LIBADD = $(top_builddir)/src/sim/libsim.la
