METASOURCES = AUTO
noinst_LTLIBRARIES = libstlstrtok.la librandom.la
noinst_HEADERS = stlStrTok.hpp Random.hpp Individual.hpp \
 Contact.hpp DataImporter.hpp EmpCovar.hpp EpiRiskException.hpp \
 McmcWriter.hpp Parameter.hpp SpatPointPop.hpp types.hpp StochasticNode.hpp

libstlstrtok_la_SOURCES = stlStrTok.cpp
librandom_la_SOURCES = Random.cpp

librandom_la_LIBADD = -lgsl -lgslcblas
