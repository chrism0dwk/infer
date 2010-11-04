INCLUDES = -I$(top_srcdir)/src/Framework -I$(top_srcdir)/src/data -I$(top_srcdir)/src/mcmc
METASOURCES = AUTO
bin_PROGRAMS = testSpatPointPop testMcmc paramTest testRandom

testSpatPointPop_SOURCES = testSpatPointPop.cpp
testSpatPointPop_LDADD = $(top_builddir)/src/data/libdata.la  $(top_builddir)/src/Framework/libstlstrtok.la -lm

testMcmc_SOURCES = testMcmc.cpp
testMcmc_LDADD = $(top_builddir)/src/Framework/librandom.la $(top_builddir)/src/data/libdata.la $(top_builddir)/src/Framework/libstlstrtok.la $(top_builddir)/src/mcmc/libmcmc.la -lgsl -lgslcblas -lm

paramTest_SOURCES = paramTest.cpp
paramTest_LDADD = -lm

testRandom_SOURCES = testRandom.cpp
testRandom_LDADD = $(top_builddir)/src/Framework/librandom.la -lgsl -lgslcblas -lm