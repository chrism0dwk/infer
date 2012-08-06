// Put GPLv3 here

#ifndef _infer_POSTERIOR_CPP
#define _infer_POSTERIOR_CPP

#include <Rcpp.h>

RcppExport SEXP readPosterior(SEXP filename, 
			SEXP burnin, 
			SEXP thin, 
			SEXP actions);

#endif
