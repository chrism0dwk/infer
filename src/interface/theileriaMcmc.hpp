#ifndef THEILERIAMCMC_HPP
#define THEILERIAMCMC_HPP

#include <R.h>
#include <Rcpp.h>

RcppExport SEXP TheileriaMcmc(const SEXP population,
			      const SEXP epidemic,
			      const SEXP contact,
			      const SEXP ticks,
			      const SEXP obsTime,
			      const SEXP init,
			      const SEXP priorParms,
			      const SEXP control,
			      const SEXP outputfile);

#endif

