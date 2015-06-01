# GPL here



.onLoad <- function(libname, pkgname)
  {
    if (!require(methods,quietly=TRUE) | !require(Rcpp, quietly=TRUE)) stop("'methods' and 'Rcpp' packages required for package 'nztheileria'")

    .initPosterior()
  }
