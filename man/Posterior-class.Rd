\name{Posterior-class}
\Rdversion{1.1}
\docType{class}
\alias{Posterior-class}
\alias{$,Posterior-method}
\alias{length,Posterior-method}
\alias{names,Posterior-method}
\alias{show,Posterior-method}
\alias{summary,Posterior-method}

\title{Class \code{"Posterior"}}
\description{
%%  ~~ A concise (1-5 lines) description of what the class is. ~~
}
\section{Objects from the Class}{
Objects can be created by calls of the form \code{new("Posterior", filename)}.
%%  ~~ describe objects here ~~ 
}
\section{Slots}{
  \describe{
    \item{\code{filename}:}{Object of class \code{"character"} ~~ }
    \item{\code{model}:}{Object of class \code{"character"} ~~ }
    \item{\code{param}:}{Object of class \code{"HD5ParamProxy"} ~~ }
    \item{\code{infec}:}{Object of class \code{"HD5InfecProxy"} ~~ }
  }
}
\section{Methods}{
  \describe{
    \item{$}{\code{signature(x = "Posterior")}: ... }
    \item{length}{\code{signature(x = "Posterior")}: ... }
    \item{names}{\code{signature(x = "Posterior")}: ... }
    \item{show}{\code{signature(object = "Posterior")}: ... }
    \item{summary}{\code{signature(object = "Posterior")}: ... }
	 }
}
\references{
%%  ~~put references to the literature/web site here~~
}
\author{
%%  ~~who you are~~
}
\note{
%%  ~~further notes~~
}

%% ~Make other sections like Warning with \section{Warning }{....} ~

\seealso{
%%  ~~objects to See Also as \code{\link{~~fun~~}}, ~~~
%%  ~~or \code{\linkS4class{CLASSNAME}} for links to other classes ~~~
}
\examples{
showClass("Posterior")
}
\keyword{classes}
