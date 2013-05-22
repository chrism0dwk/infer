\name{read.posterior}
\alias{read.posterior}
\docType{package}
\title{
Reads an InFER posterior file
}
\description{
Reads an InFER posterior file, allowing removal of burnin, and thinning to fit into memory.
}
\usage{
read.posterior(filename, burnin, thin, actions)
}
\examples{
\dontrun{
read.posterior("myfile.h5", 0, 10, "both")
}
}
