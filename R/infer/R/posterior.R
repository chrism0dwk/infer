# GPLv3 here

# Helper functions
getPosteriorLen <- function(filename)
{
	return(.Call("getPosteriorLen", filename))
}

getPosteriorModel <- function(filename)
{
	return(.Call("getPosteriorModel", filename))
}

#getPosteriorParam <- function(filename, i, j)
#{
#	rows <- NULL
#	cols <- NULL
#	
#	# Assess missingness of operators
#	if(!missing(i)) rows <- i
#	if(!missing(j)) cols <- j
#	
#	return(.Call("getPosteriorParam", filename))	
#}
#
#getPosteriorInfec <- function(filename, i, j)
#{
#	rows <- NULL
#	cols <- NULL
#	
#	# Assess missingness of operators
#	if(!missing(i)) rows <- i
#	if(!missing(j)) cols <- j
#	
#	return(.Call("getPosteriorInfec", filename))
#}

lookupInfIdxNames <- function(theVector, tags)
{
	names(theVector) <- tags[as.numeric(names(theVector))+1]
	theVector
}




# Constructor functions
Posterior <- function(filename)
{
	new("Posterior", filename=filename)
}

read.posterior <- function(filename)
  {
    Posterior(filename);
  }

write.posterior <- function(posterior, filename, ...)
  {
    file.copy(posterior@filename, filename, ...)
  }

HD5ParamProxy <- function(filename)
{
	new("HD5ParamProxy", filename=filename)
}

HD5InfecProxy <- function(filename)
{
	new("HD5InfecProxy", filename=filename)
}



.initPosterior <- function() {							 

# Proxy classes -- these classes provide an interface to the posterior in
#   the underlying disc storage.
setClass("HD5ParamProxy",representation(filename="character",tags="character"))
setMethod("initialize","HD5ParamProxy",
		function(.Object,filename) {
			.Object@filename <- filename
			info <- .Call("getPosteriorParamInfo", filename)
			.Object@tags <- info$tags
			return(.Object)
		}
)



setClass("HD5InfecProxy",representation(filename="character",tags="character"))
setMethod("initialize","HD5InfecProxy",
		function(.Object,filename) {
			.Object@filename <- filename
			info <- .Call("getPosteriorInfecInfo", filename)
			.Object@tags <- info$tags
			return(.Object)
		}
)



setClass("Posterior",representation(filename="character", model="character", param="HD5ParamProxy",infec="HD5InfecProxy"))
setMethod("initialize", "Posterior", function(.Object, filename) {
			.Object@filename <- filename
			.Object@model <- getPosteriorModel(filename)
			.Object@param <- HD5ParamProxy(filename)
			.Object@infec <- HD5InfecProxy(filename)
			return(.Object)
		})




# Metadata methods
setMethod("show","Posterior", function(object) cat("Instance of Posterior for model '",object@model,"', length=",length(object),"\n",sep=''))
setMethod("show","HD5ParamProxy", function(object) cat("Instance of HD5ParamProxy\n"))
setMethod("show","HD5InfecProxy", function(object) cat("Instance of HD5InfecProxy\n"))
setMethod("names","Posterior", function(x) c("param","infec","model","filename"))
setMethod("names","HD5ParamProxy", function(x) x@tags)
setMethod("names","HD5InfecProxy", function(x) x@tags)
#setMethod("length", "HD5ParamProxy", function(x) getPosteriorLen(x@filename))
#setMethod("length", "HD5InfecProxy", function(x) getPosteriorLen(x@filename))
setMethod("length", "Posterior", function(x) getPosteriorLen(x@filename))
setGeneric("nrow")
setMethod("nrow", "HD5ParamProxy", function(x) getPosteriorLen(x@filename))
setMethod("nrow", "HD5InfecProxy", function(x) getPosteriorLen(x@filename))
setMethod("dim", "HD5ParamProxy", function(x) c(nrow(x), length(x@tags)))
setMethod("dim", "HD5InfecProxy", function(x) c(nrow(x), length(x@tags)))
setGeneric("ncol")
setMethod("ncol", "HD5ParamProxy", function(x) dim(x)[2])
setMethod("ncol", "HD5InfecProxy", function(x) dim(x)[2])

# Data accessor methods
setMethod("$","Posterior", function(x,name) switch(name,param=x@param, infec=x@infec, filename=x@filename, model=x@model))
setMethod("[","HD5ParamProxy",
		function(x,i,j) {
			rows <- integer(0)
			cols <- integer(0)
			if(!missing(i)) {
				rows <- (0:(nrow(x)-1))[i] # May seem odd, but it delegates bounds checking to R's internals
			}
			else rows <- 0:(nrow(x)-1)
			
			if(!missing(j)) {
				if(class(j) == "character") {
					cols <- match(j,x@tags) - 1
					if(any(is.na(cols))) stop("Invalid column specification")
				}
				else cols <- (0:(length(x@tags)-1))[j]
			}
			else cols <- 0:(length(x@tags)-1)
			
			.Call("getPosteriorParams",x@filename,rows,cols)
		}
)
setMethod("$","HD5ParamProxy",
		function(x,name) {
			if(name %in% names(x)) x[,name]
			else NULL
		}
)
setMethod("[","HD5InfecProxy",
		function(x,i) {
			rows <- integer(0)
			cols <- integer(0)
			if(!missing(i)) {
				rows <- (0:(nrow(x)-1))[i]
			}
			else rows <- 0:(nrow(x)-1)
			
#			if(!missing(j)) {
#				cols <- (1:length(x@tags))[j]
#				if(any(cols < 1) | any(cols > length(x@tags))) stop("Col subscript of of bounds")
#			}
#			else cols <- 1:length(x@tags)
			
			# cols is currently unimplemented
			.Call("getPosteriorInfecs",x@filename,rows,cols)
		}
)

# Coercion
setGeneric("as.data.frame")
setMethod("as.data.frame","HD5ParamProxy", function(x) x[])
setMethod("as.data.frame","HD5InfecProxy", function(x) x[])

# Summary methods
setGeneric("summary")
setMethod("summary","Posterior", 
		function(object) {
			cat("Model:", object@model, "\n")
			cat("Num samples:", length(object@param), "\n")			
			cat("Parameters:\n")
			print(summary(object@param))
			cat("\nInfections:\n")
			print(summary(object@infec))
		}
)
setMethod("summary","HD5ParamProxy",
		function(object) {
			np <- dim(object)[2]
			z <- list()
			for(i in 1:np) z[[names(object)[i]]] <- summary(object[,i])
			fields <- names(z[[1]])
			nms <- names(z)
			z <- unlist(z,use.names=TRUE)
			dim(z) <- c(length(fields),length(nms))
			dimnames(z) <- list(fields,nms)
			as.table(z)
		})
setMethod("summary","HD5InfecProxy",
		function(object) {
			"To be implemented -- tell me what you want to know about!"
		})


} # .initPosterior


