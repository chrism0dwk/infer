# R utility functions

read.empcov <- function(filename)
{
	myFile <- file(filename,"r")
	empCovariances <- list()
	
	line <- character(1)
	while(TRUE)
	{
		line <- readLines(myFile,1)
		if(length(line) == 0) { break }
		
		toks <- unlist(strsplit(line,":"))
		name <- toks[1]
		def <- toks[2]
		
		rbracketPos=regexpr("\\]",def)
		
		nrows=as.numeric(unlist(strsplit(substr(def,2,rbracketPos-1),","))[1])
		
		matrixDef=gsub("\\(","\\c(",substring(def,rbracketPos+1))
		
		empCovariances[[name]] = matrix(eval(parse(text=matrixDef)),nrow=nrows)
		
	}
	
	close(myFile)
	
	return (empCovariances)
}




mcconvergence <- function(posterior)
  {
    require(Matrix)
    # Computes Markov chain convergence
    sigma <- cov(posterior)
    ndims <- dim(sigma)[1]
    sigmainv <- solve(sigma)

    # Normalise sigmainv
    C=diag(diag(sigmainv)^(-0.5),nrow=ndims)%*%sigmainv%*%diag(diag(sigmainv)^(-0.5),nrow=ndims)

    # Compute B = (I-L)^-1 %*% U
    L=tril(C)
    diag(L)<-rep(0,ndims)
    U=triu(C)
    diag(L)<-rep(0,ndims)
    B = solve(diag(rep(1,ndims))-L)%*%U

    # Get ratio of L2 norm to L1 norm
    sigma.eigens=eigen(sigma)$values
    adap = (sum(sigma.eigens^2)/length(sigma.eigens)) / (sum(sigma.eigens)^2/length(sigma.eigens))

    # return
    return(list(adap=adap,conveigs=eigen(B)$values))
  }
