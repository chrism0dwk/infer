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