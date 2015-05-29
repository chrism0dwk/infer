
theileria.sim <- function(population,contact,parameters=list(epsilon=1e-6, beta1=0.1, beta2=0.1, mu=60, nu=0.015, delta=2, alpha1=0.5,alpha2=0.5,alpha3=0.5,zeta=1), dLimit=50, maxtime=Inf, alg="Gillespie", timestep=1.0)
{
  require(Matrix)

  if(ncol(population) != 6) stop("Wrong number of columns in 'population'.  Should be data.frame(id,x,y,ticks,isdairy,i)")

  reqdParams <- c('epsilon','phi1','phi2','beta1','beta2','alpha1','alpha2','alpha3','nu','delta','zeta')
  if(sum(names(parameters) %in% reqdParams) != length(reqdParams))
    stop("Missing required parameters 'epsilon','beta1','beta2','alpha1','alpha2','alpha3','nu','delta','zeta'")

  params<-NULL
  with(parameters, params<<-c(epsilon,beta1,beta2,0,nu,delta,alpha1,alpha2,alpha3))
  
  names(population) <- c('id','x','y','ticks','isdairy','i')
  phi<-parameters[grep("^phi[0-9]+",names(parameters))]
  tlanums <- as.numeric(substr(names(phi), start=4, stop=10))
  population$ticks<-unlist(phi[match(population$ticks,tlanums)])
  population$ticks[population$isdairy==1] <- population$ticks[population$isdairy==1] * parameters$zeta

  if(alg == "Gillespie")
    .Call("Simulate",population=population,contact=contact,parameter=params,dLimit=dLimit,maxtime=maxtime,alg=0,timestep=timestep)
  else if(alg == "Euler")
    .Call("Simulate",population=population,contact=contact,parameter=params,dLimit=dLimit,maxtime=maxtime,alg=1,timestep=timestep)
  else
    stop("Algorithm must be one of 'Gillespie' or 'Euler'")
}

getcmatelem <- function(contact, i, j)
  {
    .Call("GetMatrixElement", contact=contact, i=i, j=j)
  }
