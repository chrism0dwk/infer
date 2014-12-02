library(infer)

pop<-read.csv("test.pop")
epi<-read.csv("test.epi")
source("test.init.r")

model<-SpatPointSINR(pop,epi)
post<-berp.fit(model, control=list(n.iter=10), init)





