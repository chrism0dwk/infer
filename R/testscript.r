library(infer)

pop<-read.csv("test.pop")
epi<-read.csv("test.epi")
init.params <- list(epsilon1=1e-6, epsilon2=1, gamma1=5e-5, gamma2=1, xi_2, xi_3, psi_1, psi_2, psi_3, zeta_2, zeta_3, phi_1, phi_2, phi_3, delta, a, b, nu, alpha, omega)

model<-SpatPointSINR(pop,epi)
post<-berp.fit(model, control=list(n.iter=10), init.params)




