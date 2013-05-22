/*
 * main.cpp
 *
 *  Created on: May 21, 2010
 *      Author: stsiab
 */

#include <cstdlib>
#include <string>
#include <R_ext/Print.h>

#include "SpatMetaPop.hpp"
#include "SmpSirMcmc.hpp"
#include "SmpSirSim.hpp"

using namespace EpiRisk::Smp;

extern "C" {
  void runSmpSirMcmc  (const char** popDataFile,
		       const char** adjDataFile,
		       const char** epiDataFile,
		       const double* priorAlpha,
		       const double* priorBeta,
		       const double* priorRho,
		       const double* priorGamma,
		       const double* IPshape,
		       const double* initAlpha,
		       const double* initBeta,
		       const double* initRho,
		       const double* initGamma,
		       const double* tuneAlpha,
		       const double* tuneBeta,
		       const double* tuneRho,
		       const double* tuneGamma,
		       const double* tuneIP,
		       const int* numIteration,
		       const int* burnin,
		       const int* thin,
		       const char** outputFile,
		       double* acceptance)
  {
    
    // Performs inference on Markov random field area-level disease incidence.
    
    string popDataFn(*popDataFile);
    string adjFn(*adjDataFile);
    string epiDataFn(*epiDataFile);
    string outputFn(*outputFile);

    SpatMetaPop* population;
    SmpSirMcmc* mcmc;

    try {
      population = new SpatMetaPop(popDataFn,adjFn);
      population->loadEpiData(epiDataFn);
    }
    catch (EpiRisk::data_exception& e) {
      cerr << "Data exception: " << e.what() << endl;
      return;
    }
    catch (...) {
      cerr << "Exception loading data" << endl;
      return;
    }


    mcmc = new SmpSirMcmc(*population);

    mcmc->initParameters(*initAlpha,*initBeta,*initRho,*initGamma); 

    mcmc->setPriorParAlpha(priorAlpha[0],priorAlpha[1]);
    mcmc->setPriorParBeta(priorBeta[0],priorBeta[1]);
    mcmc->setPriorParRho(priorRho[0],priorRho[1]);
    mcmc->setPriorParGamma(priorGamma[0],priorGamma[1]);
    mcmc->setTuneAlpha(*tuneAlpha);
    mcmc->setTuneBeta(*tuneBeta);
    mcmc->setTuneRhoLog(*tuneRho);
    mcmc->setTuneRhoLin(0.0001);
    mcmc->setTuneGamma(*tuneGamma);
    mcmc->setTuneI(*tuneIP);
    mcmc->setBurnin(*burnin);
    mcmc->setThin(*thin);

    mcmc->run(*numIteration,outputFn);

    acceptance[0] = mcmc->getAcceptAlpha();
    acceptance[1] = mcmc->getAcceptBeta();
    acceptance[2] = mcmc->getAcceptRhoLog();
    acceptance[3] = mcmc->getAcceptGamma();
    acceptance[4] = mcmc->getAcceptI();

    Rprintf("Alpha: %f\n",mcmc->getAcceptAlpha());
    Rprintf("Beta: %f\n",mcmc->getAcceptBeta());
    Rprintf("Rho: %f\n",mcmc->getAcceptRhoLog());
    Rprintf("Gamma: %f\n",mcmc->getAcceptGamma());

    delete mcmc;
    delete population;

  }




  void runSmpSirSim ( const char** popDataFile,
		      const char** adjMatrixFile,
		      const double* startTime,
		      const double* alpha,
		      const double* beta,
		      const double* rho,
		      const double* IPshape,
		      const double* gamma,
		      const int* startRegion,
		      const char** outputFile )
  {
    // Runs an instance of SmpSirSim.

    string popDataFn(*popDataFile);
    string adjMatrixFn(*adjMatrixFile);
    string outputFn(*outputFile);

    SpatMetaPop* population;
    SmpSirSim* sim;

    // Set up parameters
    SmpParams params;
    params.alpha = *alpha;
    params.beta = *beta;
    params.rho = *rho;
    params.gamma = *gamma;
    params.a = *IPshape;
    
    try {
      population = new SpatMetaPop(popDataFn,adjMatrixFn);
      population->setObsTime(0.0);
    }
    catch (EpiRisk::data_exception& e) {
      cerr << "Data exception: " << e.what() << endl;
      return;
    }
    catch (...) {
      cerr << "Exception loading data" << endl;
      return;
    }

    try {
      sim = new SmpSirSim(*population);
      sim->setParams(params);
      sim->run(*startRegion-1); // Minus 1 to correct for counting from 0
      sim->writeResults(outputFn);
    }
    catch (exception& e) {
      cerr << "Exception occurred running simulation: " << e.what() << endl;
      return;
    }
    

    delete sim;
    delete population;

  }
  
}
