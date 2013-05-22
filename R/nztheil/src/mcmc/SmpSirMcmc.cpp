/*
 * SIRMcmc.cpp
 *
 *  Created on: May 21, 2010
 *      Author: stsiab
 */

#ifdef __APPLE__&__MACH__
#define WINMACGUI
#endif

#ifdef __WINDOWS__
#define WINMACGUI
#endif


#include <queue>
#include <map>

#include <R.h>
#include <Rmath.h>
#include <R_ext/Print.h>

#include "SmpSirMcmc.hpp"

namespace EpiRisk {
  namespace Smp {
SmpSirMcmc::SmpSirMcmc(SpatMetaPop& population) :
		population_(population),
		logLikCur_(0.0),
		logLikCan_(0.0),
		tuneAlpha_(1.0),
		tuneBeta_(1.0),
		tuneRhoLog_(1.0),
		tuneRhoLin_(1.0),
		tuneGamma_(1.0),
		acceptAlpha_(0.0),
		acceptBeta_(0.0),
		acceptRhoLog_(0.0),
		acceptRhoLin_(0.0),
		acceptGamma_(0.0),
		iteration_(0.0),
		numIterations_(0),
		barLength_(20),
		spinnerPos_(0)

{
	// Init hyper parameters
	priorParAlpha_[0] = 0.1;
	priorParAlpha_[1] = 0.1;

	priorParBeta_[0] = 0.1;
	priorParBeta_[1] = 0.1;

	priorParRho_[0] = 0.1;
	priorParRho_[1] = 0.1;

	priorParGamma_[0] = 1.0;
	priorParGamma_[1] = 1.0;

	// Set up random number generator
	GetRNGstate();

	// Set up infection time sampling EDF
	size_t cumSum = 0;
	SpatMetaPop::CommuneVector::iterator iter = population_.communes.begin();
	while(iter != population_.communes.end()) {
		cumSum += iter->infections.size();
		sampleEDF.insert(pair<double,SpatMetaPop::CommuneVector::iterator>(cumSum,iter));
		iter++;
	}
}

SmpSirMcmc::~SmpSirMcmc()
{
  PutRNGstate();
}




double SmpSirMcmc::integLambda(Commune& infecteeCommune) const
{
	// Returns the integrated infectious pressure on commune

	double insidePress = 0.0;
	double outsidePress = 0.0;
	double backgroundPress = 0.0;

	// First calculate pressure from inside commune
	Infections::iterator infecteeIndiv = infecteeCommune.infections.begin();
	while(infecteeIndiv != infecteeCommune.infections.end()) {
		insidePress += infecteeCommune.infectionTime(infecteeIndiv->I) * params_.beta;
		backgroundPress += (infecteeIndiv->I - population_.getIndexCase()->I) * params_.alpha; // Background pressure
		infecteeIndiv++;
	}

	// Now eval pressure on susceptibles
	insidePress += infecteeCommune.infectionTime(population_.getObsTime()) * (double)infecteeCommune.numSuscepAt(population_.getObsTime()) * params_.beta;
	backgroundPress += (population_.getObsTime() - population_.getIndexCase()->I) * (double)infecteeCommune.numSuscepAt(population_.getObsTime()) * params_.alpha;


	// Loop through connected communes and add up infectionPressure
	Commune::ConVector::const_iterator infectorCommune = infecteeCommune.getConnections()->begin();

	while(infectorCommune != infecteeCommune.getConnections()->end()) {

		// Pressure on the infecteeIndivs
		infecteeIndiv = infecteeCommune.infections.begin();
		while(infecteeIndiv != infecteeCommune.infections.end()) {
			outsidePress += population_.communes[*infectorCommune].infectionTime(infecteeIndiv->I) * params_.rho;
			infecteeIndiv++;
		}
		outsidePress += population_.communes[*infectorCommune].infectionTime(population_.getObsTime()) * (double)infecteeCommune.numSuscepAt(population_.getObsTime()) * params_.rho;

		infectorCommune++;
	}

	return insidePress + outsidePress + backgroundPress;

}



double SmpSirMcmc::totalIntegPressure()
{
	// Returns the total integrated infection pressure for the whole epidemic

	double totalPressure = 0.0;

	int i=0;

#pragma omp parallel for default(shared) schedule(guided) reduction(+:totalPressure)
	for(i=0; i<population_.communes.size(); ++i) {
		totalPressure += integLambda(population_.communes[i]);
	}

	return totalPressure;
}

inline double SmpSirMcmc::lambda(Commune& commune, const double time) const
{
	// Returns the pressure on an individual in commune

	double insidePress = params_.beta * ((double)commune.numInfecAt(time) - 1);

	double outsidePress = 0.0;
	Commune::ConVector::const_iterator iter = commune.getConnections()->begin();
	while(iter != commune.getConnections()->end()) {
		outsidePress += (double)population_.communes.at(*iter).numInfecAt(time);
		iter++;
	}

	outsidePress *= params_.rho;

	return insidePress + outsidePress + params_.alpha;
}



double SmpSirMcmc::logProduct()
{
	// Calculates the log of the likelihood product (see SIRMcmc::likelihood)

	double sumPart = 0.0;

	int i = 0;
	Infections::iterator infectee;

#pragma omp parallel for default(shared) private(infectee) schedule(guided) reduction(+:sumPart)
	for(i=0; i<population_.communes.size(); ++i) {
		if(!population_.communes[i].infections.empty()) {
			infectee = population_.communes[i].infections.begin();
			while(infectee != population_.communes[i].infections.end()) {
				if (infectee != population_.getIndexCase()) sumPart += log(lambda(population_.communes[i],infectee->I));
				infectee++;
			}
		}
	}

	return sumPart;
}



double SmpSirMcmc::likelihood()
{
	// Returns the log(likelihood)
	// logL(\theta | X) \propto  \sum{k \in I} log(lambda(k)) - totalIntegPressure

	return logProduct() - totalIntegPressure();
}



bool SmpSirMcmc::updateAlpha()
{
	// Updates Alpha by logarithmic RWMH

	double oldAlpha = params_.alpha;

	// 1. Propose
	params_.alpha = params_.alpha * exp(rnorm(0,tuneAlpha_));

	// 2. Calculate Candidate likelihood
	logLikCan_ = likelihood();

	// 3. Calculate accept/reject ratio
	double a = logLikCan_ - logLikCur_; // Likelihood
	a += log(priorAlpha(params_.alpha) / priorAlpha(oldAlpha)); // Prior
	a += log( params_.alpha / oldAlpha);

	// 4. Perform accept/reject
	if(log(runif(0,1)) < a) {
		logLikCur_ = logLikCan_;
		return true;
	}
	else {
		params_.alpha = oldAlpha; // Reset param value
		return false;
	}

}



bool SmpSirMcmc::updateBeta()
{
	// Updates Beta by logarithmic RWMH

	double oldBeta = params_.beta;

	// 1. Propose
	params_.beta = params_.beta * exp(rnorm(0,tuneBeta_));


	// 2. Calculate Candidate likelihood
	logLikCan_ = likelihood();

	// 3. Calculate accept/reject ratio
	double a = logLikCan_ - logLikCur_;  // Likelihood
	a += log( priorBeta(params_.beta) / priorBeta(oldBeta) );  // Prior
	a += log( params_.beta / oldBeta ); // q-ratio

	// 4. Perform accept/reject
	if(log(runif(0,1)) < a) {
		logLikCur_ = logLikCan_;
		return true;
	}
	else {
		params_.beta = oldBeta;
		return false;
	}
}



bool SmpSirMcmc::updateRhoLog()
{
	// Updates Gamma by logarithmic or standard RWMH

	double oldRho = params_.rho;

		// 1. Propose
		params_.rho = params_.rho * exp(rnorm(0,tuneRhoLog_));

		// 2. Calculate Candidate likelihood
		logLikCan_ = likelihood();

		// 3. Calculate accept/reject ratio
		double a = logLikCan_ - logLikCur_; // Likelihood
		a += log( priorRho(params_.rho) / priorRho(oldRho) ); // Prior
		a += log( params_.rho / oldRho ); // q-ratio

		// 4. Perform accept/reject
		if(log(runif(0,1)) < a) {
			logLikCur_ = logLikCan_;
			return true;
		}
		else {
			params_.rho = oldRho;
			return false;
		}

}


bool SmpSirMcmc::updateRhoLin()
{
	// Updates rho by linear RW
	double oldRho = params_.rho;

	// 1. Propose
	params_.rho = rnorm(params_.rho,tuneRhoLin_);

	// 2. Calculate Candidate likelihood
	logLikCan_ = likelihood();

	// 3. Calculate accept/reject ratio
	double a = logLikCan_ - logLikCur_; // Likelihood
	a += log( priorRho(params_.rho) / priorRho(oldRho) ); // Prior

	// 4. Perform accept/reject
	if(log(runif(0,1)) < a) {
		logLikCur_ = logLikCan_;
		return true;
	}
	else {
		params_.rho = oldRho;
		return false;
	}

}



bool SmpSirMcmc::updateInfectionTime()
{
	// Updates a uniformly chosen infection time

	Commune& myCommune = pickCommune(); // Choose a commune from which to pick an infection
	size_t u = floor(runif(0,myCommune.infections.size()));
	Infections::iterator myInfection = myCommune.infections.begin();
	for(size_t i = 0; i<u; ++i) myInfection++;

	Individual oldInfec = *myInfection; // Make a copy of the old infection


	double newI = myInfection->R - (myInfection->R - myInfection->I)*exp(rnorm(0,tuneI_)); // Logarithmic RW
	Infections::iterator newInfec = myCommune.moveInfectionTime(myInfection,newI);
	population_.updateIndexCase();

	logLikCan_ = likelihood();

	// Conditional posteriors
	double logPiCan = logLikCan_ + log(priorI(newInfec->R - newInfec->I));
	double logPiCur = logLikCur_ + log(priorI(oldInfec.R - oldInfec.I));

	// q-ratio
	double qRatio = log( (newInfec->R - newInfec->I) / (oldInfec.R - oldInfec.I)); // Logarithmic RW

	// Accept/reject ratio
	double a = logPiCan - logPiCur + qRatio;

	if(log(runif(0,1)) < a)
	{
		// Accept
		logLikCur_ = logLikCan_;
		return true;
	}
	else
	{
		// Reject
		myCommune.moveInfectionTime(newInfec,oldInfec.I);
		population_.updateIndexCase();
		return false;
	}


}



bool SmpSirMcmc::updateGammaGibbs()
{
	// Updates gamma by Gibbs sampling

	double shape = ISHAPE*(double)population_.getNumInfecs() + priorParGamma_[0];
	double rate = population_.getMeanI()*(double)population_.getNumInfecs() + priorParGamma_[1];

	params_.gamma = rgamma(shape,1/rate);
	return true;
}


bool SmpSirMcmc::updateGammaNC()
{

	// Updates Gamma using PNC algorithm
	// Slight reordering of Page 91, Kypraios 2009, Lancaster University

	double oldGamma = params_.gamma; // Save old value for later
	double logPiCan = 0.0;
	double logPiCur = 0.0;

	// 1. Generate proposal for gamma by logarithmic RW
	params_.gamma = params_.gamma * exp(rnorm(0, tuneGamma_));


	// 3. For NONCENTREDPROP of individuals, transform their infection times
	//  create an undo list in case we reject this move.
	typedef queue< pair<Infections::const_iterator,SpatMetaPop::CommuneVector::iterator> > UndoList;
	UndoList undoList; // Undo list

	// Iterate over communes
	SpatMetaPop::CommuneVector::iterator commune = population_.communes.begin();
	while(commune != population_.communes.end()) {

		// Calculate prior for infection times, and build a list of individuals to non-centre.
		queue<Infections::const_iterator> toModify;
		Infections::const_iterator infection = commune->infections.begin();
		while(infection != commune->infections.end()) {
		  if(runif(0,1) < NONCENTREDPROP) {
				toModify.push(infection);
			}
			else {
			  logPiCur += log( dgamma(infection->N - infection->I,params_.a,1/oldGamma,0) );
			  logPiCan += log( dgamma(infection->N - infection->I,params_.a,1/params_.gamma,0) );
			}
			infection++;
		}

		// Non-centre the infection times
		while(!toModify.empty()) {
			double gamRatio = oldGamma / params_.gamma;
			infection = toModify.front();
			double newInfecTime = infection->N - gamRatio * (infection->N - infection->I);
			Infections::const_iterator newInfec = commune->moveInfectionTime(infection,newInfecTime);
			undoList.push(pair<Infections::const_iterator,SpatMetaPop::CommuneVector::iterator>(newInfec,commune));
			toModify.pop();
		}

		commune++;
	}

	population_.updateIndexCase();

	// Calculate conditional posterior
	logLikCan_ = likelihood();
	logPiCur += logLikCur_ + log(priorGamma(oldGamma));
	logPiCan += logLikCan_ + log(priorGamma(params_.gamma));

	// Calculate q-ratio
	double qRatio = log(params_.gamma / oldGamma);

	// Calculate accept/reject
	double a = logPiCan - logPiCur + qRatio;

	if (log(runif(0,1)) < a) {
		// Accept the move
		logLikCur_ = logLikCan_;
		return true;
	}
	else
	{ // Reject the move, undoing all the changes to the I's

		// Undo I's
		while(!undoList.empty()) {
			double oldInfecTime = undoList.front().first->N - ( params_.gamma / oldGamma * (undoList.front().first->N - undoList.front().first->I));
			undoList.front().second->moveInfectionTime(undoList.front().first,oldInfecTime);
			undoList.pop();
		}
		params_.gamma = oldGamma;
		population_.updateIndexCase();
		return false;
	}
}










Commune& SmpSirMcmc::pickCommune()
{
	// Picks an infection time uniformly
	CommuneEDF::iterator iter = sampleEDF.end();
	iter--;
	double u = runif(0.0,iter->first);
	SpatMetaPop::CommuneVector::iterator myCommune = sampleEDF.lower_bound(u)->second;

	return *myCommune;
}



void SmpSirMcmc::run(const size_t numIterations, const string outputFile)
{
	// Runs the MCMC

	numIterations_ = numIterations;

	acceptAlpha_ = 0.0;
	acceptBeta_ = 0.0;
	acceptRhoLog_ = 0.0;
	acceptRhoLin_ = 0.0;

	logLikCur_ = likelihood();

	openOutputFile(outputFile);

	for (iteration_ = 0; iteration_ < numIterations; ++iteration_) {

	  if (iteration_ % 100 == 0) progressBar();
	  acceptAlpha_ += updateAlpha();
	  acceptBeta_ += updateBeta();
	  
	  double u = runif(0,1);
	  if(u < ADDMULTRATIO) acceptRhoLog_ += updateRhoLog();
	  else acceptRhoLin_ += updateRhoLin();
	  
	  for(size_t i=0; i < NUMINFECSTOMOVE; ++i) acceptI_ += updateInfectionTime();
	  
	  // Interleaved Centred and non-centred gamma
	  updateGammaGibbs();

	  acceptGamma_ += updateGammaNC();
	  
	  writeParameters();


	}

	progressBar();

	closeOutputFile();

}


void SmpSirMcmc::openOutputFile(const string filename)
{
  // Opens the output file
  outputFile = fopen(filename.c_str(),"w");
  if(outputFile == NULL)
    {
      throw output_exception(string("Cannot open output file '" + filename + "' for writing").c_str());
    }
  
  fprintf(outputFile, "\"Alpha\",\"Beta\",\"Rho\",\"Gamma\",\"MeanI\",\"likelihood\"");
  
  map<size_t,double> infecTimes;
  SpatMetaPop::CommuneVector::iterator commune = population_.communes.begin();
  while(commune != population_.communes.end()) {
    Infections::iterator infection = commune->infections.begin();
    while(infection != commune->infections.end()) {
      infecTimes.insert(pair<size_t,double>(infection->label,infection->I));
      infection++;
    }
    commune++;
  }
  
  map<size_t,double>::iterator iter = infecTimes.begin();
  while(iter != infecTimes.end()) {
    fprintf(outputFile, ",\"%lu\"", iter->first);
    iter++;
  }
  
  fprintf(outputFile,"\n");
  
  
}


void SmpSirMcmc::writeParameters()
{
	// Write parameters
  fprintf(outputFile,"%f,%f,%f,%f,%f,%u",
	  params_.alpha,
	  params_.beta,
	  params_.rho,
	  params_.gamma,
	  population_.getMeanI(),
	  population_.getIndexCase()->label);
  
  map<size_t,double> infecTimes;
  SpatMetaPop::CommuneVector::iterator commune = population_.communes.begin();
  while(commune != population_.communes.end()) {
    Infections::iterator infection = commune->infections.begin();
    while(infection != commune->infections.end()) {
      infecTimes.insert(pair<size_t,double>(infection->label,infection->I));
      infection++;
    }
    commune++;
  }
  
  map<size_t,double>::iterator iter = infecTimes.begin();
  while(iter != infecTimes.end()) {
    fprintf(outputFile,",%f",iter->second);
    iter++;
  }
  
  fprintf(outputFile,"\n");
}


void SmpSirMcmc::closeOutputFile()
{
	// Closes the output file
	fclose(outputFile);
}


void SmpSirMcmc::progressBar()
{
	// Displays a progress bar
	const char spinner[5] = "|/-\\";
	const char BAR[] = {"--------------------"};
	char bar[barLength_+2+9]; // 2 for end caps, 9 for spinner and percent done

	float fractionDone = (float)iteration_ / (float)numIterations_;
	int barPercentDone = floor(fractionDone * barLength_);

	Rprintf("\r|%.*s%*s| %c %.0f%%    ",barPercentDone,BAR,barLength_ - barPercentDone,"",spinner[spinnerPos_],fractionDone*100);
	
	R_FlushConsole();
#ifdef  WINMACGUI
	R_ProcessEvents();
#endif
	if(spinnerPos_ == 3) spinnerPos_ = 0;
	else spinnerPos_++;
}



double SmpSirMcmc::priorAlpha(const double val) const
{
	// Prior for alpha

  return dgamma(val,priorParAlpha_[0], 1/priorParAlpha_[1],0);
}


double SmpSirMcmc::priorBeta(const double val) const
{
	// Prior for beta

  return dgamma(val, priorParBeta_[0], 1/priorParBeta_[1],0);
}


double SmpSirMcmc::priorRho(const double val) const
{
	// Prior for rho
  return dgamma(val, priorParRho_[0], 1/priorParRho_[1],0);
}



double SmpSirMcmc::priorI(const double val) const
{
	// Prior for N-I
  return dgamma(val, ISHAPE, 1/params_.gamma,0);
}


double SmpSirMcmc::priorGamma(const double val) const
{
	// Prior for Gamma
  return dgamma(val, priorParGamma_[0], 1/priorParGamma_[1],0);
}


  } }
