/*
 * SIRSim.cpp
 *
 *  Created on: May 31, 2010
 *      Author: stsiab
 */

#include <fstream>
#include <iomanip>
#include <algorithm>
#include <cstdio>

#include <R.h>
#include <Rmath.h>
#include <R_ext/Print.h>

#include "EpiRiskException.hpp"

#include "SmpSirSim.hpp"

namespace EpiRisk {
  namespace Smp {
SmpSirSim::SmpSirSim(SpatMetaPop& population) : population_(population),
						currTime_(0.0),
						totalInfecPres_(0.0)
{
  
  // Allocate rng;
  GetRNGstate();
}


SmpSirSim::~SmpSirSim()
{
  // Free the PRNG
  PutRNGstate();
}


const SmpParams* SmpSirSim::getParams() const
{
  return &params_;
}


void SmpSirSim::setParams(SmpParams& params)
{
  params_ = params;
}


void SmpSirSim::run(const size_t indexCommune)
{
  // Runs the simulation

  currTime_ = population_.getObsTime();
  
  // Reset population and indices
  infections_.clear();
  infecSN_ = 0;
  removalQueue.clear();

  // Set up simulation communes
  simCommunes.clear();
  SpatMetaPop::CommuneVector::const_iterator iter = population_.communes.begin();
  while (iter != population_.communes.end()) {
    simCommunes.push_back(iter->getPopSize());
    iter++;
  }
  
  // Check to make sure alpha is non-zero
  if(params_.alpha == 0.0) {
    throw param_exception("Alpha cannot be non-zero");
  }
  
  // Population counters
  numSusceptible=population_.getPopSize();
  numInfected=0;
  numRemoved=0;
  
  
  // Infect initial infection
  infect(InfectionEvent(currTime_,indexCommune));
  
  
  // Ready to Rock
  
  while(numInfected > 0) {
    
    
    // Draw time to next event
    InfectionEvent infectionEvent = chooseInfection();
    
    // Check for removals
    if(!removalQueue.empty() & (removalQueue.begin()->first < infectionEvent.time) )
      {
	// Perform removal
	currTime_ = removalQueue.begin()->first;
	remove();
      }
    else
      {
	// Perform infection
	if(numSusceptible == 0) continue; // Don't do anything if there are no susceptibles to infect
	currTime_ = infectionEvent.time;
	infect(infectionEvent);
      }
    
    Rprintf("\rS=%i, I=%i, R=%i                  ",numSusceptible,numInfected,numRemoved);
  }
    
}


void SmpSirSim::writeResults(const string filename)
{

  FILE* output;
  output = fopen(filename.c_str(),"w");
  if (output == NULL) {
    	string msg;
	msg = "Cannot open output file '" + filename + "' for writing";
	throw output_exception(msg.c_str());
  }

  
  fprintf(output,"\"Label\",\"Commune\",\"I\",\"R\"\n");

  SimInfections::iterator iter = infections_.begin();
  while(iter != infections_.end())
    {
      fprintf(output,"%i,%i,%f,%f\n",iter->label+1,iter->communeId+1,iter->I,iter->R);
      iter++;
    }
  
  fclose(output);
}



void SmpSirSim::infect(const InfectionEvent& infection)
{
	// Infects an individual in commune and updates the pressure index

	double removalTime = infection.time + rgamma(params_.a,1/params_.gamma);

	simCommunes.at(infection.communeId).numInfected++;
	simCommunes[infection.communeId].numSusceptible--;

	CommuneIndividual individual(infecSN_,
				     infection.communeId,
				     infection.time,
				     removalTime,
				     removalTime);
	infections_.push_back(individual);

	removalQueue.insert(RemovalQueue::value_type(removalTime,infection.communeId));

	numSusceptible--;
	numInfected++;
	infecSN_++;
}


void SmpSirSim::remove()
{
	// Set a removal time and removes infectious pressure

	size_t removalCommune = removalQueue.begin()->second;

	simCommunes[removalCommune].numInfected--;
	simCommunes[removalCommune].numRemoved++;

	removalQueue.erase(removalQueue.begin());

	numInfected--;
	numRemoved++;
}


InfectionEvent SmpSirSim::chooseInfection()
{
	// Returns the index of a commune in which the infection takes place

	map<double,const Commune*> communeCDF;
	double cumSum = 0.0;

	SpatMetaPop::CommuneVector::const_iterator commune = population_.communes.begin();
	while(commune != population_.communes.end()){

		double localSum = 0.0;

		localSum += params_.beta * simCommunes[commune->getId()].numInfected;

		Commune::ConVector::const_iterator connection = commune->getConnections()->begin();
		while(connection != commune->getConnections()->end())
		{
			localSum += params_.rho * simCommunes[*connection].numInfected;
			connection++;
		}

		//localSum *= commune->getSusceptibility();

		localSum += params_.alpha;

		localSum *= simCommunes[commune->getId()].numSusceptible;

		cumSum += localSum;

		communeCDF.insert(pair<double,const Commune*>(cumSum,&(*commune)));
		commune++;
	}

	map<double,const Commune*>::iterator chosen;

	double u = runif(0.0,cumSum);
	chosen = communeCDF.lower_bound(u);
	size_t indiv = chosen->second->getId();  // Indiv is actually a commune
	double t = currTime_ + rexp(1/cumSum);

	totalInfecPres_ = cumSum;

	return InfectionEvent(t,indiv);
}


double SmpSirSim::sampleInfecPeriod(const size_t communeId) {
  // Returns a draw from infectious periods
  return rgamma(params_.a,1/params_.gamma);
}




  } }
