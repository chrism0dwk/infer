/*
 * SmpSirSim.hpp
 *
 *  Created on: May 31, 2010
 *      Author: stsiab
 */

#ifndef SIRSIM_HPP_
#define SIRSIM_HPP_

#include <map>
#include <list>
#include <utility>
#include <vector>
#include <queue>

#include "SmpTypes.hpp"
#include "SpatMetaPop.hpp"


#define DEFAULT_INFEC_PERIOD 6.0

namespace EpiRisk {
  namespace Smp {

struct InfectionEvent {
	double time;
	size_t communeId;
	InfectionEvent(const double time, const size_t communeId) : time(time),communeId(communeId) {};
};


class CommuneIndividual : public Individual {
	// Extends Individual to include commune
public:
	CommuneIndividual(size_t label,
					  size_t communeId,
					  double I,
					  double N,
					  double R) : Individual(label,
								 I,N,R,
								 true,
								 Individual::INFECTED), communeId(communeId) {};
  size_t communeId;
};


struct SimCommune {
	size_t numSusceptible;
	size_t numInfected;
	size_t numRemoved;
	SimCommune(size_t popSize) : numSusceptible(popSize), numInfected(0), numRemoved(0) {};
};


class SmpSirSim {
public:
	SmpSirSim(SpatMetaPop& population);
	virtual ~SmpSirSim();

	const SmpParams* getParams() const;
	void setParams(SmpParams& params);
	void setSeed(const unsigned long int seed);
	void run(const size_t indexCommune);
	void writeResults(const string filename);

private:
	const SpatMetaPop& population_;
	SmpParams params_;
	double currTime_;
	size_t numSusceptible, numInfected, numRemoved;
	double totalInfecPres_;
	double chosenPres_;
	size_t infecSN_;

	typedef list<CommuneIndividual> SimInfections;
	SimInfections infections_;

	typedef map<double,size_t> RemovalQueue;
	RemovalQueue removalQueue;

	typedef vector<SimCommune> SimCommunes;
	SimCommunes simCommunes;

	void infect(const InfectionEvent& infection);
	void remove();
	InfectionEvent chooseInfection();
	double sampleInfecPeriod(const size_t communeId);

};

  } }

#endif /* SIRSIM_HPP_ */
