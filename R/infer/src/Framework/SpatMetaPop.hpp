/*
 * VietPoultry.hpp
 *
 *  Created on: May 24, 2010
 *      Author: stsiab
 */

#ifndef SPATMETAPOP_HPP_
#define SPATMETAPOP_HPP_

#include "SmpTypes.hpp"
#include "SmpIndividual.hpp"
#include "EpiRiskException.hpp"

#include <string>
#include <set>
#include <utility>

namespace EpiRisk {

  namespace Smp {
// Infections data structure
class InfecComp {
public:
	bool operator()(const Individual& lhs, const Individual& rhs) const
	{ return lhs.I < rhs.I; };
};

typedef multiset<Individual,InfecComp> Infections;


class VietIndividual : public Individual
{
  // Extends Individual with commune covariate
public:
  VietIndividual(size_t myLabel, size_t commune, eventTime_t myI, eventTime_t myN,
		 eventTime_t myR, bool isKnown = 0, infecStatus_e _status = SUSCEPTIBLE) : Individual(myLabel,myI, myN,
												      myR,
												      isKnown,
												      _status),
											   commune_(commune) {};
  size_t getCommune() const { return commune_; };
private:
  size_t commune_;
};
  
  
  // Represents the population within a commune
  class Commune
  {
    
  public:
    typedef vector<size_t> ConVector;
    
    Commune(size_t ID, size_t popSize);
    
    size_t getPopSize() const;
    const ConVector* getConnections() const;
    size_t getId() const;
    void setAdjacency(ConVector& connections);
    size_t numInfecAt(const double time) const;
    size_t numRemovedAt(const double time) const;
    size_t numSuscepAt(const double time) const;
    double infectionTime(const double time) const;
    
    Infections::iterator moveInfectionTime(Infections::iterator toMove, double newTime);
    double getMaxR() const;
    double getMinI() const;
    
    const Individual* infect(const size_t label, const double time);
    void remove(Individual*, const double time);
    
    Infections infections;
    double infectiousPressure;
    
  private:
    size_t popSize_;
    ConVector connections_;
    size_t id_;
    
  };
  
  
  class SpatMetaPop {
  public:
    SpatMetaPop(const string populationDataFile,
		const string communeAdjGraph);
    
    virtual ~SpatMetaPop();

    void loadEpiData(const string filename);
    
    typedef vector<Commune> CommuneVector;
    CommuneVector communes;
    
    size_t numInfecAt(const double time) const;
    void updateIndexCase();
    Infections::iterator getIndexCase() const { return indexCase_; };
    double getObsTime() const { return obsTime_; };
    void setObsTime(const double obsTime) { obsTime_ = obsTime; };
    void updateObsTime();
    double getMeanI() const;
    size_t getNumInfecs() const { return numInfecs_; };
    size_t getPopSize() const { return popSize_; };
    const Individual* infect(const size_t commune, const double time);
    void remove(Individual* toRemove,const double time);
    void clearInfections(); // Wipes all infections
    
    void dumpInfectives(Commune& commune) const;
    
    
  private:
    void loadPopnData(const string filename);
    void loadCommuneAdjGraph(const string filename);
    
    double obsTime_;
    Infections::iterator indexCase_;
    size_t numInfecs_;
    size_t popSize_;
  };

  
  } }

#endif /* SPATMETAPOP_HPP_ */
