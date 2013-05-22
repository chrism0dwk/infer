/*
 * VietPoultry.cpp
 *
 *  Created on: May 24, 2010
 *      Author: stsiab
 */

#include "SpatMetaPop.hpp"
#include "stlStrTok.hpp"
#include "Individual.hpp"

#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <R.h>
#include <Rmath.h>


namespace EpiRisk {
  namespace Smp {

Commune::Commune(size_t ID, size_t popSize) : popSize_(popSize),
					      id_(ID),
					      infectiousPressure(0.0)
{

}


const Commune::ConVector* Commune::getConnections() const
{
	return &connections_;
}


size_t Commune::getPopSize() const
{
	return popSize_;
}


void Commune::setAdjacency(ConVector& connections)
{
	connections_ = connections;
}


size_t Commune::numInfecAt(const double time) const
{
	// Returns the number infected at time

	size_t rv = 0;
	//Individual upperBound(0,time,time,time);
	//Infections::iterator upper = infections.upper_bound(upperBound);

	Infections::iterator iter = infections.begin();

	while(iter != infections.end()) {
		if (iter->I <= time && time < iter->R) rv++;
		iter++;
	}

	return rv;
}


size_t Commune::numRemovedAt(const double time) const
{
	// Returns the number of removed at time

	size_t rv=0;
	Infections::iterator iter = infections.begin();
	while(iter != infections.end()) {
		if (iter->R <= time) rv ++;
		iter++;
	}

	return rv;
}



size_t Commune::numSuscepAt(const double time) const
{
	// Returns the number of susceptible at time

	return popSize_ - numInfecAt(time) - numRemovedAt(time);
}



double Commune::infectionTime(const double time) const
{
	// Returns the total infection time up to time
	double infecInteg = 0.0;

	Infections::iterator iter = infections.begin();
	while(iter != infections.end()) {
	  infecInteg += min<double>(iter->R, time) - min<double>(iter->I,time);
		iter++;
	}

	return infecInteg;
}


size_t Commune::getId() const
{
	return id_;
}



const Individual* Commune::infect(const size_t label,const double time)
{
	// Infects an individual
	Individual myIndividual(label,time,R_PosInf,R_PosInf,true,Individual::INFECTED);
	Infections::iterator newInfec =  infections.insert(myIndividual);
	return &(*newInfec);
}



void Commune::remove(Individual* toRemove, const double time)
{
	// Sets a removal time
	toRemove->N = time;
	toRemove->R = time;
}



Infections::iterator Commune::moveInfectionTime(Infections::iterator toMove, double newTime)
{
	// Moves an infection time
	Individual tmpIndiv = *toMove; // Copy old infection
	infections.erase(toMove); // Delete old infection
	tmpIndiv.I = newTime;
	Infections::iterator newInfec = infections.insert(tmpIndiv);

	return newInfec;
}


double Commune::getMaxR() const
{
	// Returns the maximum removal time for any infecteds

	double maxTime = R_NegInf;
	Infections::iterator iter = infections.begin();
	while(iter != infections.end()) {
		maxTime = iter->R > maxTime ? iter->R : maxTime;
		iter++;
	}
	return maxTime;
}


SpatMetaPop::SpatMetaPop(const string populationDataFile, const string communeAdjGraph)
{
	loadPopnData(populationDataFile);
	loadCommuneAdjGraph(communeAdjGraph);
}

SpatMetaPop::~SpatMetaPop() {
	// TODO Auto-generated destructor stub
}


const Individual* SpatMetaPop::infect(const size_t commune, const double time)
{
	// Infects an individual in a commune
	const Individual* newIndiv = communes.at(commune).infect(numInfecs_,time);
	numInfecs_++;
	return newIndiv;
}


void SpatMetaPop::remove(Individual* toRemove,const double time)
{
	// Removes an individual
	// Currently does nothing.  Be Warned!
}


void SpatMetaPop::clearInfections()
{
	// Clears all infections
	CommuneVector::iterator commune = communes.begin();
	while(commune != communes.end())
	{
		commune->infections.clear();
		commune++;
	}

	numInfecs_ = 0;
}



void SpatMetaPop::loadPopnData(const string filename) {
	// Initialises the commune vector

	// File format: <commune id>,<poultry holding count>,<total head poultry>

	vector<string> tokens;
	string buffer;
	size_t totalPopSize = 0;


	ifstream dataFile;
	dataFile.open(filename.c_str(),ios::in);
	if(!dataFile.is_open()) {
		string msg = "Cannot open population data file '" + filename + "' for reading";
		throw data_exception(msg.c_str());
	}

	getline(dataFile,buffer);  // First row has col headings.  Ignore.

	while(!dataFile.eof()) {
		getline(dataFile,buffer);
		stlStrTok(tokens,buffer,",");
		if(tokens.size() < 2) break;
		size_t popSize = atoi(tokens[1].c_str());
		totalPopSize += popSize;
		size_t communeId = atoi(tokens[0].c_str()) - 1; // Count from 0
		communes.push_back(Commune(communeId,popSize));
	}

	dataFile.close();

	popSize_ = totalPopSize;

}


void SpatMetaPop::loadCommuneAdjGraph(const string filename) {
	// Loads commune adjacency data in GMRF format

	vector<string> tokens;
	string buffer;
	Commune::ConVector tmpGraph;

	ifstream dataFile;
	dataFile.open(filename.c_str(),ios::in);
	if(!dataFile.is_open()) {
		throw data_exception(string("Cannot open commune graph file:" + filename).c_str());
	}

	getline(dataFile, buffer);
	if(strtoul(buffer.c_str(),NULL,0) != communes.size()) throw data_exception("Number of individuals specified in graph file does not match population size");

	while(!dataFile.eof()) {
		getline(dataFile,buffer);
		stlStrTok(tokens,buffer," ");
		if(tokens.size() < 2) break;
		tmpGraph.clear();

		for(int i=2; i<atoi(tokens[1].c_str())+2; ++i) {
			tmpGraph.push_back(atoi(tokens[i].c_str()) - 1);
		}
		communes.at(atoi(tokens[0].c_str())-1).setAdjacency(tmpGraph);
	}

	dataFile.close();

}


void SpatMetaPop::loadEpiData(const string filename) {
	// Loads epidemic data in <indiv ID> <commune ID> <infection time> <detection time> <cull time> ...

	vector<string> tokens;
	string buffer;

	ifstream dataFile;
	dataFile.open(filename.c_str(),ios::in);
	if(!dataFile.is_open()) {
		throw data_exception(string("Cannot open epidemic file: " + filename).c_str());
	}

	getline(dataFile, buffer); // Ignore file headings

	size_t numInfecs = 0;
	while(!dataFile.eof()) {
		getline(dataFile, buffer);
		stlStrTok(tokens, buffer, ",");
		if(tokens.size() < 5) break;

		size_t myCommune = atoi(tokens[1].c_str()) - 1;
		size_t label = atoi(tokens[0].c_str());
		double I = atof(tokens[2].c_str());
		double N = atof(tokens[3].c_str());
		double R = atof(tokens[4].c_str());

		Individual tmpInfec(label,I,N,R,true,Individual::INFECTED);

		communes.at(myCommune).infections.insert(tmpInfec);
		numInfecs++;
	}

	dataFile.close();

	numInfecs_ = numInfecs;
	updateIndexCase();
	updateObsTime();
}


void SpatMetaPop::updateIndexCase()
{
	// Finds and caches the index time

	double indexTime = R_PosInf;
	Infections::iterator indexCase;

	CommuneVector::iterator iter = communes.begin();
	while(iter != communes.end()) {
		if(!iter->infections.empty()) {
			if(iter->infections.begin()->I < indexTime) {
				indexCase = iter->infections.begin();
				indexTime = indexCase->I;
			}
		}
		iter++;
	}

	indexCase_ = indexCase;
}


void SpatMetaPop::updateObsTime()
{
	// Updates the observation time

	double maxTime = R_NegInf;

	CommuneVector::iterator iter = communes.begin();
	while(iter != communes.end()) {
		if(!iter->infections.empty()) {
			maxTime = iter->getMaxR() > maxTime ? iter->getMaxR() : maxTime;
		}
		iter++;
	}

	obsTime_ = maxTime;
}


double SpatMetaPop::getMeanI() const
{
	// Returns the mean infection to notification time

	size_t numI = 0;
	double sumTime = 0.0;

	CommuneVector::const_iterator commune = communes.begin();
	while(commune != communes.end()) {

		numI += commune->infections.size();

		Infections::iterator infection = commune->infections.begin();
		while(infection != commune->infections.end()) {
			sumTime += infection->R - infection->I;
			infection++;
		}

		commune++;
	}

	return sumTime / numI;
}


void SpatMetaPop::dumpInfectives(Commune& commune) const
{
	// Dumps infectives in a commune to stdout
	Infections::iterator iter = commune.infections.begin();
	while(iter != commune.infections.end()) {
		cout << iter->label << " (" << commune.getId() << ": " << iter->I << ", " << iter->N << ", " << iter->R << endl;
		iter++;
	}
}




  } }

