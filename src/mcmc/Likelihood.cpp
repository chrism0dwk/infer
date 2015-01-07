/*
 * Likelihood.cpp
 *
 *  Created on: 7/01/2015
 *      Author: cpjewell
 */


#include "Likelihood.hpp"

namespace EpiRisk
{

  Likelihood::Likelihood(PopDataImporter& population,
      EpiDataImporter& epidemic, const size_t nSpecies, const float obsTime,
      const float dLimit, const bool occultsOnlyDC, const int gpuId) :
      popSize_(0), numSpecies_(nSpecies), obsTime_(obsTime), I1Time_(0.0), I1Idx_(
          0), covariateCopies_(0), occultsOnlyDC_(occultsOnlyDC), movtBan_(obsTime)
  {
    // Load data into host memory
    LoadPopulation(population);
    LoadEpidemic(epidemic);
    SortPopulation();
  }



  void
  GpuLikelihood::LoadPopulation(PopDataImporter& importer)
  {
    idMap_.clear();
    hostPopulation_.clear();

    importer.open();
    try
      {
        size_t idx = 0;
        while (1)
          {
            PopDataImporter::Record record = importer.next();
            Covars covars;
            covars.id = record.id;
            covars.status = SUSC;
            covars.x = record.data.x;
            covars.y = record.data.y;
            covars.I = obsTime_; //EpiRisk::POSINF;
            covars.N = obsTime_; //EpiRisk::POSINF;
            covars.R = obsTime_; //EpiRisk::POSINF;
            covars.cattle = record.data.cattle;
            covars.pigs = record.data.pigs;
            covars.sheep = record.data.sheep;
            idMap_.insert(make_pair(covars.id, idx));
            idx++;
            hostPopulation_.push_back(covars);
          }
      }
    catch (EpiRisk::fileEOF& e)
      {
        // Continue -- this is harmless condition
      }
    catch (...)
      {
        importer.close();
        throw;
      }

    importer.close();
    const_cast<size_t &>(popSize_) = hostPopulation_.size();

    return;

  }

  void
  GpuLikelihood::LoadEpidemic(EpiDataImporter& importer)
  {
    maxInfecs_ = 0;

    importer.open();
    try
      {
        while (1)
          {
            EpiDataImporter::Record record = importer.next();
            map<string, size_t>::const_iterator map = idMap_.find(record.id);
            if (map == idMap_.end())
              {
                cerr << "idMap size: " << idMap_.size() << endl;
                string msg("Key '" + record.id + "' not found in population data");
                throw range_error(msg.c_str());
              }

            Population::iterator ref = hostPopulation_.begin() + map->second;
            // Check type
            if (record.data.I == EpiRisk::POSINF)
              ref->status = DC;
            else
              ref->status = IP;

            // Check data integrity
            if (record.data.N > record.data.R)
              {
                cerr << "Individual " << record.id
                    << " has N > R.  Setting N = R\n";
                record.data.N = record.data.R;
              }
            if (record.data.R < record.data.I
                and record.data.I != EpiRisk::POSINF)
              {
                cerr << "WARNING: Individual " << record.id
                    << " has I > R!  Setting I = R-7\n";
                record.data.I = record.data.R - 7;
              }

            ref->I = record.data.I;
            ref->N = record.data.N;
            ref->R = record.data.R;

            ref->R = min(ref->R, obsTime_);
            ref->N = min(ref->N, ref->R);
            ref->I = min(ref->I, ref->N);

            if (ref->status == IP and ref->I == ref->N)
              ref->I = ref->N - 14.0f; // Todo: Get rid of this hacky fix!!

            maxInfecs_++;
          }

      }
    catch (EpiRisk::fileEOF& e)
      {
        ;
      }
    catch (...)
      {
        throw;
      }

    if (!occultsOnlyDC_) maxInfecs_ = hostPopulation_.size();
    importer.close();

  }

  void
  GpuLikelihood::SortPopulation()
  {
    // Sort individuals by disease status (IPs -> DCs -> SUSCs)
    sort(hostPopulation_.begin(), hostPopulation_.end(), CompareByStatus());
    Covars cmp;
    cmp.status = DC;
    Population::iterator topOfIPs = lower_bound(hostPopulation_.begin(),
        hostPopulation_.end(), cmp, CompareByStatus());
    numKnownInfecs_ = topOfIPs - hostPopulation_.begin();
    sort(hostPopulation_.begin(), topOfIPs, CompareByI());

    // Set up occult susceptible vector
    for (size_t i = numKnownInfecs_; i < maxInfecs_; ++i)
      hostSuscOccults_->push_back(i);

    std::cout << "Population size: " << popSize_ << "\n";
    std::cout << "Num infecs: " << numKnownInfecs_ << "\n";
    std::cout << "Max infecs: " << maxInfecs_ << "\n";

    // Rebuild population ID index
    idMap_.clear();
    Population::const_iterator it = hostPopulation_.begin();
    for (size_t i = 0; i < hostPopulation_.size(); i++)
      {
        idMap_.insert(make_pair(it->id, i));
        it++;
      }

  }








}
