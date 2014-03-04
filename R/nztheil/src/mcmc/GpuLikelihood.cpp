/*************************************************************************
 *  ./src/mcmc/GpuLikelihood.cpp
 *  Copyright Chris Jewell <chrism0dwk@gmail.com> 2012
 *
 *  This file is part of InFER.
 *
 *  InFER is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  InFER is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with InFER.  If not, see <http://www.gnu.org/licenses/>.
 *************************************************************************/
/*
 * GpuLikelihood.cpp
 *
 *  Created on: Mar 19, 2012
 *      Author: stsiab
 */

#include <iostream>
#include <fstream>

#include <boost/numeric/ublas/matrix_sparse.hpp>
using namespace boost::numeric;

#include "Data.hpp"
#include "GpuLikelihood.hpp"

namespace EpiRisk
{

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

  void
  GpuLikelihood::LoadDistanceMatrix(DistMatrixImporter& importer)
  {
    ublas::mapped_matrix<float>* Dimport = new ublas::mapped_matrix<float>(
        maxInfecs_, hostPopulation_.size());
    try
      {
        importer.open();
        while (1)
          {
            DistMatrixImporter::Record record = importer.next();
            map<string, size_t>::const_iterator i = idMap_.find(record.id);
            map<string, size_t>::const_iterator j = idMap_.find(record.data.j);
            if (i == idMap_.end() or j == idMap_.end())
              throw range_error("Key pair not found in population");

            if (i != j
                and i->second < maxInfecs_ /* Don't require distances with i known susc */)
              try
                {
                  Dimport->operator()(i->second, j->second) =
                      record.data.distance * record.data.distance;
                }
              catch (std::exception& e)
                {
                  cerr << "Inserting distance |" << i->second << " - "
                      << j->second << "| = "
                      << record.data.distance * record.data.distance
                      << " failed" << endl;
                  throw e;
                }
          }
      }
    catch (EpiRisk::fileEOF& e)
      {
        cout << "Imported " << Dimport->nnz() << " distance elements" << endl;
      }
    catch (exception& e)
      {
        throw e;
      }

    // Set up distance matrix
    dnnz_ = Dimport->nnz();
    ublas::compressed_matrix<float>* D = new ublas::compressed_matrix<float>(
        *Dimport);
    int* rowPtr = new int[D->index1_data().size()];
    for (size_t i = 0; i < D->index1_data().size(); ++i)
      rowPtr[i] = D->index1_data()[i];
    int* colInd = new int[D->index2_data().size()];
    for (size_t i = 0; i < D->index2_data().size(); ++i)
      colInd[i] = D->index2_data()[i];
    SetDistance(D->value_data().begin(), rowPtr, colInd);
    delete[] rowPtr;
    delete[] colInd;
    delete D;
    delete Dimport;
  }


  void
  GpuLikelihood::SetParameters(Parameter& epsilon1, Parameter& epsilon2, Parameter& gamma1,
      Parameter& gamma2, Parameters& xi, Parameters& psi, Parameters& zeta,
			       Parameters& phi, Parameter& delta, Parameter& omega, Parameter& nu, Parameter& alpha, Parameter& a, Parameter& b)
  {

    epsilon1_ = epsilon1.GetValuePtr();
    epsilon2_ = epsilon2.GetValuePtr();
    gamma1_ = gamma1.GetValuePtr();
    gamma2_ = gamma2.GetValuePtr();
    delta_ = delta.GetValuePtr();
    omega_ = omega.GetValuePtr();
    nu_ = nu.GetValuePtr();
    alpha_ = alpha.GetValuePtr();
    a_ = a.GetValuePtr();
    b_ = b.GetValuePtr();

    xi_.clear();
    psi_.clear();
    zeta_.clear();
    phi_.clear();
    for (size_t p = 0; p < numSpecies_; ++p)
      {
        xi_.push_back(xi[p].GetValuePtr());
        psi_.push_back(psi[p].GetValuePtr());
        zeta_.push_back(zeta[p].GetValuePtr());
        phi_.push_back(phi[p].GetValuePtr());
      }

    RefreshParameters();
  }

  void
  GpuLikelihood::SetMovtBan(const float movtBanTime)
  {
	  movtBan_ = movtBanTime;
  }

  float
  GpuLikelihood::GetMovtBan() const
  {
	  return movtBan_;
  }

  size_t
  GpuLikelihood::GetNumInfecs() const
  {
    return hostInfecIdx_->size();
  }

  size_t
  GpuLikelihood::GetNumKnownInfecs() const
  {
    return numKnownInfecs_;
  }

  size_t
  GpuLikelihood::GetMaxInfecs() const
  {
    return maxInfecs_;
  }

  size_t
  GpuLikelihood::GetNumPossibleOccults() const
  {
    return hostSuscOccults_->size();
  }

  size_t
  GpuLikelihood::GetPopulationSize() const
  {
	  return popSize_;
  }

  void
  GpuLikelihood::GetIds(std::vector<std::string>& ids) const
  {
	  ids.resize(popSize_);
	  for(size_t i=0; i<popSize_; ++i)
		  ids[i] = hostPopulation_[i].id;
  }

  size_t
  GpuLikelihood::GetNumOccults() const
  {
    return hostInfecIdx_->size() - numKnownInfecs_;
  }

  
  

} // namespace EpiRisk
