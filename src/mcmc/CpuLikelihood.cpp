/*
 * Gpulikelihood.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stsiab
 */
#include <stdexcept>
#include <string>
#include <cstddef>
#include <iostream>
#include <sstream>
#include <vector>
#include <utility>
#include <cmath>
#include <cassert>
#include <sys/time.h>
#include <gsl/gsl_cdf.h>
#include <boost/numeric/ublas/operation.hpp>
#include <omp.h>
#include "CpuLikelihood.hpp"

namespace EpiRisk
{
// Constants
  const float UNITY = 1.0;
  const float ZERO = 0.0;
  
  float
  GetDistElement(const CsrMatrix* d, const int row, const int col) {
    assert(row < d->n);
    assert(col < d->m);
    
    int start = d->rowPtr[row];
    int end = d->rowPtr[row+1];
    for(int j = start; j<end; ++j)
      if (d->colInd[j] == col) return d->val[j];
    return EpiRisk::POSINF;
  }


  inline
  float
  timeinseconds(const timeval a, const timeval b)
  {
    timeval result;
    timersub(&b, &a, &result);
    return result.tv_sec + result.tv_usec / 1000000.0;
  }


  bool
  getDistMatrixElement(const int row, const int col, const CsrMatrix* csrMatrix, float* val)
  {
    int* cols = csrMatrix->colInd + csrMatrix->rowPtr[row];
    float* vals = csrMatrix->val + csrMatrix->rowPtr[row];
    int rowlen = csrMatrix->rowPtr[row+1] - csrMatrix->rowPtr[row];

    for(int ptr=0; ptr<rowlen; ++ptr)
      {
        if(cols[ptr] == col) {
          *val = vals[ptr];
          return true;
        }
      }
    return false;
  }

  inline
  fp_t
  h(const float t, float nu, float alpha)
  {
    return t < alpha ? 0.0f : 1.0f;
  }

  inline
  fp_t
  H(const float t, const float nu, const float alpha)
  {
    float integral = t-alpha;
    return fmaxf(0.0f, integral);
  }

  inline
  fp_t
  K(const float dsq, const float delta, const float omega)
  {
    return delta / powf(delta*delta + dsq, omega);
  } 


  CpuLikelihood::CpuLikelihood(PopDataImporter& population,
      EpiDataImporter& epidemic, const size_t nSpecies, const float obsTime,
      const float dLimit, const bool occultsOnlyDC) :
      popSize_(0), numSpecies_(nSpecies), obsTime_(obsTime), I1Time_(0.0), I1Idx_(
          0), occultsOnlyDC_(occultsOnlyDC), movtBan_(obsTime)
  {

    // Load data into host memory
    LoadPopulation(population);
    LoadEpidemic(epidemic);
    SortPopulation();
    int np = omp_get_num_threads();
    omp_set_num_threads(8);
    CalcDistanceMatrix(dLimit);
    omp_set_num_threads(np);

    // Set up species and events
    SetSpecies();
    SetEvents();

    // Allocate product cache
    productCache_.resize(maxInfecs_);
    fill(productCache_.begin(), productCache_.end(), 1.0f);

    // Components
    likComponents_.bgIntegral = 0.0f;
    likComponents_.integral = 0.0f;
    likComponents_.sumI = 0.0f;
    likComponents_.logProduct = 0.0f;


#ifndef NDEBUG
    cerr << "ObsTime: " << obsTime_ << endl;
#endif

  }


  CpuLikelihood::~CpuLikelihood()
  {

  }


  void
  CpuLikelihood::LoadPopulation(PopDataImporter& importer)
  {
    idMap_.clear();
    population_.clear();

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
            population_.push_back(covars);
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
    const_cast<size_t &>(popSize_) = population_.size();

    return;

  }



  void
  CpuLikelihood::LoadEpidemic(EpiDataImporter& importer)
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

            Population::iterator ref = population_.begin() + map->second;
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

    if (!occultsOnlyDC_) maxInfecs_ = population_.size();
    importer.close();
  }

  void
  CpuLikelihood::SortPopulation()
  {
    // Sort individuals by disease status (IPs -> DCs -> SUSCs)
    sort(population_.begin(), population_.end(), CompareByStatus());
    Covars cmp;
    cmp.status = DC;
    Population::iterator topOfIPs = lower_bound(population_.begin(),
        population_.end(), cmp, CompareByStatus());
    numKnownInfecs_ = topOfIPs - population_.begin();
    sort(population_.begin(), topOfIPs, CompareByI());

    // Set up occult susceptible vector
    suscOccults_.resize(maxInfecs_ - numKnownInfecs_);
    for (size_t i = numKnownInfecs_; i < maxInfecs_; ++i)
      suscOccults_(i-numKnownInfecs_) = i;

    std::cout << "Population size: " << popSize_ << "\n";
    std::cout << "Num infecs: " << numKnownInfecs_ << "\n";
    std::cout << "Max infecs: " << maxInfecs_ << "\n";

    // Rebuild population ID index
    idMap_.clear();
    Population::const_iterator it = population_.begin();
    for (size_t i = 0; i < population_.size(); i++)
      {
        idMap_.insert(make_pair(it->id, i));
        it++;
      }
  }


  void
  CpuLikelihood::SetMovtBan(const float movtBan)
  {
    movtBan_ = movtBan;
  }


  void
  CpuLikelihood::SetParameters(Parameter& epsilon1, Parameter& epsilon2, Parameter& gamma1,
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

  }

  void
  CpuLikelihood::CalcDistanceMatrix(const float dLimit)
  {
    float dLimitSq = dLimit*dLimit;
    D_.resize(maxInfecs_,popSize_,false);
    for (size_t i = 0; i < maxInfecs_; ++i)
      {
	float xi = population_[i].x;
	float yi = population_[i].y;
	ublas::vector<float> denserow(popSize_);
#pragma omp parallel for shared(denserow)
	for (size_t j = 0; j < popSize_; ++j) {
	  float xj = population_[j].x;
	  float yj = population_[j].y;
	  float dx = xi - xj;
	  float dy = yi - yj;
	  float dsq = dx*dx + dy*dy;
	  denserow(j) = 0.0f < dsq and dsq <= dLimitSq ? dsq : 0.0f;
	}
	ublas::matrix_row< ublas::compressed_matrix<fp_t> > row(D_,i);
	row = denserow;
      }

    cout << "D sparsity = " << D_.nnz() << endl;
  }

  void
  CpuLikelihood::SetEvents()
  {

    // Set up Species and events
    eventTimes_.resize(popSize_, NUMEVENTS);
    Population::iterator it = population_.begin();
    for (size_t i = 0; i < popSize_; ++i)
      {
        eventTimes_(i,0) = it->I;
	eventTimes_(i,1) = it->N;
        eventTimes_(i,2) = it->R;
	++it;
      }

    infecIdx_.resize(numKnownInfecs_);
    for (size_t i = 0; i < numKnownInfecs_; ++i)
      {
        infecIdx_(i)=i;
      }
  }

  void
  CpuLikelihood::SetSpecies()
  {

    // Set up Species and events
    animals_.resize(popSize_,numSpecies_);
    Population::const_iterator it = population_.begin();
    int i=0;
    for (it = population_.begin(); it != population_.end(); ++it)
      {
	animals_(i,0) = it->cattle;
	if(numSpecies_ > 1)
	  animals_(i,1) = it->pigs;
	if(numSpecies_ > 2)
	  animals_(i,2) = it->sheep;
	++i;
      }

    susceptibility_.resize(popSize_);
    animalsSuscPow_.resize(popSize_,numSpecies_);
    infectivity_.resize(maxInfecs_);
    animalsInfPow_.resize(maxInfecs_,numSpecies_);
  }

  inline
  void
  CpuLikelihood::CalcInfectivityPow()
  {

    for(int i=0; i<maxInfecs_; ++i)
      {
	for(int k=0; k<numSpecies_; ++k)
	  animalsInfPow_(i,k) = powf(animals_(i,k),psi_[k]);
      }
  }
    
  inline
  void
  CpuLikelihood::CalcInfectivity()
  {
    ublas::vector<float> xi(3);
    for(int i=0; i<numSpecies_;++i)
      xi(i) = xi_[i];

    // Now calculate infectivity
    //ublas::axpy_prod(animalsInfPow_, xi, infectivity_,false);
#pragma omp parallel for
    for(int i = 0; i<maxInfecs_; ++i) {
      float infectivity = 0.0f;
      
      for(int j = 0; j<numSpecies_; ++j)
	infectivity += xi(j) * animalsInfPow_(i,j);
     
      infectivity_(i) = infectivity;
    }
  }

  inline
  void
  CpuLikelihood::CalcSusceptibilityPow()
  {
    for(int i=0; i<popSize_; ++i)
      {
	for(int k=0; k<numSpecies_; ++k)
	  animalsSuscPow_(i,k) = powf(animals_(i,k),phi_[k]);
      }
  }

  inline
  void
  CpuLikelihood::CalcSusceptibility()
  {
    // Calculates susceptibility powers and sums over suscept.
    ublas::vector<float> zeta(3);
    for(int i=0; i<numSpecies_; ++i)
      zeta(i) = zeta_[i];
    //ublas::axpy_prod(animalsSuscPow_, zeta, susceptibility_, false);
#pragma omp parallel for
    for(int i = 0; i<popSize_; ++i) {
      float susc = 0.0f;
      
      for(int j = 0; j<numSpecies_; ++j)
	susc += zeta(j) * animalsSuscPow_(i,j);
     
      susceptibility_[i] = susc;
    }


  }

  inline
  void
  CpuLikelihood::UpdateI1()
  {
    I1Idx_ = 0;
    for(int i=0; i<infecIdx_.size(); ++i)
      if(eventTimes_(infecIdx_(i).ptr,0) < eventTimes_(infecIdx_(I1Idx_).ptr,0)) I1Idx_=i;
    I1Time_ = eventTimes_(I1Idx_,0);

  }


  ////////////////////////////////////////////////////////////////////////////////////////////////
  inline
  void
  CpuLikelihood::CalcBgIntegral()
  {
    float res = 0.0f;
#pragma omp parallel for reduction(+:res)
    for(int i=0; i<popSize_; ++i)
      {
	float I = eventTimes_(i,0);
	res += *epsilon1_ * max((min(I, movtBan_) - I1Time_),0.0f);
        res += *epsilon1_ * *epsilon2_ * max(I - max(movtBan_, I1Time_),0.0f);
      }
    likComponents_.bgIntegral = res;

  }


  inline
  void
  CpuLikelihood::ReduceProductVector()
  {
    float res = 0.0f;
    productCache_(infecIdx_(I1Idx_).ptr) = 1.0f;
#pragma omp parallel for reduction(+:res)
    for(int i=0; i<productCache_.size(); ++i)
      {
	res += logf(productCache_(i));
      }

    likComponents_.logProduct = res;
  }

  inline
  void
  CpuLikelihood::CalcProduct()
  {
#pragma omp parallel for
    for(int jj=0; jj<infecIdx_.size(); ++jj)
      {
	int j = infecIdx_(jj).ptr;
	int begin = D_.index1_data()[j];
	int end = D_.index1_data()[j+1];

	float Ij = eventTimes_(j,0);
	float sumPressure = 0.0f;
	for(int ii=begin; ii<end; ++ii) {
	  int i = D_.index2_data()[ii];
	  float Ii = eventTimes_(i,0);
	  float Ni = eventTimes_(i,1);
	  float Ri = eventTimes_(i,2);
	  float idxOnj = 0.0f;
	  if(Ii < Ni)
	    {
	      if (Ii < Ij and Ij <= Ni)
		idxOnj += h(Ij - Ii, *nu_, *alpha_);
	      else if (Ni < Ij and Ij <= Ri)
		idxOnj += *gamma2_ * h(Ij - Ii, *nu_, *alpha_);
	    }
	  sumPressure += idxOnj;
	}
	float epsilon = *epsilon1_;
	epsilon *= Ij < movtBan_ ? 1.0f : *epsilon2_;
	productCache_[j] = sumPressure * *gamma1_ + epsilon;
      }

    ReduceProductVector();
  }

  inline
  void
  CpuLikelihood::CalcIntegral()
  {
    float res = 0.0f;
#pragma omp parallel for reduction(+:res)
    for(int ii=0; ii<infecIdx_.size(); ++ii)
      {
	int i = infecIdx_(ii).ptr;
	int begin = D_.index1_data()[i];
	int end = D_.index1_data()[i+1];
	float Ii = eventTimes_(i,0);
	float Ni = eventTimes_(i,1);
	float Ri = eventTimes_(i,2);
	float pressureFromI = 0.0f;
	for(int jj = begin; jj < end; jj++)
	  {
	    float Ij = eventTimes_(D_.index2_data()[jj],0);
	    float betaij = H(fminf(Ni, Ij) - fminf(Ii, Ij), *nu_, *alpha_);
	    betaij += *gamma2_ * (H(fminf(Ri, Ij) - Ii,*nu_,*alpha_));
	    betaij *= K(D_.value_data()[jj], *delta_, *omega_);
	    betaij *= susceptibility_[D_.index2_data()[jj]];
	    pressureFromI += betaij;
	  }

	res += pressureFromI * infectivity_[i];
      }

    likComponents_.integral = res;
  }

  void
  CpuLikelihood::FullCalculate()
  {

    CalcInfectivityPow();
    CalcInfectivity();
    CalcSusceptibilityPow();
    CalcSusceptibility();

    UpdateI1();
    CalcIntegral();
    CalcProduct();
    CalcBgIntegral();

    logLikelihood_ = likComponents_.logProduct
        - (likComponents_.integral* *gamma1_ + likComponents_.bgIntegral);


#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
    << timeinseconds(start, end) << std::endl;
    std::cerr << "Likelihood (" << __PRETTY_FUNCTION__ << "): " << logLikelihood_
    << std::endl;
#endif


  }

  void
  CpuLikelihood::Calculate()
  {
#ifdef GPUTIMING
    timeval start, end;
    gettimeofday(&start, NULL);
#endif

    CalcInfectivity();
    CalcSusceptibility();

    UpdateI1();
    CalcIntegral();
    CalcProduct();
    CalcBgIntegral();

    likComponents_.integral;

    logLikelihood_ = likComponents_.logProduct
        - (likComponents_.integral* *gamma1_ + likComponents_.bgIntegral);

#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
    << timeinseconds(start, end) << std::endl;
#endif

  }



  float
  CpuLikelihood::GetLogLikelihood() const
  {
    return logLikelihood_;
  }


  void
  CpuLikelihood::PrintLikelihoodComponents() const
  {
    cout << "Background: " << likComponents_.bgIntegral << "\n";
    cout << "Integral: " << likComponents_.integral << "\n";
    cout << "Product: " << likComponents_.logProduct << "\n";
  }

  void CpuLikelihood::PrintParameters() const
  {
    cerr << "Epsilon1,2: " << *epsilon1_ << ", " << *epsilon2_ << "\n";
    cerr << "Gamma1,2: " << *gamma1_ << ", " << *gamma2_ << "\n";
    cerr << "Delta: " << *delta_ << "\n";
    cerr << "Omega: " << *omega_ << "\n";
    for(int i = 0; i<numSpecies_; ++i) cerr << "Xi,Zeta,Phi,Psi[" << i << "]: " << xi_[i] << ", " << zeta_[i] << ", " << phi_[i] << ", " << psi_[i] << "\n";
    cerr << "alpha: " << *alpha_ << "\n";
    cerr << "a: " << *a_ << "\n";
    cerr << "b: " << *b_ << endl;
    cerr << "ObsTime: " << obsTime_ << "\n";
    cerr << "I1Idx = " << I1Idx_ << "\n";
    cerr << "I1Time = " << I1Time_ << "\n";
  }

  void CpuLikelihood::PrintProdCache() const
  {
    for(int i=0; i<productCache_.size(); ++i)
      {
	cout << population_[i].id << ": " << productCache_(i) << "\n";
      }
  }


} // namespace EpiRisk

