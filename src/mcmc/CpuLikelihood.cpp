/*
 * Gpulikelihood.cpp
 *
 *  Created on: Feb 13, 2012
 *      Author: stsiab
 */
#include <stdexcept>
#include <string>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>
#include <numeric>
#include <utility>
#include <cmath>
#include <cassert>
#include <sys/time.h>
#include <gsl/gsl_cdf.h>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#ifdef HAVEOMP
#include <omp.h>
#endif

#include "CpuLikelihood.hpp"


#define VECSIZE 8

namespace EpiRisk
{

  float
  sumPairwise(const float* x, const size_t n)
  {
    float s;
    if (n <= 16)
      {
        s = 0.0f;
        for (size_t i = 0; i < n; ++i)
          s += x[i];
      }
    else
      {
        size_t m = n / 2;
        s = sumPairwise(x, m) + sumPairwise(x + m, n - m);
      }
    return s;
  }

  inline fp_t
  h(const float t, float nu, float alpha)
  {
    return t < alpha ? 0.0f : 1.0f;
  }

  inline fp_t
  H(const float t, const float nu, const float alpha)
  {
    float integral = t - alpha;
    return fmaxf(0.0f, integral);
  }

  inline fp_t
  K(const float dsq, const float delta, const float omega)
  {
    return delta / powf(delta * delta + dsq, omega);
  }

  CpuLikelihood::CpuLikelihood(PopDataImporter& population,
      EpiDataImporter& epidemic, const size_t nSpecies, const float obsTime,
      const float dLimit, const bool occultsOnlyDC) :
    Likelihood(population, epidemic, nSpecies, obsTime, occultsOnlyDC),
      I1Time_(0.0), I1Idx_(0)
  {

    // Load data into host memory
#ifdef HAVEOMP
    int np;
#pragma omp parallel
      {
        np = omp_get_num_threads();
      }
    cout << "Using " << np << " threads" << endl;
    omp_set_num_threads(8);
#endif

    CalcDistanceMatrix(dLimit);

#ifdef HAVEOMP
    omp_set_num_threads(np);
#endif

    // Set up species and events
    SetSpecies();
    SetEvents();

    // Set up occult susceptible vector
    suscOccults_.resize(maxInfecs_ - numKnownInfecs_);
    for (size_t i = numKnownInfecs_; i < maxInfecs_; ++i)
      suscOccults_[i - numKnownInfecs_] = i;

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

  CpuLikelihood::CpuLikelihood(const CpuLikelihood& other) :
    Likelihood(other),
    I1Time_(other.I1Time_),
    I1Idx_(other.I1Idx_),
    infecIdx_(other.infecIdx_),
    suscOccults_(other.suscOccults_),
    productCache_(other.productCache_),
    likComponents_(other.likComponents_),
    logLikelihood_(other.logLikelihood_),
    animals_(other.animals_),
    D_(other.D_),
    dnnz_(other.dnnz_),
    animalsInfPow_(other.animalsInfPow_),
    animalsSuscPow_(other.animalsSuscPow_),
    eventTimes_(other.eventTimes_),
    susceptibility_(other.susceptibility_),
    infectivity_(other.infectivity_)
  {
    
  }

  CpuLikelihood*
  CpuLikelihood::clone() const
  {
    return new CpuLikelihood(*this);
  }

  const Likelihood&
  CpuLikelihood::assign(const Likelihood& other)
  {
    const CpuLikelihood& x = static_cast<const CpuLikelihood&>(other);
    animalsInfPow_ = x.animalsInfPow_;
    animalsSuscPow_ = x.animalsSuscPow_;

    memcpy(&eventTimes_(0,0), &x.eventTimes_(0,0), sizeof(fp_t)*popSize_);
    susceptibility_ = x.susceptibility_;
    infectivity_ = x.infectivity_;
    infecIdx_ = x.infecIdx_;
    suscOccults_ = x.suscOccults_;
    I1Idx_ = x.I1Idx_;
    I1Time_ = x.I1Time_;
    productCache_ = x.productCache_;
    likComponents_ = x.likComponents_;
    logLikelihood_ = x.logLikelihood_;
    return *this;
  }

  void
  CpuLikelihood::InfecCopy(const Likelihood& other)
  {
    const CpuLikelihood& x = static_cast<const CpuLikelihood&>(other);
    eventTimes_ = x.eventTimes_;
    infecIdx_ = x.infecIdx_;
    suscOccults_ = x.suscOccults_;
    I1Idx_ = x.I1Idx_;
    I1Time_ = x.I1Time_;
    productCache_ = x.productCache_;
    likComponents_ = x.likComponents_;
    logLikelihood_ = x.logLikelihood_;
  }
    



  CpuLikelihood::~CpuLikelihood()
  {
  }

  void
  CpuLikelihood::CalcDistanceMatrix(const float dLimit)
  {
    float dLimitSq = dLimit * dLimit;
    D_.resize(maxInfecs_, popSize_, false);
    for (size_t i = 0; i < maxInfecs_; ++i)
      {
        float xi = population_[i].x;
        float yi = population_[i].y;
        ublas::vector<float> denserow(popSize_);
#pragma omp parallel for shared(denserow)
        for (size_t j = 0; j < popSize_; ++j)
          {
            float xj = population_[j].x;
            float yj = population_[j].y;
            float dx = xi - xj;
            float dy = yi - yj;
            float dsq = dx * dx + dy * dy;
            denserow(j) = 0.0f < dsq and dsq <= dLimitSq ? dsq : 0.0f;
          }
        ublas::matrix_row<ublas::compressed_matrix<fp_t> > row(D_, i);
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
        eventTimes_(i, 0) = it->I;
        eventTimes_(i, 1) = it->N;
        eventTimes_(i, 2) = it->R;
        ++it;
      }
    infecIdx_.resize(numKnownInfecs_);
    for (size_t i = 0; i < numKnownInfecs_; ++i)
      {
        infecIdx_[i] = i;
      }
  }

  void
  CpuLikelihood::SetSpecies()
  {

    // Set up Species and events
    animals_.resize(popSize_, numSpecies_);

    Population::const_iterator it = population_.begin();
    int i = 0;
    for (it = population_.begin(); it != population_.end(); ++it)
      {
        animals_(i, 0) = it->cattle;
        if (numSpecies_ > 1)
          animals_(i, 1) = it->pigs;
        if (numSpecies_ > 2)
          animals_(i, 2) = it->sheep;
        ++i;
      }

    susceptibility_.resize(popSize_);
    animalsSuscPow_.resize(popSize_, numSpecies_);

    infectivity_.resize(maxInfecs_);
    animalsInfPow_.resize(maxInfecs_, numSpecies_);
  }

  inline
  void
  CpuLikelihood::CalcInfectivityPow()
  {
#pragma omp parallel for
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
#pragma omp parallel for
    for(int i = 0; i<maxInfecs_; ++i) {
      float infectivity = 0.0f;      
      for(int j = 0; j<numSpecies_; ++j)
	infectivity += xi_[j] * animalsInfPow_(i,j);
     
      infectivity_(i) = infectivity;
    }
  }


  inline
  void
  CpuLikelihood::CalcSusceptibilityPow()
  {
#pragma omp parallel for
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
#pragma omp parallel for
    for(int i = 0; i<popSize_; ++i) {
      float susc = 0.0f;
      
      for(int j = 0; j<numSpecies_; ++j)
	susc += zeta_[j] * animalsSuscPow_(i,j);
     
      susceptibility_[i] = susc;
    }
  }

  inline
  void
  CpuLikelihood::UpdateI1()
  {
    I1Idx_ = 0;
    for (int i = 0; i < infecIdx_.size(); ++i)
      if (eventTimes_(infecIdx_[i].ptr, 0)
          < eventTimes_(infecIdx_[I1Idx_].ptr, 0))
        I1Idx_ = i;
    I1Time_ = eventTimes_(I1Idx_, 0);

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
    productCache_(infecIdx_[I1Idx_].ptr) = 1.0f;
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
	int j = infecIdx_[jj].ptr;
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
	  sumPressure += idxOnj * infectivity_[i] * K(D_.value_data()[ii], *delta_, *omega_);
	}
	float epsilon = *epsilon1_;
	epsilon *= Ij < movtBan_ ? 1.0f : *epsilon2_;
	productCache_[j] = sumPressure * susceptibility_[j] * *gamma1_ + epsilon;
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
	int i = infecIdx_[ii].ptr;
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
	    betaij += *gamma2_ * (H(fminf(Ri, Ij) - Ii,*nu_,*alpha_)
				  - H(fminf(Ni, Ij) - Ii, *nu_, *alpha_));
	    betaij *= K(D_.value_data()[jj], *delta_, *omega_);
	    betaij *= susceptibility_[D_.index2_data()[jj]];
	    pressureFromI += betaij;
	  }

	res += pressureFromI * infectivity_[i];
      }

    likComponents_.integral = res * *gamma1_;
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
        - (likComponents_.integral + likComponents_.bgIntegral);

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

    logLikelihood_ = likComponents_.logProduct
        - (likComponents_.integral + likComponents_.bgIntegral);

#ifdef GPUTIMING
    gettimeofday(&end, NULL);
    std::cerr << "Time (" << __PRETTY_FUNCTION__ << "): "
    << timeinseconds(start, end) << std::endl;
#endif

  }

  void
  CpuLikelihood::UpdateInfectionTimeInteg(const unsigned int i,
      const fp_t newTime)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];

    fp_t buff = 0.0;  // Accumulate pressure difference

    for (unsigned int jj = begin; jj < end; ++jj)
      {
        unsigned int j = D_.index2_data()[jj];

        fp_t Ii = eventTimes_(i, 0);
        fp_t Ni = eventTimes_(i, 1);
        fp_t Ri = eventTimes_(i, 2);

        fp_t Ij = eventTimes_(j, 0);
        fp_t Nj = eventTimes_(j, 1);
        fp_t Rj = eventTimes_(j, 2);

        // Recalculate pressure from j on idx
        fp_t jOnIdx = 0.0;
        if (Ij < Nj)
          {
            jOnIdx = H(min(Nj, newTime) - min(Ij, newTime), *nu_, *alpha_)
                + *gamma2_
                    * (H(min(Rj, newTime) - Ij, *nu_, *alpha_)
                        - H(min(Nj, newTime) - Ij, *nu_, *alpha_)); // New pressure
            jOnIdx -= H(min(Nj, Ii) - min(Ii, Ij), *nu_, *alpha_)
                + *gamma2_
                    * (H(min(Rj, Ii) - Ij, *nu_, *alpha_)
                        - H(min(Nj, Ii) - Ij, *nu_, *alpha_)); // Old pressure
                // Apply infec and suscep
            jOnIdx *= susceptibility_(i);
            jOnIdx *= infectivity_(j);
          }

        // Recalculate pressure from idx on j
        float IdxOnj = H(min(Ni, Ij) - min(newTime, Ij), *nu_, *alpha_);
        IdxOnj -= H(min(Ni, Ij) - min(Ii, Ij), *nu_, *alpha_);
        IdxOnj += *gamma2_
            * (H(min(Ri, Ij) - newTime, *nu_, *alpha_)
                - H(min(Ni, Ij) - newTime, *nu_, *alpha_));
        IdxOnj -= *gamma2_
            * (H(min(Ri, Ij) - Ii, *nu_, *alpha_)
                - H(min(Ni, Ij) - Ii, *nu_, *alpha_));
        IdxOnj *= susceptibility_(j);
        IdxOnj *= infectivity_(i);

        buff += (IdxOnj + jOnIdx) * K(D_.value_data()[jj], *delta_, *omega_);
      }

    likComponents_.integral += buff * *gamma1_;
  }

  void
  CpuLikelihood::UpdateInfectionTimeProd(unsigned int i, fp_t newTime)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];

    for (unsigned int jj = begin; jj < end; ++jj)
      {
        unsigned int j = D_.index2_data()[jj];
        fp_t Ij = eventTimes_(j, 0);
        fp_t Nj = eventTimes_(j, 1);

        if (Ij < Nj)
          {
            fp_t Ii = eventTimes_(i, 0);
            fp_t Ni = eventTimes_(i, 1);
            fp_t Ri = eventTimes_(i, 2);
            fp_t Rj = eventTimes_(j, 2);

            // Adjust product cache from idx on others
            fp_t idxOnj = 0.0f;
            if (Ii < Ij and Ij <= Ni)
              idxOnj -= h(Ij - Ii, *nu_, *alpha_);
            else if (Ni < Ij and Ij <= Ri)
              {
                idxOnj -= *gamma2_ * h(Ij - Ii, *nu_, *alpha_);
                idxOnj += *gamma2_ * h(Ij - newTime, *nu_, *alpha_);
              }
            if (newTime < Ij and Ij <= Ni)
              idxOnj += h(Ij - newTime, *nu_, *alpha_);

            idxOnj *= *gamma1_ * infectivity_[i] * susceptibility_[j]
                * K(D_.value_data()[jj], *delta_, *omega_);
            productCache_[j] += idxOnj;

            // Recalculate instantaneous pressure on idx
            float jOnIdx = 0.0f;
            if (Ij < newTime and newTime <= Nj)
              jOnIdx = h(newTime - Ij, *nu_, *alpha_);
            else if (Nj < newTime and newTime <= Rj)
              jOnIdx = *gamma2_ * h(newTime - Ij, *nu_, *alpha_);

            jOnIdx *= susceptibility_[i] * infectivity_[j]
                * K(D_.value_data()[jj], *delta_, *omega_);
            productCache_[i] += jOnIdx * *gamma1_;
          }
      }
    productCache_[i] +=
        newTime < movtBan_ ? *epsilon1_ : (*epsilon1_ * *epsilon2_);
  }

  void
  CpuLikelihood::UpdateInfectionTime(const unsigned int idx, const float inTime)
  {

    if (idx >= infecIdx_.size())
      throw std::range_error(
          "Invalid idx in CpuLikelihood::UpdateInfectionTime");

    fp_t savedIntegral = likComponents_.integral;
    unsigned int i = infecIdx_[idx].ptr;
    float newTime = eventTimes_(i, 1) - inTime;
    float oldTime = eventTimes_(i, 0);

    bool haveNewI1 = false;
    if (newTime < I1Time_ or i == I1Idx_)
      {
        haveNewI1 = true;
        productCache_[I1Idx_] =
            newTime < movtBan_ ? *epsilon1_ : (*epsilon1_ * *epsilon2_);
      }
    productCache_[i] = 0.0;

    UpdateInfectionTimeInteg(i, newTime);
    UpdateInfectionTimeProd(i, newTime);

    eventTimes_(i, 0) = newTime;

    if (haveNewI1)
      {
        UpdateI1();
        CalcBgIntegral();
      }
    else
      {
        likComponents_.bgIntegral += *epsilon1_
            * (min(movtBan_, newTime) - min(movtBan_, oldTime));
        likComponents_.bgIntegral += *epsilon1_ * *epsilon2_
            * (max(movtBan_, newTime) - max(movtBan_, oldTime));
      }

    ReduceProductVector();

    logLikelihood_ = likComponents_.logProduct
        - (likComponents_.integral + likComponents_.bgIntegral);

  }

  void
  CpuLikelihood::AddInfectionTimeProd(const unsigned int i, const fp_t newTime)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];

    for (unsigned int jj = begin; jj < end; ++jj)
      {
        unsigned int j = D_.index2_data()[jj];

        fp_t Ij = eventTimes_(j, 0);
        fp_t Nj = eventTimes_(j, 1);

        if (Ij < Nj) // Only look at infected individuals
          {
            fp_t Ni = eventTimes_(i, 1);
            fp_t Ri = eventTimes_(i, 2);
            fp_t Rj = eventTimes_(j, 2);

            // Adjust product cache from idx on others
            float idxOnj = 0.0;
            if (newTime < Ij and Ij <= Ni)
              idxOnj += h(Ij - newTime, *nu_, *alpha_);
            else if (Ni < Ij and Ij <= Ri)
              idxOnj += *gamma2_ * h(Ij - newTime, *nu_, *alpha_);

            idxOnj *= *gamma1_ * infectivity_[i] * susceptibility_[j]
                * K(D_.value_data()[jj], *delta_, *omega_);
            productCache_[j] += idxOnj;

            // Calculate instantaneous pressure on idx
            float jOnIdx = 0.0f;
            if (Ij < newTime and newTime <= Nj)
              jOnIdx = h(newTime - Ij, *nu_, *alpha_);
            else if (Nj < newTime and newTime <= Rj)
              jOnIdx = *gamma2_ * h(newTime - Ij, *nu_, *alpha_);

            jOnIdx *= *gamma1_ * infectivity_[j] * susceptibility_[i]
                * K(D_.value_data()[jj], *delta_, *omega_);

            productCache_[i] += jOnIdx;
          }

      }

    float epsilon = newTime < movtBan_ ? *epsilon1_ : (*epsilon1_ * *epsilon2_);
    productCache_[i] += epsilon;
  }

  void
  CpuLikelihood::AddInfectionTimeInteg(const unsigned int i, const fp_t newTime)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];

    fp_t buff = 0.0;
    for (unsigned int jj = begin; jj < end; ++jj)
      {
        unsigned int j = D_.index2_data()[jj];

        fp_t Ii = eventTimes_(i, 0);
        fp_t Ni = eventTimes_(i, 1);
        fp_t Ri = eventTimes_(i, 2);

        fp_t Ij = eventTimes_(j, 0);
        fp_t Nj = eventTimes_(j, 1);
        fp_t Rj = eventTimes_(j, 2);

        fp_t jOnIdx = 0.0;
        if (Ij < Nj)  // Pressure from j on i
          {
            jOnIdx -= H(fminf(Nj, Ii) - fminf(Ij, Ii), *nu_, *alpha_);
            jOnIdx -= *gamma2_
                * (H(fminf(Rj, Ii) - Ij, *nu_, *alpha_)
                    - H(fminf(Nj, Ii) - Ij, *nu_, *alpha_));
            jOnIdx += H(fminf(Nj, newTime) - fminf(Ij, newTime), *nu_, *alpha_);
            jOnIdx += *gamma2_
                * (H(fminf(Rj, newTime) - Ij, *nu_, *alpha_)
                    - H(fminf(Nj, newTime) - Ij, *nu_, *alpha_));

            // Apply infec and suscep
            jOnIdx *= susceptibility_[i];
            jOnIdx *= infectivity_[j];
          }

        // Add pressure from i on j
        float IdxOnj = H(fminf(Ni, Ij) - fminf(newTime, Ij), *nu_, *alpha_);
        IdxOnj += *gamma2_
            * (H(fminf(Ri, Ij) - newTime, *nu_, *alpha_)
                - H(fminf(Ni, Ij) - newTime, *nu_, *alpha_));
        IdxOnj *= susceptibility_[j];
        IdxOnj *= infectivity_[i];

        buff += (IdxOnj + jOnIdx) * K(D_.value_data()[jj], *delta_, *omega_);
      }

    likComponents_.integral += buff * *gamma1_;
  }

  void
  CpuLikelihood::AddInfectionTime(const unsigned int idx, const fp_t inTime)
  {
    if (idx >= suscOccults_.size())
      throw std::range_error("Invalid idx in CpuLikelihood::AddInfectionTime");

    unsigned int i = suscOccults_[idx].ptr;
    fp_t Ni = eventTimes_(i, 1);
    fp_t newTime = Ni - inTime;

    // Update indices
    infecIdx_.push_back(i);
    suscOccults_.erase(suscOccults_.begin() + idx);

    productCache_[i] = 0.0;
    bool haveNewI1 = false;
    if (newTime < I1Time_)
      {
        productCache_[I1Idx_] =
            newTime < movtBan_ ? *epsilon1_ : *epsilon1_ * *epsilon2_;
        haveNewI1 = true;
      }

    unsigned int addIdx = infecIdx_.size() - 1;

    AddInfectionTimeInteg(i, newTime);
    AddInfectionTimeProd(i, newTime);

    // Update population
    eventTimes_(i, 0) = newTime;
    if (haveNewI1)
      {
        UpdateI1();
        CalcBgIntegral();
      }
    else
      {
        likComponents_.bgIntegral += *epsilon1_
            * (min(movtBan_, newTime) - min(movtBan_, Ni));
        likComponents_.bgIntegral += *epsilon1_ * *epsilon2_
            * (max(movtBan_, newTime) - max(movtBan_, Ni));
      }

    ReduceProductVector();

    logLikelihood_ = likComponents_.logProduct
        - (likComponents_.integral + likComponents_.bgIntegral);

  }

  void
  CpuLikelihood::DelInfectionTimeProd(const unsigned int i)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];

    for (unsigned int jj = begin; jj < end; ++jj)
      {
        unsigned int j = D_.index2_data()[jj];

        fp_t Ij = eventTimes_(j, 0);
        fp_t Nj = eventTimes_(j, 1);

        if (Ij < Nj)
          {
            fp_t Ii = eventTimes_(i, 0);
            fp_t Ni = eventTimes_(i, 1);
            fp_t Ri = eventTimes_(i, 2);

            // Adjust product cache from idx on others
            fp_t idxOnj = 0.0;
            if (Ii < Ij and Ij <= Ni)
              idxOnj -= h(Ij - Ii, *nu_, *alpha_);
            else if (Ni < Ij and Ij <= Ri)
              idxOnj -= *gamma2_ * h(Ij - Ii, *nu_, *alpha_);

            idxOnj *= *gamma1_ * infectivity_[i] * susceptibility_[j]
                * K(D_.value_data()[jj], *delta_, *omega_);
            productCache_[j] += idxOnj;
          }
      }
  }

  void
  CpuLikelihood::DelInfectionTimeInteg(const unsigned int i)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];

    fp_t buff = 0.0;
    for (unsigned int jj = begin; jj < end; ++jj)
      {
        unsigned int j = D_.index2_data()[jj];

        fp_t Ii = eventTimes_(i, 0);
        fp_t Ni = eventTimes_(i, 1);
        fp_t Ri = eventTimes_(i, 2);

        fp_t Ij = eventTimes_(j, 0);
        fp_t Nj = eventTimes_(j, 1);
        fp_t Rj = eventTimes_(j, 2);

        fp_t jOnIdx = 0.0;
        if (Ij < Nj)
          {
            // Recalculate pressure from j on idx
            jOnIdx -= H(fminf(Nj, Ii) - fminf(Ii, Ij), *nu_, *alpha_)
                + *gamma2_
                    * (H(fminf(Rj, Ii) - Ij, *nu_, *alpha_)
                        - H(fminf(Nj, Ii) - Ij, *nu_, *alpha_)); // Old pressure
            jOnIdx += H(fminf(Nj, Ni) - fminf(Ij, Ni), *nu_, *alpha_)
                + *gamma2_
                    * (H(fminf(Rj, Ni) - Ij, *nu_, *alpha_)
                        - H(fminf(Nj, Ni) - Ij, *nu_, *alpha_)); // New pressure
                // Apply infec and suscep
            jOnIdx *= susceptibility_[i];
            jOnIdx *= infectivity_[j];
          }

        // Subtract pressure from idx on j
        fp_t IdxOnj = 0.0f;
        IdxOnj -= H(fminf(Ni, Ij) - fminf(Ii, Ij), *nu_, *alpha_);
        IdxOnj -= *gamma2_
            * (H(fminf(Ri, Ij) - Ii, *nu_, *alpha_)
                - H(fminf(Ni, Ij) - Ii, *nu_, *alpha_));
        IdxOnj *= susceptibility_[j];
        IdxOnj *= infectivity_[i];

        buff += (IdxOnj + jOnIdx)
            * K(D_.value_data()[jj], *delta_, *omega_);
      }

    likComponents_.integral += buff * *gamma1_;
  }

  void
  CpuLikelihood::DeleteInfectionTime(const unsigned int idx)
  {
    if (idx >= infecIdx_.size() - numKnownInfecs_)
      throw std::range_error(
          "Invalid idx in CpuLikelihood::DeleteInfectionTime");

    unsigned int ii = idx + numKnownInfecs_;
    unsigned int i = infecIdx_[ii].ptr;

    fp_t Ni = eventTimes_(i, 1);
    fp_t oldI = eventTimes_(i, 0);

    DelInfectionTimeInteg(i);
    DelInfectionTimeProd(i);

    // Make the change to the population
    bool haveNewI1 = false;
    infecIdx_.erase(infecIdx_.begin() + ii);
    suscOccults_.push_back(i);
    eventTimes_(i, 0) = eventTimes_(i, 1);
    productCache_(i) = 1.0;

    if (i == I1Idx_)
      {
        UpdateI1();
        CalcBgIntegral();
        haveNewI1 = true;
      }
    else
      {
        likComponents_.bgIntegral += *epsilon1_
            * (min(movtBan_, Ni) - min(movtBan_, oldI));
        likComponents_.bgIntegral += *epsilon1_ * *epsilon2_
            * (max(movtBan_, Ni) - max(movtBan_, oldI));
      }

    ReduceProductVector();

    logLikelihood_ = likComponents_.logProduct
        - (likComponents_.integral + likComponents_.bgIntegral);
  }

  fp_t
  CpuLikelihood::NonCentreInfecTimes(const fp_t oldGamma, 
				     const fp_t newGamma, const fp_t prob)
  {
    // Rescale infection times
    for(size_t ii = 0; ii < numKnownInfecs_; ++ii)
      {
	unsigned int i = infecIdx_[ii].ptr;
	fp_t d = eventTimes_(i,1) - eventTimes_(i,0);
	d *= oldGamma/newGamma;
	eventTimes_(i,0) = eventTimes_(i,1) - d;
      }

    // Calculate likelihood for occults (they've not been non-centred)
    fp_t val = 0.0;
    for(size_t ii = numKnownInfecs_; ii < infecIdx_.size(); ++ii)
      {
	unsigned int i = infecIdx_[ii].ptr;
	fp_t d = eventTimes_(i,1) - eventTimes_(i,0);
	val += logf(gsl_cdf_gamma_Q(d, *a_, (fp_t)1.0 / newGamma))
	  - logf(gsl_cdf_gamma_Q(d, *a_, (fp_t)1.0 / oldGamma));
      }

    return val;
  }

  fp_t
  CpuLikelihood::InfectionPart()
  {
    fp_t val = 0.0;
    for(size_t ii = 0; ii < numKnownInfecs_; ++ii)
      {
	unsigned int i = infecIdx_[ii].ptr;
	fp_t d = eventTimes_(i,1) - eventTimes_(i,0);
	val += log(powf(*b_, *a_) * powf(d, *a_ - 1) * expf(-d * *b_));
      }
    for(size_t ii = numKnownInfecs_; ii < infecIdx_.size(); ++ii)
      {
	unsigned int i = infecIdx_[ii].ptr;
	fp_t d = obsTime_ - eventTimes_(i,0);
	val += log(gsl_cdf_gamma_Q(d, (fp_t)*a_, (fp_t)1.0 / (fp_t)*b_));
      }
    return val;
  }

  float
  CpuLikelihood::GetLogLikelihood() const
  {
    return logLikelihood_;
  }

  size_t
  CpuLikelihood::GetNumInfecs() const
  {
    return infecIdx_.size();
  }

  size_t
  CpuLikelihood::GetNumOccults() const
  {
    return infecIdx_.size() - numKnownInfecs_;
  }

  size_t
  CpuLikelihood::GetNumPossibleOccults() const
  {
    return suscOccults_.size();
  }


  void 
  colSums(fp_t* result, const ublas::matrix<fp_t, ublas::column_major>& m)
  {
    for(int j = 0; j < m.size2(); ++j)
      {
	result[j] = 0.0;
	for(int i = 0; i < m.size1(); ++i)
	  {
	    result[j] += m(i,j);
	  }
      }
  }

  void
  CpuLikelihood::GetSumInfectivityPow(fp_t* result) const
  {
    colSums(result, animalsInfPow_);
  }

  void
  CpuLikelihood::GetSumSusceptibilityPow(fp_t* result) const
  {
    colSums(result, animalsSuscPow_);
  }

  fp_t
  CpuLikelihood::GetIN(size_t idx)
  {
    unsigned int i = infecIdx_[idx].ptr;
    return eventTimes_(i,1) - eventTimes_(i,0);
  }

  fp_t
  CpuLikelihood::GetMeanI2N() const
  {
    fp_t mean = 0.0;
    for(int ii = 0; ii < numKnownInfecs_; ++ii)
      {
	unsigned int i = infecIdx_[ii].ptr;
	fp_t x = eventTimes_(i,1) - eventTimes_(i,0);
	mean = mean + (x - mean)/(i+1);
      }
    return mean;
  }

  fp_t
  CpuLikelihood::GetMeanOccI() const
  {
    fp_t mean = 0.0;
    for(int ii = numKnownInfecs_; ii < infecIdx_.size(); ++ii)
      {
	unsigned int i = infecIdx_[ii].ptr;
	fp_t x = eventTimes_(i,1) - eventTimes_(i,0);
	mean = mean + (x - mean)/i+1;
      }
    return mean;
  }

  void
  CpuLikelihood::GetInfectiousPeriods(std::vector<IPTuple_t>& result)
  {
    result.resize(eventTimes_.size1());
    for(int i = 0; i < eventTimes_.size1(); ++i)
      {
	IPTuple_t ip;
	ip.idx = i; ip.val = eventTimes_(i,1) - eventTimes_(i,0);
	result[i] = ip;
      }
  }
  

  void
  CpuLikelihood::PrintLikelihoodComponents() const
  {
    cout.precision(15);
    cout << "Background: " << likComponents_.bgIntegral << "\n";
    cout << "Integral: " << likComponents_.integral << "\n";
    cout << "Product: " << likComponents_.logProduct << "\n";
  }

  void
  CpuLikelihood::PrintParameters() const
  {
    cerr << "Epsilon1,2: " << *epsilon1_ << ", " << *epsilon2_ << "\n";
    cerr << "Gamma1,2: " << *gamma1_ << ", " << *gamma2_ << "\n";
    cerr << "Delta: " << *delta_ << "\n";
    cerr << "Omega: " << *omega_ << "\n";
    for (int i = 0; i < numSpecies_; ++i)
      cerr << "Xi,Zeta,Phi,Psi[" << i << "]: " << xi_[i] << ", " << zeta_[i]
          << ", " << phi_[i] << ", " << psi_[i] << "\n";
    cerr << "alpha: " << *alpha_ << "\n";
    cerr << "a: " << *a_ << "\n";
    cerr << "b: " << *b_ << endl;
    cerr << "ObsTime: " << obsTime_ << "\n";
    cerr << "I1Idx = " << I1Idx_ << "\n";
    cerr << "I1Time = " << I1Time_ << "\n";
  }

  void
  CpuLikelihood::PrintProdCache() const
  {
    for (int i = 0; i < popSize_; ++i)
      {
        cout << population_[i].id << ": " << productCache_(i) << "\n";
      }
  }

  const ublas::vector<fp_t>&
  CpuLikelihood::GetProdCache() const
  {
    return productCache_;
  }

} // namespace EpiRisk

