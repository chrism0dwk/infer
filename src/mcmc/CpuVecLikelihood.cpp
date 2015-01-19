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
#include <omp.h>

#include "vectorclass/vectorclass.h"
#include "vectorclass/vectormath_exp.h"
#include "CpuVecLikelihood.hpp"

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

  inline Vec8f
  hVec8f(const Vec8f t, float nu, float alpha)
  {
    return select(t < alpha, 0.0f, 1.0f);
  }

  inline fp_t
  H(const float t, const float nu, const float alpha)
  {
    float integral = t - alpha;
    return fmaxf(0.0f, integral);
  }

  inline Vec8f
  HVec8f(const Vec8f t, const float nu, const float alpha)
  {
    Vec8f integral = t - alpha;
    return max(0.0f, integral);
  }

  inline fp_t
  K(const float dsq, const float delta, const float omega)
  {
    return delta / powf(delta * delta + dsq, omega);
  }

  inline Vec8f
  KVec8f(const Vec8f dsq, const float delta, const float omega)
  {
    return delta / pow(delta * delta + dsq, omega);
  }

  CpuLikelihood::CpuLikelihood(PopDataImporter& population,
      EpiDataImporter& epidemic, const size_t nSpecies, const float obsTime,
      const float dLimit, const bool occultsOnlyDC) :
    Likelihood(population, epidemic, nSpecies, obsTime, occultsOnlyDC),
      I1Time_(0.0), I1Idx_(0)
  {

    // Load data into host memory
#pragma omp parallel
      {
        numThreads_ = omp_get_num_threads();
      }
    cout << "Using " << numThreads_ << " threads" << endl;
    omp_set_num_threads(8);
    CalcDistanceMatrix(dLimit);
    omp_set_num_threads(numThreads_);

    // Set up species and events
    SetSpecies();
    SetEvents();

    // Set up occult susceptible vector
    suscOccults_.resize(maxInfecs_ - numKnownInfecs_);
    for (size_t i = numKnownInfecs_; i < maxInfecs_; ++i)
      suscOccults_[i - numKnownInfecs_] = i;

    // Allocate product cache
    productCache_.resize(maxInfecsPitch_);
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
    popSizePitch_(other.popSizePitch_),
    maxInfecsPitch_(other.maxInfecsPitch_),
    numThreads_(other.numThreads_),
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
    eventTimes_.resize(popSizePitch_, NUMEVENTS);
    Population::iterator it = population_.begin();
    for (size_t i = 0; i < popSize_; ++i)
      {
        eventTimes_(i, 0) = it->I;
        eventTimes_(i, 1) = it->N;
        eventTimes_(i, 2) = it->R;
        ++it;
      }
    for (size_t i = popSize_; i < popSizePitch_; ++i)
      {
        eventTimes_(i, 0) = NEGINF;
        eventTimes_(i, 1) = NEGINF;
        eventTimes_(i, 2) = NEGINF;
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
    popSizePitch_ = (popSize_ + VECSIZE - 1) & (-VECSIZE);
    animals_.resize(popSizePitch_, numSpecies_);

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
    for (int i = popSize_; i < popSizePitch_; ++i)
      {
        animals_(i, 0) = 0.0f;
        animals_(i, 1) = 0.0f;
        animals_(i, 2) = 0.0f;
      }

    susceptibility_.resize(popSizePitch_);
    animalsSuscPow_.resize(popSizePitch_, numSpecies_);

    maxInfecsPitch_ = (maxInfecs_ + VECSIZE - 1) & (-VECSIZE);
    infectivity_.resize(maxInfecsPitch_);
    animalsInfPow_.resize(maxInfecsPitch_, numSpecies_);
  }

  inline
  void
  CpuLikelihood::CalcInfectivityPow()
  {

    for (size_t k = 0; k < numSpecies_; ++k)
      {

#pragma omp parallel for
        for (size_t i = 0; i < maxInfecsPitch_; i += VECSIZE)
          {
            Vec8f input, output;
            input.load(&animals_(i, k));
            output = pow(input, psi_[k]);
            output.store(&animalsInfPow_(i, k));
          }

      }
  }

  inline
  void
  CpuLikelihood::CalcInfectivity()
  {

    // Now calculate infectivity
#pragma omp parallel for
    for (size_t i = 0; i < maxInfecsPitch_; i += VECSIZE)
      {
        Vec8f output = 0.0f;
        for (size_t k = 0; k < numSpecies_; ++k)
          {
            Vec8f input;
            input.load(&animalsInfPow_(i, k));
            output += input * xi_[k];
          }

        output.store(&infectivity_[i]);
      }

  }

  inline
  void
  CpuLikelihood::CalcSusceptibilityPow()
  {
    for (size_t k = 0; k < numSpecies_; ++k)
      {

#pragma omp parallel for
        for (int i = 0; i < popSizePitch_; i += VECSIZE)
          {
            Vec8f input, output;
            input.load(&animals_(i, k));
            output = pow(input, phi_[k]);
            output.store(&animalsSuscPow_(i, k));
          }

      }
  }

  inline
  void
  CpuLikelihood::CalcSusceptibility()
  {
    // Calculates susceptibility powers and sums over suscept.

#pragma omp parallel for
    for (int i = 0; i < popSizePitch_; i += VECSIZE)
      {
        Vec8f output = 0.0f;
        for (int k = 0; k < numSpecies_; ++k)
          {
            Vec8f input;
            input.load(&animalsSuscPow_(i, k));
            output += input * zeta_[k];
          }
        output.store(&susceptibility_[i]);
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
    //std::vector<float> res(popSize_);
    // for(int i=0; i<popSize_; ++i)
    //   {
    // 	float I = eventTimes_(i,0);
    // 	res[i] = *epsilon1_ * max((min(I, movtBan_) - I1Time_), 0.0f);
    // 	res[i] += *epsilon1_ * *epsilon2_ * max(I - max(movtBan_, I1Time_), 0.0f);
    //   }
    // likComponents_.bgIntegral = std::accumulate(res.begin(), res.end(), 0.0f);

    float res = 0.0f;

#pragma omp parallel for reduction(+:res)
    for (int i = 0; i < popSizePitch_; i += VECSIZE)
      {
        Vec8f I;
        I.load(&eventTimes_(i, 0));
        Vec8f partial = *epsilon1_ * max((min(I, movtBan_) - I1Time_), 0.0f);
        partial += *epsilon1_ * *epsilon2_
            * max(I - max(movtBan_, I1Time_), 0.0f);
        res += horizontal_add(partial);
      }

    likComponents_.bgIntegral = res;
  }

  inline
  void
  CpuLikelihood::ReduceProductVector()
  {

    double logprod = 0.0;
    productCache_(infecIdx_[I1Idx_].ptr) = 1.0f;

#pragma omp parallel for reduction(+:logprod)
    for (int i = 0; i < productCache_.size(); i += VECSIZE)
      {
        Vec8f prod, input;
        input.load(&productCache_(i));
        prod = log(input);
        logprod += horizontal_add(prod);
      }

    likComponents_.logProduct = logprod;

  }

  inline
  void
  CpuLikelihood::CalcProduct()
  {

    float* workA;
    float* workB;
    float* workC;

#pragma omp parallel private(workA, workB, workC)
      {
        workA = new float[popSize_];
        workB = new float[popSize_];
        workC = new float[popSize_];
#pragma omp for 
        for (int jj = 0; jj < infecIdx_.size(); ++jj)
          {
            int j = infecIdx_[jj].ptr;
            int begin = D_.index1_data()[j];
            int end = D_.index1_data()[j + 1];

            float Ij = eventTimes_(j, 0);

            /////////// VECTORIZATION ATTEMPT ////////////

            // Load data
            size_t dataSize = end - begin;
            size_t arrSize = (dataSize + VECSIZE - 1) & (-VECSIZE);

            size_t shortDataSize = 0;
            for (size_t ii = 0; ii < dataSize; ii++)
              {
                size_t i = D_.index2_data()[ii + begin];
                if (eventTimes_(i, 0) < eventTimes_(i, 1))
                  {
                    workA[shortDataSize] = eventTimes_(i, 0);
                    workA[shortDataSize + arrSize] = eventTimes_(i, 1);
                    workA[shortDataSize + arrSize * 2] = eventTimes_(i, 2);
                    workB[shortDataSize] = infectivity_(i);
                    workC[shortDataSize] = D_.value_data()[ii + begin];
                    shortDataSize++;
                  }
              }
            size_t shortArrSize = (shortDataSize + VECSIZE - 1) & (-VECSIZE);
            for (size_t ii = shortDataSize; ii < shortArrSize; ii++)
              {
                workA[ii] = 0.0f;
                workA[ii + arrSize] = 0.0f;
                workA[ii + arrSize * 2] = 0.0f;
                workB[ii] = 0.0f;
                workC[ii] = 0.0f;
              }

            float sumPressure = 0.0f;
            for (int i = 0; i < shortArrSize; i += VECSIZE)
              {
                Vec8f Ii;
                Ii.load(workA + i);
                Vec8f Ni;
                Ni.load(workA + i + arrSize);
                Vec8f Ri;
                Ri.load(workA + i + arrSize * 2);
                Vec8f infec;
                infec.load(workB + i);
                Vec8f d;
                d.load(workC + i);
                Vec8f idxOnj = 0.0f;

                idxOnj = hVec8f(Ij - Ii, *nu_, *alpha_) * infec
                    * KVec8f(d, *delta_, *omega_);
                Vec8f state = select(Ii < Ij & Ij <= Ni, 1.0f, 0.0f);
                state = select(Ni < Ij & Ij <= Ri, *gamma2_, state);
                idxOnj *= state;
                sumPressure += horizontal_add(idxOnj);
              }
            ///////////////////////////////////////////////

            float epsilon = *epsilon1_;
            epsilon *= Ij < movtBan_ ? 1.0f : *epsilon2_;
            productCache_[j] = sumPressure * *gamma1_ * susceptibility_(j)
                + epsilon;
          }
        delete[] workA;
        delete[] workB;
        delete[] workC;
      }

    ReduceProductVector();
  }

  inline
  void
  CpuLikelihood::CalcIntegral()
  {
    float res = 0.0f;

    float* workA;
    float* workB;
    float* workC;

    // Each thread allocates memory here
#pragma omp parallel private(workA, workB, workC)
      {
        workA = new float[popSize_];
        workB = new float[popSize_];
        workC = new float[popSize_];

#pragma omp for reduction(+:res)
        for (size_t ii = 0; ii < infecIdx_.size(); ++ii)
          {
            size_t i = infecIdx_[ii].ptr;
            size_t begin = D_.index1_data()[i];
            size_t end = D_.index1_data()[i + 1];
            float Ii = eventTimes_(i, 0);
            float Ni = eventTimes_(i, 1);
            float Ri = eventTimes_(i, 2);

            // Load data
            size_t dataSize = end - begin;
            size_t arrSize = (dataSize + VECSIZE - 1) & (-VECSIZE);

            for (size_t jj = 0; jj < dataSize; ++jj)
              {
                size_t j = D_.index2_data()[jj + begin];
                workA[jj] = susceptibility_[j];
                workB[jj] = eventTimes_(j, 0);
                workC[jj] = D_.value_data()[jj + begin];
              }
            for (size_t jj = dataSize; jj < arrSize; ++jj)
              {
                workA[jj] = 0.0f;
                workB[jj] = 0.0f;
                workC[jj] = POSINF;
              }

            // Vector processing
            float pressureFromI = 0.0f;
            for (size_t j = 0; j < dataSize; j += VECSIZE)
              {
                // Load into vector types
                Vec8f Ij, suscep, d;
                Ij.load(workB + j);
                suscep.load(workA + j);
                d.load(workC + j);

                // Calculate
                Vec8f betaij = HVec8f(min(Ni, Ij) - min(Ii, Ij), *nu_, *alpha_);
                betaij += *gamma2_
                    * (HVec8f(min(Ri, Ij) - Ii, *nu_, *alpha_)
                        - HVec8f(min(Ni, Ij) - Ii, *nu_, *alpha_));
                betaij *= KVec8f(d, *delta_, *omega_);
                betaij *= suscep;
                betaij.store(workA + j); // TODO: Necessary?
                pressureFromI += horizontal_add(betaij);
              }

            res += pressureFromI * infectivity_[i];
          }

        delete[] workA;
        delete[] workB;
        delete[] workC;
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
  CpuLikelihood::PackData(const size_t begin, const size_t end, fp_t** data, size_t* pitch)
  {
    size_t dataSize = end - begin;
    size_t vecLen = (dataSize + VECSIZE - 1) & (-VECSIZE);
    fp_t* jData = new fp_t[vecLen * 5]; // Treat as column-major
#pragma omp parallel for
    for (unsigned int jj = begin; jj < end; ++jj)
      {
        unsigned int j = D_.index2_data()[jj];

	jData[jj-begin]          = eventTimes_(j,0);
	jData[jj-begin + vecLen] = eventTimes_(j,1);
	jData[jj-begin + vecLen*2] = eventTimes_(j,2);
	jData[jj-begin + vecLen*3] = susceptibility_[j];
	jData[jj-begin + vecLen*4] = infectivity_[j];
      }
    // Not worth parallelising the following
    for(unsigned int col=0; col < 5; ++col) 
      memset(jData+vecLen*col+dataSize,0.0f,(vecLen-dataSize)*sizeof(fp_t));

    *pitch = vecLen; 
    *data = jData;
  }

  void
  CpuLikelihood::UpdateInfectionTimeInteg(const unsigned int i,
      const fp_t newTime)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];
    unsigned int dataSize = end - begin;
    size_t vecLen;
    fp_t Ii = eventTimes_(i, 0);
    fp_t Ni = eventTimes_(i, 1);
    fp_t Ri = eventTimes_(i, 2);

    // Pack data
    fp_t* jData;
    PackData(begin, end, &jData, &vecLen);
    
    // Vector calculation
    fp_t buff = 0.0f;
#pragma omp parallel for reduction(+:buff)
      for(unsigned int jj = 0; jj < vecLen; jj += VECSIZE) {
	// Recalculate pressure from j on idx

	Vec8f Ij; Ij.load(jData+jj);
	Vec8f Nj; Nj.load(jData+jj+vecLen);
	Vec8f Rj; Rj.load(jData+jj+vecLen*2);
	Vec8f jSusc; jSusc.load(jData+jj+vecLen*3);
	Vec8f jInf;  jInf.load(jData+jj+vecLen*4);
	Vec8f d;     d.load(&(D_.value_data()[jj+begin]));

	Vec8f jOnIdx = 0.0;
	jOnIdx = HVec8f(min(Nj, newTime) - min(Ij, newTime), *nu_, *alpha_)
	  + *gamma2_
	  * (HVec8f(min(Rj, newTime) - Ij, *nu_, *alpha_)
	     - HVec8f(min(Nj, newTime) - Ij, *nu_, *alpha_)); // New pressure
	jOnIdx -= HVec8f(min(Nj, Ii) - min(Ii, Ij), *nu_, *alpha_)
	  + *gamma2_
	  * (HVec8f(min(Rj, Ii) - Ij, *nu_, *alpha_)
	     - HVec8f(min(Nj, Ii) - Ij, *nu_, *alpha_)); // Old pressure
	// Apply infec and suscep
	jOnIdx *= susceptibility_(i);
	jOnIdx *= jInf;
	jOnIdx = select(Ij < Nj, jOnIdx, 0.0f);

        // Recalculate pressure from idx on j
	Vec8f IdxOnj = 0.0f;
	IdxOnj += HVec8f(min(Ni, Ij) - min(newTime, Ij), *nu_, *alpha_);
	IdxOnj -= HVec8f(min(Ni, Ij) - min(Ii, Ij), *nu_, *alpha_);
	IdxOnj += *gamma2_
	  * (HVec8f(min(Ri, Ij) - newTime, *nu_, *alpha_)
	     - HVec8f(min(Ni, Ij) - newTime, *nu_, *alpha_));
	IdxOnj -= *gamma2_
	  * (HVec8f(min(Ri, Ij) - Ii, *nu_, *alpha_)
	     - HVec8f(min(Ni, Ij) - Ii, *nu_, *alpha_));
	IdxOnj *= jSusc;
	IdxOnj *= infectivity_(i);

	buff += horizontal_add((IdxOnj + jOnIdx) * KVec8f(d, *delta_, *omega_));
      }

    likComponents_.integral += buff * *gamma1_;

    delete[] jData;
  }

  void
  CpuLikelihood::UpdateInfectionTimeProd(unsigned int i, fp_t newTime)
  {
    unsigned int begin = D_.index1_data()[i];
    unsigned int end = D_.index1_data()[i + 1];
    unsigned int dataSize = end - begin;
    size_t vecLen;
    fp_t Ii = eventTimes_(i, 0);
    fp_t Ni = eventTimes_(i, 1);
    fp_t Ri = eventTimes_(i, 2);

    // Pack the j data
    fp_t* jData;
    PackData(begin, end, &jData, &vecLen);

    // Vectorised calculation
    for (unsigned int jj = 0; jj < vecLen; jj+=VECSIZE)
      {
        Vec8f Ij;    Ij.load(jData+jj);
	Vec8f Nj;    Nj.load(jData+jj+vecLen);
	Vec8f Rj;    Rj.load(jData+jj+vecLen*2);
	Vec8f jSusc; jSusc.load(jData+jj+vecLen*3);
	Vec8f jInf;  jInf.load(jData+jj+vecLen*4);
	Vec8f d;     d.load(&(D_.value_data()[jj+begin]));
	
	// Adjust product cache from idx on others
	Vec8f idxOnj = 0.0f;
	//if (Ii < Ij and Ij <= Ni)
	idxOnj -= hVec8f(Ij - Ii, *nu_, *alpha_) * select(Ii < Ij & Ij <= Ni,1.0f,0.0f);
	//else if (Ni < Ij and Ij <= Ri)
	idxOnj += *gamma2_ * (hVec8f(Ij - newTime, *nu_, *alpha_) 
			      - hVec8f(Ij - Ii, *nu_, *alpha_))
	  * select(Ni < Ij & Ij <= Ri, 1.0f, 0.0f);	  
	//if (newTime < Ij and Ij <= Ni)
	idxOnj += hVec8f(Ij - newTime, *nu_, *alpha_) * select(newTime < Ij & Ij <= Ni, 1.0f, 0.0f);
	idxOnj *= *gamma1_ * infectivity_[i] * jSusc
	  * KVec8f(d, *delta_, *omega_);

	// Mask off uninfected j's
	idxOnj = select(Ij < Nj, idxOnj, 0.0f);

	for(unsigned j=0; j < VECSIZE; ++j)
	  productCache_[D_.index2_data()[jj+begin+j]] += idxOnj.extract(j);

	// Recalculate instantaneous pressure on idx
	Vec8f jOnIdx = 0.0f;
	jOnIdx = hVec8f(newTime - Ij, *nu_, *alpha_) 
	  * select(Ij < newTime & newTime <= Nj, 1.0f, 0.0f);
	jOnIdx += *gamma2_ * hVec8f(newTime - Ij, *nu_, *alpha_) 
	  * select(Nj < newTime & newTime <= Rj, 1.0f, 0.0f);
	
	jOnIdx *= susceptibility_[i] * jInf
	  * KVec8f(d, *delta_, *omega_);

	// Mask uninfected j's
	jOnIdx = select(Ij < Nj, jOnIdx, 0.0f);
	productCache_[i] += horizontal_add(jOnIdx) * *gamma1_;
      }
    productCache_[i] +=
        newTime < movtBan_ ? *epsilon1_ : (*epsilon1_ * *epsilon2_);

    delete[] jData;
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
    size_t begin = D_.index1_data()[i];
    size_t end = D_.index1_data()[i + 1];
    size_t dataSize = end - begin;
    size_t vecLen;

    fp_t Ni = eventTimes_(i, 1);
    fp_t Ri = eventTimes_(i, 2);

    fp_t* jData;
    PackData(begin, end, &jData, &vecLen);

#pragma omp parallel for    
    for (unsigned int jj = 0; jj < vecLen; jj+=VECSIZE)
      {
        unsigned int j = D_.index2_data()[jj];

        Vec8f Ij;    Ij.load(jData+jj);
	Vec8f Nj;    Nj.load(jData+jj+vecLen);
	Vec8f Rj;    Rj.load(jData+jj+vecLen*2);
	Vec8f jSusc; jSusc.load(jData+jj+vecLen*3);
	Vec8f jInf;  jInf.load(jData+jj+vecLen*4);
	Vec8f d;     d.load(&(D_.value_data()[jj+begin]));

	// Adjust product cache from idx on others
	Vec8f idxOnj = 0.0;
	idxOnj += hVec8f(Ij - newTime, *nu_, *alpha_) 
	  * select(newTime < Ij & Ij <= Ni, 1.0f, 0.0f);
	idxOnj += *gamma2_ * hVec8f(Ij - newTime, *nu_, *alpha_) 
	  * select(Ni < Ij & Ij <= Ri, 1.0f, 0.0f);

	idxOnj *= *gamma1_ * infectivity_[i] * jSusc
	  * KVec8f(d, *delta_, *omega_);

	// Mask uninfected j's
	idxOnj = select(Ij < Nj, idxOnj, 0.0f);

	for(size_t j=0; j < VECSIZE; ++j)
	  productCache_[D_.index2_data()[jj+begin+j]] += idxOnj.extract(j);

	// Calculate instantaneous pressure on idx
	Vec8f jOnIdx = 0.0f;
	jOnIdx = hVec8f(newTime - Ij, *nu_, *alpha_) 
	  * select(Ij < newTime & newTime <= Nj, 1.0f, 0.0f);
	jOnIdx += *gamma2_ * hVec8f(newTime - Ij, *nu_, *alpha_) 
	  * select(Nj < newTime & newTime <= Rj, 1.0f, 0.0f);

	jOnIdx *= jInf * susceptibility_[i]
	  * KVec8f(d, *delta_, *omega_);

	// Mask uninfected j's
	jOnIdx = select(Ij < Nj, jOnIdx, 0.0f);

	productCache_[i] += horizontal_add(jOnIdx) * *gamma1_;
          
      }

    float epsilon = newTime < movtBan_ ? *epsilon1_ : (*epsilon1_ * *epsilon2_);
    productCache_[i] += epsilon;

    delete[] jData;
  }

  void
  CpuLikelihood::AddInfectionTimeInteg(const unsigned int i, const fp_t newTime)
  {
    size_t begin = D_.index1_data()[i];
    size_t end = D_.index1_data()[i + 1];
    size_t dataLen = end - begin;
    size_t vecLen;

    fp_t Ii = eventTimes_(i, 0);
    fp_t Ni = eventTimes_(i, 1);
    fp_t Ri = eventTimes_(i, 2);

    fp_t* jData;
    PackData(begin, end, &jData, &vecLen);

    fp_t buff = 0.0;
#pragma omp parallel for reduction(+:buff)
    for (unsigned int jj = 0; jj < vecLen; jj += VECSIZE)
      {

        Vec8f Ij;    Ij.load(jData+jj);
        Vec8f Nj;    Nj.load(jData+jj+vecLen);
        Vec8f Rj;    Nj.load(jData+jj+vecLen*2);
	Vec8f jSusc; jSusc.load(jData+jj+vecLen*3);
	Vec8f jInf;  jInf.load(jData+jj+vecLen*4);
	Vec8f d;     d.load(&(D_.value_data()[jj+begin]));

        Vec8f jOnIdx = 0.0;
	jOnIdx -= HVec8f(min(Nj, Ii) - min(Ij, Ii), *nu_, *alpha_);
	jOnIdx -= *gamma2_
	  * (HVec8f(min(Rj, Ii) - Ij, *nu_, *alpha_)
	     - HVec8f(min(Nj, Ii) - Ij, *nu_, *alpha_));
	jOnIdx += HVec8f(min(Nj, newTime) - min(Ij, newTime), *nu_, *alpha_);
	jOnIdx += *gamma2_
	  * (HVec8f(min(Rj, newTime) - Ij, *nu_, *alpha_)
	     - HVec8f(min(Nj, newTime) - Ij, *nu_, *alpha_));
	
	// Apply infec and suscep
	jOnIdx *= susceptibility_[i];
	jOnIdx *= jInf;
	
	jOnIdx = select(Ij < Nj, jOnIdx, 0.0f);

        // Add pressure from i on j
        Vec8f IdxOnj = 0.0f;
	IdxOnj += HVec8f(min(Ni, Ij) - min(newTime, Ij), *nu_, *alpha_);
        IdxOnj += *gamma2_
            * (HVec8f(min(Ri, Ij) - newTime, *nu_, *alpha_)
                - HVec8f(min(Ni, Ij) - newTime, *nu_, *alpha_));
        IdxOnj *= jSusc;
        IdxOnj *= infectivity_[i];

        buff += horizontal_add((IdxOnj + jOnIdx) * KVec8f(d, *delta_, *omega_));
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
    size_t begin = D_.index1_data()[i];
    size_t end = D_.index1_data()[i + 1];
    size_t dataSize = end - begin;
    size_t vecLen;
    fp_t* jData;
    PackData(begin, end, &jData, &vecLen);

    fp_t Ii = eventTimes_(i, 0);
    fp_t Ni = eventTimes_(i, 1);
    fp_t Ri = eventTimes_(i, 2);

#pragma omp parallel for
    for (unsigned int jj = 0; jj < vecLen; jj+=VECSIZE)
      {
        Vec8f Ij; Ij.load(jData+jj);
        Vec8f Nj; Nj.load(jData+jj+vecLen);
	Vec8f jSusc; jSusc.load(jData+jj+vecLen*3);
	Vec8f d;     d.load(&(D_.value_data()[jj+begin]));
 
	// Adjust product cache from idx on others
	Vec8f idxOnj = 0.0;
	idxOnj -= hVec8f(Ij - Ii, *nu_, *alpha_) 
	  * select(Ii < Ij & Ij <= Ni, 1.0f, 0.0f);
	idxOnj -= *gamma2_ * hVec8f(Ij - Ii, *nu_, *alpha_) 
	  * select(Ni < Ij & Ij <= Ri, 1.0f, 0.0f);

	idxOnj *= *gamma1_ * infectivity_[i] * jSusc
	  * KVec8f(d, *delta_, *omega_);

	// Mask uninfected j's
	idxOnj = select(Ij < Nj, idxOnj, 0.0f);

	for(size_t j=0; j<VECSIZE; ++j)
	  productCache_[D_.index2_data()[jj+begin+j]] += idxOnj.extract(j);
       }
  }

  void
  CpuLikelihood::DelInfectionTimeInteg(const unsigned int i)
  {
    size_t begin = D_.index1_data()[i];
    size_t end = D_.index1_data()[i + 1];
    size_t dataLen = end - begin;
    size_t vecLen;

    fp_t Ii = eventTimes_(i, 0);
    fp_t Ni = eventTimes_(i, 1);
    fp_t Ri = eventTimes_(i, 2);

    fp_t* jData;
    PackData(begin, end, &jData, &vecLen);

    fp_t buff = 0.0;
#pragma omp parallel for reduction(+:buff)
    for (unsigned int jj = 0; jj < vecLen; jj+=VECSIZE)
      {
        Vec8f Ij;    Ij.load(jData+jj);
        Vec8f Nj;    Nj.load(jData+jj+vecLen);
        Vec8f Rj;    Rj.load(jData+jj+vecLen*2);
	Vec8f jSusc; jSusc.load(jData+jj+vecLen*3);
	Vec8f jInf;  jInf.load(jData+jj+vecLen*4);
	Vec8f d;     d.load(&(D_.value_data()[jj+begin]));

        Vec8f jOnIdx = 0.0;
	// Recalculate pressure from j on idx
	jOnIdx -= HVec8f(min(Nj, Ii) - min(Ii, Ij), *nu_, *alpha_)
	  + *gamma2_
	  * (HVec8f(min(Rj, Ii) - Ij, *nu_, *alpha_)
	     - HVec8f(min(Nj, Ii) - Ij, *nu_, *alpha_)); // Old pressure
	jOnIdx += HVec8f(min(Nj, Ni) - min(Ij, Ni), *nu_, *alpha_)
	  + *gamma2_
	  * (HVec8f(min(Rj, Ni) - Ij, *nu_, *alpha_)
	     - HVec8f(min(Nj, Ni) - Ij, *nu_, *alpha_)); // New pressure

	// Apply infec and suscep
	jOnIdx *= susceptibility_[i];
	jOnIdx *= jInf;

	// Mask uninfected j's
	jOnIdx = select(Ij < Nj, jOnIdx, 0.0f);

        // Subtract pressure from idx on j
        Vec8f IdxOnj = 0.0f;
        IdxOnj -= HVec8f(min(Ni, Ij) - min(Ii, Ij), *nu_, *alpha_);
        IdxOnj -= *gamma2_
            * (HVec8f(min(Ri, Ij) - Ii, *nu_, *alpha_)
                - HVec8f(min(Ni, Ij) - Ii, *nu_, *alpha_));
        IdxOnj *= jSusc;
        IdxOnj *= infectivity_[i];

        buff += horizontal_add((IdxOnj + jOnIdx)
			       * KVec8f(d, *delta_, *omega_));
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

