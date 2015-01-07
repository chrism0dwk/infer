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
#include <omp.h>

#include "vectorclass/vectorclass.h"
#include "vectorclass/vectormath_exp.h"
#include "CpuLikelihood.hpp"

#define VECSIZE 8

namespace EpiRisk
{
// Constants
  const float UNITY = 1.0;
  const float ZERO = 0.0;

  float
  GetDistElement(const CsrMatrix* d, const int row, const int col)
  {
    assert(row < d->n);
    assert(col < d->m);

    int start = d->rowPtr[row];
    int end = d->rowPtr[row + 1];
    for (int j = start; j < end; ++j)
      if (d->colInd[j] == col)
        return d->val[j];
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
  getDistMatrixElement(const int row, const int col, const CsrMatrix* csrMatrix,
      float* val)
  {
    int* cols = csrMatrix->colInd + csrMatrix->rowPtr[row];
    float* vals = csrMatrix->val + csrMatrix->rowPtr[row];
    int rowlen = csrMatrix->rowPtr[row + 1] - csrMatrix->rowPtr[row];

    for (int ptr = 0; ptr < rowlen; ++ptr)
      {
        if (cols[ptr] == col)
          {
            *val = vals[ptr];
            return true;
          }
      }
    return false;
  }

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
      popSize_(0), numSpecies_(nSpecies), obsTime_(obsTime), I1Time_(0.0), I1Idx_(
          0), occultsOnlyDC_(occultsOnlyDC), movtBan_(obsTime), workspaceA_(
      NULL), workspaceB_(NULL), workspaceC_(NULL)
  {

    // Load data into host memory
    int np;
#pragma omp parallel
      {
        np = omp_get_num_threads();
      }
    cout << "Using " << np << " threads" << endl;
    LoadPopulation(population);
    LoadEpidemic(epidemic);
    SortPopulation();
    omp_set_num_threads(8);
    CalcDistanceMatrix(dLimit);
    omp_set_num_threads(np);

    // Set up species and events
    SetSpecies();
    SetEvents();

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

  CpuLikelihood::~CpuLikelihood()
  {
    if (workspaceA_)
      delete[] workspaceA_;
    if (workspaceB_)
      delete[] workspaceB_;
    if (workspaceC_)
      delete[] workspaceC_;
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
                string msg(
                    "Key '" + record.id + "' not found in population data");
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

    if (!occultsOnlyDC_)
      maxInfecs_ = population_.size();
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
      suscOccults_[i - numKnownInfecs_] = i;

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
  CpuLikelihood::SetParameters(Parameter& epsilon1, Parameter& epsilon2,
      Parameter& gamma1, Parameter& gamma2, Parameters& xi, Parameters& psi,
      Parameters& zeta, Parameters& phi, Parameter& delta, Parameter& omega,
      Parameter& nu, Parameter& alpha, Parameter& a, Parameter& b)
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

#pragma omp parallel private(workA, workB, workC)
      {
        workA = new float[popSize_];
        workB = new float[popSize_];
        workC = new float[popSize_];

#pragma omp for ordered schedule(static) reduction(+:res)
        for (size_t ii = 0; ii < infecIdx_.size(); ++ii)
          {
            size_t i = infecIdx_[ii].ptr;
            size_t begin = D_.index1_data()[i];
            size_t end = D_.index1_data()[i + 1];
            float Ii = eventTimes_(i, 0);
            float Ni = eventTimes_(i, 1);
            float Ri = eventTimes_(i, 2);

            //////// VECTORIZATION ATTEMPT ////////

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
                betaij.store(workA + j);
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

    cout << "Move individual " << i << " from " << oldTime << " to " << newTime
        << endl;

    bool haveNewI1 = false;
    if (newTime < I1Time_ or i == I1Idx_)
      {
        cout << "Have New I1" << endl;
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

  float
  CpuLikelihood::GetLogLikelihood() const
  {
    return logLikelihood_;
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

