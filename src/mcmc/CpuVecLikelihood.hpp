/*************************************************************************
 *  ./src/mcmc/CpuLikelihood.hpp
 *  Copyright Chris Jewell <c.p.jewell@massey.ac.nz> 2013
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
 * GpuLikelihood.hpp
 *
 *  Created on: Nov 20th, 2013
 *      Author: stsiab
 */

#ifndef CPULIKELIHOOD_HPP_
#define CPULIKELIHOOD_HPP_

#include <map>
#include <ostream>
#include <vector>
#include <string>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix_sparse.hpp>

#include "Likelihood.hpp"


namespace EpiRisk
{

  using namespace boost::numeric;

  class CpuLikelihood : public Likelihood
  {
  public:
    explicit
    CpuLikelihood(PopDataImporter& population, EpiDataImporter& epidemic,
        const size_t nSpecies,
		  const float obsTime, const float dLimit, const bool occultsOnlyDC = false);
    CpuLikelihood(const CpuLikelihood& rhs);
    ~CpuLikelihood();
    CpuLikelihood*
    clone() const;
    void
    InfecCopy(const Likelihood& rhs);
    void
    CalcDistanceMatrix(const float dLimit);
    void
    SetEvents();
    void
    SetSpecies();
    void
    SetDistance(const float* data, const int* rowptr, const int* colind);
    void
    RefreshParameters();
    size_t
    GetNumInfecs() const;
    size_t
    GetNumPossibleOccults() const;
    size_t
    GetNumOccults() const;
    void
    FullCalculate();
    void
    Calculate();
    void
    UpdateInfectionTime(const unsigned int idx, const float inTime);
    void
    AddInfectionTime(const unsigned int idx, const float inTime);
    void
    DeleteInfectionTime(const unsigned int idx);
    fp_t
    NonCentreInfecTimes(const fp_t oldGamma, const fp_t newGamma,
			const fp_t prob);
    fp_t
    InfectionPart();
    fp_t
    GetIN(const size_t index);
    fp_t
    GetMeanI2N() const;
    fp_t
    GetMeanOccI() const;
    void
    GetInfectiousPeriods(std::vector<IPTuple_t>& periods);
    void
    GetSumInfectivityPow(fp_t* result) const;
    void
    GetSumSusceptibilityPow(fp_t* result) const;
    fp_t
    GetLogLikelihood() const;
    LikelihoodComponents
    GetLikelihoodComponents() const
    {
      return likComponents_;
    }

    void
    PrintLikelihoodComponents() const;
    void
    PrintParameters() const;
    void
    PrintEventTimes() const;
    void
    PrintProdCache() const;
    const ublas::vector<fp_t>&
    GetProdCache() const;

  private:

    // Helper methods
    const Likelihood&
    assign(const Likelihood& rhs);
    void
    ReduceProductVector();
    void
    PackData(const size_t begin, const size_t end, fp_t** data, size_t* pitch);
    void
    UpdateInfectionTimeInteg(const unsigned int i, const fp_t newTime);
    void
    UpdateInfectionTimeProd(const unsigned int i, const fp_t newTime);
    void
    AddInfectionTimeProd(const unsigned int i, const fp_t newTime);
    void
    AddInfectionTimeInteg(const unsigned int i, const fp_t newTime);
    void
    DelInfectionTimeProd(const unsigned int i);
    void
    DelInfectionTimeInteg(const unsigned int i);
    void
    CalcSusceptibilityPow();
    void
    CalcSusceptibility();
    void
    CalcInfectivityPow();
    void
    CalcInfectivity();
    void
    UpdateI1();
    void
    CalcBgIntegral();
    void
    CalcProduct();
    void
    CalcIntegral();

    // Host vars
    size_t popSizePitch_;
    size_t maxInfecsPitch_;

    size_t numThreads_;

    std::vector<InfecIdx_t> infecIdx_;
    std::vector<InfecIdx_t> suscOccults_;
    fp_t logLikelihood_;
    fp_t I1Time_;
    unsigned int I1Idx_;
    ublas::vector<fp_t> productCache_;
    LikelihoodComponents likComponents_;

    // Covariate data is shared over a copy
    ublas::matrix<fp_t,ublas::column_major> animals_;
    ublas::compressed_matrix<fp_t> D_;

    size_t dnnz_; //CRS

    ublas::matrix<fp_t,ublas::column_major> animalsInfPow_;
    ublas::matrix<fp_t,ublas::column_major> animalsSuscPow_;
    ublas::matrix<fp_t,ublas::column_major> eventTimes_;

    ublas::vector<fp_t> susceptibility_;
    ublas::vector<fp_t> infectivity_;

  };


} // namespace EpiRisk

#endif /* CPULIKELIHOOD_HPP_ */
