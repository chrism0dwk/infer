/*
 * SIRMcmc.hpp
 *
 *  Created on: May 21, 2010
 *      Author: stsiab
 */

#ifndef SIRMCMC_HPP_
#define SIRMCMC_HPP_

#include <string>
#include <map>
#include <math.h>

#include "SmpTypes.hpp"
#include "SpatMetaPop.hpp"

#define ADDMULTRATIO 0.8
#define NONCENTREDPROP 0.25
#define ISHAPE 6.0
#define MU 1.0
#define NU 1.0


namespace EpiRisk {
  namespace Smp {

#define NUMINFECSTOMOVE 15


class SmpSirMcmc {
public:
	SmpSirMcmc(SpatMetaPop& population);
	virtual ~SmpSirMcmc();

	void run(size_t numIterations, const string outputFilename);
    void getPriorParAlpha(double* lambda, double* nu) const
    {
        *lambda = priorParAlpha_[0];
        *nu = priorParAlpha_[1];
    }

    void getPriorParBeta(double* lambda, double* nu) const
    {
        *lambda = priorParBeta_[0];
        *nu = priorParBeta_[1];
    }

    void getPriorParGamma(double* lambda, double* nu) const
    {
        *lambda = priorParGamma_[0];
        *nu = priorParGamma_[1];
    }

    void getPriorParRho(double* lambda, double* nu) const
    {
        *lambda = priorParRho_[0];
        *nu = priorParRho_[1];
    }

    void setPriorParAlpha(const double lambda, const double nu)
    {
        this->priorParAlpha_[0] = lambda;
        this->priorParAlpha_[1] = nu;
    }

    void setPriorParBeta(const double lambda, const double nu)
    {
        this->priorParBeta_[0] = lambda;
        this->priorParBeta_[1] = nu;
    }

    void setPriorParGamma(const double lambda, const double nu)
    {
        this->priorParGamma_[0] = lambda;
        this->priorParGamma_[1] = nu;
    }

    void setPriorParRho(const double lambda, const double nu)
    {
        this->priorParRho_[0] = lambda;
        this->priorParRho_[1] = nu;
    }

    size_t getBurnin() const
    {
        return burnin_;
    }

    size_t getThin() const
    {
        return thin_;
    }

    void setBurnin(size_t burnin_)
    {
        this->burnin_ = burnin_;
    }

    void setThin(size_t thin_)
    {
        this->thin_ = thin_;
    }

    double getAcceptAlpha() const
    {
        return acceptAlpha_ / iteration_;
    }

    double getAcceptBeta() const
    {
        return acceptBeta_ / iteration_;
    }

    double getAcceptRhoLog() const
    {
    	return acceptRhoLog_ / (iteration_*ADDMULTRATIO);
    }

    double getAcceptRhoLin() const
    {
        return acceptRhoLin_ / (iteration_*(1-ADDMULTRATIO));
    }

    double getAcceptI() const
    {
    	return acceptI_ / (iteration_ * NUMINFECSTOMOVE);
    }

    double getAcceptGamma() const
    {
    	return acceptGamma_ / iteration_;
    }

    double getTuneAlpha() const
    {
        return tuneAlpha_;
    }

    double getTuneBeta() const
    {
        return tuneBeta_;
    }

    double getTuneRhoLog() const
    {
        return tuneRhoLog_;
    }

    double getTuneI() const
    {
    	return tuneI_;
    }

    double getTuneGamma() const
    {
    	return tuneGamma_;
    }

    void setTuneAlpha(double tuneAlpha)
    {
        this->tuneAlpha_ = tuneAlpha;
    }

    void setTuneBeta(double tuneBeta)
    {
        this->tuneBeta_ = tuneBeta;
    }

    void setTuneRhoLog(double tuneRho)
    {
        this->tuneRhoLog_ = tuneRho;
    }

    void setTuneRhoLin(double tuneRho)
    {
        this->tuneRhoLin_ = tuneRho;
    }

    void setTuneI(double tuneI)
    {
    	this->tuneI_ = tuneI;
    }

    void setTuneGamma(double tuneGamma)
    {
    	this->tuneGamma_ = tuneGamma;
    }

    void initParameters(double alpha, double beta, double rho, double gamma)
    {
    	params_.alpha = alpha;
    	params_.beta = beta;
    	params_.rho = rho;
    	params_.gamma = gamma;
    }

private:
	SpatMetaPop& population_;
	SmpParams params_;

	double logLikCur_;
	double logLikCan_;

	double priorParAlpha_[2];
	double priorParBeta_[2];
	double priorParRho_[2];
	double priorParGamma_[2];

	double tuneAlpha_;
	double tuneBeta_;
	double tuneRhoLog_;
	double tuneRhoLin_;
	double tuneI_;
	double tuneGamma_;

	double acceptAlpha_;
	double acceptBeta_;
	double acceptRhoLog_;
	double acceptRhoLin_;
	double acceptI_;
	double acceptGamma_;

	size_t burnin_;
	size_t thin_;
	size_t iteration_;
	FILE* outputFile;

	typedef multimap<double,SpatMetaPop::CommuneVector::iterator> CommuneEDF;
	CommuneEDF sampleEDF;

	Commune& pickCommune();

	bool updateBeta();
	bool updateRhoLog();
	bool updateRhoLin();
	bool updateAlpha();
	bool updateInfectionTime();
	bool updateGammaGibbs();
	bool updateGammaNC();
	void makeUiTransform();

	double lambda(Commune& commune, const double time) const;
	double integLambda(Commune& commune) const;
	double totalIntegPressure();
	double logProduct();
	double likelihood();

	double priorAlpha(const double val) const;
	double priorBeta(const double val) const;
	double priorRho(const double val) const;
	double priorI(const double val) const;
	double priorGamma(const double val) const;

	void progressBar();
	void openOutputFile(const string filename);
	void writeParameters();
	void closeOutputFile();


	// Progress bar
	size_t numIterations_;
	const size_t barLength_;
	size_t spinnerPos_;

};

  } }

#endif /* SIRMCMC_HPP_ */
