// GPLv3 Here

#include <iostream>
#include <cstdlib>
#include <unistd.h>
#include <string>
#include <boost/bind.hpp>

#include "Mcmc.hpp"
#include "McmcFactory.hpp"
#include "MCMCUpdater.hpp"
#include "Data.hpp"
#include "Parameter.hpp"
#include "GpuLikelihood.hpp"
#include "PosteriorHDF5Writer.hpp"

#include "SpSINRMcmc.hpp"

#define RNUM(x) Rcpp::NumericVector(x)

class GammaPrior : public EpiRisk::Prior
{
  float shape_;
  float rate_;
public:
  GammaPrior(const float shape, const float rate)
  {
    shape_ = shape;
    rate_ = rate;
  }
  float
  operator()(const float x)
  {
    return gsl_ran_gamma_pdf(x, shape_, 1 / rate_);
  }
  Prior*
  create() const
  {
    return new GammaPrior(shape_, rate_);
  }
  Prior*
  clone() const
  {
    return new GammaPrior(*this);
  }
};

class BetaPrior : public EpiRisk::Prior
{
  float a_;
  float b_;
public:
  BetaPrior(const float a, const float b) :
    a_(a), b_(b)
  {
  }
  ;
  float
  operator()(const float x)
  {
    return gsl_ran_beta_pdf(x, a_, b_);
  }
  Prior*
  create() const
  {
    return new BetaPrior(a_, b_);
  }
  Prior*
  clone() const
  {
    return new BetaPrior(*this);
  }
};




class PopRImporter : public EpiRisk::PopDataImporter
{
public:
  PopRImporter(Rcpp::DataFrame population) : _rownum(0) { 
    _ids = population[0];
    _x = population[1];
    _y = population[2];

    for(int i=3; i<population.size(); ++i)
      species_.push_back(population[i]);
    }
  virtual 
  ~PopRImporter() {};
  void open() {};
  void close() {};
  Record 
  next() { 
    Record record;
    
    if(_rownum >= _ids.size()) throw EpiRisk::fileEOF();
    
    record.id = std::string(_ids[_rownum]);
    record.data.x = _x[_rownum];
    record.data.y = _y[_rownum];
    record.data.cattle = species_[0][_rownum];
    if(species_.size() > 1)
      record.data.pigs = species_[1][_rownum];
    if(species_.size() > 2)
      record.data.sheep = species_[2][_rownum];

    _rownum++;

    return record;
  };
  void reset() { _rownum = 0; };

private:
  int _rownum;
  Rcpp::CharacterVector _ids;
  Rcpp::NumericVector _x, _y;
  std::vector<Rcpp::NumericVector> species_;
};


class EpiRImporter : public EpiRisk::EpiDataImporter
{
public:
  EpiRImporter(Rcpp::DataFrame epidemic) : _rownum(0)
  {
    _ids = epidemic["id"];
    _i = epidemic["i"];
    _n = epidemic["n"];
    _r = epidemic["r"];
    _type = epidemic["type"];
  }
  virtual
  ~EpiRImporter() {}
  void open() {};
  void close() {};
  Record
  next() {
    Record record;

    if(_rownum >= _ids.size()) throw EpiRisk::fileEOF();

    record.id = std::string(_ids[_rownum]);

    if (_type[_rownum] == "DC") record.data.I = EpiRisk::POSINF;
    else record.data.I = _i[_rownum];

    record.data.N = _n[_rownum];
    record.data.R = _r[_rownum];
    record.data.type = _type[_rownum];

    _rownum++;

    return record;
  }
  void
  reset() { _rownum = 0; }

private:
  int _rownum;
  Rcpp::CharacterVector _ids;
  Rcpp::NumericVector _i;
  Rcpp::NumericVector _n;
  Rcpp::NumericVector _r;
  Rcpp::CharacterVector _type;
};

RcppExport SEXP SpSINRMcmc(const SEXP population, 
			   const SEXP epidemic,
			   const SEXP obsTime,
			   const SEXP movtBan,
			   const SEXP init,
			   const SEXP priorParms,
			   const SEXP control,
			   const SEXP outputfile)
{

  cout << "Starting..." << endl;
  try {

  Rcpp::DataFrame _population(population);
  Rcpp::DataFrame _epidemic(epidemic);
  Rcpp::NumericVector _obsTime(obsTime);
  Rcpp::NumericVector _movtBan(movtBan);
  Rcpp::List _init(init);
  Rcpp::List _priorParms(priorParms);
  Rcpp::List _control(control);
  Rcpp::CharacterVector _outputfile(outputfile);
  
  Rcpp::IntegerVector numIter = _control["n.iter"];
  Rcpp::IntegerVector gpuId = _control["gpuid"];
  Rcpp::LogicalVector doMovtBan = _control["movtban"];
  Rcpp::CharacterVector occults = _control["occults"];
  Rcpp::LogicalVector doPowers = _control["powers"];
  Rcpp::IntegerVector seed = _control["seed"];    
  Rcpp::NumericVector ncratio = _control["ncratio"]; 
  Rcpp::NumericVector tuneI = _control["tune.I"];    
  Rcpp::IntegerVector repsI = _control["reps.I"];    
  Rcpp::LogicalVector doLatentPeriodScale = _control["infer.latent.period.scale"];
  Rcpp::NumericVector dLimit = _control["dlimit"];

  size_t nSpecies = _population.size() - 3;
  bool dcOnly = false;
  bool doOccults = false;

  if(string(occults[0]) == "dconly") {
    dcOnly = true;
    doOccults = true;
  }
  else if(string(occults[0]) == "yes")
    doOccults = true;

  // Set up likelihood
  PopRImporter* popRImporter = new PopRImporter(_population);
  EpiRImporter* epiRImporter = new EpiRImporter(_epidemic);

  EpiRisk::GpuLikelihood likelihood(*popRImporter, *epiRImporter,
  				    nSpecies, _obsTime[0], dLimit[0], 
  				    dcOnly, gpuId[0]);

  delete popRImporter;
  delete epiRImporter;

  
  // Set up parameters
  Rcpp::NumericVector prior = _priorParms["epsilon1"];
  Rcpp::NumericVector startval =_init["epsilon1"];
  EpiRisk::Parameter epsilon1(startval[0], GammaPrior(prior[0], prior[1]), "epsilon1");

  if(doMovtBan[0]) { 
    startval = _init["epsilon2"]; 
    prior = _priorParms["epsilon2"];
  }
  else {
    startval[0] = 1.0;
    prior[0] = 1.0; prior[1] = 1.0;
  }
  EpiRisk::Parameter epsilon2(startval[0], GammaPrior(prior[0], prior[1]), "epsilon2");
  startval = _init["gamma1"]; prior = _priorParms["gamma1"];
  EpiRisk::Parameter gamma1(startval[0], GammaPrior(prior[0], prior[1]), "gamma1");
  startval = _init["gamma2"]; prior = _priorParms["gamma2"];
  EpiRisk::Parameter gamma2(startval[0], GammaPrior(prior[0], prior[1]), "gamma2");

  EpiRisk::Parameters xi(nSpecies);
  EpiRisk::Parameters psi(nSpecies);
  EpiRisk::Parameters zeta(nSpecies);
  EpiRisk::Parameters phi(nSpecies);

  xi[0] = EpiRisk::Parameter(1.0, GammaPrior(1, 1), "xi_1");
  zeta[0] = EpiRisk::Parameter(1.0, GammaPrior(1, 1), "zeta_1");
  startval = _init["psi_1"]; prior = _priorParms["psi_1"];
  psi[0] = EpiRisk::Parameter(startval[0], BetaPrior(prior[0], prior[1]), "psi_1");
  startval = _init["phi_1"]; prior = _priorParms["phi_1"];
  phi[0] = EpiRisk::Parameter(startval[0], BetaPrior(prior[0], prior[1]), "phi_1");

  if (nSpecies > 1) {
    startval = _init["xi_2"]; prior = _priorParms["xi_2"];
    xi[1] = EpiRisk::Parameter(startval[0], GammaPrior(prior[0], prior[1]), "xi_2");
    startval = _init["psi_2"]; prior = _priorParms["psi_2"];
    psi[1] = EpiRisk::Parameter(startval[0], BetaPrior(prior[0], prior[1]), "psi_2");
    startval = _init["zeta_2"]; prior = _priorParms["zeta_2"];
    zeta[1] = EpiRisk::Parameter(startval[0], GammaPrior(prior[0], prior[1]), "zeta_2");
    startval = _init["phi_2"]; prior = _priorParms["phi_2"];
    phi[1] = EpiRisk::Parameter(startval[0], BetaPrior(prior[0], prior[1]), "phi_2");
  }
  else {
    xi[1] = EpiRisk::Parameter(1.0f, GammaPrior(1, 1), "xi_2");
    psi[1] = EpiRisk::Parameter(0.0f, BetaPrior(1, 1), "psi_2");
    zeta[1] = EpiRisk::Parameter(1.0f, GammaPrior(1, 1), "zeta_2");
    psi[1] = EpiRisk::Parameter(0.0f, BetaPrior(1, 1), "psi_2");
  }
  if (nSpecies > 2) {
    startval = _init["xi_3"]; prior = _priorParms["xi_3"];
    xi[2] = EpiRisk::Parameter(startval[0], GammaPrior(prior[0], prior[1]), "xi_3");
    startval = _init["psi_3"]; prior = _priorParms["psi_3"];
    psi[2] = EpiRisk::Parameter(startval[0], BetaPrior(prior[0], prior[1]), "psi_3");
    startval = _init["zeta_3"]; prior = _priorParms["zeta_3"];
    zeta[2] = EpiRisk::Parameter(startval[0], GammaPrior(prior[0], prior[1]), "zeta_3");
    startval = _init["phi_3"]; prior = _priorParms["phi_3"];
    phi[2] = EpiRisk::Parameter(startval[0], BetaPrior(prior[0], prior[1]), "phi_3");
  }
  else {
    xi[2] = EpiRisk::Parameter(1.0f, GammaPrior(1, 1), "xi_3");
    psi[2] = EpiRisk::Parameter(0.0f, BetaPrior(1, 1), "psi_3");
    zeta[2] = EpiRisk::Parameter(1.0f, GammaPrior(1, 1), "zeta_3");
    psi[2] = EpiRisk::Parameter(0.0f, BetaPrior(1, 1), "psi_3");
  }

  startval = _init["delta"]; prior = _priorParms["delta"];
  EpiRisk::Parameter delta(startval[0], GammaPrior(prior[0], prior[1]), "delta");
  EpiRisk::Parameter nu(0.001, GammaPrior(1, 1), "nu");
  startval = _init["alpha"];
  EpiRisk::Parameter alpha(startval[0], GammaPrior(1, 1), "alpha");
  startval = _init["a"];
  EpiRisk::Parameter a(startval[0], GammaPrior(1,1), "a");
  startval = _init["b"]; prior = _priorParms["b"];
  EpiRisk::Parameter b(startval[0], GammaPrior(prior[0], prior[1]), "b");

  likelihood.SetMovtBan(_movtBan[0]);
  likelihood.SetParameters(epsilon1, epsilon2, gamma1, gamma2, xi, psi, zeta, phi, delta,
      nu, alpha, a, b);

  // Set up MCMC algorithm
  cout << "Initializing MCMC" << endl;
  EpiRisk::Mcmc::Initialize();

  EpiRisk::Mcmc::McmcRoot mcmc(likelihood, seed[0]);

  EpiRisk::UpdateBlock txDelta;
  txDelta.add(epsilon1);
  if(doMovtBan[0]) txDelta.add(epsilon2);
  txDelta.add(gamma1);
  txDelta.add(gamma2);
  txDelta.add(delta);

  EpiRisk::Mcmc::AdaptiveMultiLogMRW* updateDistance =
    (EpiRisk::Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW",
          "txBase");
  updateDistance->SetParameters(txDelta);

  EpiRisk::UpdateBlock txPsi;
  EpiRisk::UpdateBlock txPhi;

  if(doPowers[0]) {
    for(int i = 0; i<nSpecies; i++) {
      txPsi.add(psi[i]);
      txPhi.add(phi[i]);
    }

    if(nSpecies > 1) {
    EpiRisk::Mcmc::AdaptiveMultiLogMRW* updatePsi =
      (EpiRisk::Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW", "txPsi");
    updatePsi->SetParameters(txPsi);

    EpiRisk::Mcmc::AdaptiveMultiLogMRW* updatePhi =
      (EpiRisk::Mcmc::AdaptiveMultiLogMRW*) mcmc.Create("AdaptiveMultiLogMRW", "txPhi");
    updatePhi->SetParameters(txPhi);
    }
    else {
      EpiRisk::Mcmc::AdaptiveSingleMRW* updatePsi = 
  	(EpiRisk::Mcmc::AdaptiveSingleMRW*) mcmc.Create("AdaptiveSingleLogMRW", "txPsi");
      updatePsi->SetParameters(txPsi);

      EpiRisk::Mcmc::AdaptiveSingleMRW* updatePhi = 
  	(EpiRisk::Mcmc::AdaptiveSingleMRW*) mcmc.Create("AdaptiveSingleMRW", "txPhi");
      updatePhi->SetParameters(txPhi);
    }
  }

  // Make decisions here based on number of species

    EpiRisk::UpdateBlock txInfec;
    EpiRisk::UpdateBlock txSuscep;
    
    if(nSpecies > 1) {
      
      txInfec.add(gamma1);
      txInfec.add(xi[1]);
      txInfec.add(xi[2]);
      
      EpiRisk::Mcmc::InfectivityMRW* updateInfec = 
  	(EpiRisk::Mcmc::InfectivityMRW*) mcmc.Create("InfectivityMRW", "txInfec");
      updateInfec->SetParameters(txInfec);
      
      txSuscep.add(gamma1);
      txSuscep.add(zeta[1]);
      txSuscep.add(zeta[2]);
      EpiRisk::Mcmc::SusceptibilityMRW* updateSuscep =
  	(EpiRisk::Mcmc::SusceptibilityMRW*) mcmc.Create("SusceptibilityMRW", "txSuscep");
      updateSuscep->SetParameters(txSuscep);
    }

  EpiRisk::UpdateBlock infecPeriod;
  infecPeriod.add(a);
  infecPeriod.add(b);
  EpiRisk::Mcmc::InfectionTimeUpdate* updateInfecTime =
    (EpiRisk::Mcmc::InfectionTimeUpdate*) mcmc.Create("InfectionTimeUpdate",
          "infecTimes");
  updateInfecTime->SetParameters(infecPeriod);
  updateInfecTime->SetUpdateTuning(tuneI[0]);
  updateInfecTime->SetReps(repsI[0]);
  updateInfecTime->SetOccults(doOccults);

  EpiRisk::UpdateBlock bUpdate;
  if(doLatentPeriodScale[0]) {
    bUpdate.add(b);
    EpiRisk::Mcmc::InfectionTimeGammaCentred* updateBC =
      (EpiRisk::Mcmc::InfectionTimeGammaCentred*) mcmc.Create("InfectionTimeGammaCentred", "b_centred");
    updateBC->SetParameters(bUpdate);
    updateBC->SetTuning(0.014);

    EpiRisk::Mcmc::InfectionTimeGammaNC* updateBNC =
      (EpiRisk::Mcmc::InfectionTimeGammaNC*)mcmc.Create("InfectionTimeGammaNC", "b_ncentred");
    updateBNC->SetParameters(bUpdate);
    updateBNC->SetTuning(0.0007);
    updateBNC->SetNCRatio(ncratio[0]);
  }

    //// Output ////

    // Make output directory
  string outputFile(_outputfile[0]);
  EpiRisk::PosteriorHDF5Writer output(outputFile, likelihood);
  output.AddParameter(epsilon1); output.AddParameter(epsilon2);
  output.AddParameter(gamma1);
  output.AddParameter(gamma2);
  output.AddParameter(xi[1]);   output.AddParameter(xi[2]);
  output.AddParameter(psi[0]);  output.AddParameter(psi[1]);
  output.AddParameter(psi[2]);
  output.AddParameter(zeta[1]); output.AddParameter(zeta[2]);
  output.AddParameter(phi[0]);  output.AddParameter(phi[1]);
  output.AddParameter(phi[2]);  output.AddParameter(delta);
  output.AddParameter(b);
  

  boost::function< float () > getnuminfecs = boost::bind(&EpiRisk::GpuLikelihood::GetNumInfecs, &likelihood);
  output.AddSpecial("numInfecs",getnuminfecs);
  boost::function< float () > getmeanI2N = boost::bind(&EpiRisk::GpuLikelihood::GetMeanI2N, &likelihood);
  output.AddSpecial("meanI2N", getmeanI2N);
  boost::function< float () > getmeanOccI = boost::bind(&EpiRisk::GpuLikelihood::GetMeanOccI, &likelihood);
  output.AddSpecial("meanOccI", getmeanOccI);
   boost::function< float () > getlikelihood = boost::bind(&EpiRisk::GpuLikelihood::GetLogLikelihood, &likelihood);
  output.AddSpecial("loglikelihood",getlikelihood);
 
  // Run the chain
  Rcpp::Rcout << "Running MCMC" << std::endl;
  for(int k=0; k<numIter[0]; ++k)
    {
      mcmc.Update();
      output.write();

      if(k % 500 == 0)
  	{
	  Rcpp::Rcout << "Iteration " << k << std::endl;
  	  output.flush();
  	}

    }
  
  // Wrap up
  map<string, float> acceptance = mcmc.GetAcceptance();
  
  for(map<string, float>::const_iterator it = acceptance.begin();
     it != acceptance.end(); ++it)
   {
     Rcpp::Rcout << it->first << ": " << it->second << "\n";
   }
  
  // cout << "Covariances\n";
  // cout << updateDistance->GetCovariance() << "\n";
  // cout << updatePsi->GetCovariance() << "\n";
  // cout << updatePhi->GetCovariance() << "\n";
  // cout << updateInfec->GetCovariance() << "\n";
  // cout << updateSuscep->GetCovariance() << "\n";

  }
  catch (std::exception& __ex__)
    {
      forward_exception_to_r(__ex__);
    }
  catch (H5::Exception& e)
    {
      ::Rf_error(e.getCDetailMsg());
    }
  catch (...)
    {
      ::Rf_error("c++ exception (unknown reason)");
    }
  
  cout << "Finishing..." << endl;

  return outputfile;

}





