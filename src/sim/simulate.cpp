#include <algorithm>
#include <iterator>
#include <map>
#include <list>
#include <stdexcept>
#include <iostream>
#include <cassert>

#include <math.h>

#include "RUtils.hpp"
#include "simulate.hpp"


float _h(const float t, const float I, float nu, float alpha1, float alpha2, float alpha3)
{
  // Periodic piece-wise square wave
  const float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
  float Y[] = {1.0f, 1.0f, 1.0f, 1.0f,1.0f};

  assert(t-I >= 0);

  // Re-scale time to unit period
  float tAdj = (t+nu)/365.0f;
  tAdj = tAdj - floorf(tAdj);
    
  // Set up parameters
  Y[0] = alpha1; Y[2] = alpha2; Y[3] = alpha3; Y[4] = alpha1;
    
  // Calculate which epoch we are in
  int epoch = 0;
  assert(tAdj <= 1.0f);
  while(tAdj > T[epoch+1]) epoch++;
    
  return Y[epoch];
}




namespace EpiRisk {

  Simulator::Simulator(DataFrame population, S4 contact, NumericVector parameter, NumericVector dLimit) :
    population_(population),
    contact_(new S4CMatrixView(contact)),
    parameter_(parameter),
    dLimit_(dLimit[0])
  {
    x_ = population_["x"];
    y_ = population_["y"];
    ticks_ = population_["ticks"];
    infecTime_ = population_["i"];
    popSize_ = infecTime_.size();
    Initialize();
  }
    
  Simulator::~Simulator() 
  {
    delete contact_;
  };

  void
  Simulator::Initialize()
  {
        // Find infectives
    time_ = R_NegInf;
    infecList_.clear();
    for(size_t i = 0; i<infecTime_.size(); ++i)
       {
	if(infecTime_[i] < R_PosInf) {
	  infecList_.push_back(i);
	  if (infecTime_[i] > time_) time_ = infecTime_[i];
	}
      }
  }

  void
  Simulator::Gillespie(const double maxtime)
  {
    RNGScope scope;
        
    int interruptCheckTimer = 0;

    Initialize();

    // Calculate initial pairwise pressures and add to CDF
    double cumPressure = 0.0;
    for(size_t j = 0; j < infecTime_.size(); j++) {
            
      if(infecTime_[j] == R_PosInf) {
	cumPressure += background();
	for(IntPtrList::const_iterator i = infecList_.begin();
	    i != infecList_.end(); ++i)
	  {
	    cumPressure += beta(*i,j);
	  }
	pressure_.insert(PressureCDF::value_type(cumPressure, j));
      }
    }

#ifndef NDEBUG
    std::cerr << "Pressure map size = " << pressure_.size() << std::endl;
    std::cerr << "Maxtime is " << maxtime << std::endl;
    std::cerr << "epsilon: " << parameter_[0] << "\n"
	      << "beta1: " << parameter_[1] << "\n"
	      << "beta2: " << parameter_[2] << "\n"
	      << "mu: " << parameter_[3] << "\n"
	      << "nu: " << parameter_[4] << "\n"
	      << "delta: " << parameter_[5] << "\n"
	      << "Pressure: " << GetMaxPressure() << std::endl;
#endif

    while (GetMaxPressure() > 0.0) {
            
      interruptCheckTimer++;
      if(interruptCheckTimer == 100) {
	if(checkInterrupt()) throw std::runtime_error("Execution cancelled by user");
	interruptCheckTimer = 0;
      }
      // Draw time to next infection

#ifndef NDEBUG
      std::cerr << "Max pressure: " << GetMaxPressure() << std::endl;
#endif
            
      double timeOfNextI = rexp(1,GetMaxPressure())[0];
            
      time_ += timeOfNextI;
            
      if(time_ > maxtime) break;
            
      // Draw next infectee
      double u = runif(1,0,GetMaxPressure())[0];
      PressureCDF::iterator infectee = pressure_.lower_bound(u);
            
      // Construct infector CDF and calculate h() function
      PressureCDF infectors;
      double cumPressure = 0.0;
            
      for (IntPtrList::iterator i = infecList_.begin();
	   i != infecList_.end();
	   ++i) {
	double b = beta(*i,infectee->second);
	if(b > 0.0) {
	  cumPressure += b;
	  infectors.insert(PressureCDF::value_type(cumPressure,*i));
	}
      }
            
      // Choose if to have a real infection or ghost
      bool isInfectious = false;
      if(infectors.empty()) isInfectious = true; // Background infection
      else {
	u = runif(1,0,cumPressure)[0];
	size_t infector = infectors.lower_bound(u)->second;
	u = runif(1,0,1)[0];
	if(u < h(infector, time_)) isInfectious = true;
      }
            
#ifndef NDEBUG
      std::cerr << "time_ = " << time_ <<std::endl;
#endif
            
      if(isInfectious) {
                
	// Save infection time in population and add to infecList
	infecTime_[infectee->second] = time_;
	infecList_.push_back(infectee->second);
                
	// Add new pressure, and delete the old
	double oldPressure;
	if(infectee == pressure_.begin()) oldPressure = infectee->first;
	else {
	  PressureCDF::iterator tmp(infectee);
	  --tmp;
	  oldPressure = infectee->first - tmp->first;
	}
                
	PressureCDF::iterator it;
	cumPressure = 0.0;
	for(it = pressure_.begin(); it != infectee; ++it)
	  {
	    cumPressure += beta(infectee->second, it->second);
	    const_cast<double&> (it->first) += cumPressure;
	  }
	it = infectee;
	++it;
	for(; it != pressure_.end(); ++it)
	  {
	    cumPressure += beta(infectee->second, it->second);
	    const_cast<double&> (it->first) += cumPressure - oldPressure;
	  }
                
	pressure_.erase(infectee);
      }
            
    }
  }

  DataFrame
  Simulator::GetPopulation() const
  {
    return population_;
  }

  inline
  double
  Simulator::h(const size_t i, const double time) const
  {
    double ans = 0.0;
    double alpha1 = parameter_[6];
    double alpha2 = parameter_[7];
    double alpha3 = parameter_[8];
    double nu = parameter_[4];

    if (time > infecTime_[i])
    {
      ans = _h(time, time-1.0f, nu, alpha1, alpha2, alpha3);
    }

    return ans;
  }


  inline
  double
  Simulator::K(const size_t i, const size_t j) const
  {
    double dx = x_[i] - x_[j];
    double dy = y_[i] - y_[j];
    double d = dx*dx + dy*dy;
    if(d < dLimit_*dLimit_)
      return parameter_[5] / pow(parameter_[5]*parameter_[5] + d,1.2);
    //return expf(-parameter_[5] * (sqrtf(d) - 5.0f));
    else return 0.0;
  }
    
  inline
  double
  Simulator::beta(const size_t i, const size_t j) const
  {
    double beta = ticks_[j] * (parameter_[2]*(*contact_)(i,j) + parameter_[1]*K(i,j));
        
    return beta;
  }

  inline
  double
  Simulator::background() const
  {
    return parameter_[0];
  }
    
  inline
  double
  Simulator::GetMaxPressure() const
  {
    PressureCDF::const_iterator tmp = pressure_.end();
    --tmp;
    return tmp->first;
  }

}

SEXP GetMatrixElement(SEXP contact, SEXP i, SEXP j)
{
  using namespace Rcpp;
  using namespace EpiRisk;
  IntegerVector i_(i);
  IntegerVector j_(j);
  S4CMatrixView mat(contact);
  return wrap(mat(i_[0],j_[0]));
}


SEXP Simulate(SEXP population, SEXP contact, SEXP parameter, SEXP dLimit, SEXP maxtime)
{
  using namespace Rcpp ;
  using namespace EpiRisk ;
    
  try {
    Simulator sim(population, contact,parameter, dLimit);
    NumericVector t(maxtime);
    
    sim.Gillespie(t[0]);

    return wrap(sim.GetPopulation());
  }
  catch (std::exception& e) {
    forward_exception_to_r(e);
  }
  catch (...) {
    ::Rf_error("unknown C++ exception");
  }
  return NumericVector();
}
