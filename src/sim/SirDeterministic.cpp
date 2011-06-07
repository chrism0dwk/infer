/*
 *  sirDeterministic.cpp
 *  sirDeterministic
 *
 *  Created by Chris Jewell on 19/05/2011.
 *  Copyright 2011 University of Warwick. All rights reserved.
 *
 */

#include "SirDeterministic.hpp"

#include <iostream>

#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>

namespace EpiRisk {
	
	static
	int
	model (double t, const double y[], double f[], void* params)
	{
		double beta = *(double *)params;
		double gamma = *((double *)params + 1);
		
		f[0] = -beta*y[0]*y[1];
		f[1] = beta*y[0]*y[1] - gamma*y[1];
		f[2] = gamma*y[1];
		
		return GSL_SUCCESS;
	}	
	
	
	SirDeterministic::SirDeterministic(const double N, const double delta) : N_(N),delta_(delta),maxT_(0.0),maxIntegral_(0.0)
	{
	};
	
	SirDeterministic::~SirDeterministic()
	{
		
	}
	

	
	void
	SirDeterministic::simulate(const double beta, const double gamma, const double I0)
	{
		
		iGraph_.clear();
		
		// Parameters
		double parms[2]; 
		parms[0] = beta; 
		parms[1] = gamma;
		
		double y[3] = {N_,I0,0.0};
		
		gsl_odeiv2_system sys = {model, NULL, 3, parms};
		gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 1e-4, 1e-4, 0.0);
		
		double t = 0.0;

		// Store initial conditions
		CurveVal tmp;
		tmp.value  = 1.0;
		tmp.integral = 0.0;
		iGraph_.insert(std::make_pair(0.0,tmp));
						   
		// Run model forward
		while (y[1]>=1.0)
		{
			
			int status = gsl_odeiv2_driver_apply (driver, &t, t+delta_, y);
			if(status != GSL_SUCCESS)
			{
				std::cerr << "Error, return value = " << status << std::endl;
			}
			
			double integral = (--iGraph_.end())->second.integral; // Integral at the previous time point
			integral += ( (--iGraph_.end())->second.value + y[1] )/2 * delta_;
	
			tmp.value = y[1];
			tmp.integral = integral;
			iGraph_.insert(std::make_pair(t,tmp));
		}
		
		gsl_odeiv2_driver_free(driver);

		// Cache max time
		maxT_ = (--iGraph_.end())->first;
		maxIntegral_ = (--iGraph_.end())->second.integral;
		
	}
	
	
	
	double
	SirDeterministic::numInfecAt(const double time) const
	{
	        if (time > maxT_) return 0.0;

		EpiGraph::const_iterator upper, lower;
		upper = iGraph_.upper_bound(time);
		
		//if(upper == iGraph_.end()) return 0.0; // Return if we're past the end of the epidemic
		
		// Set lower to the previous timepoint
		lower = upper; 
		lower--;
		
		// Now interpolate between the timepoints
		double rv = lower->second.value + (time - lower->first) * ( upper->second.value - lower->second.value) / delta_;
		
		return rv;
	}
	
	
	
	double
	SirDeterministic::integNumInfecAt(const double time) const
	{
	        if(time > maxT_)  return maxIntegral_;

		EpiGraph::const_iterator upper,lower;
		upper = iGraph_.upper_bound(time);
		
		//if(upper == iGraph_.end()) return (--iGraph_.end())->second.integral; // Return whole integral if epidemic has finished
		
		// Set lower to previous timepoint
		lower = upper;
		lower--;
		
		// Now interpolate between timepoints
		double numInfecAtTime = lower->second.value + (time - lower->first) * ( upper->second.value - lower->second.value) / delta_;
		double rv = lower->second.integral + (lower->second.value + numInfecAtTime)/2 * (time - lower->first);
		
		return rv;
	}

	double
	SirDeterministic::getMaxTime() const
	{
	  return maxT_;
	}

	double
	SirDeterministic::getMaxIntegral() const
	{
	  return maxIntegral_;
	}

}
