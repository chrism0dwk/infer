/*************************************************************************
 *  ./src/sim/SirDeterministic.cpp
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
/* ./src/sim/SirDeterministic.cpp
 *
 * Copyright 2012 Chris Jewell <chrism0dwk@gmail.com>
 *
 * This file is part of InFER.
 *
 * InFER is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * InFER is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with InFER.  If not, see <http://www.gnu.org/licenses/>. 
 */
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
		double alpha = *((double *)params + 1);
		double gamma = *((double *)params + 2);
		
		f[0] = -beta*y[0]*y[2];
		f[1] = beta*y[0]*y[2] - alpha*y[1];
		f[2] = alpha*y[1] - gamma*y[2];
		f[3] = gamma*y[2];
		
		return GSL_SUCCESS;
	}	
	
	
	SirDeterministic::SirDeterministic(const double N, const double delta) : N_(N),delta_(delta),maxT_(0.0),maxIntegral_(0.0)
	{
	};
	
	SirDeterministic::~SirDeterministic()
	{
		
	}
	

	
	void
	SirDeterministic::simulate(const double beta, const double alpha, const double gamma, const double I0)
	{
		
		iGraph_.clear();
		
		// Parameters
		double parms[3];
		parms[0] = beta / N_; // Density independent!
		parms[1] = alpha;
		parms[2] = gamma;
		
		double y[4] = {N_-I0,I0,0.0,0.0};
		
		gsl_odeiv2_system sys = {model, NULL, 4, parms};
		gsl_odeiv2_driver* driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 1e-4, 1e-4, 0.0);
		
		double t = 0.0;

		// Store initial conditions
		CurveVal tmp;
		tmp.value  = 1.0;
		tmp.integral = 0.0;
		iGraph_.insert(std::make_pair(0.0,tmp));

		// Run model forward
		while (y[2]>=0.5 or y[1]>=0.5)
		{
			
			int status = gsl_odeiv2_driver_apply (driver, &t, t+delta_, y);
			if(status != GSL_SUCCESS)
			{
				std::cerr << "Error, return value = " << status << std::endl;
			}
			
			double integral = (--iGraph_.end())->second.integral; // Integral at the previous time point
			integral += ( (--iGraph_.end())->second.value + y[2] )/2 * delta_;
	
			tmp.value = y[2];
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
