/*
 *  sirDeterministic.h
 *  sirDeterministic
 *
 *  Created by Chris Jewell on 19/05/2011.
 *  Copyright 2011 University of Warwick. All rights reserved.
 *
 */

#ifndef SIRDETERMINISTIC_H
#define SIRDETERMINISTIC_H

#include <map>


namespace EpiRisk {
	
	class SirDeterministic
	{
		struct CurveVal
		{
			double value;
			double integral;
		};
		typedef std::map<double,CurveVal> EpiGraph;
		
	public:
		explicit SirDeterministic(const double N, const double delta = 0.5);
		virtual ~SirDeterministic();
		void simulate(const double beta, const double gamma, const double I0 = 1);
		double numInfecAt(const double time) const;
		double integNumInfecAt(const double time) const;
		double getMaxTime() const;
		double getMaxIntegral() const;
	
	private:
		double N_;
		double delta_;
		EpiGraph iGraph_;
		double maxT_;
		double maxIntegral_;
	};
	
}
#endif
