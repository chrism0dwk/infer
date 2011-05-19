/*************************************************************************
 *  ./src/sim/SirDeterministic.hpp
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
		SirDeterministic(const double N, const double delta = 0.5);
		virtual ~SirDeterministic();
		void simulate(const double beta, const double gamma, const double I0 = 1);
		double numInfecAt(const double time);
		double integNumInfecAt(const double time);
	
	private:
		double N_;
		double delta_;
		EpiGraph iGraph_;		
	};
	
}
#endif