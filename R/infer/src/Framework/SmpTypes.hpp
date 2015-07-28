/***************************************************************************
 *   Copyright (C) 2009 by Chris Jewell                                    *
 *   chris.jewell@warwick.ac.uk                                            *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

// Types used for epidemic MCMC code


#ifndef INCLUDE_AITYPES_HPP
#define INCLUDE_AITYPES_HPP

namespace EpiRisk {
  namespace Smp {
typedef double eventTime_t;
typedef unsigned int Ilabel_t;

typedef unsigned int Slabel_t;
typedef unsigned int Spos_t;
typedef unsigned int Ipos_t;
typedef float freq_t;


struct SmpParams {
  double alpha;
  double beta;
  double rho;
  double a;
  double gamma;
  
  SmpParams() : alpha(2.2e-6), beta(1.0e-4), rho(2e-6), a(6.0), gamma(2.2) {};
};


  } } // namespace EpiRisk


#endif