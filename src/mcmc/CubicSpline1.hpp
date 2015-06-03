//////////////////////////////////////////////////////////////////////////
// Copyright 2015 Chris Jewell                                          //
// 								        //
// This file is part of nztheileria.                                    //
//                                                                      //
// nztheileria is free software: you can redistribute it and/or modify  //
// it under the terms of the GNU General Public License as published by //
// the Free Software Foundation, either version 3 of the License, or    //
// (at your option) any later version.                                  //
//                                                                      //
// nztheileria is distributed in the hope that it will be useful,       //
// but WITHOUT ANY WARRANTY; without even the implied warranty of       //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the        //
// GNU General Public License for more details.                         //
//                                             			        //
// You should have received a copy of the GNU General Public License    //
// along with nztheileria.  If not, see <http://www.gnu.org/licenses/>. //
//////////////////////////////////////////////////////////////////////////

/*
 * CubicSpline1
 * Created: 08-08-2014
 * Author: Chris Jewell <c.p.jewell@massey.ac.nz> (c) 2014
 * Co-author: Richard Brown <r.g.brown@massey.ac.nz>
 * Purpose: Implements CUDA periodic cubic spline functions
 *    given knots at t=0, 0.25, 0.5, 0.75, and 1.
 *
 * Parameters: alpha1 -- height at t=0,1
 * Parameters: alpha2 -- height at t=0.5
 * Parameters: alpha3 -- height at t=0.75
 * Parameters: nu -- phase
 *
 * NB. Requires a integral cache to be allocated on GPU
 *  float cache[5]
 * Cache must be initialised using CalcFuncIntegCache before _H(.)
 *  can be used.
 */

#ifndef CUBICSPLINE1_HPP
#define CUBICSPLINE1_HPP

namespace EpiRisk {

  
  // extern __device__ float
  // _s(const float t, const float I, float nu, float alpha1, float alpha2, float alpha3);
  
  // extern __device__ float
  // _S(const float b, const float a, const float nu, const float alpha1, const float alpha2, const float alpha3, const float* hCache);
  
  // extern void
  // CalcSIntegCache(const float alpha1, const float alpha2, const float alpha3, float* cache, const bool setZero=false);
  
  // __device__ float
  // _SIntegrand(float t, const float* T, const float* Y);
  
  // __global__ void
  // _SIntegConst(const float alpha1, const float alpha2, const float alpha3, float* cache);
    __device__ float
  _s(const float t, const float I, float nu, float alpha1, float alpha2, float alpha3)
  {
    // Periodic piece-wise cubic spline
    float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float Y[] = {1.0f, 1.0f, 1.0f, 1.0f,1.0f};
    float delta = 0.25;
    
    assert(t-I >= 0);
    
    // Re-scale time to unit period
    float tAdj = (t+nu)/365.0f;
    tAdj = tAdj - floorf(tAdj);
    
    // Set up parameters
    Y[0] = alpha1; Y[2] = alpha2; Y[3] = alpha3; Y[4] = alpha1;
    
    // Calculate spline value
    int epoch = (int)(tAdj*4.0f);
    
    float a = -6.0f*(Y[epoch+1]-Y[epoch])/(delta*delta);
    float b = -a;
    
    float h = a/(6.0f*delta) * powf(tAdj - T[epoch], 3);
    h      += (Y[epoch+1]/delta - (a*delta)/6.0f) *  (tAdj - T[epoch]);
    h      += b/(6.0f*delta) * powf(T[epoch+1] - tAdj, 3);
    h      += (Y[epoch]/delta - (b*delta)/6.0f) * (T[epoch+1] - tAdj);
    
    return h;
  }
  
  
  __device__ float
  _SIntegrand(float t, const float* T, const float* Y) {
    // Calculates cubic spline integral between t1 and t2
    float delta = 0.25f;
    float a = -6.0f * (Y[1] - Y[0])/(delta * delta);
    float b = -a;
    
    float h;
    h =  a/(24.0f*delta) * powf(t - T[0], 4);
    h += (Y[1]/(2.0f*delta) - (a*delta)/12.0f) * powf(t - T[0],2);
    h -= b/(24.0f*delta) * powf(T[1] - t, 4);
    h -= (Y[0]/(2.0f*delta) - (b*delta)/12.0f) * powf(T[1] - t,2);
    
    return h;
  }
  
  
  __global__ void
  _SIntegConst(const float alpha1, const float alpha2, const float alpha3, float* cache)
  {
    // Calculates cached integral -- requires only 4 threads
    float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float Y[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    
    __shared__ float buff[4];
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    Y[0] = alpha1; Y[2] = alpha2; Y[3] = alpha3; Y[4] = alpha1;
    
    
    if(tid < 4) {
      buff[tid] = _SIntegrand(T[tid+1], T+tid, Y+tid)
	- _SIntegrand(T[tid], T+tid, Y+tid);
      
      __syncthreads();
      
      // Reduce cummulative sum here -- needs parallelising
      if(tid == 0) {
	buff[1] += buff[0];
	buff[2] += buff[1];
	buff[3] += buff[2];
      }
      __syncthreads();
      
      cache[tid+1] = buff[tid];  // Cache[0] is set to 0
    }
  }
  
  
  void
  CalcSIntegCache(const float alpha1, const float alpha2, const float alpha3, float* cache, const bool setZero)
  {
    // Calculates the CUDA H function integral cache

    if(setZero)
      checkCudaError(cudaMemset(cache, 0, sizeof(float)));
    
    _SIntegConst<<<1, 4>>>(alpha1, alpha2, alpha3, cache);

    cudaDeviceSynchronize();
  }
  
  __device__ float
  _S(const float b, const float a, const float nu, const float alpha1, const float alpha2, const float alpha3, const float* hCache)
  {
    // Returns the integral of the 'h' function over [a,b]
    float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    float Y[] = {1.0f, 1.0f,  1.0f, 1.0f,  1.0f};
    
    Y[0] = alpha1; Y[2] = alpha2; Y[3] = alpha3; Y[4] = alpha1;
    
    
    if(b <= a) return 0.0f;
    
    float t1 = (a+nu)/365.0f;
    float t2 = (b+nu)/365.0f;
    
    // Set relative to the beginning of t1's period
    t2 = t2 - floorf(t1);
    t1 = t1 - floorf(t1);
    
    int epoch1 = t1*4;
    int period2 = t2;
    int epoch2 = (t2-floorf(t2))*4;
    
    float integrand1 = hCache[epoch1] + _SIntegrand(t1, T+epoch1, Y+epoch1) - _SIntegrand(epoch1*0.25f, T+epoch1, Y+epoch1);
    float integrand2 = hCache[4]*period2 + hCache[epoch2] + _SIntegrand(t2-period2, T+epoch2, Y+epoch2) - _SIntegrand(epoch2*0.25f, T+epoch2, Y+epoch2);
    
    return 365.0f*(integrand2 - integrand1);
    
  }
  


}

#endif
