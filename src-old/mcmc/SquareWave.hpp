/*
 * CubicSpline1
 * Created: 08-08-2014
 * Author: Chris Jewell <c.p.jewell@massey.ac.nz> (c) 2014
 * Co-author: Richard Brown <r.g.brown@massey.ac.nz>
 * Purpose: Implements CUDA periodic square wave functions
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

#ifndef SQUAREWAVE_HPP
#define SQUAREWAVE_HPP

namespace EpiRisk {

  static __device__  const float T[] = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};

    __device__ float
  _s(const float t, const float I, float nu, float alpha1, float alpha2, float alpha3)
  {
    // Periodic piece-wise square wave
    float Y[] = {1.0f, 1.0f, 1.0f, 1.0f,1.0f};
    float delta = 0.25;
    
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
    
    //int epoch = (int)(tAdj*4.0f);
    
    return Y[epoch];
  }
  
  
  __device__ float
  _SIntegrand(float t, const float* T, const float* Y) {
    // Calculates epoch integral
    return (t-T[0])*Y[0];
  }
  
  
  __global__ void
  _SIntegConst(const float alpha1, const float alpha2, const float alpha3, float* cache)
  {
    // Calculates cached integral -- requires only 4 threads
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
    float Y[] = {1.0f, 1.0f,  1.0f, 1.0f,  1.0f};
    Y[0] = alpha1; Y[2] = alpha2; Y[3] = alpha3; Y[4] = alpha1;
    
    
    if(b <= a) return 0.0f;
    
    float t1 = (a+nu)/365.0f;
    float t2 = (b+nu)/365.0f;
    
    // Set relative to the beginning of t1's period
    t2 = t2 - floorf(t1);
    t1 = t1 - floorf(t1);
    
    //int epoch1 = t1*4;
    int epoch1 = 0; while(t1 > T[epoch1+1]) epoch1++;
    int period2 = t2;
    //int epoch2 = (t2-floorf(t2))*4;
    int epoch2 = 0; while((t2-floorf(t2)) > T[epoch2+1]) epoch2++;
    
    float integrand1 = hCache[epoch1] + _SIntegrand(t1, T+epoch1, Y+epoch1) - _SIntegrand(epoch1*0.25f, T+epoch1, Y+epoch1);
    float integrand2 = hCache[4]*period2 + hCache[epoch2] + _SIntegrand(t2-period2, T+epoch2, Y+epoch2) - _SIntegrand(epoch2*0.25f, T+epoch2, Y+epoch2);
    
    return 365.0f*(integrand2 - integrand1);
    
  }
  


}

#endif
