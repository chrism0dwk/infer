#define STRINGIFY(A) #A

std::string kernel_source = STRINGIFY(

				      float _h(const float t, const float nu, const float alpha)
				      {
					return t < alpha ? 0.0f : 1.0f;
				      }

				      float _K(const float dsq, const float delta, const float omega)
				      {
					return delta / powf(delta*delta + dsq, omega);
				      }

				      template<typename T>
				      float localReduce(T* buff)
				      {
					// Reduce buffer into output
					__syncthreads();
					
					for (unsigned int size = blockDim.x / 2; size > 32; size >>= 1)
					  {
					    if (threadIdx.x < size)
					      buff[threadIdx.x] += buff[threadIdx.x + size];
					    __syncthreads();
					  }
					if (threadIdx.x < 32)
					  {
					    volatile T* vbuff = buff;
					    vbuff[threadIdx.x] += vbuff[threadIdx.x + 32];
					    vbuff[threadIdx.x] += vbuff[threadIdx.x + 16];
					    vbuff[threadIdx.x] += vbuff[threadIdx.x + 8];
					    vbuff[threadIdx.x] += vbuff[threadIdx.x + 4];
					    vbuff[threadIdx.x] += vbuff[threadIdx.x + 2];
					    vbuff[threadIdx.x] += vbuff[threadIdx.x + 1];
					  }
					
					__syncthreads();
				      }
				      
				      
				      __kernel void
				      _calcProduct(const InfecIdx_t* infecIdx, const int infecSize,
						   const CsrMatrix distance, const float* eventTimes,
						   const int eventTimesPitch, const float* susceptibility,
						   const float* infectivity, const float epsilon1, const float epsilon2,
						   const float gamma1, const float gamma2, const float delta, const float omega, const float nu,
						   const float alpha, const float movtBan, float* prodCache)
				      {
					// Each warp calculates a row of the sparse matrix

					int tid = threadIdx.x + blockDim.x * blockIdx.x;
					int gwid = tid / 32; // Global Warp id
					int lane = tid & (32 - 1); // Id within a warp

					__shared__
					  float threadProdCache[THREADSPERBLOCK];
					threadProdCache[threadIdx.x] = 0.0f;
					int row = gwid;

					if (row < infecSize)
					  {
					    int j = infecIdx[row].ptr;

					    int begin = distance.rowPtr[j];
					    int end = distance.rowPtr[j + 1];

					    float Ij = eventTimes[j];

					    for (int ii = begin + lane; ii < end; ii += 32)
					      {
						int i = distance.colInd[ii];
						float Ii = eventTimes[i];
						float Ni = eventTimes[eventTimesPitch + i];
						float Ri = eventTimes[eventTimesPitch * 2 + i];

						if (Ii < Ni)
						  {
						    float idxOnj = 0.0f;
						    if (Ii < Ij and Ij <= Ni)
						      idxOnj += _h(Ij - Ii, nu, alpha);
						    else if (Ni < Ij and Ij <= Ri)
						      idxOnj += gamma2 * _h(Ij - Ii, nu, alpha);
						    threadProdCache[threadIdx.x] += idxOnj * infectivity[i] * _K(distance.val[ii],delta,omega);
						  }
					      }
					    __syncthreads();

					    // Reduce semi-products into productCache
					    volatile float* vThreadProdCache = threadProdCache;
					    if (lane < 16)
					      {
						vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 16];
						vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 8];
						vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 4];
						vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 2];
						vThreadProdCache[threadIdx.x] += vThreadProdCache[threadIdx.x + 1];
					      }
					    __syncthreads();

					    // Write out to global memory
					    if (lane == 0)
					      {
						float epsilon = Ij < movtBan ? epsilon1 : (epsilon1*epsilon2);
						prodCache[j] = threadProdCache[threadIdx.x] * susceptibility[j]
								     * gamma1 + epsilon;
					      }
					  }
				      }
				      );
