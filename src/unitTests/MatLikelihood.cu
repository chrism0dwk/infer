/*
 * MatLikelihood.cpp
 *
 *  Created on: 30 Jan 2012
 *      Author: stsiab
 */
#include <set>
#include <vector>
#include <iterator>
#include <map>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <sys/time.h>



#include "MatLikelihood.hpp"

__global__ void distanceKernel(float* input, float delta, size_t N, float* output)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < N)
    output[idx] = delta / (delta*delta + input[idx]);
}

__global__ void pointwiseMult(float* A, float* B, float* C, size_t N)
{
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if(idx < N)
    C[idx] = A[idx] * B[idx];
}


void
addreduction(ublas::vector<float>& rb, size_t level=1)
{
  int reductionLevel = 2 << (level-1);
  if (reductionLevel/2 > rb.size()) return;

  for(size_t i=reductionLevel/2; i<rb.size(); i += reductionLevel)
    {
      rb[i-reductionLevel/2] += rb[i];
    }

  addreduction(rb,level+1);

  return;
}


class CmpIndivIdxOnInfection
{
  const EpiRisk::Population<TestCovars>& population_;
public:
  CmpIndivIdxOnInfection(const EpiRisk::Population<TestCovars>& population) :
    population_(population)
  {
  }
  ;
  bool
  operator()(const size_t lhs, const size_t rhs) const
  {
    return population_[lhs].getI() < population_[rhs].getI();
  }
  ;
};

MatLikelihood::MatLikelihood(const EpiRisk::Population<TestCovars>& population,
    EpiRisk::Parameters& txparams) :
  txparams_(txparams),infectivesSz_(population.numInfected()), obsTime_(population.getObsTime()),
      population_(population),zero_(0.0),unity_(1.0)
{
  // Get all susceptibles relevant to current epidemic
  set<size_t> tmp; // Stored in infection time order

  // Get list of individuals involved in the epidemic
  for (int i = 0; i < population.size(); ++i)
    {
      tmp.insert(population[i].getConnectionList().begin(),
          population[i].getConnectionList().end());
    }

  // Order subpopulation by infection time
  SubPopulation subpop(tmp.begin(),tmp.end());
  CmpIndivIdxOnInfection comp(population);
  sort(subpop.begin(),subpop.end(),comp);
  subPopSz_ = subpop.size();

  // Animals, susceptibility, infectivity, and D_ sparse matrix
  animals_.resize(subPopSz_, 3);
  animalsSuscPow_.resize(subPopSz_,3);
  animalsInfPow_.resize(infectivesSz_,3);
  susceptibility_.resize(subPopSz_);
  infectivity_.resize(infectivesSz_);
  product_.resize(infectivesSz_);
  icoord_.resize(infectivesSz_);
  jcoord_.resize(subPopSz_);
  infecTimes_.resize(infectivesSz_,3);

  E_.resize(infectivesSz_,infectivesSz_,false);
  D_.resize(infectivesSz_, subPopSz_,false);
  T_.resize(infectivesSz_, subPopSz_,false);

  cerr << "Populating data tables..." << endl;
  std::map<size_t,size_t> rawtoblas;
  size_t j_idx = 0;
  for (SubPopulation::const_iterator j = subpop.begin(); j != subpop.end(); ++j)
    {
      // Enter into jcoord
      jcoord_(j_idx) = *j;
      rawtoblas.insert(make_pair(*j,j_idx));

      // Copy covariates
      const Individual::CovarsType& covars = population_[*j].getCovariates();
      animals_(j_idx, 0) = covars.cattle;
      animals_(j_idx, 1) = covars.pigs;
      animals_(j_idx, 2) = covars.sheep;
      animalsSuscPow_(j_idx,0) = pow(covars.cattle,txparams_(13));
      animalsSuscPow_(j_idx,1) = pow(covars.pigs,txparams_(14));
      animalsSuscPow_(j_idx,2) = pow(covars.sheep,txparams_(15));
      j_idx++;
    }

  // Set up infectives
  cerr << "Setting up infectives" << endl;

  for(size_t i = 0; i < infectivesSz_; ++i)
    {

          icoord_(i) = jcoord_(i);
          infecTimes_(i, 0) = population_[jcoord_[i]].getI();
          infecTimes_(i, 1) = population_[jcoord_[i]].getN();
          infecTimes_(i, 2) = population_[jcoord_[i]].getR();

          for(size_t k = 0; k < 3; ++k)
            {
              animalsInfPow_(i,k) = powf(animals_(i,k),txparams_(k+10));
            }
    }

  // Cache I1
  matrix_column< matrix<float,column_major> > col(infecTimes_,0);
  I1_ = col(0); I1idx_ = 0;
  for(size_t i = 1; i < col.size(); ++i)
    if (col(i) < I1_) {
        I1_ = col(i); I1idx_=i;
    }


  // Calculate product mask and exposure times
  for(size_t i = 0; i < infectivesSz_; ++i) {
          // Populate row of D_ and T_
          for (Individual::ConnectionList::const_iterator con =
              population_[jcoord_[i]].getConnectionList().begin(); con
              != population_[jcoord_[i]].getConnectionList().end(); ++con)
            {
              float dx = population_[jcoord_[i]].getCovariates().x
                  - population_[*con].getCovariates().x;
              float dy = population_[jcoord_[i]].getCovariates().y
                  - population_[*con].getCovariates().y;
              float sqDist = dx * dx + dy * dy;

              // Get con's index in jcoord
              size_t j = rawtoblas[*con];

              // Product mask
              if(j < infectivesSz_ and i != j) {
                  if (infecTimes_(i,0) < infecTimes_(j,0) and infecTimes_(j,0) <= infecTimes_(i,1)) E_(i,j) = 1.0f;
                  else if (infecTimes_(i,1) < infecTimes_(j,0) and infecTimes_(j,0) <= infecTimes_(i,2)) E_(i,j) = txparams_(1);
              }

              // Integral of infection time
              float jMaxSuscepTime = 0.0;
              if(j < infectivesSz_)
                {
                  if(infecTimes_(i,0) < infecTimes_(j,0)) jMaxSuscepTime = infecTimes_(j,0);
                }
              else
                {
                  jMaxSuscepTime = min((float)population_[*con].getN(),obsTime_);
                }
              D_(i, j) = sqDist;
              float exposureTime;
              exposureTime = min(infecTimes_(i,1),jMaxSuscepTime) - min(infecTimes_(i,0),jMaxSuscepTime);
              exposureTime += txparams_(1)*(min(infecTimes_(i,2),jMaxSuscepTime)-min(infecTimes_(i,1),jMaxSuscepTime));
              if (exposureTime < 0.0) cerr << "T_(" << population_[icoord_[i]].getId() << "," << population_[*con].getId() << ") = " << exposureTime << endl;
              T_(i, j) = exposureTime;
            }
    }

  DT_ = D_;
  cerr << "Initialised D_ with nnz=" << D_.nnz() << endl;
  cerr << "Initialised T_ with nnz=" << T_.nnz() << endl;
  cerr << "Initialised E_ with nnz=" << E_.nnz() << endl;

  // Initialize GPU datasets
  
  // Data sizes
  size_t animSuscSize = subPopSz_*3;
  size_t animInfSize = infectivesSz_*3;
  size_t infecTimesSize = infectivesSz_*3;
  size_t nnz = D_.nnz();
  size_t csrRowPtrSize = D_.index1_data().size();
  size_t txparamsSize = txparams_.size();
  cerr << "Allocating GPU memory" << endl;
  int rv = 0;
  rv |= cudaMalloc(&devAnimalsInfPow_,animInfSize*sizeof(float));
  rv |= cudaMalloc(&devAnimalsSuscPow_,(animSuscSize*sizeof(float)));
  rv |= cudaMalloc(&devInfecTimes_,infecTimesSize*sizeof(float));
  rv |= cudaMalloc(&devSusceptibility_,subPopSz_*sizeof(float));
  rv |= cudaMalloc(&devInfectivity_,infectivesSz_*sizeof(float));

  rv |= cudaMalloc(&devDVal_,nnz*sizeof(double));
  rv |= cudaMalloc(&devDRowPtr_,csrRowPtrSize*sizeof(int));
  rv |= cudaMalloc(&devDColInd_,nnz*sizeof(int));

  rv |= cudaMalloc(&devTVal_,nnz*sizeof(float));

  rv |= cudaMalloc(&devDTVal_,nnz*sizeof(float));

  rv |= cudaMalloc(&devEVal_,E_.nnz()*sizeof(float));
  rv |= cudaMalloc(&devEColPtr_,E_.size1()*sizeof(int));
  rv |= cudaMalloc(&devERowInd_,E_.nnz()*sizeof(int));

  rv |= cudaMalloc(&devModelParams_,txparamsSize*sizeof(float));
  rv |= cudaMalloc(&devTmp_,infectivesSz_*sizeof(float));

  if(rv != cudaSuccess) {
      cerr << "CUDA ERROR " << cudaGetErrorString((cudaError_t)rv) << endl;
      throw runtime_error("Error allocating GPU device memory");
  }

  // Copy data structures to device
  cerr << "Copying host memory to device memory" << endl;
  rv = 0;
  rv |= cudaMemcpy(devAnimalsInfPow_, animalsInfPow_.data().begin(), animInfSize * sizeof(float), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devAnimalsSuscPow_, animalsSuscPow_.data().begin(), (animSuscSize * sizeof(float)), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devInfecTimes_, infecTimes_.data().begin(),infecTimesSize * sizeof(float), cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) {
      cerr << "Error copying covariates to GPU device: " << cudaGetErrorString((cudaError_t)rv) << endl;
      throw runtime_error("Error copying data to GPU device");
  }
  
  // Sparse matrices -- convert indices from size_t to int -- limitation of CUSPARSE!
  DRowPtr_.resize(D_.index1_data().size());
  for(size_t i=0; i<D_.index1_data().size(); ++i) DRowPtr_[i] = D_.index1_data()[i];
  DColInd_.resize(D_.index2_data().size());
  for(size_t i=0; i<D_.index2_data().size(); ++i) DColInd_[i] = D_.index2_data()[i];
  
  // Copy sparse matrices to GPU -- a lot of this can be implemented on the GPU itself!
  rv = 0;
  rv |= cudaMemcpy(devDVal_, D_.value_data().begin(),nnz * sizeof(float), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devDRowPtr_, DRowPtr_.begin(),csrRowPtrSize * sizeof(int), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devDColInd_, DColInd_.begin(),nnz * sizeof(int), cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) {
      cerr << "Error copying D_ to GPU device: " << cudaGetErrorString((cudaError_t)rv) << endl;
      throw runtime_error("Error copying data to GPU device");
  }

  rv |= cudaMemcpy(devTVal_, T_.value_data().begin(),nnz * sizeof(float), cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) {
      cerr << "Error copying T_ to GPU device: " << cudaGetErrorString((cudaError_t)rv) << endl;
      throw runtime_error("Error copying data to GPU device");
  }

  rv |= cudaMemcpy(devDTVal_, DT_.value_data().begin(),nnz * sizeof(float), cudaMemcpyHostToDevice);
  if(rv != cudaSuccess) {
      cerr << "Error copying DT_ to GPU device: " << cudaGetErrorString((cudaError_t)rv) << endl;
      throw runtime_error("Error copying data to GPU device");
  }

  rv |= cudaMemcpy(devEVal_, E_.value_data().begin(),(size_t)(E_.nnz() * sizeof(float)), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devEColPtr_, E_.index1_data().begin(),(size_t)(E_.size1() * sizeof(int)), cudaMemcpyHostToDevice);
  rv |= cudaMemcpy(devERowInd_, E_.index2_data().begin(),(size_t)(E_.nnz() * sizeof(int)), cudaMemcpyHostToDevice);

  for(size_t i=0; i<txparams_.size();++i) {
      float val = txparams_(i);
      rv |= cudaMemcpy(devModelParams_+i,&val,sizeof(float),cudaMemcpyHostToDevice);
  }

  if(rv != cudaSuccess) {
      cerr << "Error copying parameters to GPU device: " << cudaGetErrorString((cudaError_t)rv) << endl;
      throw runtime_error("Error copying data to GPU device");
  }
  
  
  // BLAS handles
  blasStat_ = cublasCreate(&cudaBLAS_);
  if(blasStat_ != CUBLAS_STATUS_SUCCESS) throw runtime_error("CUBLAS init failed");

  sparseStat_ = cusparseCreate(&cudaSparse_);
  if(sparseStat_ != CUSPARSE_STATUS_SUCCESS) throw runtime_error("CUSPARSE init failed");
  
  sparseStat_ = cusparseCreateMatDescr(&crsDescr_);
  if(sparseStat_ != CUSPARSE_STATUS_SUCCESS) throw runtime_error("CUSPARSE matrix descriptor init failed");
  
  cusparseSetMatType(crsDescr_,CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(crsDescr_,CUSPARSE_INDEX_BASE_ZERO);

}

MatLikelihood::~MatLikelihood()
{
  int rv = 0;
  rv |= cudaFree(devAnimalsInfPow_);
  rv |= cudaFree(devAnimalsSuscPow_);
  rv |= cudaFree(devInfecTimes_);
  rv |= cudaFree(devSusceptibility_);
  rv |= cudaFree(devInfectivity_);
  rv |= cudaFree(devDVal_);
  rv |= cudaFree(devDRowPtr_);
  rv |= cudaFree(devDColInd_);
  rv |= cudaFree(devTVal_);
  rv |= cudaFree(devDTVal_);
  rv |= cudaFree(devEVal_);
  rv |= cudaFree(devEColPtr_);
  rv |= cudaFree(devERowInd_);

  if(rv != cudaSuccess) throw runtime_error("Error freeing GPU device memory");

  cublasDestroy(cudaBLAS_);
  cusparseDestroy(cudaSparse_);

}

double
MatLikelihood::calculate()
{

  // Temporaries
  ublas::vector<float> tmp(infectivesSz_);
  
  // Parameters
  float deltasq = txparams_(2) * txparams_(2);
  ublas::vector<float> infecParams(3);
  ublas::vector<float> suscepParams(3);
  for (size_t i = 0; i < 3; ++i)
    {
      infecParams(i) = txparams_(i + 4);
      suscepParams(i) = txparams_(i + 7);
    }

  // Susceptibility
  axpy_prod(animalsSuscPow_, suscepParams, susceptibility_, true);


  // Infectivity
  axpy_prod(animalsInfPow_, infecParams, infectivity_, true);

  
  // Calculate product
  float lp = 0.0;
  compressed_matrix<float,column_major> QE(E_);
  for(size_t j = 0; j != QE.size1(); j++) // Iterate over COLUMNS j
    {
      size_t begin = QE.index1_data()[j];
      size_t end = QE.index1_data()[j+1];
      for(size_t i = begin; i < end; ++i) // Non-zero ROWS i
        {
          QE.value_data()[i] *= txparams_(2) / (deltasq + D_(QE.index2_data()[i],j));
        }
    }
  axpy_prod(infectivity_,QE,tmp);

  tmp *= txparams_(0);  // Gamma1

  for(size_t i = 0; i < I1idx_; ++i)
    {
      float subprod = susceptibility_(i)*tmp(i) + txparams_(3);
      product_(i) = subprod; lp += logf(subprod);
    }
  product_(I1idx_) = 1.0; // Loop unrolled to skip I1
  for(size_t i = I1idx_+1; i < tmp.size(); ++i)
    {
      float subprod = susceptibility_(i)*tmp(i) + txparams_(3);
      product_(i) = subprod; lp += logf(subprod);
    }

  // Apply distance kernel to D_ and calculate DT
  for(size_t i = 0; i < D_.size1(); ++i)
    {
      size_t begin = D_.index1_data()[i];
      size_t end = D_.index1_data()[i+1];

      for(size_t j = begin; j < end; ++j)
        {
          DT_.value_data()[j] = txparams_(2) / (deltasq + D_.value_data()[j]) * T_.value_data()[j];
        }
    }
 
  // Calculate the integral
  axpy_prod(DT_,susceptibility_,tmp);

  float integral = txparams_(0) * inner_prod(infectivity_,tmp);

  // Calculate background pressure
  //matrix_column< matrix<float,column_major> > col(infecTimes_,0);
  //float bg = sum(col) - I1_*infectivesSz_;
  //bg += (obsTime_ - I1_)*(population_.size() - infectivesSz_);
  //bg *= txparams_(3);

  //integral += bg;

  return /*lp */- integral;
}


double
MatLikelihood::calculateGPU()
{
  // GPU version of likelihood calculation

  // Susceptibility
  blasStat_ = cublasSgemv(cudaBLAS_,CUBLAS_OP_N,
              subPopSz_,3,
              &unity_,
              devAnimalsSuscPow_, subPopSz_,
              (devModelParams_+7), 1,
              &zero_,
              devSusceptibility_, 1);
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      cerr << "Error in susceptibility: " << blasStat_ << endl;
    }

  
  // Infectivity
  blasStat_ = cublasSgemv(cudaBLAS_,CUBLAS_OP_N,
              infectivesSz_,3,
              &unity_,
              devAnimalsInfPow_, infectivesSz_,
              (devModelParams_+4), 1,
              &zero_,
              devInfectivity_, 1);
  if (blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      cerr << "Error in infectivity: " << blasStat_ << endl;
    }
  
  // Apply distance kernel to D_, place result in DT_.
  size_t threadsPerBlock = 512;
  size_t blocksPerGrid = (D_.nnz() + threadsPerBlock - 1) / threadsPerBlock;
  float delta = txparams_(2);
  size_t nnz = D_.nnz();
  distanceKernel<<<blocksPerGrid,threadsPerBlock>>>(devDVal_,delta,nnz,devDTVal_);
  pointwiseMult<<<blocksPerGrid,threadsPerBlock>>>(devDTVal_,devTVal_,devDTVal_,nnz);
  

  // DT * Susceptibility
  sparseStat_ = cusparseScsrmv(cudaSparse_, CUSPARSE_OPERATION_NON_TRANSPOSE,
      infectivesSz_,subPopSz_,nnz,&unity_,
      crsDescr_, devDTVal_,
      devDRowPtr_, devDColInd_,
      devSusceptibility_,&zero_,
      devTmp_);
  if(sparseStat_ != CUSPARSE_STATUS_SUCCESS)
    {
      cerr << "Error in cusparseScsrmv() " << sparseStat_ << endl;
    }
  

  // infectivity * devTmp
  float result;
  blasStat_ = cublasSdot(cudaBLAS_, infectivesSz_,
                           devInfectivity_, 1,
                           devTmp_, 1,
                           &result);  // May have an issue with 1-based indexing here!
  if(blasStat_ != CUBLAS_STATUS_SUCCESS)
    {
      cerr << "Error in cublasSdot() " << blasStat_ << endl;
    }
  cerr << "Waiting for kernels to finish" << endl;
  cudaDeviceSynchronize();
  
  // Gamma1
  result *= txparams_(0);
  
  return -result;
}
