///////////////////////////////////////////////////////////////////////
// Name: DistanceMatrix.hpp					     //
// Created: 2015-07-27						     //
// Author: Chris Jewell <c.jewell@lancaster.ac.uk>		     //
// Copyright: Chris Jewell 2015					     //
// Purpose: Implements a truncated distance-weighted network given a //
//          list of 2D coordinates.				     //
///////////////////////////////////////////////////////////////////////

#ifndef DISTANCEMATRIX_HPP
#define DISTANCEMATRIX_HPP

#define DM_THREADSPERBLOCK 128

#include <vector_types.h>

enum DM_PLATFORM
  {
    CPU,
    GPU
  };

struct CsrMatrix
{
  int* rowPtr;
  int* colInd;
  float* val;
  int nnz;
  int n;
  int m;
  DM_PLATFORM platform;
};

unsigned long long
distanceNumValid(const float2* devCoords, const unsigned long long n, const float dLimit);

CsrMatrix*
makeSparseDistance(const float2* coords, const size_t n, const float dLimit, const size_t chunkSize = 128);

float
getDistElement(const CsrMatrix* d, const int row, const int col);

CsrMatrix*
allocCsrMatrix(const size_t n, const size_t m, const size_t nnz, const DM_PLATFORM platform);

void
destroyCsrMatrix(CsrMatrix* const& csrMatrix);

#endif
