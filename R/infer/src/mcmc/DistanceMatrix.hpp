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

struct CsrMatrix
{
  int* rowPtr;
  int* colInd;
  float* val;
  int nnz;
  int n;
  int m;
};

unsigned long long
distanceNumValid(const float2* devCoords, const unsigned long long n, const float dLimit);

CsrMatrix*
makeSparseDistance(const float2* coords, const size_t n, const float dLimit);

float
getDistElement(const CsrMatrix* d, const int row, const int col);

CsrMatrix*
allocCsrMatrix(const unsigned long long n, const unsigned long long m, const unsigned long long nnz);

void
destroyCsrMatrix(CsrMatrix* const& csrMatrix);

#endif
