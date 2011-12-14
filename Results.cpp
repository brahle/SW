#include "Results.h"

#include <cstring>
#include "cuda_runtime.h"

#include "utilities.h"

Results::Results()
  : n_(0), 
    m_(0),
    results_(0),
    previous_row_(0),
    previous_column_(0),
    special_(0),
    on_cuda_(false)
{
}


Results::Results(int n, int m, ResultsTypePtr results,
                 ResultsTypePtr previous_row, ResultsTypePtr previous_column,
                 ResultsType special, bool on_cuda=false)
  : n_(n), 
    m_(m),
    results_(results),
    previous_row_(previous_row),
    previous_column_(previous_column),
    special_(special),
    on_cuda_(on_cuda)
{
}


Results::~Results()
{
  if (on_cuda_) {
    cudaFree(results_);
    cudaFree(previous_row_);
    cudaFree(previous_column_);
  } else {
    delete [] results_;
    delete [] previous_row_;
    delete [] previous_column_;
  }
}


void Results::Init(int n, int m) {
  n_ = n;
  m_ = m;
  results_ = new ResultsType[n*m];
  memset(results_, 0, sizeof(ResultsType)*n_*m_);
  previous_row_ = new ResultsType[m];
  memset(previous_row_, 0, sizeof(ResultsType)*m_);
  previous_column_ = new ResultsType[n];
  memset(previous_column_, 0, sizeof(ResultsType)*n_);
  special_ = 0;
  on_cuda_ = false;
}


void Results::AdvanceToNewRow(const ResultsType special,
                              const ResultsTypePtr previous_row) {
  special_ = special;
  memcpy(previous_row_, previous_row, sizeof(ResultsType)*m_);
  memset(previous_column_, 0, sizeof(ResultsType)*n_);
  memset(results_, 0, sizeof(ResultsType)*n_*m_);
}


void Results::Advance(const ResultsType special,
                      const ResultsTypePtr previous_row) {
  special_ = special;
  memcpy(previous_row_, previous_row, sizeof(ResultsType)*m_);
  for (int i = 0; i < n_; ++i) {
    previous_column_[i] = results_[i*m_ + m_-1];
  }
  memset(results_, 0, sizeof(ResultsType)*n_*m_);
}


Results Results::CreateCopyOnDevice() {
  return Results(
    n_,
    m_,
    copyArrayToDevice(results_, n_*m_),
    copyArrayToDevice(previous_row_, m_),
    copyArrayToDevice(previous_column_, n_),
    special_,
    true
  );
}


Results Results::CreateCopyOnHost() {
  return Results(
    n_,
    m_,
    copyArrayToHost(results_, n_*m_),
    copyArrayToHost(previous_row_, m_),
    copyArrayToHost(previous_column_, n_),
    special_,
    false
  );
}


/*
// Implemented in the kernel.cu file
__device__ __host__ ResultsType Results::GetResult(int i, int j) const {
  if (i < -1 || j < -1 || i >= n_ || j >= m_) return 0;
  if (i == -1) {
    if (j == -1) return special_;
    return previous_column_[j];
  }
  if (j == -1) {
    return previous_row_[i];
  }
  return results_[i*m_ + j];
}
*/

ResultsTypePtr Results::GetLastRow() const {
  return results_ + (n_-1)*m_;
}


void Results::CopyLastRow(ResultsTypePtr dest) const {
  memcpy(dest, GetLastRow(), sizeof(ResultsType)*m_);
}


/*
// Implemented in the kernel.cu file
__device__ __host__ void Results::SetResult(int i, int j, ResultsType value) {
  results_[i*m_ + j] = value;
}
*/