#pragma once
#include "typedefs.h"
#include "cuda_runtime.h"

class Results
{
 public:
  Results(void);
  Results(int, int, ResultsTypePtr, ResultsTypePtr, ResultsTypePtr,
          ResultsType, bool);
  ~Results(void);

  void Init(int, int);
  void AdvanceToNewRow(const ResultsType, const ResultsTypePtr);
  void Advance(const ResultsType, const ResultsTypePtr);
  Results CreateCopyOnDevice();
  Results CreateCopyOnHost();

  __device__ __host__ ResultsType GetResult(int, int) const;
  ResultsTypePtr GetLastRow() const;
  void CopyLastRow(const ResultsTypePtr) const;

  __device__ __host__ void SetResult(int i, int j, ResultsType value) {
    results_[i*m_ + j] = value;
  }

 private:
  int n_, m_;
  ResultsTypePtr results_;
  ResultsTypePtr previous_row_;
  ResultsTypePtr previous_column_;
  ResultsType special_; // TODO: better name
  bool on_cuda_;
};

/***********************
  special
  v
  # p r e v   r o w  
  p . . . . . . . . -> row (n)
  r . . . . . . . .
  e . . . . . . . .
  v . . . . . . . .
    . . . . . . . .
  c . . . . . . . .
  o . . . . . . . .
  l . . . . . . . .
  u . . . . . . . .
  m . . . . . . . .
  n . . . . . . . .
    ^
    column (m)
 */