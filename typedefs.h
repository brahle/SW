typedef double ResultValue;
typedef struct RESULTTYPE {
  ResultValue value;
  int move;

  __device__ __host__ RESULTTYPE(); // : value(0), move(-1) {}
  __device__ __host__ RESULTTYPE(ResultValue v, int m); //: value(v), move(m) {}
} ResultType;
typedef ResultType* ResultTypePtr;