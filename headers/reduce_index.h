#ifndef REDUCE_INDEX_H_
#define REDUCE_INDEX_H_


template <typename Device, typename T>
struct ReduceIndexFunctor {
  void operator()(const Device& d, const T* input, T* out, const int* indexes, const size_t maxInd, const size_t indLength);
};

#endif //REDUCE_INDEX_H_