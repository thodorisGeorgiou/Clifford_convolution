#ifndef POOL_BY_INDEX_H_
#define POOL_BY_INDEX_H_


template <typename Device, typename T>
struct PoolByIndexFunctor {
  void operator()(const Device& d, const T* input, T* out, const int* indexes, const size_t indLength, const size_t inbStep, const size_t outbStep);
};

#endif //POOL_BY_INDEX_H_