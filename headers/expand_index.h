#ifndef EXPAND_INDEX_H_
#define EXPAND_INDEX_H_


template <typename Device, typename T>
struct ExpandIndexFunctor {
  void operator()(const Device& d, const T* input, T* out, const int* indexes, const size_t maxInd, const size_t indLength);
};

#endif //EXPAND_INDEX_H_