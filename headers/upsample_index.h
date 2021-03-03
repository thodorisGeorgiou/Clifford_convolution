#ifndef UPSAMPLE_INDEX_H_
#define UPSAMPLE_INDEX_H_


template <typename Device, typename T>
struct UpsampleIndexFunctor {
  void operator()(const Device& d, const T* input, T* out, const int* indexes, const size_t indLength,
  	const size_t outLength, const size_t inbStep, const size_t outbStep);
};

#endif //UPSAMPLE_INDEX_H_