#ifndef CALCULATE_PROPORTIONS_H_
#define CALCULATE_PROPORTIONS_H_


template <typename Device, typename T>
struct CalculateProportionsFunctor {
  void operator()(OpKernelContext *context, const T* input, T* out, const int* indexes, T *sums, const int maxInd, const size_t indLength);
};

#endif //CALCULATE_PROPORTIONS_H_