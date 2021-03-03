#ifndef FILL_MISSING_VALUES_H_
#define FILL_MISSING_VALUES_H_

template <typename Device, typename T>
struct FillMissingValuesFunctor{
  void operator()(OpKernelContext *context, const Tensor& input, T* output, const int* mask, const int depth, const int batch, const int rows, const int cols);
};

#endif //FILL_MISSING_VALUES_H_