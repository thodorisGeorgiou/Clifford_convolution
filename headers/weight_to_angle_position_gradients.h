#ifndef WEIGHT_TO_ANGLE_GRADIENTS_H_
#define WEIGHT_TO_ANGLE_GRADIENTS_H_

#ifndef M_PI
#define M_PI 3.14159265358979323846  /* pi */
#endif //M_PI


template <typename Device, typename T>
struct CalculateWeightToAngleGradientsFunctor<Device, T> {
  void operator()(OpKernelContext *context, const Tensor& input, T* output, const int* indexes, const T *weights, const T* gradients, T *woa, \
    const int filter_rows, const int filter_cols, const int num_angles, const int in_depth, const int out_depth, \
    const int batch, const int rows, const int cols,Padding padding_,TensorFormat data_format_);
};

#endif //WEIGHT_TO_ANGLE_GRADIENTS_H_