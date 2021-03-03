#ifndef CONV_BY_INDEX_INPUT_GRADS_2D_H_
#define CONV_BY_INDEX_INPUT_GRADS_2D_H_

template <typename Device, typename T>
struct ConvByIndexInputGrads2DFunctor{
  void operator()(OpKernelContext *context, const Tensor& filter, const Tensor& out_backprop, const Tensor& indexes,
  		  Tensor* in_backprop, int batch, int out_rows, int out_cols, int out_depth,
  		  int in_depth, int in_rows, int in_cols, int filter_rows, int filter_cols,
          Padding padding_, TensorFormat data_format_);
};

#endif //CONV_BY_INDEX_INPUT_GRADS_2D_H_