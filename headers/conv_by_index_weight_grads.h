#ifndef CONV_BY_INDEX_WEIGHT_GRADS_2D_H_
#define CONV_BY_INDEX_WEIGHT_GRADS_2D_H_

template <typename Device, typename T>
struct ConvByIndexWeightGrads2DFunctor{
  void operator()(OpKernelContext *context, const Tensor& input, const Tensor& out_backprop, const Tensor& indexes,
  		  Tensor* weight_backprop, int batch, int out_rows, int out_cols, int out_depth,
  		  int in_depth, int in_rows, int in_cols, int filter_rows, int filter_cols, int num_indeces,
          Padding padding_, TensorFormat data_format_);
};

#endif //CONV_BY_INDEX_WEIGHT_GRADS_2D_H_