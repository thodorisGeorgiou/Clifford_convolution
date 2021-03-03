#ifndef CONV_BY_INDEX_2D_H_
#define CONV_BY_INDEX_2D_H_

template <typename Device, typename T>
struct ConvByIndexFanctor {
  void operator()(OpKernelContext *context, const Tensor& input, Tensor* output, const Tensor& filter, const Tensor& indexes, const Tensor& mask,
	      int batch, int out_rows, int out_cols, int out_depth, int in_depth,
	      int in_rows, int in_cols, int stride_cols, int stride_rows,
	      int num_filter_sets, int filter_rows, int filter_cols,
	      Padding padding_, TensorFormat data_format_);
};

#endif //CONV_BY_INDEX_2D_H_