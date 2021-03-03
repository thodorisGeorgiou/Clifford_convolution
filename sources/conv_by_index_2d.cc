#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
// #include "third_party/eigen3/unsupported/Eigen/CXX11/ThreadPool"
// #include "tensorflow/core/framework/resource_mgr.h"
// #include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/use_cudnn.h"
// #include "tensorflow/core/kernels/conv_ops.h"
// #include "tensorflow/core/kernels/conv_2d.h"
// #include "tensorflow/core/util/tensor_format.h"

#include <iostream>
#include <omp.h>

using namespace tensorflow;

REGISTER_OP("ConvByIndex2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("indexes: int32")
    .Input("mask: int32")
    .Input("angles: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "conv_by_index_2d.h"

typedef void (*func_ptr)(float* inpt, const float *filter, float *out, int filIndex, int in_channels, int in_cols, int filter_rows,
    int filter_cols, int out_channels);

void setZeroOutput(float* inpt, const float *filter, float *out, int filIndex, int in_channels, int in_cols, int filter_rows,
    int filter_cols, int out_channels){
    *out = 0;
}


void singlePointConv(float* inpt, const float *filter, float *out, int filIndex, int in_channels, int in_cols, int filter_rows,
    int filter_cols, int out_channels){
    out[0] = 0;
    int fcStep = in_channels*out_channels;
    int frStep = filter_cols*fcStep;
    int irStep = in_cols*in_channels;
    for (int r=0; r < filter_rows; r++){
        int frp = r*frStep;
        int irp = (r-filter_rows/2)*irStep;
        for(int c=0; c < filter_cols; c++){
            const float *fcp = filter+(size_t)(frp + c*fcStep);
            float *icp = inpt+(size_t)(irp + (c-filter_cols/2)*in_channels);
            for(int d=0; d < in_channels; d++) out[0] += icp[d]*fcp[d*out_channels + filIndex];
        }
    }
}

void copyTo(const float *inpt, float *out, int size){
    for (int i = 0; i < size; i++) out[i] = inpt[i];
}

// CPU specialization of actual computation.
template <typename T>
struct ConvByIndexFanctor<CPUDevice, T> {
  void operator()(OpKernelContext *context, const Tensor& input, Tensor* output, const Tensor& filter, const Tensor& indexes, const Tensor& mask,
          int batch, int out_rows, int out_cols, int out_depth, int in_depth,
          int in_rows, int in_cols, int stride_cols, int stride_rows,
          int num_filter_sets, int filter_rows, int filter_cols,
          Padding padding_, TensorFormat data_format_) {

        const T *in = input.flat<T>().data();
        T *out = output->flat<T>().data();
        const T *fil = filter.flat<T>().data();
        const int *ind = indexes.flat<int32>().data();
        const int *msk = mask.flat<int32>().data();

        int padding_rows = 0;
        int padding_cols = 0;
        Tensor new_input = input;
        int new_in_rows = in_rows;
        int new_in_cols = in_cols;
        int paddingOffsetRows = 0;
        int paddingOffsetCols = 0;

        if (padding_ == SAME) {
            padding_rows = std::max<int>(0, (out_rows - 1) * stride_rows + filter_rows - in_rows);
            padding_cols = std::max<int>(0, (out_cols - 1) * stride_cols + filter_cols - in_cols);

            // const bool rows_odd = (padding_rows % 2 != 0);
            // const bool cols_odd = (padding_cols % 2 != 0);

            Tensor transformed_input;
            new_in_rows = in_rows + padding_rows;
            new_in_cols = in_cols + padding_cols;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                        ShapeFromFormat(data_format_, batch, new_in_rows,
                        new_in_cols, in_depth), &transformed_input));

            paddingOffsetRows = padding_rows/2;
            paddingOffsetCols = padding_cols/2;
            T *tInput = transformed_input.flat<T>().data();
            int bStep = new_in_cols*new_in_rows*in_depth;
            int rStep = new_in_cols*in_depth;

            int orStep = in_cols*in_depth;
            int obStep = orStep*in_rows;
            /*Set padding rows and cols*/
            #pragma omp parallel for collapse(2)
            for(int b = 0; b < batch; b++){
                for(int r = 0; r < in_rows; r++){
                    T *tp = tInput + (size_t)(b*bStep + (r+paddingOffsetRows)*rStep + paddingOffsetCols*in_depth);
                    const T *op = in + (size_t)(b*obStep + r*orStep);
                    copyTo(op, tp, orStep);
                }
            }
            #pragma omp parallel for
            for(int b = 0; b < batch; b++){
                float * tbp = tInput + (size_t)(b*bStep);
                // const float * ibp = in + (size_t)(b*obStep);
                for(int cbr=0; cbr < paddingOffsetRows*rStep; cbr++) tbp[cbr] = 0;
                for(int cbr=(paddingOffsetRows+in_rows)*rStep; cbr < new_in_rows*rStep; cbr++) tbp[cbr] = 0;
                for(int r=paddingOffsetRows*rStep; r < (paddingOffsetRows+in_rows)*rStep; r+=rStep){
                    for(int cb=r; cb<r+in_depth*paddingOffsetCols; cb++) tbp[cb] = 0;
                    for(int cb=r+in_depth*(paddingOffsetCols+in_cols); cb<r+rStep; cb++) tbp[cb] = 0;
                }
            }
            new_input = transformed_input;
        } /*Padding done*/

        func_ptr convByMask[2] = {&setZeroOutput, &singlePointConv};
        T *newIn = new_input.flat<T>().data();
        int filSetSize = filter_rows*filter_cols*in_depth*out_depth;
        int inrStep = new_in_cols*in_depth;
        int inbStep = inrStep*new_in_rows;
        int outrStep = out_cols*out_depth;
        int outbStep = outrStep*out_rows;
        #pragma omp parallel for collapse(4)
        for(int n = 0; n < batch; n++){
            for(int r = 0; r < out_rows; r++){
                for(int c = 0; c < out_cols; c++){
                    for(int d = 0; d < out_depth; d++){
                        float *inpt = newIn + (size_t)(n*inbStep + (r+paddingOffsetRows)*inrStep + (c+paddingOffsetCols)*in_depth);
                        size_t outIndex = (size_t)(n*outbStep + r*outrStep + c*out_depth + d);
                        const float *fltr = fil + (size_t)(filSetSize*ind[outIndex]);
                        convByMask[msk[outIndex]](inpt, fltr, out+outIndex, d, in_depth, new_in_cols, filter_rows, filter_cols, out_depth);
                        // singlePointConv(inpt, fltr, out+outIndex, in_depth, new_in_cols, filter_rows, filter_cols, out_depth);
                    }
                }
            }      
        }
  }
};


template <typename Device, typename T>
class ConvByIndex2DOp : public OpKernel {
 public:
  explicit ConvByIndex2DOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = strides_[0];
    const int64 stride_c = strides_[3];
    const int64 stride_h = strides_[1];
    const int64 stride_w = strides_[2];
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        context, stride_h == 1 && stride_w == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the row and column dimensions."));
    // OP_REQUIRES(context, stride_h > 0 && stride_w > 0,
    //             errors::InvalidArgument(
    //                 "Row and column strides should be larger than 0."));

    const int64 dilation_n = dilations_[0];
    const int64 dilation_c = dilations_[3];
    const int64 dilation_h = dilations_[1];
    const int64 dilation_w = dilations_[2];
    OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h == 1 && dilation_w == 1,
        errors::InvalidArgument("Current implementation does not support dilations"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }


  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& filter = context->input(1);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& indexes = context->input(2);

    const Tensor& mask = context->input(3);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, indexes.dims() == 4,
                errors::InvalidArgument("input must be 4-dimensional",
                                        input.shape().DebugString()));
    // For 2D convolution, there should be 4 dimensions.
    // printf("Weight dimensions: %d\n", filter.dims());
    OP_REQUIRES(context, filter.dims() == 5,
                errors::InvalidArgument("filter must be 5-dimensional: ",
                                        filter.shape().DebugString()));

    OP_REQUIRES(context, mask.dims() == 4,
                errors::InvalidArgument("Mask must be 4-dimensional: ",
                                        mask.shape().DebugString()));


    for(int i = 0; i < 5; i++) {
      OP_REQUIRES(
          context,
          FastBoundsCheck(filter.dim_size(i), std::numeric_limits<int>::max()),
          errors::InvalidArgument("filter too large"));
    }

    // The last dimension for input is in_depth. It must be the same as the
    // filter's in_depth or be evenly divisible by filter's in_depth.
    const int64 in_depth = input.dim_size(3);
    const int64 patch_depth = filter.dim_size(3);
    OP_REQUIRES(context, in_depth % patch_depth == 0,
                errors::InvalidArgument(
                    "input depth must be evenly divisible by filter depth: ",
                    in_depth, " vs ", patch_depth));

    // The last dimension for filter is out_depth.
    const int out_depth = static_cast<int>(filter.dim_size(4));

    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = input.dim_size(1);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);
    const int filter_rows = static_cast<int>(filter.dim_size(1));
    const int num_filter_sets = static_cast<int>(filter.dim_size(0));

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = input.dim_size(2);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);
    const int filter_cols = static_cast<int>(filter.dim_size(2));

    // The first dimension for input is batch.
    const int64 batch_raw = input.dim_size(0);
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // For now we take the stride and dilation from the second and third
    // dimensions only (we do not support striding or dilation on the batch or
    // depth dimension).
    const int stride_rows = strides_[1];
    const int stride_cols = strides_[2];

    const int dilation_rows = dilations_[1];
    const int dilation_cols = dilations_[2];

    int64 out_rows = 0, out_cols = 0, pad_rows = 0, pad_cols = 0;
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_rows, filter_rows, dilation_rows,
                                stride_rows, padding_, &out_rows, &pad_rows));
    OP_REQUIRES_OK(context, GetWindowedOutputSizeV2(
                                input_cols, filter_cols, dilation_cols,
                                stride_cols, padding_, &out_cols, &pad_cols));

    OP_REQUIRES(
        context, batch == indexes.dim_size(0), 
        errors::InvalidArgument("indexes must have the same batch size as output"));

    OP_REQUIRES(
        context, out_rows == indexes.dim_size(1), 
        errors::InvalidArgument("indexes must have the same number of rows as output"));

    OP_REQUIRES(
        context, out_cols == indexes.dim_size(2), 
        errors::InvalidArgument("indexes must have the same number of columns as output"));

    OP_REQUIRES(
        context, out_depth == indexes.dim_size(3), 
        errors::InvalidArgument("indexes must have the same number of channels as output"));

    OP_REQUIRES(
        context, batch == mask.dim_size(0), 
        errors::InvalidArgument("Mask must have the same batch size as output"));

    OP_REQUIRES(
        context, out_rows == mask.dim_size(1), 
        errors::InvalidArgument("Mask must have the same number of rows as output"));

    OP_REQUIRES(
        context, out_cols == mask.dim_size(2), 
        errors::InvalidArgument("Mask must have the same number of columns as output"));

    OP_REQUIRES(
        context, out_depth == mask.dim_size(3), 
        errors::InvalidArgument("Mask must have the same number of channels as output"));


    TensorShape out_shape = indexes.shape();

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "ConvByIndex2D: in_depth = " << in_depth
            << ", patch_depth = " << patch_depth
            << ", input_cols = " << input_cols
            << ", filter_cols = " << filter_cols
            << ", input_rows = " << input_rows
            << ", filter_rows = " << filter_rows
            << ", stride_rows = " << stride_rows
            << ", stride_cols = " << stride_cols
            << ", dilation_rows = " << dilation_rows
            << ", dilation_cols = " << dilation_cols
            << ", out_depth = " << out_depth;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

	ConvByIndexFanctor<Device, T>()(
	       context, input, output, filter, indexes, mask,
	       batch, out_rows, out_cols, out_depth,
	       in_depth, input_rows, input_cols, stride_cols, stride_rows,
           num_filter_sets, filter_rows, filter_cols,
           padding_, data_format_);

  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  // LaunchConv2DOp<Device, T> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConvByIndex2DOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvByIndex2D").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      ConvByIndex2DOp<CPUDevice, T>);
REGISTER_CPU(float);
// REGISTER_CPU(int32);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvByIndex2D").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      ConvByIndex2DOp<GPUDevice, T>);
REGISTER_GPU(float);
// REGISTER_GPU(int32);
#endif  // GOOGLE_CUDA


// Register the GPU kernels.
// #ifdef GOOGLE_CUDA
// #define REGISTER_GPU(T)                                          \
//   /* Declare explicit instantiations in kernel_example.cu.cc. */ \
//   extern template ConvByIndexFunctor<GPUDevice, T>;                  \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("ConvByIndex2D").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
//       ConvByIndex2DOp<GPUDevice, T>);
// REGISTER_GPU(float);
// REGISTER_GPU(int32);
// #endif  // GOOGLE_CUDA