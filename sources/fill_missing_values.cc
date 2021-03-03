#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/use_cudnn.h"

#include <iostream>
#include <omp.h>

using namespace tensorflow;

REGISTER_OP("OffsetCorrect")
    .Input("input: T")
    .Input("mask: int32")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("use_cudnn_on_gpu: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "fill_missing_values.h"


// CPU specialization of actual computation.
template <typename T>
struct FillMissingValuesFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* input, T* out, const T *offset, int* mask, const size_t dataLength) {
    T ofst = *offset;
    #pragma omp parallel for
    for(size_t p = 0; p<dataLength; p++){
        if(std::abs(input[p])<ofst){
            mask[p] = 1;
            out[p] = input[p];
        }
        else{
            mask[p] = 0;
            out[p] = 0;
        }
    }
  }
};


template <typename Device, typename T>
class FillMissingValuesOp : public OpKernel {
 public:
  explicit FillMissingValuesOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
  }


  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);
    const Tensor& mask = context->input(1);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == 4,
                errors::InvalidArgument("Input must have rank 4",
                                        input.shape().DebugString()));

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, mask.dims() == 4,
                errors::InvalidArgument("Mask must have rank 4",
                                        mask.shape().DebugString()));
    for(int i=0; i<input.dims(); i++){
      OP_REQUIRES(context, input.dim_size(i) == mask.dim_size(i),
                  errors::InvalidArgument("Mask and Input must have the same dimenstions",
                                          mask.shape().DebugString()));      
    }
    // The second dimension for input is rows/height.
    // The first dimension for filter is rows/height.
    const int64 input_rows_raw = input.dim_size(1);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);

    // The third dimension for input is columns/width.
    // The second dimension for filter is columns/width.
    const int64 input_cols_raw = input.dim_size(2);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);

    // The first dimension for input is batch.
    const int64 batch_raw = input.dim_size(0);
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    const int batch = static_cast<int>(batch_raw);

    // The first dimension for input is batch.
    const int64 depth_raw = input.dim_size(3);
    OP_REQUIRES(context,
                FastBoundsCheck(depth_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("depth is too large"));
    const int depth = static_cast<int>(depth_raw);


    TensorShape out_shape = input.shape();
        // ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "FilllMissingValues: "
            << "input_cols = " << input_cols
            << ", input_rows = " << input_rows;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) return;

    const T *in = input.flat<T>().data();
    T *out = output->flat<T>().data();
    int *msk = mask->flat<int32>().data();
    const size_t dataLength = (size_t)input.NumElements();

	FillMissingValuesFunctor<Device, T>()(context->eigen_device<Device>(), in, out, msk, dataLength);
  }

 private:
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(FillMissingValuesOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("FilllMissingValues").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      FillMissingValuesOp<CPUDevice, T>);
REGISTER_CPU(float);
// REGISTER_CPU(int32);

// Register the GPU kernels.
// #ifdef GOOGLE_CUDA
// #define REGISTER_GPU(T)                                          \
//   REGISTER_KERNEL_BUILDER(                                       \
//       Name("OffsetCorrect").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
//       OffsetCorrectOp<GPUDevice, T>);
// REGISTER_GPU(float);
// REGISTER_GPU(int32);
// #endif  // GOOGLE_CUDA
