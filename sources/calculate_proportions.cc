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

REGISTER_OP("CalculateProportions")
    .Input("input: T")
    .Input("indexes: int32")
    .Input("weights: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("use_cudnn_on_gpu: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "calculate_proportions.h"

// CPU specialization of actual computation.
template <typename T>
struct CalculateProportionsFunctor<CPUDevice, T> {
  void operator()(OpKernelContext *context, const T* input, T* out, const int* indexes, T *sums, const int maxInd, const size_t indLength) {
        #pragma omp parallel for
        for (int i=0; i<maxInd; i++){
            const int *ind = indexes;
            const T *in = input;
            sums[i] = 0;
            for(;ind<indexes+indLength;){
                if(*ind==i) sums[i] += *in;
                in++;
                ind++;
            }
            if(sums[i]==0) sums[i] = 1;
        }
        #pragma omp parallel for
        for(int i=0;i<indLength;i++) out[i] = input[i]/sums[indexes[i]];
  }
};


template <typename Device, typename T>
class CalculateProportionsOp : public OpKernel {
 public:
  explicit CalculateProportionsOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
  }


  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);

    // Input filter is of the following dimensions:
    // [ filter_rows, filter_cols, in_depth, out_depth]
    const Tensor& indexes = context->input(1);
    
    const Tensor& weights = context->input(2);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == indexes.dims(),
                errors::InvalidArgument("input must the same dimensions with indexes",
                                        input.shape().DebugString()));

    // OP_REQUIRES(context, numBins.shape().num_elements() == 1,
    //             errors::InvalidArgument("Number of bins should be one integer",
    //                                     numBins.shape().DebugString()));

    for(int i = 0; i < indexes.dims(); i++) {
      OP_REQUIRES(
          context,
          indexes.dim_size(i) == input.dim_size(i),
          errors::InvalidArgument("input and indexes must have corresponding dimensions"));
    }

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


    TensorShape out_shape = indexes.shape();

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "calculateProportions: "
            << "input_cols = " << input_cols
            << ", input_rows = " << input_rows;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    const T *in = input.flat<T>().data();
    const int maxInd = static_cast<int>(weights.dim_size(0));
    T *out = output->flat<T>().data();
    const int *ind = indexes.flat<int32>().data();
    const size_t indLength = (size_t)indexes.NumElements();
    Tensor sumsPerAngle;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, TensorShape({maxInd}), &sumsPerAngle));
    T * sums = sumsPerAngle.flat<T>().data();

    CalculateProportionsFunctor<Device, T>()(context, in, out, ind, sums, maxInd, indLength);

  }

 private:
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(CalculateProportionsOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CalculateProportions").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      CalculateProportionsOp<CPUDevice, T>);
REGISTER_CPU(float);
// REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CalculateProportions").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      CalculateProportionsOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
