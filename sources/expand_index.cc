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

REGISTER_OP("ExpandIndex")
    .Input("input: T")
    .Input("indexes: int32")
    .Input("target_shaped_tensor: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("use_cudnn_on_gpu: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(2));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "expand_index.h"

// CPU specialization of actual computation.
template <typename T>
struct ExpandIndexFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* input, T* out, const int* indexes, const size_t maxInd, const size_t indLength) {
        #pragma omp parallel for
        for(size_t i=0; i<maxInd*indLength ; i++){
          out[i] = 0;
        }
        #pragma omp parallel
        {
            size_t threadID = (size_t)omp_get_thread_num();
            size_t numThreads = (size_t)omp_get_num_threads();
            size_t istart = threadID*indLength/numThreads;
            size_t iend;
            if(threadID == numThreads - 1) iend = indLength;
            else iend = (threadID+1)*indLength/numThreads;
            const T* in = input + istart;
            T* o = out + maxInd*istart;
            const int *i = indexes + istart;
            const int *end = indexes + iend;
            for(;i<end; o+=maxInd)
                *(o + (size_t)*i++) = *in++;
        }
  }
};


template <typename Device, typename T>
class ExpandIndexOp : public OpKernel {
 public:
  explicit ExpandIndexOp(OpKernelConstruction* context) : OpKernel(context) {
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

    const Tensor& target_shaped_tensor = context->input(2);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == indexes.dims(),
                errors::InvalidArgument("input must the same dimensions with indexes",
                                        input.shape().DebugString()));

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, target_shaped_tensor.dims() == indexes.dims() + 1,
                errors::InvalidArgument("output must have 1 more dimensions than indexes",
                                        target_shaped_tensor.shape().DebugString()));

    for(int i = 0; i < indexes.dims(); i++) {
      OP_REQUIRES(
          context,
          indexes.dim_size(i) == input.dim_size(i),
          errors::InvalidArgument("input and indexes must have corresponding dimensions"));

      OP_REQUIRES(
          context,
          indexes.dim_size(i) == target_shaped_tensor.dim_size(i),
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


    TensorShape out_shape = target_shaped_tensor.shape();
        // ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "expandIndex: "
            << "input_cols = " << input_cols
            << ", input_rows = " << input_rows;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    const T *in = input.flat<T>().data();
    T *out = output->flat<T>().data();
    const int *ind = indexes.flat<int32>().data();
    const size_t indLength = (size_t)indexes.NumElements();
    const size_t maxInd = target_shaped_tensor.dim_size(target_shaped_tensor.dims()-1);

	ExpandIndexFunctor<Device, T>()(context->eigen_device<Device>(), in, out, ind, maxInd, indLength);

  }

 private:
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExpandIndexOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ExpandIndex").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      ExpandIndexOp<CPUDevice, T>);
REGISTER_CPU(float);
// REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ExpandIndex").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      ExpandIndexOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
