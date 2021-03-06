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

REGISTER_OP("PoolByIndex")
    .Input("input: T")
    .Input("indexes: int32")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("use_cudnn_on_gpu: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "pool_by_index.h"

// CPU specialization of actual computation.
template <typename T>
struct PoolByIndexFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* input, T* out, const int* indexes, const size_t indLength, const size_t inbStep, const size_t outbStep) {
        #pragma omp parallel
        {
            size_t threadID = (size_t)omp_get_thread_num();
            size_t numThreads = (size_t)omp_get_num_threads();
            size_t istart = threadID*indLength/numThreads;
            size_t iend;
            if(threadID == numThreads - 1) iend = indLength;
            else iend = (threadID+1)*indLength/numThreads;
            // const T* in = input + maxInd*istart;
            T* o = out + istart;
            T* end = out + iend;
            const int * i = indexes + istart;
            for(;o<end;)
                *o++ = *(input + (size_t)*i++);
        }
  }
};


template <typename Device, typename T>
class PoolByIndexOp : public OpKernel {
 public:
  explicit PoolByIndexOp(OpKernelConstruction* context) : OpKernel(context) {
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

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, input.dims() == indexes.dims(),
                errors::InvalidArgument("input must have 1 more dimensions than indexes",
                                        input.shape().DebugString()));

    for(int i = 0; i < indexes.dims(); i++) {
      OP_REQUIRES(
          context,
          indexes.dim_size(i) <= input.dim_size(i),
          errors::InvalidArgument("input and indexes must have corresponding dimensions"));
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


    TensorShape out_shape = indexes.shape();
        // ShapeFromFormat(data_format_, batch, out_rows, out_cols, out_depth);

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "poolByIndex: "
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
    const size_t outbStep = (size_t)(indexes.dim_size(1)*indexes.dim_size(2)*indexes.dim_size(3));
    const size_t inbStep = (size_t)(input.dim_size(1)*input.dim_size(2)*input.dim_size(3));

	PoolByIndexFunctor<Device, T>()(context->eigen_device<Device>(), in, out, ind, indLength, inbStep, outbStep);

  }

 private:
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(PoolByIndexOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("PoolByIndex").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      PoolByIndexOp<CPUDevice, T>);
REGISTER_CPU(float);
// REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("PoolByIndex").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      PoolByIndexOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
