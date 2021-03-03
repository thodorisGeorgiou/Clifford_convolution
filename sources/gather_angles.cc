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

REGISTER_OP("GatherAngles")
    .Input("angles: T")
    .Input("indexes: int32")
    .Input("thetas: T")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("use_cudnn_on_gpu: bool = true")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "gather_angles.h"

// CPU specialization of actual computation.
template <typename T>
struct GatherAnglesFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const T* angs, const int* indexes, T* out, const size_t indLength) {
        #pragma omp parallel
        {
            size_t threadID = (size_t)omp_get_thread_num();
            size_t numThreads = (size_t)omp_get_num_threads();
            size_t istart = threadID*indLength/numThreads;
            size_t iend;
            if(threadID == numThreads - 1) iend = indLength;
            else iend = (threadID+1)*indLength/numThreads;
            const int* in = indexes + istart;
            T* o = out + istart;
            T* end = out + iend;
            for(;o<end;) *o++ = angs[*in++];
        }
  }
};


template <typename Device, typename T>
class GatherAnglesOp : public OpKernel {
 public:
  explicit GatherAnglesOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
  }


  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [num_bins]
    const Tensor& angles = context->input(0);

    // Input indexes is of the following dimensions:
    // [batch, in_rows, in_cols, in_depth]
    const Tensor& indexes = context->input(1);

    // Input thetas is of the following dimensions:
    const Tensor& thetas = context->input(2);

    // Angles should be one dimensional
    OP_REQUIRES(context, angles.dims() == 1,
                errors::InvalidArgument("input angles must be 1 dimensional",
                                        angles.shape().DebugString()));

    // Indexes and thetas should hae the same dimensionality
    OP_REQUIRES(context, indexes.dims() == thetas.dims(),
                errors::InvalidArgument("Indexes and thetas should have the same dimensionality",
                                        indexes.shape().DebugString()));

    for(int i = 0; i < indexes.dims(); i++) {
      OP_REQUIRES(
          context,
          indexes.dim_size(i) == thetas.dim_size(i),
          errors::InvalidArgument("Indexes and thetas should have the same dimensionality"));
    }


    // The second dimension for indexes is rows/height.
    const int64 input_rows_raw = indexes.dim_size(1);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_rows_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input rows too large"));
    const int input_rows = static_cast<int>(input_rows_raw);

    // The third dimension for input is columns/width.
    const int64 input_cols_raw = indexes.dim_size(2);
    OP_REQUIRES(
        context,
        FastBoundsCheck(input_cols_raw, std::numeric_limits<int>::max()),
        errors::InvalidArgument("Input cols too large"));
    const int input_cols = static_cast<int>(input_cols_raw);

    // The first dimension for indexes is batch.
    const int64 batch_raw = indexes.dim_size(0);
    OP_REQUIRES(context,
                FastBoundsCheck(batch_raw, std::numeric_limits<int>::max()),
                errors::InvalidArgument("batch is too large"));
    
    const int batch = static_cast<int>(batch_raw);


    TensorShape out_shape = indexes.shape();

    // Output tensor is of the following dimensions:
    // [ in_batch, out_rows, out_cols, out_depth ]
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    VLOG(2) << "GatherAngles: "
            << "input_cols = " << input_cols
            << ", input_rows = " << input_rows;

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    const T *angs = angles.flat<T>().data();
    const int *inds = indexes.flat<int32>().data();
    T *out = output->flat<T>().data();
    const size_t indLength = (size_t)indexes.NumElements();

	GatherAnglesFunctor<Device, T>()(context->eigen_device<Device>(), angs, inds, out, indLength);

  }

 private:
  bool use_cudnn_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(GatherAnglesOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("GatherAngles").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      GatherAnglesOp<CPUDevice, T>);
REGISTER_CPU(float);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("GatherAngles").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      GatherAnglesOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA