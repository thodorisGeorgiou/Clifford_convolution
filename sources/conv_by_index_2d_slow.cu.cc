#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "conv_by_index_2d.h"

// Define the CUDA kernel.


__device__ void singlePointConv(float* inpt, const float *filter, float *out, size_t filIndex, size_t in_channels, size_t in_cols, size_t filter_rows,
    size_t filter_cols, size_t out_channels, size_t frStep, size_t irStep, size_t fcStep){
    float result = 0;
    for(size_t r=0; r < filter_rows; r++){
        const float *frp = r*frStep + filter;
        float *irp = inpt+(r-filter_rows/2)*irStep - (in_channels*(filter_cols/2));
        for(size_t c=0; c < filter_cols; c++){
            const float *fcp = frp + c*fcStep;
            for(size_t d=0; d < in_channels; d++) result += (*irp++)*fcp[d*out_channels + filIndex];
        }
    }
    *out = result;
}
            // float *icp = irp + (c-filter_cols/2)*in_channels;

__device__ void cudaCopyTo(const float *inpt, float *out, size_t size){
    for (int i = 0; i < size; i++) out[i] = inpt[i];
}

template <typename T>
__global__ void ConvByIndexCudaKernel(T *input, T *out, const T *filters, const int *ind, const int *msk, size_t outbStep, size_t outrStep, size_t out_depth,
  size_t inbStep, size_t irStep, size_t in_depth, size_t filSetSize, size_t filter_rows, size_t filter_cols, size_t in_cols, size_t frStep, size_t fcStep,
  size_t paddingOffsetRows, size_t paddingOffsetCols, size_t numJobs){
      size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
      size_t num_threads = (size_t)(blockDim.x*gridDim.x);
      for(size_t outIndex=(size_t)threadID; outIndex<numJobs;outIndex+=num_threads){
          size_t n = outIndex / outbStep;
          size_t r = (outIndex % outbStep)/outrStep;
          size_t c = ((outIndex % outbStep) % outrStep) / out_depth;
          size_t f = ((outIndex % outbStep) % outrStep) % out_depth;
          T *inpt = input + n*inbStep + (r+paddingOffsetRows)*irStep + (c+paddingOffsetCols)*in_depth;
          const T *fltr = filters + (size_t)(filSetSize*ind[outIndex]);
          if(msk[outIndex]) singlePointConv(inpt, fltr, out+outIndex, f, in_depth, in_cols, filter_rows, filter_cols, out_depth, frStep, irStep, fcStep);
          else *(out+outIndex) = 0;
      }
  }

template <typename T>
__global__ void padCudaKernel(T * tInput, const T* in, size_t depth, size_t bStep, size_t rStep, size_t obStep, size_t orStep, size_t oNumRows, size_t paddingOffsetRows, size_t paddingOffsetCols){
    size_t b = (size_t)blockIdx.x;
    size_t r = (size_t)threadIdx.x;
    T * tbp = tInput + b*bStep + r*rStep;
    if(r < paddingOffsetRows || r >= paddingOffsetRows + oNumRows)
      for(T *c = tbp; c<tbp+rStep;) *c++ = 0;
    else{
      T *tp = tbp + paddingOffsetCols*depth;
      const T *op = in + b*obStep + (r - paddingOffsetRows)*orStep;
      cudaCopyTo(op, tp, orStep);
      for(T *c=tbp; c < tp;) *c++ = 0;
      for(T *c=tp+orStep; c < tbp+rStep;) *c++ = 0;
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct ConvByIndexFanctor<GPUDevice, T>{
  void operator()(OpKernelContext *context,
          const Tensor& input, Tensor* output, const Tensor& filter, const Tensor& indexes,
          const Tensor& mask, int batch, int out_rows, int out_cols, int out_depth, int in_depth,
          int in_rows, int in_cols, int stride_cols, int stride_rows,
          int num_filter_sets, int filter_rows, int filter_cols,
          Padding padding_, TensorFormat data_format_) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.

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
      size_t rStep = new_in_cols*in_depth;
      size_t bStep = new_in_rows*rStep;

      size_t orStep = in_cols*in_depth;
      size_t obStep = orStep*in_rows;
      /*Set padding rows and cols*/
      padCudaKernel<T><<<batch, new_in_rows, 0, context->eigen_device<GPUDevice>().stream()>>>
      (tInput, in, (size_t)in_depth, bStep, rStep, obStep, orStep, (size_t)in_rows, paddingOffsetRows, paddingOffsetCols);
      new_input = transformed_input;
  } /*Padding done*/

  T *newIn = new_input.flat<T>().data();
  int irStep = new_in_cols*in_depth;
  int inbStep = irStep*new_in_rows;
  int outrStep = out_cols*out_depth;
  int outbStep = outrStep*out_rows;
  int numJobs = batch*outbStep;

  int fcStep = in_depth*out_depth;
  int frStep = filter_cols*fcStep;
  int filSetSize = filter_rows*frStep;

  CudaLaunchConfig cfg = GetCudaLaunchConfig(numJobs, context->eigen_device<GPUDevice>());
  // printf("block count: %d\nThread per block: %d\nNum operations: %d\n", (int)cfg.block_count, (int)cfg.thread_per_block, batch*out_rows*out_cols*out_depth);
  cudaDeviceSynchronize();
  ConvByIndexCudaKernel<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  (newIn, out, fil, ind, msk, (size_t)outbStep, (size_t)outrStep, (size_t)out_depth, (size_t)inbStep, (size_t)irStep, (size_t)in_depth, (size_t)filSetSize,
    (size_t)filter_rows, (size_t)filter_cols, (size_t)new_in_cols, (size_t)frStep, (size_t)fcStep, (size_t)paddingOffsetRows, (size_t)paddingOffsetCols, (size_t)numJobs);
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ConvByIndexFanctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA