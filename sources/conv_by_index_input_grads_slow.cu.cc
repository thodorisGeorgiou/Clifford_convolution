#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "conv_by_index_input_grads.h"

// Define the CUDA kernel.

__device__ void singlePointGatherGradients(const float *outBp, const int *indexes, const float *filters, float *inBackprop, size_t out_depth, size_t out_cols,
  size_t filter_rows, size_t filter_cols, size_t in_depth, size_t fcStep, size_t frStep, size_t foStep, size_t orStep) {
    float result = 0;
    for (size_t r=0; r < filter_rows; r++){
        size_t frp = (filter_rows -1 - r)*frStep;
        size_t out = r*orStep;
        const float *orp = outBp+out;
        const int *pInds = indexes+out;
        for(size_t c=0; c < filter_cols; c++){
            size_t fcp = frp + (filter_cols -1 - c)*fcStep;
            for(size_t d=0; d < out_depth; d++)
              result += (*orp++)*filters[(*pInds++)*foStep + (fcp++)];
        }
    }
    *inBackprop = result;
}


template <typename K>
__device__ void cudaCopyTo(const K *inpt, K *out, size_t size){
    for (int i = 0; i < size; i++) out[i] = inpt[i];
}

template <typename T>
__global__ void ConvByIndexInputGradsCudaKernel(T *outBprop, T *inBprop, const T *filters, int *ind, size_t outbStep, size_t outrStep, size_t out_depth,
  size_t inbStep, size_t irStep, size_t in_depth, size_t filSetSize, size_t filter_rows, size_t filter_cols, size_t out_cols, size_t frStep, size_t fcStep, size_t numJobs){
      size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
      size_t num_threads = (size_t)(blockDim.x*gridDim.x);
      for(size_t inIndex=threadID; inIndex<numJobs; inIndex+=num_threads){
          size_t n = inIndex / inbStep;
          size_t r = (inIndex % inbStep)/irStep;
          size_t c = ((inIndex % inbStep) % irStep) / in_depth;
          size_t f = ((inIndex % inbStep) % irStep) % in_depth;
          size_t outIndex = n*outbStep + r*outrStep + c*out_depth;
          T *outBp = outBprop + outIndex;
          int *indp = ind + outIndex;
          singlePointGatherGradients(outBp, indp, filters+f*out_depth, inBprop+inIndex, out_depth, out_cols, filter_rows, filter_cols, in_depth, fcStep, frStep, filSetSize, outrStep);
      }
  }

template <typename K>
__global__ void padCudaKernel(K * tInput, const K* in, size_t depth, size_t bStep, size_t rStep, size_t obStep, size_t orStep, size_t oNumRows, size_t paddingOffsetRows, size_t paddingOffsetCols){
    size_t b = (size_t)blockIdx.x;
    size_t r = (size_t)threadIdx.x;
    K * tbp = tInput + b*bStep + r*rStep;
    if(r < paddingOffsetRows || r >= paddingOffsetRows + oNumRows)
      for(K *c = tbp; c<tbp+rStep;) *c++ = 0;
    else{
      K *tp = tbp + paddingOffsetCols*depth;
      const K *op = in + b*obStep + (r - paddingOffsetRows)*orStep;
      cudaCopyTo(op, tp, orStep);
      for(K *c=tbp; c < tp;) *c++ = 0;
      for(K *c=tp+orStep; c < tbp+rStep;) *c++ = 0;
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct ConvByIndexInputGrads2DFunctor<GPUDevice, T>{
  void operator()(OpKernelContext *context, const Tensor& filter, const Tensor& out_backprop, const Tensor& indexes,
        Tensor* in_backprop, int batch, int out_rows, int out_cols, int out_depth,
        int in_depth, int in_rows, int in_cols, int filter_rows, int filter_cols,
          Padding padding_, TensorFormat data_format_) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.

  const T *outBackprop = out_backprop.flat<T>().data();
  T *inBackprop = in_backprop->flat<T>().data();
  const T *fil = filter.flat<T>().data();
  const int *ind = indexes.flat<int32>().data();

  int padding_rows = 0;
  int padding_cols = 0;
  Tensor paddedOutBackprop = out_backprop;
  Tensor paddedIndexes = indexes;
  int new_out_rows = out_rows;
  int new_out_cols = out_cols;
  int paddingOffsetRows = 0;
  int paddingOffsetCols = 0;


  if (padding_ == SAME) {
      padding_rows = std::max<int>(0, (in_rows - 1) + filter_rows - out_rows);
      padding_cols = std::max<int>(0, (in_cols - 1) + filter_cols - out_cols);

      Tensor transformed_outBackprop;
      Tensor transformed_indexes;
      new_out_rows = out_rows + padding_rows;
      new_out_cols = out_cols + padding_cols;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                  ShapeFromFormat(data_format_, batch, new_out_rows,
                  new_out_cols, out_depth), &transformed_outBackprop));

      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int32>::value,
                  ShapeFromFormat(data_format_, batch, new_out_rows,
                  new_out_cols, out_depth), &transformed_indexes));

      paddingOffsetRows = padding_rows/2;
      paddingOffsetCols = padding_cols/2;
      T *tOutBackprop = transformed_outBackprop.flat<T>().data();
      int *tIndexes = transformed_indexes.flat<int32>().data();
      int rStep = new_out_cols*out_depth;
      int bStep = rStep*new_out_rows;

      int orStep = out_cols*out_depth;
      int obStep = orStep*out_rows;

      /*Set padding rows and cols*/
      padCudaKernel<T><<<batch, new_out_rows, 0, context->eigen_device<GPUDevice>().stream()>>>
      (tOutBackprop, outBackprop, (size_t)out_depth, bStep, rStep, obStep, orStep, (size_t)out_rows, paddingOffsetRows, paddingOffsetCols);
      paddedOutBackprop = transformed_outBackprop;
      padCudaKernel<int><<<batch, new_out_rows, 0, context->eigen_device<GPUDevice>().stream()>>>
      (tIndexes, ind, (size_t)out_depth, bStep, rStep, obStep, orStep, (size_t)out_rows, paddingOffsetRows, paddingOffsetCols);
      paddedIndexes = transformed_indexes;
  } /*Padding done*/

  T *padOutBp = paddedOutBackprop.flat<T>().data();
  int *padInds = paddedIndexes.flat<int32>().data();
  int inrStep = in_cols*in_depth;
  int inbStep = inrStep*in_rows;
  int outrStep = new_out_cols*out_depth;
  int outbStep = outrStep*new_out_rows;
  int numJobs = batch*inbStep;

  int fcStep = in_depth*out_depth;
  int frStep = filter_cols*fcStep;
  int filSetSize = filter_rows*frStep;

  CudaLaunchConfig cfg = GetCudaLaunchConfig(numJobs, context->eigen_device<GPUDevice>());
  // printf("block count: %d\nThread per block: %d\nNum operations: %d\n", (int)cfg.block_count, (int)cfg.thread_per_block, batch*out_rows*out_cols*out_depth);
  cudaDeviceSynchronize();
  ConvByIndexInputGradsCudaKernel<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  (padOutBp, inBackprop, fil, padInds, (size_t)outbStep, (size_t)outrStep, (size_t)out_depth, (size_t)inbStep, (size_t)inrStep, (size_t)in_depth, (size_t)filSetSize,
    (size_t)filter_rows, (size_t)filter_cols, (size_t)new_out_cols, (size_t)frStep, (size_t)fcStep, (size_t)numJobs);
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ConvByIndexInputGrads2DFunctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA