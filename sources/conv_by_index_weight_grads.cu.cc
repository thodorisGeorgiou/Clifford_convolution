#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU

#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/framework/common_shape_fns.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "conv_by_index_weight_grads.h"

// Define the CUDA kernel.


template <typename K>
__global__ void initializeArrayCuda(K *out, int size, K val){
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x*gridDim.x;
    for (int i = threadID; i < size; i+=num_threads) out[i] = val;
}

// __device__ void singleWeightGatherGradients(float *w, float *padIn, const float *outBackprop, const int *indeces, float *result, int fr, int fc,
//     int num_indeces, int filSetSize, int inrStep, int in_depth, int out_depth, int batch, int out_rows, int out_cols, int paddingOffsetRows, int paddingOffsetCols){
__device__ void singleWeightGatherGradients(float *padIn, const float *outBackprop, const int *indeces, float *result, int fr, int fc,
    int num_indeces, int filSetSize, int inrStep, int in_depth, int out_depth, int batch, int out_rows, int out_cols, int paddingOffsetRows, int paddingOffsetCols){
    int outIndex = 0;
    int inIndex = fr*inrStep + fc*in_depth;
    for(float *f = result; f < (result+(size_t)num_indeces); f++) *f = 0;
    // for(int f = 0; f < num_indeces; f++) *(w + (size_t)(f*filSetSize)) = 0;
    for (int b=0; b < batch; b++){
        for(int r = 0; r < out_rows; r++){
            for(int c=0; c < out_cols; c++){
                *(result + (size_t)(indeces[outIndex])) += outBackprop[outIndex] * padIn[inIndex];
                // *(w + (size_t)(indeces[outIndex] * filSetSize)) += outBackprop[outIndex] * padIn[inIndex];
                inIndex += in_depth;
                outIndex += out_depth;
            }
            inIndex += 2*paddingOffsetCols*in_depth;
        }
        inIndex += 2*paddingOffsetRows*inrStep;
    }
    // for(int f = 0; f < num_indeces; f++) *(w + (size_t)(f*filSetSize)) = result[f];
}

template <typename K>
__device__ void cudaCopyTo(const K *inpt, K *out, size_t size){
    for (size_t i = 0; i < size; i++) out[i] = inpt[i];
}

template <typename T>
// __global__ void ConvByIndexWeightGradsCudaKernel(T *padIn, const T *outBackprop, const int *ind, T *weightBackprop, T * res, int out_depth,
__global__ void ConvByIndexWeightGradsCudaKernel(T *padIn, const T *outBackprop, const int *ind, T * res, int out_depth,
  int irStep, int in_depth, int filSetSize, int out_rows, int out_cols, int frStep, int fcStep, int num_indeces, int batch, int paddingOffsetRows, int paddingOffsetCols){
      int threadID = blockIdx.x * blockDim.x + threadIdx.x;
      int num_threads = blockDim.x*gridDim.x;
      for(int wIndex=threadID; wIndex<filSetSize; wIndex+=num_threads){
          int r = wIndex / frStep;
          int c = (wIndex % frStep)/fcStep;
          int din = ((wIndex % frStep) % fcStep) / out_depth;
          int dout = ((wIndex % frStep) % fcStep) % out_depth;
          singleWeightGatherGradients(padIn+(size_t)din, outBackprop+(size_t)dout, ind+(size_t)dout, res+(size_t)(wIndex*num_indeces), r, c, num_indeces, filSetSize, irStep, in_depth,
            out_depth, batch, out_rows, out_cols, paddingOffsetRows, paddingOffsetCols);
          // singleWeightGatherGradients(weightBackprop+(size_t)wIndex, padIn+(size_t)din, outBackprop+(size_t)dout, ind+(size_t)dout, res+(size_t)(wIndex*num_indeces), r, c, num_indeces, filSetSize, irStep, in_depth,
          //   out_depth, batch, out_rows, out_cols, paddingOffsetRows, paddingOffsetCols);
      }
  }

template <typename T>
__global__ void reshapeWeightsCudaKernel(T *res, T* weightBackprop, int out_depth, int in_depth, int filSetSize, int frStep, int fcStep, int num_indeces, int numJobs){
      int threadID = blockIdx.x * blockDim.x + threadIdx.x;
      int num_threads = blockDim.x*gridDim.x;
      for(int wIndex=threadID; wIndex<numJobs; wIndex+=num_threads){
          int in = wIndex / filSetSize;
          int resIndex = (wIndex % filSetSize)*num_indeces + in;
          weightBackprop[wIndex] = res[resIndex];
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
struct ConvByIndexWeightGrads2DFunctor<GPUDevice, T>{
  void operator()(OpKernelContext *context, const Tensor& input, const Tensor& out_backprop, const Tensor& indexes,
        Tensor* weight_backprop, int batch, int out_rows, int out_cols, int out_depth,
        int in_depth, int in_rows, int in_cols, int filter_rows, int filter_cols, int num_indeces,
          Padding padding_, TensorFormat data_format_) {

  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  const T *outBackprop = out_backprop.flat<T>().data();
  T *weightBackprop = weight_backprop->flat<T>().data();
  const T *inpt = input.flat<T>().data();
  const int *ind = indexes.flat<int32>().data();

  int padding_rows = 0;
  int padding_cols = 0;
  Tensor paddedInput = input;
  int new_in_rows = in_rows;
  int new_in_cols = in_cols;
  int paddingOffsetRows = 0;
  int paddingOffsetCols = 0;


  if (padding_ == SAME) {
      padding_rows = std::max<int>(0, (out_rows - 1) + filter_rows - in_rows);
      padding_cols = std::max<int>(0, (out_cols - 1) + filter_cols - in_cols);
      Tensor transformed_input;
      new_in_rows = in_rows + padding_rows;
      new_in_cols = in_cols + padding_cols;
      OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                  ShapeFromFormat(data_format_, batch, new_in_rows,
                  new_in_cols, in_depth), &transformed_input));

      paddingOffsetRows = padding_rows/2;
      paddingOffsetCols = padding_cols/2;
      T *tInput = transformed_input.flat<T>().data();
      int rStep = new_in_cols*in_depth;
      int bStep = rStep*new_in_rows;

      int orStep = in_cols*in_depth;
      int obStep = orStep*in_rows;
      /*Set padding rows and cols*/
      padCudaKernel<T><<<batch, new_in_rows, 0, context->eigen_device<GPUDevice>().stream()>>>
      (tInput, inpt, (size_t)in_depth, (size_t)bStep, (size_t)rStep, (size_t)obStep, (size_t)orStep, (size_t)in_rows, (size_t)paddingOffsetRows, (size_t)paddingOffsetCols);
      paddedInput = transformed_input;
  } /*Padding done*/

  T *padIn = paddedInput.flat<T>().data();
  int inrStep = new_in_cols*in_depth;

  int fcStep = in_depth*out_depth;
  int frStep = filter_cols*fcStep;
  int filSetSize = frStep*filter_rows;

  Tensor result;
  int numJobs = filSetSize*num_indeces;
  OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
              TensorShape({numJobs}), &result));

  T *res = result.flat<T>().data();

  CudaLaunchConfig cfg = GetCudaLaunchConfig(numJobs, context->eigen_device<GPUDevice>());
  initializeArrayCuda<T> <<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>(res, numJobs, 0);

  cfg = GetCudaLaunchConfig(filSetSize, context->eigen_device<GPUDevice>());
  cudaDeviceSynchronize();
  ConvByIndexWeightGradsCudaKernel<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  // (padIn, outBackprop, ind, weightBackprop, res, out_depth, inrStep, in_depth, filSetSize,
  (padIn, outBackprop, ind, res, out_depth, inrStep, in_depth, filSetSize,
    out_rows, out_cols, frStep, fcStep, num_indeces, batch, paddingOffsetRows, paddingOffsetCols);
  cfg = GetCudaLaunchConfig(numJobs, context->eigen_device<GPUDevice>());
  cudaDeviceSynchronize();
  reshapeWeightsCudaKernel<T><<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>
  (res, weightBackprop, out_depth, in_depth, filSetSize, frStep, fcStep, num_indeces, numJobs);
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct ConvByIndexWeightGrads2DFunctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA