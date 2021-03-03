#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "calculate_proportions.h"

// Define the CUDA kernel.
template <typename T>
__global__ void CalculateProportionsCudaKernel(const T* input, T* out, T *sums, const int* indexes, const size_t indLength) {
  size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
  size_t num_threads = (size_t)(blockDim.x*gridDim.x);
  for(size_t p=threadID; p<indLength;p+=num_threads)
    out[p] = input[p]/sums[indexes[p]];
}

template <typename T>
__global__ void calculateMaxCudaKernel(T *sums, const T *in, const int *ind, const size_t indLength){
  int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  T *s = sums + (size_t)threadID;
  *s = 0;
  const int *end = ind + indLength;
  for(;ind<end;){
      if(*ind==threadID) *s += *in;
      in++;
      ind++;
  }
  if(*s==0) *s = 1;
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct CalculateProportionsFunctor<GPUDevice, T>{
  void operator()(OpKernelContext *context, const T* input, T* out, const int* indexes, T *sums, const int maxInd, const size_t indLength) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.

  calculateMaxCudaKernel<T> <<<1, maxInd, 0, context->eigen_device<GPUDevice>().stream()>>>(sums, input, indexes, indLength);
  CudaLaunchConfig cfg = GetCudaLaunchConfig((int)indLength, context->eigen_device<GPUDevice>());
  cudaDeviceSynchronize();
  CalculateProportionsCudaKernel<T> <<<cfg.block_count, cfg.thread_per_block, 0, context->eigen_device<GPUDevice>().stream()>>>(input, out, sums, indexes, indLength);
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CalculateProportionsFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA