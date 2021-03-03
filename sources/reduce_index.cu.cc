#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "reduce_index.h"

// Define the CUDA kernel.
template <typename T>
__global__ void ReduceIndexCudaKernel(const T* input, T* out, const int* indexes, const size_t maxInd, const size_t indLength) {
  size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
  size_t num_threads = (size_t)(blockDim.x*gridDim.x);
  for(size_t p=threadID; p<indLength;p+=num_threads)
      out[p] = input[p*maxInd + indexes[p]];
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct ReduceIndexFunctor<GPUDevice, T>{
  void operator()(const GPUDevice& d, const T* input, T* out, const int* indexes, const size_t maxInd, const size_t indLength) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  CudaLaunchConfig cfg = GetCudaLaunchConfig((int)indLength, d);
  ReduceIndexCudaKernel<T> <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(input, out, indexes, maxInd, indLength);
}
};
// Explicitly instantiate functors for the types of OpKernels registered.
template struct ReduceIndexFunctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA