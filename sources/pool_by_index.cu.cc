#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "pool_by_index.h"

// Define the CUDA kernel.
template <typename T>
__global__ void PoolByIndexCudaKernel(const T* input, T* out, const int* indexes, const size_t indLength, const size_t inbStep, const size_t outbStep) {
  size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
  size_t num_threads = (size_t)(blockDim.x*gridDim.x);
  for(size_t p=threadID; p<indLength;p+=num_threads){
      size_t b = p/outbStep;
      out[p] = *(input + b*inbStep + (size_t)indexes[p]);
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct PoolByIndexFunctor<GPUDevice, T>{
  void operator()(const GPUDevice& d, const T* input, T* out, const int* indexes, const size_t indLength, const size_t inbStep, const size_t outbStep) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  CudaLaunchConfig cfg = GetCudaLaunchConfig((int)indLength, d);
  PoolByIndexCudaKernel<T> <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(input, out, indexes, indLength, inbStep, outbStep);
}
};
// Explicitly instantiate functors for the types of OpKernels registered.
template struct PoolByIndexFunctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA