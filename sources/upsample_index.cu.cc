#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "upsample_index.h"

template <typename K>
__global__ void initializeArrayCuda(K *out, int size, K val){
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x*gridDim.x;
    for (int i = threadID; i < size; i+=num_threads) out[i] = val;
}

// Define the CUDA kernel.
template <typename T>
__global__ void UpsampleIndexCudaKernel(const T* input, T* out, const int* indexes, const size_t indLength, const size_t inbStep, const size_t outbStep) {
  size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
  size_t num_threads = (size_t)(blockDim.x*gridDim.x);
  for(size_t p=threadID; p<indLength;p+=num_threads){
      size_t b = p/inbStep;
      *(out + b*outbStep + indexes[p]) = input[p];
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct UpsampleIndexFunctor<GPUDevice, T>{
  void operator()(const GPUDevice& d, const T* input, T* out, const int* indexes, const size_t indLength, const size_t outLength, const size_t inbStep, const size_t outbStep) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  CudaLaunchConfig cfg = GetCudaLaunchConfig((int)outLength, d);
  initializeArrayCuda<T> <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(out, outLength, 0);
  cfg = GetCudaLaunchConfig((int)indLength, d);
  cudaDeviceSynchronize();
  UpsampleIndexCudaKernel<T> <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(input, out, indexes, indLength, inbStep, outbStep);
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct UpsampleIndexFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA