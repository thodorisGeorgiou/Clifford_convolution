#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"
// #include "tensorflow/core/util/gpu_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "gather_angles.h"

// Define the CUDA kernel.
template <typename T>
__global__ void GatherAnglesCudaKernel(const T* angs, const int* indexes, T* out, const size_t indLength) {
  size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
  size_t num_threads = (size_t)(blockDim.x*gridDim.x);
  for(size_t p=threadID; p<indLength; p+=num_threads)
    out[p] = angs[indexes[p]];
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct GatherAnglesFunctor<GPUDevice, T>{
  void operator()(const GPUDevice& d, const T* angs, const int* indexes, T* out, const size_t indLength){
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  CudaLaunchConfig cfg = GetCudaLaunchConfig((int)indLength, d);
  GatherAnglesCudaKernel<<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(angs, indexes, out, indLength);
}
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct GatherAnglesFunctor<GPUDevice, float>;

#endif  // GOOGLE_CUDA