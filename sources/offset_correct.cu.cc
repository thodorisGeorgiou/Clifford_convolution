#ifdef GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

using GPUDevice = Eigen::GpuDevice;

#include "offset_correct.h"

// Define the CUDA kernel.
template <typename T>
__global__ void OffsetCorrectCudaKernel(const T* input, T* out, const T *offset, int* mask, const size_t dataLength) {
    T ofst = *offset;
    size_t threadID = (size_t)(blockIdx.x * blockDim.x + threadIdx.x);
    size_t num_threads = (size_t)(blockDim.x*gridDim.x);
    for(size_t p = threadID; p<dataLength; p+=num_threads){
        if(fabsf(input[p])<ofst){
            mask[p] = 1;
            out[p] = input[p];
        }
        else{
            mask[p] = 0;
            out[p] = 0;
        }
    }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct OffsetCorrectFunctor<GPUDevice, T>{
  void operator()(const GPUDevice& d, const T* input, T* out, const T *offset, int* mask, const size_t dataLength) {
  // Launch the cuda kernel.
  //
  // See core/util/cuda_kernel_helper.h for example of computing
  // block count and thread_per_block count.
  CudaLaunchConfig cfg = GetCudaLaunchConfig((int)dataLength, d);
  // printf("block count: %d\nThread per block: %d\nNum jobs: %d\n", (int)cfg.block_count, (int)cfg.thread_per_block, (int)dataLength);
  OffsetCorrectCudaKernel<T> <<<cfg.block_count, cfg.thread_per_block, 0, d.stream()>>>(input, out, offset, mask, dataLength);
}
};
// Explicitly instantiate functors for the types of OpKernels registered.
template struct OffsetCorrectFunctor<GPUDevice, float>;
// template struct ExampleFunctor<GPUDevice, int32>;

#endif  // GOOGLE_CUDA