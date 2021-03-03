#ifndef OFFSET_CORRECT_H_
#define OFFSET_CORRECT_H_

template <typename Device, typename T>
struct OffsetCorrectFunctor {
  void operator()(const Device& d, const T* input, T* out, const T* offset, int* mask, const size_t dataLength);
};

// #if GOOGLE_CUDA
// // Partially specialize functor for GpuDevice.
// template <typename Eigen::GpuDevice, typename T>
// struct OffsetCorrectFanctor {
//   void operator()(const Device& d, const T* input, T* out, const T* offset, int* mask, const size_t dataLength);
// };
// #endif

#endif //OFFSET_CORRECT_H_