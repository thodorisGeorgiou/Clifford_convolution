#ifndef GATHER_ANGLES_H_
#define GATHER_ANGLES_H_

template <typename Device, typename T>
struct GatherAnglesFunctor {
  void operator()(const Device& d, const T* angs, const int* indexes, T* out, const size_t indLength);
};

#endif //GATHER_ANGLES_H_