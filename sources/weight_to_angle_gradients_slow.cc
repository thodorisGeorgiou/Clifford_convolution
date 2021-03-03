#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/bounds_check.h"
#include "tensorflow/core/util/use_cudnn.h"

#include <iostream>
#include <omp.h>

using namespace tensorflow;

REGISTER_OP("WeightToAngleGradients")
    .Input("input: T")
    .Input("gradients: T")
    .Input("weights: T")
    .Input("indexes: int32")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(3));
      return Status::OK();
    });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "weight_to_angle_gradients.h"

template <typename T>
void copyTo(const T *inpt, T *out, int size){
    for (int i = 0; i < size; i++) out[i] = inpt[i];
}

template <typename T>
void singlePointGatherAngleGradients(T *padIn, T *output, const T gradient, T *woa, const T *weights, int inbStep, int inrStep, int in_depth, \
  int outbStep, int outrStep, int out_depth, int filter_rows, int filter_cols, int fcStep, int frStep, int weightSetSize, int inIndex, int outIndex){
    *output = 0;
    for(int r=0; r<filter_rows; r++){
      int inrInd = r*inrStep;
      int frInd = r*frStep;
      for(int c=0; c<filter_cols; c++){
        int incInd = inrInd + c*in_depth;
        int fcInd = frInd + c*fcStep;
        for(int din=0; din<in_depth;din+=2){
          size_t inOffset = (size_t)(incInd + din);
          T EoverW = *(padIn+inOffset)*gradient;
          int fInd = fcInd + din*out_depth;
          *output += EoverW * (woa[fInd] - weights[fInd+out_depth]);
        }
        for(int din=1; din<in_depth;din+=2){
          size_t inOffset = (size_t)(incInd + din);
          T EoverW = *(padIn+inOffset)*gradient;
          int fInd = fcInd + din*out_depth;
          *output += EoverW * (woa[fInd] + weights[fInd-out_depth]);
        }
      }
    }

}


// CPU specialization of actual computation.
template <typename T>
struct CalculateWeightToAngleGradientsFunctor<CPUDevice, T> {
  void operator()(OpKernelContext *context, const Tensor& input, T* output, const int* indexes, const T *weights, const T* gradients, T *woa, \
    const int filter_rows, const int filter_cols, const int num_angles, const int in_depth, const int out_depth, \
    const int batch, const int rows, const int cols,Padding padding_,TensorFormat data_format_) {

        const T *inpt = input.flat<T>().data();

        int fcStep = out_depth*in_depth;
        int frStep = filter_cols*fcStep;
        int weightSetSize = filter_rows*frStep;
        float normFactor = M_PI*4/(float)(num_angles-1);
        #pragma omp parallel for collapse(5)
        for(int f=0; f<num_angles-1; f++){
            for(int r=0; r<filter_rows; r++){
                for(int c=0; c<filter_cols; c++){
                    for(int din=0;din<in_depth;din++){
                        for(int dout=0; dout<out_depth; dout++){
                            int weightIndex = f*weightSetSize + r*frStep + c*fcStep + din*out_depth + dout;
                            int correction = -1;
                            if(f==0) correction = num_angles-2;
                            woa[weightIndex] = (weights[weightIndex+weightSetSize]-weights[weightIndex+correction*weightSetSize])/normFactor;
                        }
                    }
                }
            }
        }
        copyTo(woa, woa+(size_t)((num_angles-1)*weightSetSize), weightSetSize);
        int outrStep = out_depth*cols;
        int outbStep = rows*outrStep;

        int padding_rows = 0;
        int padding_cols = 0;
        Tensor paddedInput = input;
        int new_in_rows = rows;
        int new_in_cols = cols;
        int paddingOffsetRows = 0;
        int paddingOffsetCols = 0;

        if (padding_ == SAME) {
            padding_rows = std::max<int>(0, (rows - 1) + filter_rows - rows);
            padding_cols = std::max<int>(0, (cols - 1) + filter_cols - cols);

            Tensor transformed_input;
            new_in_rows = rows + padding_rows;
            new_in_cols = cols + padding_cols;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                        ShapeFromFormat(data_format_, batch, new_in_rows,
                        new_in_cols, in_depth), &transformed_input));

            paddingOffsetRows = padding_rows/2;
            paddingOffsetCols = padding_cols/2;
            T *tInput = transformed_input.flat<T>().data();
            int rStep = new_in_cols*in_depth;
            int bStep = rStep*new_in_rows;

            int orStep = cols*in_depth;
            int obStep = orStep*rows;
            /*Set padding rows and cols*/
            #pragma omp parallel for collapse(2)
            for(int b = 0; b < batch; b++){
                for(int r = 0; r < rows; r++){
                  size_t rowIndex = (size_t)(b*bStep + (r+paddingOffsetRows)*rStep + paddingOffsetCols*in_depth);
                    T *tp = tInput + rowIndex;
                    size_t orRowIndex = (size_t)(b*obStep + r*orStep);
                    const T *op = inpt + orRowIndex;
                    copyTo(op, tp, orStep);
                }
            }
            #pragma omp parallel for
            for(int b = 0; b < batch; b++){
              size_t batchIndex = (size_t)(b*bStep);
                float *tbp = tInput + batchIndex;
                for(int cbr=0; cbr < paddingOffsetRows*rStep; cbr++) tbp[cbr] = 0;
                for(int cbr=(paddingOffsetRows+rows)*rStep; cbr < new_in_rows*rStep; cbr++) tbp[cbr] = 0;
                for(int r=paddingOffsetRows*rStep; r < (paddingOffsetRows+rows)*rStep; r+=rStep){
                    for(int cb=r; cb<r+in_depth*paddingOffsetCols; cb++) tbp[cb] = 0;
                    for(int cb=r+in_depth*(paddingOffsetCols+cols); cb<r+rStep; cb++) tbp[cb] = 0;
                }
            }
            paddedInput = transformed_input;
        } /*Padding done*/

        T *padIn = paddedInput.flat<T>().data();
        int inrStep = in_depth*new_in_cols;
        int inbStep = inrStep*new_in_rows;

        #pragma omp parallel for collapse(4)
        for(int b=0; b<batch; b++){
          for(int r=0; r<rows; r++){
            for(int c=0; c<cols; c++){
              for(int dout=0; dout<out_depth;dout++){
                int inIndex = b*inbStep + r*inrStep + c*in_depth;
                int outIndex = b*outbStep + r*outrStep + c*out_depth + dout;
                singlePointGatherAngleGradients(padIn+(size_t)inIndex, output+(size_t)outIndex, gradients[outIndex], \
                  woa+(size_t)(weightSetSize*indexes[outIndex] + dout), weights+(size_t)(weightSetSize*indexes[outIndex] + dout), inbStep, inrStep, in_depth, outbStep, outrStep,\
                  out_depth, filter_rows, filter_cols, fcStep, frStep, weightSetSize, inIndex, outIndex);
              }
            }
          }
        }

  }
};


template <typename Device, typename T>
class WeightToAngleGradientsOp : public OpKernel {
 public:
  explicit WeightToAngleGradientsOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("dilations", &dilations_));
    OP_REQUIRES_OK(context, context->GetAttr("strides", &strides_));
    string data_format;
    OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
    OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
                errors::InvalidArgument("Invalid data format"));
    OP_REQUIRES_OK(context, context->GetAttr("use_cudnn_on_gpu", &use_cudnn_));
    use_cudnn_ &= CanUseCudnn();
    cudnn_use_autotune_ = CudnnUseAutotune();
    OP_REQUIRES(context, dilations_.size() == 4,
                errors::InvalidArgument("Sliding window dilations field must "
                                        "specify 4 dimensions"));
    OP_REQUIRES(context, strides_.size() == 4,
                errors::InvalidArgument("Sliding window strides field must "
                                        "specify 4 dimensions"));
    const int64 stride_n = strides_[0];
    const int64 stride_c = strides_[3];
    const int64 stride_h = strides_[1];
    const int64 stride_w = strides_[2];
    OP_REQUIRES(
        context, stride_n == 1 && stride_c == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the batch and depth dimensions."));
    OP_REQUIRES(
        context, stride_h == 1 && stride_w == 1,
        errors::InvalidArgument("Current implementation does not yet support "
                                "strides in the row and column dimensions."));

    const int64 dilation_n = dilations_[0];
    const int64 dilation_c = dilations_[3];
    const int64 dilation_h = dilations_[1];
    const int64 dilation_w = dilations_[2];
    OP_REQUIRES(context, dilation_n == 1 && dilation_c == 1,
                errors::InvalidArgument(
                    "Current implementation does not yet support "
                    "dilations in the batch and depth dimensions."));
    OP_REQUIRES(
        context, dilation_h == 1 && dilation_w == 1,
        errors::InvalidArgument("Current implementation does not support dilations"));
    OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
  }


  void Compute(OpKernelContext* context) override {
    // Input tensor is of the following dimensions:
    // [ batch, in_rows, in_cols, in_depth ]

    const Tensor& input = context->input(0);
    const Tensor& gradients = context->input(1);
    const Tensor& weights = context->input(2);
    const Tensor& indexes = context->input(3);

    // For 2D convolution, there should be 4 dimensions.
    OP_REQUIRES(context, gradients.dims() == indexes.dims(),
                errors::InvalidArgument("proportions must the same dimensions with indexes",
                                        gradients.shape().DebugString()));

    for(int i = 0; i < indexes.dims(); i++) {
      OP_REQUIRES(
          context,
          indexes.dim_size(i) == gradients.dim_size(i),
          errors::InvalidArgument("proportions and indexes must have corresponding dimensions"));
    }

    OP_REQUIRES(
      context,
      weights.dim_size(4) == indexes.dim_size(3),
      errors::InvalidArgument("weights last dim must be equal to output depth"));

    OP_REQUIRES(
      context,
      weights.dim_size(3) == input.dim_size(3),
      errors::InvalidArgument("weights last dim must be equal to output depth"));

    for(int i = 0; i < input.dims()-1; i++) {
      OP_REQUIRES(
        context,
        input.dim_size(i) == indexes.dim_size(i),
        errors::InvalidArgument("weights last dim must be equal to output depth"));
    }
  
    const int batch = static_cast<int>(indexes.dim_size(0));
    const int rows = static_cast<int>(indexes.dim_size(1));
    const int cols = static_cast<int>(indexes.dim_size(2));
    const int out_depth = static_cast<int>(indexes.dim_size(3));

    const int num_angles = static_cast<int>(weights.dim_size(0));
    const int filter_rows = static_cast<int>(weights.dim_size(1));
    const int filter_cols = static_cast<int>(weights.dim_size(2));
    const int in_depth = static_cast<int>(weights.dim_size(3));

    Tensor weightOverAngle;
    OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value, weights.shape(), &weightOverAngle));
    T *woa = weightOverAngle.flat<T>().data();
    
    TensorShape out_shape = indexes.shape();
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

    // If there is nothing to compute, return.
    if (out_shape.num_elements() == 0) {
      return;
    }

    // const T *in = input.flat<T>().data();
    T *out = output->flat<T>().data();
    const int *ind = indexes.flat<int32>().data();
    const T *w = weights.flat<T>().data();
    const T *g = gradients.flat<T>().data();

    CalculateWeightToAngleGradientsFunctor<Device, T>()(context, input, out, ind, w, g, woa, \
      filter_rows, filter_cols, num_angles, in_depth, out_depth, batch, rows, cols, padding_, data_format_);

  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  // LaunchConv2DOp<Device, T> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(WeightToAngleGradientsOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("WeightToAngleGradients").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      WeightToAngleGradientsOp<CPUDevice, T>);
REGISTER_CPU(float);
// REGISTER_CPU(int32);

//Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("WeightToAngleGradients").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      WeightToAngleGradientsOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif  // GOOGLE_CUDA
