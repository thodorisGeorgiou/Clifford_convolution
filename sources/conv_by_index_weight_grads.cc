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

REGISTER_OP("ConvByIndexWeightGrads2D")
    .Input("input: T")
    .Input("filter: T")
    .Input("out_backprop: T")
    .Input("indexes: int32")
    .Output("output: T")
    .Attr("T: {float}")
    .Attr("strides: list(int)")
    .Attr("use_cudnn_on_gpu: bool = true")
    .Attr(GetPaddingAttrString())
    .Attr(GetConvnetDataFormatAttrString())
    .Attr("dilations: list(int) = [1, 1, 1, 1]")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(1));
      return Status::OK();
    });

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "conv_by_index_weight_grads.h"



void singleWeightGatherGradients(float *w, float *padIn, const float *outBackprop, const int *indeces, int fr, int fc,
    int num_indeces, int filSetSize, int inbStep, int inrStep, int in_depth, int outbStep, int outrStep, int out_depth, int batch,
    int out_rows, int out_cols){
    for(int f = 0; f < num_indeces; f++) *(w + (size_t)(f*filSetSize)) = 0;
    for (int b=0; b < batch; b++){
        int outbIndex = b*outbStep;
        int inbIndex = b*inbStep;
        for(int r = 0; r < out_rows; r++){
            int outrIndex = outbIndex + r * outrStep;
            int inrIndex = inbIndex + (r + fr)*inrStep;
            for(int c=0; c < out_cols; c++){
                int outcIndex = outrIndex + c*out_depth;
                *(w + (size_t)(indeces[outcIndex] * filSetSize)) += outBackprop[outcIndex] * padIn[inrIndex + (c+fc)*in_depth];
            }
        }
    }
}

void copyTo(const float *inpt, float *out, int size){
    for (int i = 0; i < size; i++) out[i] = inpt[i];
}

// CPU specialization of actual computation.
template <typename T>
struct ConvByIndexWeightGrads2DFunctor<CPUDevice, T> {
  void operator()(OpKernelContext *context, const Tensor& input, const Tensor& out_backprop, const Tensor& indexes,
  		  Tensor* weight_backprop, int batch, int out_rows, int out_cols, int out_depth,
  		  int in_depth, int in_rows, int in_cols, int filter_rows, int filter_cols, int num_indeces,
          Padding padding_, TensorFormat data_format_) {

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
            #pragma omp parallel for collapse(2)
            for(int b = 0; b < batch; b++){
                for(int r = 0; r < in_rows; r++){
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
                for(int cbr=(paddingOffsetRows+in_rows)*rStep; cbr < new_in_rows*rStep; cbr++) tbp[cbr] = 0;
                for(int r=paddingOffsetRows*rStep; r < (paddingOffsetRows+in_rows)*rStep; r+=rStep){
                    for(int cb=r; cb<r+in_depth*paddingOffsetCols; cb++) tbp[cb] = 0;
                    for(int cb=r+in_depth*(paddingOffsetCols+in_cols); cb<r+rStep; cb++) tbp[cb] = 0;
                }
            }
            paddedInput = transformed_input;
        } /*Padding done*/

        T *padIn = paddedInput.flat<T>().data();
        int inrStep = new_in_cols*in_depth;
        int inbStep = inrStep*new_in_rows;
        int outrStep = out_cols*out_depth;
        int outbStep = outrStep*out_rows;

        int fcStep = in_depth*out_depth;
        int frStep = filter_cols*fcStep;
        int foStep = frStep*filter_rows;
        int filSetSize = frStep*filter_rows;

        #pragma omp parallel for
        for(int i = 0; i < num_indeces*filSetSize; i++){
            weightBackprop[i] = 0;
        }
        #pragma omp parallel for collapse(4)
        for(int r = 0; r < filter_rows; r++){
            for(int c = 0; c < filter_cols; c++){
                for(int din = 0; din < in_depth; din++){
                    for(int dout = 0; dout < out_depth; dout++){
                    	float *w = weightBackprop + (size_t)(r*frStep + c*fcStep + din*out_depth + dout);
                        singleWeightGatherGradients(w, padIn+(size_t)din, outBackprop+(size_t)dout, ind+(size_t)dout, r, c, num_indeces, filSetSize, 
                            inbStep, inrStep, in_depth, outbStep, outrStep, out_depth, batch,
                            out_rows, out_cols);
                    }
                }
            }      
        }
  }
};


template <typename Device, typename T>
class ConvByIndexWeightGrads2DOp : public OpKernel {
 public:
  explicit ConvByIndexWeightGrads2DOp(OpKernelConstruction* context) : OpKernel(context) {
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

    const Tensor& input = context->input(0);
    const Tensor& filter = context->input(1);
    const Tensor& out_backprop = context->input(2);
    const Tensor& indexes = context->input(3);
    TensorShape filter_shape = filter.shape();

    int batch = static_cast<int>(indexes.dim_size(0));
    int out_rows = static_cast<int>(indexes.dim_size(1));
    int out_cols = static_cast<int>(indexes.dim_size(2));
    int out_depth = static_cast<int>(indexes.dim_size(3));

    int num_indeces = static_cast<int>(filter.dim_size(0));
    int filter_rows = static_cast<int>(filter.dim_size(1));
    int filter_cols = static_cast<int>(filter.dim_size(2));

    Tensor* weight_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, filter_shape, &weight_backprop));

    int in_rows = static_cast<int>(input.dim_size(1));
    int in_cols = static_cast<int>(input.dim_size(2));
    int in_depth = static_cast<int>(input.dim_size(3));

    // If there is nothing to compute, return.
    if (filter_shape.num_elements() == 0) {
      return;
    }

	ConvByIndexWeightGrads2DFunctor<Device, T>()(
	       context, input, out_backprop, indexes, weight_backprop,
	       batch, out_rows, out_cols, out_depth,
	       in_depth, in_rows, in_cols,
	       filter_rows, filter_cols, num_indeces,
	       padding_, data_format_);

  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConvByIndexWeightGrads2DOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvByIndexWeightGrads2D").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      ConvByIndexWeightGrads2DOp<CPUDevice, T>);
REGISTER_CPU(float);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvByIndexWeightGrads2D").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      ConvByIndexWeightGrads2DOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif  // GOOGLE_CUDA
