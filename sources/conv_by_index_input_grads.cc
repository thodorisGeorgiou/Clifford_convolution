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

REGISTER_OP("ConvByIndexInputGrads2D")
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
      c->set_output(0, c->input(0));
      return Status::OK();
    });
  //   .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  //   	::tensorflow::shape_inference::ShapeHandle s;
		// TF_RETURN_IF_ERROR(c->MakeShapeFromShapeTensor(0, &s));
		// TF_RETURN_IF_ERROR(c->WithRank(s, 4, &s));
		// c->set_output(0, s);
		// return Status::OK();
  //   });


using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

#include "conv_by_index_input_grads.h"

void singlePointGatherGradients(const float *outBp, const int *indexes, const float *filters, float *inBackprop, int out_depth,
    int filter_rows, int filter_cols, int fcStep, int frStep, int foStep, int orStep){
    *inBackprop = 0;
    for (int r=0; r < filter_rows; r++){
        int frp = (filter_rows-1-r)*frStep;
        size_t out = (size_t)(r*orStep);
        const float *orp = outBp+out;
        const int *pInds = indexes+out;
        for(int c=filter_cols-1; c > -1; c--){
            int fcp = frp + c*fcStep;
            for(int d=0; d < out_depth; d++)
            	*inBackprop += (*orp++)*filters[(*pInds++)*foStep + (fcp++)];
        }
    }
}
            // const int *pInds = irp + (size_t)(c*out_depth);
            // float *ocp = outBp+(size_t)(orp + (c-filter_cols/2)*out_depth);

void copyTo(const float *inpt, float *out, int size){
    for (int i = 0; i < size; i++) out[i] = inpt[i];
}

void copyToInt(const int *inpt, int *out, int size){
    for (int i = 0; i < size; i++) out[i] = inpt[i];
}

// CPU specialization of actual computation.
template <typename T>
struct ConvByIndexInputGrads2DFunctor<CPUDevice, T> {
  void operator()(OpKernelContext *context, const Tensor& filter, const Tensor& out_backprop, const Tensor& indexes,
  		  Tensor* in_backprop, int batch, int out_rows, int out_cols, int out_depth,
  		  int in_depth, int in_rows, int in_cols, int filter_rows, int filter_cols,
          Padding padding_, TensorFormat data_format_) {

        const T *outBackprop = out_backprop.flat<T>().data();
        T *inBackprop = in_backprop->flat<T>().data();
        const T *fil = filter.flat<T>().data();
        const int *ind = indexes.flat<int32>().data();

        int padding_rows = 0;
        int padding_cols = 0;
        Tensor paddedOutBackprop = out_backprop;
        Tensor paddedIndexes = indexes;
        int new_out_rows = out_rows;
        int new_out_cols = out_cols;
        int paddingOffsetRows = 0;
        int paddingOffsetCols = 0;

        if (padding_ == SAME) {
            padding_rows = std::max<int>(0, (in_rows - 1) + filter_rows - out_rows);
            padding_cols = std::max<int>(0, (in_cols - 1) + filter_cols - out_cols);

            Tensor transformed_outBackprop;
            Tensor transformed_indexes;
            new_out_rows = out_rows + padding_rows;
            new_out_cols = out_cols + padding_cols;
            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::value,
                        ShapeFromFormat(data_format_, batch, new_out_rows,
                        new_out_cols, out_depth), &transformed_outBackprop));

            OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<int32>::value,
                        ShapeFromFormat(data_format_, batch, new_out_rows,
                        new_out_cols, out_depth), &transformed_indexes));

            paddingOffsetRows = padding_rows/2;
            paddingOffsetCols = padding_cols/2;
            T *tOutBackprop = transformed_outBackprop.flat<T>().data();
            int *tIndexes = transformed_indexes.flat<int32>().data();
            int rStep = new_out_cols*out_depth;
            int bStep = rStep*new_out_rows;

            int orStep = out_cols*out_depth;
            int obStep = orStep*out_rows;
            /*Set padding rows and cols*/
            #pragma omp parallel for collapse(2)
            for(int b = 0; b < batch; b++){
                for(int r = 0; r < out_rows; r++){
                	size_t rowIndex = (size_t)(b*bStep + (r+paddingOffsetRows)*rStep + paddingOffsetCols*out_depth);
                    T *tp = tOutBackprop + rowIndex;
                    int *tip = tIndexes + rowIndex;
                    size_t orRowIndex = (size_t)(b*obStep + r*orStep);
                    const T *op = outBackprop + orRowIndex;
                    copyTo(op, tp, orStep);
                    const int *ip = ind + orRowIndex;
                    copyToInt(ip, tip, orStep);
                }
            }
            #pragma omp parallel for
            for(int b = 0; b < batch; b++){
            	size_t batchIndex = (size_t)(b*bStep);
                float *tbp = tOutBackprop + batchIndex;
                int *tibp = tIndexes + batchIndex;
                for(int cbr=0; cbr < paddingOffsetRows*rStep; cbr++){
                	tbp[cbr] = 0;
                	tibp[cbr] = 0;
                }
                for(int cbr=(paddingOffsetRows+out_rows)*rStep; cbr < new_out_rows*rStep; cbr++){
                	tbp[cbr] = 0;
                	tibp[cbr] = 0;
                }
                for(int r=paddingOffsetRows*rStep; r < (paddingOffsetRows+out_rows)*rStep; r+=rStep){
                    for(int cb=r; cb<r+out_depth*paddingOffsetCols; cb++){
                    	tbp[cb] = 0;
                    	tibp[cb] = 0;
                    }
                    for(int cb=r+out_depth*(paddingOffsetCols+out_cols); cb<r+rStep; cb++){
                    	tbp[cb] = 0;
                    	tibp[cb] = 0;
                    }
                }
            }
            paddedOutBackprop = transformed_outBackprop;
            paddedIndexes = transformed_indexes;
        } /*Padding done*/

        T *padOutBp = paddedOutBackprop.flat<T>().data();
        int *padInds = paddedIndexes.flat<int32>().data();
        int inrStep = in_cols*in_depth;
        int inbStep = inrStep*in_rows;
        int outrStep = new_out_cols*out_depth;
        int outbStep = outrStep*new_out_rows;

        int fcStep = in_depth*out_depth;
        int frStep = filter_cols*fcStep;
        int foStep = frStep*filter_rows;
        int filSetSize = frStep*filter_rows;

        #pragma omp parallel for collapse(4)
        for(int n = 0; n < batch; n++){
            for(int r = 0; r < in_rows; r++){
                for(int c = 0; c < in_cols; c++){
                    for(int d = 0; d < in_depth; d++){
                    	size_t outIndex = (size_t)(n*outbStep + r*outrStep + c*out_depth);
                        float *outBp = padOutBp + outIndex;
                        int *indexPp = padInds + outIndex;
                        size_t inIndex = (size_t)(n*inbStep + r*inrStep + c*in_depth + d);
                        singlePointGatherGradients(outBp, indexPp, fil+(size_t)(d*out_depth), inBackprop+inIndex, out_depth, filter_rows, filter_cols, fcStep, frStep, filSetSize, outrStep);
                    }
                }
            }      
        }
  }
};


template <typename Device, typename T>
class ConvByIndexInputGrads2DOp : public OpKernel {
 public:
  explicit ConvByIndexInputGrads2DOp(OpKernelConstruction* context) : OpKernel(context) {
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
    // OP_REQUIRES(
    //     context, TensorShapeUtils::IsVector(input_sizes.shape()),
    //     errors::InvalidArgument(
    //         "Conv2DBackpropInput: input_sizes input must be 1-dim, not ",
    //         input_sizes.dims()));
    // TensorShape input_shape;
    TensorShape input_shape = input.shape();
    // OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(
    //                             input_sizes.vec<int32>(), &input_shape));
    // OP_REQUIRES_OK(context, MakeShape(input_sizes, &input_shape));

    int batch = static_cast<int>(indexes.dim_size(0));
    int out_rows = static_cast<int>(indexes.dim_size(1));
    int out_cols = static_cast<int>(indexes.dim_size(2));
    int out_depth = static_cast<int>(indexes.dim_size(3));

    int filter_rows = static_cast<int>(filter.dim_size(1));
    int filter_cols = static_cast<int>(filter.dim_size(2));

    Tensor* in_backprop = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input_shape, &in_backprop));

    int in_rows = static_cast<int>(in_backprop->dim_size(1));
    int in_cols = static_cast<int>(in_backprop->dim_size(2));
    int in_depth = static_cast<int>(in_backprop->dim_size(3));

    // If there is nothing to compute, return.
    if (input_shape.num_elements() == 0) {
      return;
    }

	ConvByIndexInputGrads2DFunctor<Device, T>()(
	       context, filter, out_backprop, indexes, in_backprop,
	       batch, out_rows, out_cols, out_depth,
	       in_depth, in_rows, in_cols,
	       filter_rows, filter_cols,
	       padding_, data_format_);

  }

 private:
  std::vector<int32> dilations_;
  std::vector<int32> strides_;
  bool use_cudnn_;
  Padding padding_;
  TensorFormat data_format_;
  // LaunchConv2DOp<Device, T> launcher_;
  bool cudnn_use_autotune_;

  TF_DISALLOW_COPY_AND_ASSIGN(ConvByIndexInputGrads2DOp);

};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvByIndexInputGrads2D").Device(DEVICE_CPU).TypeConstraint<float>("T"), \
      ConvByIndexInputGrads2DOp<CPUDevice, T>);
REGISTER_CPU(float);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("ConvByIndexInputGrads2D").Device(DEVICE_GPU).TypeConstraint<float>("T"), \
      ConvByIndexInputGrads2DOp<GPUDevice, T>);
REGISTER_GPU(float);

#endif  // GOOGLE_CUDA
