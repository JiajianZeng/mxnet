/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

 /*!
 * Copyright (c) 2017 by Contributors
 * \file positional_convolution-inl.h
 * \brief
 * \ref: https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
 * \author Jiajian Zeng
 */

#ifndef MXNET_OPERATOR_CONTRIB_POSITIONAL_CONVOLUTION_INL_H_
#define MXNET_OPERATOR_CONTRIB_POSITIONAL_CONVOLUTION_INL_H_

#include <mxnet/io.h>
#include <mxnet/base.h>
#include <mxnet/ndarray.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <dmlc/logging.h>
#include <dmlc/optional.h>
#include <algorithm>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../operator_common.h"
#include "../nn/im2col.h"
#include "../linalg.h"

namespace mxnet{
namespace op {

namespace conv{
  enum PositionalConvolutionOpInputs {kData, kScale, kWeight, kBias};
  enum PositionalConvolutionOpOutputs {kOut};
  enum PositionalConvolutionOpResource {kTempSpace};
}

struct PositionalConvolutionParam : public dmlc::Parameter<PositionalConvolutionParam> {
  TShape kernel;
  TShape stride;
  TShape dilate;
  TShape pad;
  uint32_t num_filter;
  uint32_t num_group;
  uint64_t workspace;
  bool no_bias;
  dmlc::optional<int> layout;
  DMLC_DECLARE_PARAMETER(PositionalConvolutionParam) {
    DMLC_DECLARE_FIELD(kernel).describe("Convolution kernel size: (h, w) or (d, h, w)");
    DMLC_DECLARE_FIELD(stride).set_default(TShape())
      .describe("Convolution stride: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(dilate).set_default(TShape())
      .describe("Convolution dilate: (h, w) or (d, h, w). Defaults to 1 for each dimension.");
    DMLC_DECLARE_FIELD(pad).set_default(TShape())
      .describe("Zero pad for convolution: (h, w) or (d, h, w). Defaults to no padding.");
    DMLC_DECLARE_FIELD(num_filter).set_range(1, 100000)
      .describe("Convolution filter(channel) number");
    DMLC_DECLARE_FIELD(num_group).set_default(1)
      .describe("Number of group partitions.");
    DMLC_DECLARE_FIELD(workspace).set_default(1024).set_range(0, 8192)
      .describe("Maximum temporary workspace allowed for convolution (MB).");
    DMLC_DECLARE_FIELD(no_bias).set_default(false)
      .describe("Whether to disable bias parameter.");
    DMLC_DECLARE_FIELD(layout)
      .add_enum("NCW", mshadow::kNCW)
      .add_enum("NCHW", mshadow::kNCHW)
      .add_enum("NCDHW", mshadow::kNCDHW)
      .set_default(dmlc::optional<int>())
      .describe("Set layout for input, output and weight. Empty for\n    "
        "default layout: NCW for 1d, NCHW for 2d and NCDHW for 3d.");
  }
};


template<typename xpu, typename DType>
class PositionalConvolutionOp : public Operator {
 public:
  explicit PositionalConvolutionOp(PositionalConvolutionParam p) {
    this->param_ = p;
    // convert MBytes first to Bytes and then to elements.
    param_.workspace = (param_.workspace << 20) / sizeof(DType);
    CHECK(param_.layout.value() == mshadow::kNCW ||
      param_.layout.value() == mshadow::kNCHW ||
      param_.layout.value() == mshadow::kNCDHW)
      << "Only support NCW, NCHW and NCDHW layout";
  }

  virtual void Forward(const OpContext &ctx,
    const std::vector<TBlob> &in_data,
    const std::vector<OpReqType> &req,
    const std::vector<TBlob> &out_data,
    const std::vector<TBlob> &aux_args) {

  }

  virtual void Backward(const OpContext &ctx,
    const std::vector<TBlob>& out_grad,
    const std::vector<TBlob>& in_data,
    const std::vector<TBlob>& out_data,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& in_grad,
    const std::vector<TBlob>& aux_args) {

  }

 private:
  void LayerSetUp(const TShape& ishape, const TShape& scale_shape, const TShape& oshape) {
    channel_axis_ = 1;  // hard code channel axis
    const index_t first_spatial_axis = channel_axis_ + 1;
    const index_t num_axes = param_.kernel.ndim() + 2;
    num_spatial_axes_ = num_axes - first_spatial_axis;
    is_1x1_ = true;
    for (index_t i = 0; i < param_.kernel.ndim(); ++i) {
      is_1x1_ &= param_.kernel[i] == 1 && param_.stride[i] == 1 && param_.pad[i] == 0;
      if (!is_1x1_) break;
    }

    // batch size
    num_ = ishape[0];
    // number of input channels
    channels_ = ishape[1];
    group_ = param_.num_group;
    conv_out_channels_ = param_.num_filter;
    conv_in_channels_ = channels_;
    bias_term_ = !param_.no_bias;
    kernel_dim_ = conv_in_channels_ / group_ * param_.kernel.Size();
    weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    conv_out_spatial_dim_ = oshape.ProdShape(2, oshape.ndim());
    col_offset_ = kernel_dim_ * conv_out_spatial_dim_;
    output_offset_ = conv_out_channels_ * conv_out_spatial_dim_ / group_;
    // size of the column buffer used for storing im2col-ed pixels
    col_buffer_size_ = kernel_dim_ * group_ * conv_out_spatial_dim_;
    // input/output image size (#channels * height * width)
    input_dim_ = ishape.ProdShape(1, ishape.ndim());
    input_scale_dim_ = scale_shape.ProdShape(1, scale_shape.ndim());
    output_dim_ = oshape.ProdShape(1, oshape.ndim());
    num_kernels_im2col_ = conv_in_channels_ * conv_out_spatial_dim_;
    num_kernels_col2im_ = input_dim_;
  }

 private:
  PositionalConvolutionParam param_;
  index_t channel_axis_;  // channel axis of the input
  index_t channels_;  // number of channels of input image
  index_t num_spatial_axes_;  // number of spatial axes
  index_t num_;  // batch size
  index_t group_;  // number of groups
  index_t conv_out_channels_;  // number of output channels (num_filter)
  index_t conv_out_spatial_dim_;  // number of pixels of output images per channel
  index_t conv_in_channels_;  // number of input channels
  index_t kernel_dim_;  // number of input channels per group * kernel size
  index_t weight_offset_;  // number of output channels per group * kernel_dim_
  index_t col_offset_;
  index_t output_offset_;
  index_t col_buffer_size_;
  index_t input_dim_;
  index_t input_scale_dim_;
  index_t output_dim_;
  index_t num_kernels_im2col_;
  index_t num_kernels_col2im_;
  bool bias_term_;  // has bias term?
  bool is_1x1_;
};  // class PositionalConvolutionOp

template<typename xpu>
Operator* CreateOp(PositionalConvolutionParam param, int dtype,
  std::vector<TShape> *in_shape,
  std::vector<TShape> *out_shape,
  Context ctx);

#if DMLC_USE_CXX11
class PositionalConvolutionProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    if (!param_.no_bias) {
      return{ "data", "scale", "weight", "bias" };
    } else {
      return{ "data", "scale", "weight" };
    }
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
      param_.layout = param_.layout ? param_.layout.value() : mshadow::kNCHW;
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.dilate.ndim() == 0) param_.dilate = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      LOG(FATAL) << "not implemented";
    }
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    if (!param_.no_bias) {
      CHECK_EQ(in_shape->size(), 4U) << "Input:[data, scale, weight, bias]";
    } else {
      CHECK_EQ(in_shape->size(), 3U) << "Input:[data, scale, weight]";
    }
    out_shape->resize(1, TShape());
    const TShape &dshp = (*in_shape)[conv::kData];
    const TShape &sshp = (*in_shape)[conv::kScale];
    if (dshp.ndim() == 0) return false;
    if (param_.kernel.ndim() == 2) {
      // 2d conv
      CHECK_EQ(dshp.ndim(), 4U) \
        << "Input data should be 4D in batch-num_filter-y-x";
      CHECK_EQ(sshp.ndim(), 4U) \
        << "Input scale should be 4D in batch-num_filter-y-x";
      Shape<4> dshape = ConvertLayout(dshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> scaleshape = ConvertLayout(sshp.get<4>(), param_.layout.value(), kNCHW);
      Shape<4> wshape = Shape4(param_.num_filter / param_.num_group, dshape[1] / param_.num_group,
        param_.kernel[0], param_.kernel[1]);
      wshape = ConvertLayout(wshape, kNCHW, param_.layout.value());
      wshape[0] *= param_.num_group;
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kWeight, wshape);
      if (!param_.no_bias) {
        SHAPE_ASSIGN_CHECK(*in_shape, conv::kBias, Shape1(param_.num_filter));
      }

      const index_t ksize_y = static_cast<index_t>(param_.kernel[0]);
      const index_t ksize_x = static_cast<index_t>(param_.kernel[1]);
      CHECK_EQ(dshape[1] % param_.num_group, 0U) \
        << "input num_filter must divide group size";
      CHECK_EQ(param_.num_filter % param_.num_group, 0U) \
        << "output num_filter must divide group size";
      CHECK_GT(param_.kernel.Size(), 0U) \
        << "incorrect kernel size: " << param_.kernel;
      CHECK_GT(param_.stride.Size(), 0U) \
        << "incorrect stride size: " << param_.stride;
      CHECK_GT(param_.dilate.Size(), 0U) \
        << "incorrect dilate size: " << param_.dilate;
      Shape<4> oshape;
      oshape[0] = dshape[0];
      oshape[1] = param_.num_filter;
      oshape[2] = (dshape[2] + 2 * param_.pad[0] -
        (param_.dilate[0] * (ksize_y - 1) + 1)) / param_.stride[0] + 1;
      oshape[3] = (dshape[3] + 2 * param_.pad[1] -
        (param_.dilate[1] * (ksize_x - 1) + 1)) / param_.stride[1] + 1;
      SHAPE_ASSIGN_CHECK(*out_shape, 0, ConvertLayout(oshape, kNCHW, param_.layout.value()));
      CHECK_EQ(oshape[2], scaleshape[2]) \
        << "output height must equal to scale map height";
      CHECK_EQ(oshape[3], scaleshape[3]) \
        << "output width must equal to scale map width";
      CHECK_EQ(scaleshape[1], dshape[1]) \
        << "scale num_filter must equal to input num_channel";
      // Perform incomplete shape inference. Fill in the missing values in data shape.
      // 1) We can always fill in the batch_size.
      // 2) We can back-calculate the input height/width if the corresponding stride is 1.
      oshape = ConvertLayout((*out_shape)[0].get<4>(), param_.layout.value(), kNCHW);
      dshape[0] = oshape[0];
      if (param_.stride[0] == 1) {
        dshape[2] = oshape[2] + param_.dilate[0] * (ksize_y - 1) - 2 * param_.pad[0];
      }
      if (param_.stride[1] == 1) {
        dshape[3] = oshape[3] + param_.dilate[1] * (ksize_x - 1) - 2 * param_.pad[1];
      }
      SHAPE_ASSIGN_CHECK(*in_shape, conv::kData,
        ConvertLayout(dshape, kNCHW, param_.layout.value()));
      // Check whether the kernel sizes are valid
      if (dshape[2] != 0) {
        CHECK_LE(ksize_y, dshape[2] + 2 * param_.pad[0]) << "kernel size exceed input";
      }
      if (dshape[3] != 0) {
        CHECK_LE(ksize_x, dshape[3] + 2 * param_.pad[1]) << "kernel size exceed input";
      }
      return true;
    } else {
      LOG(FATAL) << "not implemented";
      return false;
    }
  }

  bool InferType(std::vector<int> *in_type,
    std::vector<int> *out_type,
    std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        UNIFORM_TYPE_CHECK((*in_type)[i], dtype, ListArguments()[i]);
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new PositionalConvolutionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_PositionalConvolution";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return{ out_grad[conv::kOut], in_data[conv::kData],
            in_data[conv::kScale], in_data[conv::kWeight] };
  }

  std::vector<ResourceRequest> ForwardResource(
    const std::vector<TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }

  std::vector<ResourceRequest> BackwardResource(
    const std::vector<TShape> &in_shape) const override {
    return{ ResourceRequest::kTempSpace };
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
    std::vector<int> *in_type) const override;

 private:
  PositionalConvolutionParam param_;
};  // class PositionalConvolutionProp
#endif  // DMLC_CXX_11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_POSITIONAL_CONVOLUTION_INL_H_