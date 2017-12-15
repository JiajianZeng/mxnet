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
 * \file positional_pooling-inl.h
 * \brief positional pooling operator and symbol
 * \author Jiajian Zeng
 */

#ifndef MXNET_OPERATOR_CONTRIB_POSITIONAL_POOLING_INL_H_
#define MXNET_OPERATOR_CONTRIB_POSITIONAL_POOLING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "../mshadow_op.h"
#include "../operator_common.h"

namespace mxnet{
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace ppool {
enum PositionalPoolingOpInputs {kData, kMap};
enum PositionalPoolingOpOutputs {kOut};
enum PositionalPoolingOpType {kNormal, kProd};
enum PositionalPoolingOpPadConventionType {kValid, kFull};
}  // ppool

struct PositionalPoolingParam : public dmlc::Parameter<PositionalPoolingParam> {
  TShape kernel;
  TShape stride;
  TShape pad;
  int pooling_convention;
  int pool_type;
  DMLC_DECLARE_PARAMETER(PositionalPoolingParam) {
    DMLC_DECLARE_FIELD(kernel)
    .enforce_nonzero()
    .describe("Pooling kernel size: (y, x)");

    DMLC_DECLARE_FIELD(pooling_convention).set_default(ppool::kValid)
    .add_enum("full", ppool::kFull)
    .add_enum("valid", ppool::kValid)
    .describe("Pooling convention to be applied.");

    DMLC_DECLARE_FIELD(pool_type).set_default(ppool::kNormal)
    .add_enum("normal", ppool::kNormal)
    .add_enum("prod", ppool::kProd)
    .describe("Pooling type to be applied.");

    DMLC_DECLARE_FIELD(stride).set_default(TShape())
    .enforce_nonzero()
    .describe("Stride for pooling: (y, x). Defaults to 1 for each dimension.");

    DMLC_DECLARE_FIELD(pad).set_default(TShape())
    .describe("Pad for pooling: (y, x). Defaults to no padding.");
  }
};

template<typename xpu, typename DType>
class PositionalPoolingOp : public Operator {
 public:
  explicit PositionalPoolingOp(PositionalPoolingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext& ctx,
                       const std::vector<TBlob>& in_data,
                       const std::vector<OpReqType>& req,
                       const std::vector<TBlob>& out_data,
                       const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 1U);
    Stream<xpu>* s = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> data = in_data[ppool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> map = in_data[ppool::kMap].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out = out_data[ppool::kOut].get<xpu, 4, DType>(s);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(map.CheckContiguous(), true);
    CHECK_EQ(out.CheckContiguous(), true);
    PositionalPoolForward(s, out, data, map, param_.kernel, param_.pad, param_.stride, param_.pool_type);
  }

  virtual void Backward(const OpContext& ctx,
                        const std::vector<TBlob>& out_grad,
                        const std::vector<TBlob>& in_data,
                        const std::vector<TBlob>& out_data,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& in_grad,
                        const std::vector<TBlob>& aux_args) {
    using namespace mshadow;
    CHECK_EQ(out_grad.size(), 1U);
    CHECK_EQ(in_data.size(), 2U);
    CHECK_EQ(out_data.size(), 1U);
    CHECK_EQ(req.size(), 2U);
    CHECK_EQ(in_grad.size(), 2U);
    Stream<xpu>*s  = ctx.get_stream<xpu>();

    Tensor<xpu, 4, DType> in_data_grad = in_grad[ppool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> out_data_grad = out_grad[ppool::kOut].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> in_map_grad = in_grad[ppool::kMap].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> data = in_data[ppool::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 4, DType> map = in_data[ppool::kMap].get<xpu, 4, DType>(s);
    CHECK_EQ(in_data_grad.CheckContiguous(), true);
    CHECK_EQ(out_data_grad.CheckContiguous(), true);
    CHECK_EQ(in_map_grad.CheckContiguous(), true);
    CHECK_EQ(data.CheckContiguous(), true);
    CHECK_EQ(map.CheckContiguous(), true);
    PositionalPoolBackward(s, in_data_grad, out_data_grad, in_map_grad, data, map,
                           param_.kernel, param_.pad, param_.stride, param_.pool_type);
  }
 private:
  PositionalPoolingParam param_;
};  // class PositionalPoolingOP

// Declare Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(PositionalPoolingParam param, int dtype);

#if DMLC_USE_CXX11
class PositionalPoolingProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "map"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    using namespace mshadow;
    param_.Init(kwargs);
    if (param_.kernel.ndim() == 2) {
      if (param_.stride.ndim() == 0) param_.stride = Shape2(1, 1);
      if (param_.pad.ndim() == 0) param_.pad = Shape2(0, 0);
    } else {
      LOG(FATAL) << "not implemented";
    }

    CHECK_EQ(param_.kernel.ndim(), param_.stride.ndim())
      << "kernel and stride should have the same dimension (2D)";
    CHECK_EQ(param_.kernel.ndim(), param_.pad.ndim())
      << "kernel and pad should have the same dimension (2D)";
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    CHECK_EQ(in_shape->size(), 2U) << "Input: [data, map]";
    const TShape &dshape = in_shape->at(ppool::kData);
    const TShape &mshape = in_shape->at(ppool::kMap);

    CHECK_EQ(dshape.ndim(), 4U) << "Pooling: Input data should be 4D in (batch, channel, y, x)";
    CHECK_GE(mshape.ndim(), 4U) << "Pooling: Input map should be 4D in (batch, channel, y, x)";

    CHECK_EQ(mshape[1], 1U) << "Pooling: The channel dimension of input map must has size 1.";
    CHECK_EQ(mshape[2], dshape[2]) << "Pooling: The height of the input data and map should be the same.";
    CHECK_EQ(mshape[3], dshape[3]) << "Pooling: The width of the input data and map should be the same.";

    TShape oshape = dshape;
    if (param_.kernel.ndim() == 2) {
      CHECK(param_.kernel[0] <= dshape[2] + 2 * param_.pad[0])
          << "kernel size (" << param_.kernel[0] << ") exceeds input (" << dshape[2]
          << " padded to " << (dshape[2] + 2 * param_.pad[0]) << ")";
      CHECK(param_.kernel[1] <= dshape[3] + 2 * param_.pad[1])
          << "kernel size (" << param_.kernel[1] << ") exceeds input (" << dshape[3]
          << " padded to " << (dshape[3] + 2 * param_.pad[1]) << ")";
      if (param_.pooling_convention == ppool::kValid) {
        oshape[2] = 1 + (dshape[2] + 2 * param_.pad[0] - param_.kernel[0]) /
                         param_.stride[0];
        oshape[3] = 1 + (dshape[3] + 2 * param_.pad[1] - param_.kernel[1]) /
                         param_.stride[1];
      } else {
        oshape[2] = 1 + static_cast<int>(ceil(static_cast<float>(
                          dshape[2] + 2 * param_.pad[0] -
                          param_.kernel[0]) / param_.stride[0]));
        oshape[3] = 1 + static_cast<int>(ceil(static_cast<float>(
                          dshape[3] + 2 * param_.pad[1] -
                          param_.kernel[1]) / param_.stride[1]));
      }
      out_shape->clear();
      out_shape->push_back(oshape);  // save output shape
    } else {
      LOG(FATAL) << "not implemented.";
      return false;
    }
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_EQ(in_type->size(), 2U);
    int dtype = (*in_type)[0];
    CHECK_EQ(dtype, (*in_type)[1]);
    CHECK_NE(dtype, -1) << "Input must have specified type";

    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    PositionalPoolingProp *prop_sym = new PositionalPoolingProp();
    prop_sym->param_ = this->param_;
    return prop_sym;
  }

  std::string TypeString() const override {
    return "_contrib_PositionalPooling";
  }

  // declare dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    // for normal positional pooling
    if (param_.pool_type == ppool::kNormal){
      return {out_grad[ppool::kOut], in_data[ppool::kMap]};
    } else {  // for prod positional pooling
      return {out_grad[ppool::kOut], in_data[ppool::kData],
              in_data[ppool::kMap]};
    }
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

   private:
    PositionalPoolingParam param_;
};   // class PositionalPoolingProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_POSITIONAL_POOLING_INL_H_