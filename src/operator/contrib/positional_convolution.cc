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
 * Copyright (c) 2017 By Contributors
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file positional_convolution.cc
 * \brief
 * \author Jiajian Zeng
 */

#include "./positional_convolution-inl.h"

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(PositionalConvolutionParam);

template<>
Operator* CreateOp<cpu>(PositionalConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PositionalConvolutionOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *PositionalConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(_contrib_PositionalConvolution, PositionalConvolutionProp)
.describe(R"code(Compute 2-D positional convolution on 4-D input.

The positional convolution operation is described in https://

For 2-D positional convolution, the shapes are

- **data**: *(batch_size, channel, height, width)*
- **scale**: *(batch_size, channel, out_height, out_width)*
- **weight**: *(num_filter, channel, kernel[0], kernel[1])*
- **bias**: *(num_filter,)*
- **out**: *(batch_size, num_filter, out_height, out_width)*.

Define::

  f(x,k,p,s,d) = floor((x+2*p-d*(k-1)-1)/s)+1

then we have::

  out_height=f(height, kernel[0], pad[0], stride[0], dilate[0])
  out_width=f(width, kernel[1], pad[1], stride[1], dilate[1])

If ``no_bias`` is set to be true, then the ``bias`` term is ignored.

The default data ``layout`` is *NCHW*, namely *(batch_size, channle, height,
width)*.

If ``num_group`` is larger than 1, denoted by *g*, then split the input ``data``
evenly into *g* parts along the channel axis, and also evenly split ``weight``
along the first dimension (the second dimension of the input ``data`` will change
accordingly). Next compute the convolution on the *i*-th part of
the data with the *i*-th weight part. The output is obtained by concatenating all
the *g* results.


Both ``weight`` and ``bias`` are learnable parameters.


)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the PositionalConvolutionOp.")
.add_argument("scale", "NDArray-or-Symbol", "Input scale to the PositionalConvolutionOp.")
.add_argument("weight", "NDArray-or-Symbol", "Weight matrix.")
.add_argument("bias", "NDArray-or-Symbol", "Bias parameter.")
.add_arguments(PositionalConvolutionParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet