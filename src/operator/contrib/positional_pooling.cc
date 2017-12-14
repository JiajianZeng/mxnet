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
 * Licensed under The Apache-2.0 License [see LICENSE for details]
 * \file positional_pooling.cc
 * \brief positional pooling operator
 * \author Jiajian Zeng
 */
#include <mshadow/base.h>
#include <mshadow/tensor.h>
#include <mshadow/packet-inl.h>
#include <mshadow/dot_engine-inl.h>
#include <cassert>
#include "./positional_pooling-inl.h"

namespace mshadow {
template<typename DType>
inline void PositionalPoolForward(const Tensor<cpu, 4, DType> &out,
                                  const Tensor<cpu, 4, DType> &data,
                                  const Tensor<cpu, 4, DType> &map,
                                  const TShape& kernel,
                                  const TShape& pad,
                                  const TShape& stride,
                                  const int pool_type) {
  // NOT_IMPLEMENTED
  return;
}

template<typename DType>
inline void PositionalPoolBackward(const Tensor<cpu, 4, DType>& in_grad,
                                   const Tensor<cpu, 4, DType>& out_grad,
                                   const Tensor<cpu, 4, DType>& map_grad,
                                   const Tensor<cpu, 4, DType>& in_data,
                                   const Tensor<cpu, 4, DType>& map_data,
                                   const TShape& kernel,
                                   const TShape& pad,
                                   const TShape& stride,
                                   const int pool_type) {
  // NOT_IMPLEMENTED
  return;
}
}  // namespace mshadow

namespace mxnet{
namespace op {

template<>
Operator *CreateOp<cpu>(PositionalPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PositionalPoolingOp<cpu, DType>(param);
  });
  return op;
}

Operator *PositionalPoolingProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                  std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(PositionalPoolingParam);

MXNET_REGISTER_OP_PROPERTY(PositionalPooling, PositionalPoolingProp)
.describe(R"code(Performs positional pooling on the input.

The shapes for 2-D positional pooling are

- **data**: *(batch_size, channel, height, width)*
- **map**: *(batch_size, 1, height, width)*
- **out**: *(batch_size, num_filter, out_height, out_width)*, with::

    out_height = f(height, kernel[0], pad[0], stride[0])
    out_width = f(width, kernel[1], pad[1], stride[1])

The definition of *f* depends on ``pooling_convention``, which has two options:

- **valid** (default)::

    f(x, k, p, s) = floor((x+2*p-k)/s)+1

- **full**, which is compatible with Caffe::

    f(x, k, p, s) = ceil((x+2*p-k)/s)+1

Two pooling options are supported by ``pool_type``:

- **normal**: normal positional pooling
- **prod**: prod positional pooling

)code" ADD_FILELINE)
.add_argument("data", "NDArray-or-Symbol", "Input data to the pooling operator.")
.add_argument("map", "NDArray-or-Symbol", "Input map to the pooling operator.")
.add_arguments(PoolingParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet