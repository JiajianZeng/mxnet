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
 * \file positional_convolution.cu
 * \brief
 * \author Jiajian Zeng
 */

#include "./positional_convolution-inl.h"
#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"

namespace mshadow{
namespace cuda{

/*!
 * \brief duplicate row gpu kernel
 * Do not call this kernel directly. Use the interface DuplicateRow().
 */
template <typename DType>
__global__ void DuplicateRowKernel(const int nthreads,
                                   const DType* in_data,
                                   const int height,
                                   const int width,
                                   const int dfactor,
                                   DType* out_data) {
  const int duplicated_height = height * dfactor;
  const int duplicated_width = width;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int dw = index % duplicated_width;
    const int dh = (index / duplicated_width) % duplicated_height;
    const int w = dw;
    const int h = dh / dfactor;
    out_data[index] = in_data[h * width + w];
  }
}

template <typename DType>
inline void DuplicateRow(Stream<gpu>* s,
                         const Tensor<gpu, 2, DType>& out,
                         const Tensor<gpu, 2, DType>& in,
                         const int dfactor) {
  const DType* pin = in.dptr_;
  DType* pout = out.dptr_;
  const int height = in.size(0);
  const int width = in.size(1);
  const int count = out.shape_.Size();
  DuplicateRowKernel<DType><<< mxnet::op::mxnet_op::cuda_get_num_blocks(count),
    kBaseThreadNum, 0, Stream<gpu>::GetStream(s) >>>(
      count, pin, height, width, dfactor, pout);
  MSHADOW_CUDA_POST_KERNEL_CHECK(DuplicateRowKernel);
}

/*!
 * \brief sum over rows gpu kernel
 * Do not call this kernel directly. Use the interface SumOverRows().
 */
template<typename DType>
__global__ void SumOverRowsKernel(const int nthreads,
                                  const DType* in_data,
                                  const int height,
                                  const int width,
                                  const int sfactor,
                                  DType* out_data) {
  const int summed_height = height / sfactor;
  const int summed_width = width;
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int sw = index % summed_width;
    const int sh = (index / summed_width) % summed_height;
    const int w = sw;
    const int hstart = sh * sfactor;
    const int hend = hstart + sfactor;
    out_data[index] = 0;
    for (int i = hstart; i < hend; i++) {
      out_data[index] += in_data[i * width + w];
    }
  }
}

template <typename DType>
inline void SumOverRows(Stream<gpu>* s,
                        const Tensor<gpu, 2, DType>& out,
                        const Tensor<gpu, 2, DType>& in,
                        const int sfactor) {
  const DType* pin = in.dptr_;
  DType* pout = out.dptr_;
  const int height = in.size(0);
  const int width = in.size(1);
  const int count = out.shape_.Size();
  SumOverRowsKernel<DType><<< mxnet::op::mxnet_op::cuda_get_num_blocks(count),
    kBaseThreadNum, 0, Stream<gpu>::GetStream(s) >>>(
      count, pin, height, width, sfactor, pout);
  MSHADOW_CUDA_POST_KERNEL_CHECK(SumOverRowsKernel);
}

}  // namespace cuda

template<typename DType>
inline void DuplicateRow(Stream<gpu>* s,
                         const Tensor<gpu, 2, DType>& out,
                         const Tensor<gpu, 2, DType>& in,
                         const int dfactor) {
  cuda::DuplicateRow(s, out, in, dfactor);
}

template<typename DType>
inline void SumOverRows(Stream<gpu>* s,
                        const Tensor<gpu, 2, DType>& out,
                        const Tensor<gpu, 2, DType>& in,
                        const int sfactor) {
  cuda::SumOverRows(s, out, in, sfactor);
}
}  // namespace mshadow

namespace mxnet {
namespace op {

  template<>
  Operator* CreateOp<gpu>(PositionalConvolutionParam param, int dtype,
    std::vector<TShape> *in_shape,
    std::vector<TShape> *out_shape,
    Context ctx) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
      op = new PositionalConvolutionOp<gpu, DType>(param);
    })
    return op;
  }

}  // namespace op
}  // namespace mxnet