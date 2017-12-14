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
 * \file positional_pooling.cu
 * \brief positional pooling operator
 * \author Jiajian Zeng
 */

#include <mshadow/tensor.h>
#include <mshadow/cuda/reduce.cuh>
#include <algorithm>
#include <vector>
#include "../../common/cuda_utils.h"
#include "../mxnet_op.h"
#include "./positional_pooling-inl.h"

#define POSITIONALPOOLING_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n); \
      i += blockDim.x * gridDim.x)

namespace mshadow{
namespace cuda {

/*!
 * \brief Positional pooling gpu kernel for 2-D images.
 * Do not call this kernel directly. Use the interface PositionalPoolForward().
 */
template <typename DType>
__global__ void PositionalPoolForwardKernel(const int nthreads, const DType* in_data,
                                            const DType* in_map, const int channels,
                                            const int height, const int width,
                                            const int pooled_height, const int pooled_width,
                                            const int kernel_h, const int kernel_w,
                                            const int stride_h, const int stride_w,
                                            const int pad_h, const int pad_w,
                                            const int pool_type,
                                            DType* out_data) {
  using mshadow::red::limits::MinValue;
  using mxnet::op::ppool;
  // index is the output image's pixel index in NCHW
  // suppose a pixel in the output image's location is (n, c, ph, pw)
  // then index = pooled_height * pooled_width * (n * channels + c) +
  // ph * pooled_width + pw
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    const DType* map_slice =
        in_map + n * height * width;
    const DType* data_slice =
        in_data + (n * channels + c) * height * width;
    DType max_map_val = MinValue<DType>();
    DType data_val = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const DType map_val = map_slice[h * width + w];
        if (map_val >= max_map_val) {  // NOTE: use >= here to facilitate the backward computation
          max_map_val = map_val;
          data_val = data_slice[h * width + w];
          if (pool_type == ppool::kProd) {
            data_val *= map_val;
          }
        }
      }
    }
    out_data[index] = data_val;
  }
}

template<typename DType>
inline void PositionalPoolForward(const Tensor<gpu, 4, DType> &out,
                                  const Tensor<gpu, 4, DType> &data,
                                  const Tensor<gpu, 4, DType> &map,
                                  const TShape& kernel,
                                  const TShape& pad,
                                  const TShape& stride,
                                  const int pool_type) {
  const DType* pdata = data.dptr_;
  const DType* pmap = map.dptr_;
  DType* pout = out.dptr_;
  const channels = data.size(1);
  const height = data.size(2);
  const width = data.size(3);
  const pooled_height = out.size(2);
  const pooled_width = out.size(3);
  const kernel_h = kernel[0];
  const kernel_w = kernel[1];
  const stride_h = stride[0];
  const stride_w = stride[1];
  const pad_h = pad[0];
  const pad_w = pad[1];
  const count = out.shape_.Size();

  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  PositionalPoolForwardKernel<DType><<<mxnet::op::mxnet_op::cuda_get_num_blocks(count),
    kBaseThreadNum, 0, stream>>>(
      count, pdata, pmap, channels, height, width,
      pooled_height, pooled_width, kernel_h, kernel_w,
      stride_h, stride_w, pad_h, pad_w,
      pool_type, pout
    );
  POSITIONALPOOLING_CUDA_CHECK(cudaPeekAtLastError());
}

/*!
 * \brief Positional pooling backward gpu kernel for 2-D images.
 * Do not call this kernel directly. Use the interface PositionalPoolBackward().
 */
template <typename DType>
__global__ void PositionalPoolBackwardKernel(const int nthreads, const DType* out_grad,
                                             const DType* in_data, const DType* map_data,
                                             const int channels, const int height, const int width,
                                             const int pooled_height, const int pooled_width,
                                             const int kernel_h, const int kernel_w,
                                             const int stride_h, const int stride_w,
                                             const int pad_h, const int pad_w, const int pool_type,
                                             DType* in_grad, DType* map_grad) {
  using mshadow::red::limits::MinValue;
  using mxnet::op::ppool;
  // index is the output image's pixel index in NCHW
  // the order has to be consistent with pooling max
  // to avoid adding out_grad to the wrong in_grad
  // in the case where there are multiple max pixels
  // covered by a kernel window
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int pw = index % pooled_width;
    const int ph = (index / pooled_width) % pooled_height;
    const int c = (index / pooled_width / pooled_height) % channels;
    const int n = index / pooled_width / pooled_height / channels;
    int hstart = ph * stride_h - pad_h;
    int wstart = pw * stride_w - pad_w;
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);

    // in data/grad offset batch and channel dims
    int map_offset = n * height * width;
    const DType* map_slice = map_data + map_offset;
    int max_map_idx = -1;
    DType max_map_val = MinValue<DType>();
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        const int map_idx = h * width + w;
        if (map_slice[map_idx] >= max_map_val) {  // NOTE: use >= here to make backward consistent with forward
          max_map_val = map_slice[map_idx];
          max_map_idx = map_idx;
        }
      }
    }

    // In the case where pad > 0 and kernel = 1, for example,
    // max_idx can be -1 reaching this step.
    int in_offset = (n * channels + c) * height * width;
    if (max_map_idx >= 0) {
      // Normal positional pooling
      if (pool_type == ppool::kNormal) {
        atomicAdd(&in_grad[in_offset + max_map_idx], out_grad[index]);
      } else {  // Prod positional pooling
        atomicAdd(&in_grad[in_offset + max_map_idx], out_grad[index] * max_map_val);
        atomicAdd(&map_grad[map_offset + max_map_idx],
                  out_grad[index] * in_data[in_offset + max_map_idx]);
      }
    }
  }
}

template<typename DType>
inline void PositionalPoolBackward(const Tensor<gpu, 4, DType>& in_grad,
                                   const Tensor<gpu, 4, DType>& out_grad,
                                   const Tensor<gpu, 4, DType>& map_grad,
                                   const Tensor<gpu, 4, DType>& in_data,
                                   const Tensor<gpu, 4, DType>& map_data,
                                   const TShape& kernel,
                                   const TShape& pad,
                                   const TShape& stride,
                                   const int pool_type) {
  const DType* pdata = data.dptr_;
  const DType* pmap = map.dptr_;
  DType* pout = out.dptr_;
  const channels = data.size(1);
  const height = data.size(2);
  const width = data.size(3);
  const pooled_height = out.size(2);
  const pooled_width = out.size(3);
  const kernel_h = kernel[0];
  const kernel_w = kernel[1];
  const stride_h = stride[0];
  const stride_w = stride[1];
  const pad_h = pad[0];
  const pad_w = pad[1];
  const count = out.shape_.Size();
}

}  // namespace cuda

template<typename DType>
inline void PositionalPoolForward(const Tensor<gpu, 4, DType> &out,
                                  const Tensor<gpu, 4, DType> &data,
                                  const Tensor<gpu, 4, DType> &map,
                                  const TShape& kernel,
                                  const TShape& pad,
                                  const TShape& stride,
                                  const int pool_type) {
  cuda::PositionalPoolForward(out, data, map, kernel, pad, stride, pool_type);
}

template<typename DType>
inline void PositionalPoolBackward(const Tensor<gpu, 4, DType>& in_grad,
                                   const Tensor<gpu, 4, DType>& out_grad,
                                   const Tensor<gpu, 4, DType>& map_grad,
                                   const Tensor<gpu, 4, DType>& in_data,
                                   const Tensor<gpu, 4, DType>& map_data,
                                   const TShape& kernel,
                                   const TShape& pad,
                                   const TShape& stride,
                                   const int pool_type) {
  cuda::PositionalPoolBackward(in_grad, out_grad, map_grad, in_data, map_data, kernel, pad, stride, pool_type);
}

}  // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator* CreateOp<gpu>(PositionalPoolingParam param, int dtype) {
  Operator* op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new PositionalPoolingOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet