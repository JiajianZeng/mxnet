# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: skip-file
from __future__ import print_function
import numpy as np
import mxnet as mx
from mxnet.test_utils import *
import unittest
import itertools


def test_positional_pooling_forward(ctx):
    """
    Test positional pooling forward.
    """
    # num_batch * channel * height * width input
    # i.e. (2, 2, 6, 6)
    in_data = \
    mx.nd.array(
    [
    [[[1, 2, -1, 0, 1, 1],
      [3, 6, -5, 4, 2, -2],
      [9, 6, -1, 3, 1, 3],
      [4, 2, 5, 7, 3, 1],
      [0, 1, 1, 2, 2, 1],
      [3, 1, 2, 4, 3, 3]],

     [[3, 1, 2, 4, 3, 3],
      [0, 1, 1, 2, 2, 1],
      [4, 2, 5, 7, 3, 1],
      [9, 6, -1, 3, 1, 3],
      [3, 6, -5, 4, 2, -2],
      [1, 2, -1, 0, 1, 1]]],
    [[[1, 2, 3, 4, 5, 6],
      [6, 5, 4, 3, 2, 1],
      [0, 0, 1, 1, 2, 2],
      [3, 3, 0, -1, -1, -2],
      [3, 1, 0, 3, 3, 2],
      [5, 6, 7, -1, -2, 0]],

     [[5, 6, 7, -1, -2, 0],
      [3, 1, 0, 3, 3, 2],
      [3, 3, 0, -1, -1, -2],
      [0, 0, 1, 1, 2, 2],
      [6, 5, 4, 3, 2, 1],
      [1, 2, 3, 4, 5, 6]]]
    ], ctx=ctx)

    in_map = \
    mx.nd.array(
    [
    [[[0.9, 0.8, 0.7, 0.5, 0.4, 0.3],
      [0.7, 1.0, 0.9, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.2, 0.3, 0.3, 0.4],
      [0.4, 0.3, 0.3, 0.2, 0.2, 0.1]]],

    [[[0.4, 0.3, 0.3, 0.2, 0.2, 0.1],
      [0.1, 0.2, 0.2, 0.3, 0.3, 0.4],
      [0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
      [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
      [0.7, 1.0, 0.9, 0.3, 0.2, 0.1],
      [0.9, 0.8, 0.7, 0.5, 0.4, 0.3]]]
    ], ctx=ctx)

    in_data_var = mx.symbol.Variable(name="in_data")
    in_map_var = mx.symbol.Variable(name="in_map")

    op_normal = mx.symbol.contrib.PositionalPooling(name='test_positional_pooling',
                                                    data=in_data_var,
                                                    map=in_map_var,
                                                    pool_type="normal",
                                                    pooling_convention="valid",
                                                    kernel=(2, 2), stride=(2, 2), pad=(0, 0)
                                                    )

    be_normal = op_normal.bind(ctx=ctx, args={'in_data': in_data,
                               'in_map': in_map})
    be_normal.forward(True)
    out_o_normal = be_normal.outputs[0].asnumpy()
    print("out_normal")
    print(out_o_normal)

    op_prod = mx.symbol.contrib.PositionalPooling(name='test_positional_pooling',
                                                  data=in_data_var,
                                                  map=in_map_var,
                                                  pool_type="prod",
                                                  pooling_convention="valid",
                                                  kernel=(2, 2), stride=(2, 2), pad=(0, 0)
                                                  )
    be_prod = op_prod.bind(ctx=ctx, args={'in_data': in_data,
                                          'in_map': in_map})
    be_prod.forward(True)
    out_o_prod = be_prod.outputs[0].asnumpy()
    print("out_prod")
    print(out_o_prod)


def test_positional_pooling_backward():
    """
    Test positional convolution backward.
    """
    for num_batch in [1, 2, 4]:
        for num_channel_data in [4, 8, 12]:
            for input_height, input_width in itertools.product([5, 6, 9], [5, 6, 9]):
                for pool_type in ["normal", "prod"]:
                    for pooling_convention in ["valid", "full"]:
                        for kernel_w, kernel_h in itertools.product([2, 3], [2, 3]):
                            for stride_w, stride_h in itertools.product([1, 2, 3], [1, 2, 3]):
                                for pad_w, pad_h in itertools.product([0, 1], [0, 1]):
                                    for grad_nodes in [['in_data'], ['in_map']]:
                                        in_data = np.random.rand(num_batch, num_channel_data, input_height, input_width)
                                        in_map = \
                                            np.random.rand(num_batch, 1, input_height, input_width)\
                                            * 0.8 + 0.1

                                        in_data_var = mx.symbol.Variable(name="in_data")
                                        in_map_var = mx.symbol.Variable(name="in_map")
                                        op = mx.sym.contrib.PositionalPooling(name='test_op', data=in_data_var,
                                                                              map=in_map_var, pool_type=pool_type,
                                                                              pooling_convention=pooling_convention,
                                                                              pad=(pad_h, pad_w),
                                                                              kernel=(kernel_h, kernel_w),
                                                                              stride=(stride_h, stride_w),
                                                                              )
                                        rtol, atol = 1e-5, 1e-8
                                        # By now we only have gpu implementation
                                        if mx.Context.default_ctx.device_type == 'gpu':
                                            check_numeric_gradient(op, [in_data, in_map], rtol=rtol, atol=atol,
                                                                   grad_nodes=grad_nodes, ctx=mx.gpu(0))

if __name__ == '__main__':
    test_positional_pooling_forward(mx.gpu(0))
    test_positional_pooling_backward()
    print("positional pooling backward works correctly.")
