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

def test_positional_convolution(ctx):
    """
    Test normal positional convolution forward and backward.
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

    # num_filter * channel * K * K weight
    # i.e. (2, 2, 3, 3)
    weight = \
    mx.nd.array(
    [
    [[[1, 0, 1],
     [0, 2, -1],
     [2, 3, 1]],

    [[1, 1, 0],
     [2, -1, 2],
     [3, -2, 4]]],

    [[[0, 1, 2],
      [-1, 2, 3],
      [4, 1, -5]],

     [[3, 0, -1],
      [-1, 2, 1],
      [5, 6, 2]]]
    ], ctx=ctx)

    # num_batch * channel * out_height * out_width scale
    # i.e. (2, 2, 6, 6)
    scale = \
    mx.nd.array(
    [
    [[[1, 1, 1, 1, 1, 1],
     [1, -1, 1, -1, 1, -1],
     [-1, 1, -1, 1, -1, 1],
     [-1, -1, -1, -1, -1, -1],
     [2, 1, 2, 2, 1, 1],
     [1, 2, 1, 2, 1, 2]],

     [[1, 1, 1, 1, 1, 1],
      [1, -1, -1, 1, 1, 1],
      [-1, 1, -1, 1, -1, 1],
      [1, -1, -1, -1, -1, 1],
      [2, -1, 2, -2, 1, 1],
      [1, 2, 1, 2, 1, 2]]],

    [[[6, 5, 4, 3, 2, 1],
      [1, 2, 3, 4, 5, 6],
      [1, -1, 2, -2, 3, -3],
      [4, -4, 5, -5, 6, -6],
      [1, 1, 1, 1, 1, 1],
      [-1, -1, -1, -1, -1, -1]],

     [[-1, -1, -1, -1, -1, -1],
      [1, 1, 1, 1, 1, 1],
      [4, -4, 5, -5, 6, -6],
      [1, -1, 2, -2, 3, -3],
      [1, 2, 3, 4, 5, 6],
      [6, 5, 4, 3, 2, 1]]],
    ], ctx=ctx)

    # num_filter bias
    # i.e. (2, )
    bias = \
    mx.nd.array(
        [1, 2], ctx=ctx)

    in_data_var = mx.symbol.Variable(name="in_data")
    weight_var = mx.symbol.Variable(name="weight")
    scale_var = mx.symbol.Variable(name="scale")
    bias_var = mx.symbol.Variable(name="bias")

    op = mx.symbol.contrib.PositionalConvolution(name='test_positional_convolution',
                                                 data=in_data_var,
                                                 scale=scale_var,
                                                 weight=weight_var,
                                                 bias=bias_var,
                                                 num_filter=2,
                                                 pad=(1, 1), kernel=(3, 3), stride=(1, 1))
    be = op.bind(ctx=ctx, args={'in_data': in_data,
                                'scale': scale,
                                'weight': weight,
                                'bias': bias})
    be.forward(True)
    out_o = be.outputs[0].asnumpy()
    print(out_o)


if __name__ == '__main__':
    test_positional_convolution(default_context())