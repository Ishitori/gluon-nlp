# coding: utf-8

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

# pylint: disable=
"""BiDAF model blocks"""
__all__ = ['BiDAFOutputLayer']

import mxnet as mx
from mxnet.gluon import Block, nn
from mxnet.gluon.rnn import LSTM


class BiDAFOutputLayer(Block):
    """
    ``BiDAFOutputLayer`` produces the final prediction of an answer. The output is a tuple of
    start index and end index of the answer in the paragraph per each batch.

    It accepts 2 inputs:
        `x` : the output of Attention layer of shape:
        seq_max_length x batch_size x 8 * span_start_input_dim

        `m` : the output of Modeling layer of shape:
         seq_max_length x batch_size x 2 * span_start_input_dim

    Parameters
    ----------
    span_start_input_dim : `int`, default 100
        The number of features in the hidden state h of LSTM
    units : `int`, default 10 * ``span_start_input_dim``
        Number of hidden units of `Dense` layer
    nlayers : `int`, default 1
        Number of recurrent layers.
    biflag: `bool`, default True
        If `True`, becomes a bidirectional RNN.
    dropout: `float`, default 0
        If non-zero, introduces a dropout layer on the outputs of each
        RNN layer except the last layer.
    prefix : `str` or None
        Prefix of this `Block`.
    params : `ParameterDict` or `None`
        Shared Parameters for this `Block`.
    """
    def __init__(self, span_start_input_dim=100, units=None, nlayers=1, biflag=True,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFOutputLayer, self).__init__(prefix=prefix, params=params)

        units = 10 * span_start_input_dim if units is None else units

        self._start_index_dense = nn.Dense(units=units)
        self._end_index_lstm = LSTM(hidden_size=span_start_input_dim,
                                    num_layers=nlayers, dropout=dropout, bidirectional=biflag)
        self._end_index_dense = nn.Dense(units=units)

    def forward(self, x, m):  # pylint: disable=arguments-differ
        # setting batch size as the first dimension
        start_index_input = mx.nd.transpose(mx.nd.concat(x, m, dim=2), axes=(1, 0, 2))
        start_index_dense_output = self._start_index_dense(start_index_input)
        start_index_softmax_output = start_index_dense_output.softmax(axis=1)
        start_index = mx.nd.argmax(start_index_softmax_output, axis=1)

        end_index_input_part = self._end_index_lstm(m)
        end_index_input = mx.nd.transpose(mx.nd.concat(x, end_index_input_part, dim=2),
                                          axes=(1, 0, 2))

        end_index_dense_output = self._end_index_dense(end_index_input)
        end_index_softmax_output = end_index_dense_output.softmax(axis=1)
        end_index = mx.nd.argmax(end_index_softmax_output, axis=1)

        # producing output in shape 2 x batch_size
        output = mx.nd.concat(mx.nd.expand_dims(start_index, axis=0),
                              mx.nd.expand_dims(end_index, axis=0), dim=0)

        # transposing it to batch_size x 2
        return mx.nd.transpose(output, axes=(1, 0))
