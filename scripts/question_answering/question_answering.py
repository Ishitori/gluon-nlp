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
__all__ = ['BiDAFModelingLayer']

from mxnet.gluon import Block
from mxnet.gluon.rnn import LSTM


class BiDAFModelingLayer(Block):
    """BiDAFModelingLayer implements modeling layer of BiDAF paper. It is used to scan over context
    produced by Attentional Flow Layer via 2 layer bi-LSTM.

    The input data for the forward should be of dimension 8 * hidden_size (default hidden_size
    is 100).

    Parameters
    ----------

    input_dim : `int`, default 100
        The number of features in the hidden state h of LSTM
    nlayers : `int`, default 2
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
    def __init__(self, input_dim=100, nlayers=2, biflag=True,
                 dropout=0.2, prefix=None, params=None):
        super(BiDAFModelingLayer, self).__init__(prefix=prefix, params=params)

        self._modeling_layer = LSTM(hidden_size=input_dim, num_layers=nlayers, dropout=dropout,
                                    bidirectional=biflag)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self._modeling_layer(x)
        return out
