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
"""Exponential Moving Average"""
import mxnet as mx
from mxnet import gluon


class PolyakAveraging:
    """Class to do Polyak averaging based on this paper
    http://www.meyn.ece.ufl.edu/archive/spm_files/Courses/ECE555-2011/555media/poljud92.pdf"""
    def __init__(self, params, decay):
        self._params = params
        self._decay = decay

        self._polyak_params_dict = gluon.ParameterDict()

        for param in self._params.values():
            polyak_param = self._polyak_params_dict.get(param.name, shape=param.shape)
            polyak_param.initialize(mx.init.Constant(self._param_data_to_cpu(param)), ctx=mx.cpu())

    def update(self):
        """
        Updates currently held saved parameters with current state of network.

        All calculations for this average occur on the cpu context.
        """
        for param in self._params.values():
            polyak_param = self._polyak_params_dict.get(param.name)
            polyak_param.set_data(
                (1 - self._decay) * self._param_data_to_cpu(param) +
                self._decay * polyak_param.data(mx.cpu()))

    def get_params(self):
        """ Provides averaged parameters

        Returns
        -------
        gluon.ParameterDict
            Averaged parameters
        """
        return self._polyak_params_dict

    def _param_data_to_cpu(self, param):
        """Returns a copy (on CPU context) of the data held in some context of given parameter.

        Parameters
        ----------
        param: gluon.Parameter
            Parameter's whose data needs to be copied.

        Returns
        -------
        NDArray
            Copy of data on CPU context.
        """
        return param.list_data()[0].copyto(mx.cpu())
