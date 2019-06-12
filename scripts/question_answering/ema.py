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

"""Exponential moving average"""


class ExponentialMovingAverage(object):
    r"""An implement of Exponential Moving Average.

        shadow variable = decay * shadow variable + (1 - decay) * variable

    Parameters
    ----------
    decay : float, default 0.9999
        The axis to sum over when computing softmax and entropy.
    """

    def __init__(self, decay=0.9999, **kwargs):
        super(ExponentialMovingAverage, self).__init__(**kwargs)
        self.decay = decay
        self.shadow = {}

    def add(self, name, parameters):
        r"""Update the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.
        parameters : NDArray
            the init value of shadow variable.
        Returns
        --------
        return : None
        """
        self.shadow[name] = parameters.copy()

    def __call__(self, name, x):
        r"""Update the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.
        x : NDArray
            the value of shadow variable.
        Returns
        --------
        return : None
        """
        assert name in self.shadow
        self.shadow[name] = self.decay * \
            self.shadow[name] + (1.0 - self.decay) * x

    def get(self, name):
        r"""Return the shadow variable.

        Parameters
        -----------
        name : string
            the name of shadow variable.

        Returns
        --------
        return : NDArray
            the value of shadow variable.
        """
        return self.shadow[name]
