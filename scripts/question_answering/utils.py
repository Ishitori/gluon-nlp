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

"""Various utility methods for Question Answering"""
import collections
import itertools
import math
import multiprocessing as mp


def warm_up_lr(base_lr, iteration, lr_warmup_steps, resume_training=False):
    """Returns learning rate based on current iteration. Used to implement learning rate warm up
    technique

    Parameters
    ----------
    base_lr : float
        Initial learning rage
    iteration : int
        Current iteration number
    lr_warmup_steps : int
        Learning rate warm up steps

    Returns
    -------
    learning_rate : float
        Learning rate
    """
    if resume_training:
        return base_lr

    return min(base_lr, base_lr * (math.log(iteration) / math.log(lr_warmup_steps)))


class MapReduce:
    def __init__(self, map_func, reduce_func, num_workers=None):
        self._map_func = map_func
        self._reduce_func = reduce_func
        self._num_workers = num_workers

    def __call__(self, inputs, pool=None):
        if pool:
            map_responses = pool.map(self._map_func, inputs)
        else:
            with mp.Pool(self._num_workers) as p:
                map_responses = p.map(self._map_func, inputs)

        partitions = self._partition(
            itertools.chain(*map_responses)
        )

        if pool:
            reduced_values = pool.map(self._reduce_func, partitions)
        else:
            with mp.Pool(self._num_workers) as p:
                reduced_values = p.map(self._reduce_func, partitions)

        return reduced_values

    @staticmethod
    def _partition(mapped_values):
        partitioned_data = collections.defaultdict(list)
        for key, value in mapped_values:
            partitioned_data[key].append(value)
        return partitioned_data.items()
