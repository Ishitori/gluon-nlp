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

"""Question id mapper to and from int"""


class QuestionIdMapper:
    def __init__(self, dataset):
        self._question_id_to_context = {item[1]: item[3] for item in dataset}
        self._question_id_to_idx = {item[1]: item[0] for item in dataset}
        self._idx_to_question_id = {v: k for k, v in self._question_id_to_idx.items()}

    @property
    def question_id_to_context(self):
        return self._question_id_to_context

    @property
    def idx_to_question_id(self):
        return self._idx_to_question_id

    @property
    def question_id_to_idx(self):
        return self._question_id_to_idx
