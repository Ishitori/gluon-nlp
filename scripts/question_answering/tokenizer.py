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
import re

from gluonnlp.data import SpacyTokenizer


class BiDAFTokenizer:
    def __init__(self, base_tokenizer=SpacyTokenizer(), lower_case=False):
        self._base_tokenizer = base_tokenizer
        self._lower_case = lower_case

    def __call__(self, sample):
        """

        Parameters
        ----------
        sample: str
            The sentence to tokenize

        Returns
        -------
        ret : list of strs
            List of tokens
        """
        tokens = [token.replace("''", '"').replace("``", '"') for token in
                  self._base_tokenizer(sample)]

        if self._lower_case:
            tokens = [token.lower() for token in tokens]

        tokens = BiDAFTokenizer._process_tokens(tokens)
        return tokens

    @staticmethod
    def _process_tokens(temp_tokens):
        tokens = []
        splitters = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C",
                     "\u2019", "\u201D", "\u2018", "\u00B0")

        for token in temp_tokens:
            tokens.extend(re.split("([{}])".format("".join(splitters)), token))

        return tokens
