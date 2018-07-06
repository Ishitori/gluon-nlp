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

"""BiDAF model blocks"""
__all__ = ['BiDAFEmbedding']

from mxnet import nd, init
from mxnet.gluon import Block
from mxnet.gluon import nn
from mxnet.gluon.rnn import LSTM

from gluonnlp.model import ConvolutionalEncoder, Highway


class BiDAFEmbedding(Block):
    """BiDAFEmbedding is a class describing embeddings that are separately applied to question
    and context of the datasource. Both question and context are passed in two NDArrays:
    1. Matrix of words: batch_size x words_per_question/context
    2. Tensor of characters: batch_size x words_per_question/context x chars_per_word
    """
    def __init__(self, word_vocab, char_vocab, prefix=None, params=None):
        super(BiDAFEmbedding, self).__init__(prefix=prefix, params=params)

        self._char_dense_embedding = nn.Embedding(input_dim=len(char_vocab), output_dim=8)
        self._char_conv_embedding = ConvolutionalEncoder(
            embed_size=8,
            num_filters=(100,),
            ngram_filter_sizes=(5,),
            num_highway=None,
            conv_layer_activation='relu',
            output_size=None
        )

        self._word_embedding = nn.Embedding(input_dim=len(word_vocab), output_dim=100,
                                            weight_initializer=init.Constant(
                                                word_vocab.embedding.idx_to_vec))

        self._highway_network = Highway(200, num_layers=2)
        self._contextual_embedding = LSTM(hidden_size=100, num_layers=2,
                                          bidirectional=True)

    def forward(self, x, contextual_embedding_state=None):  # pylint: disable=arguments-differ
        # Changing shape from NTC to TNC as most MXNet blocks work with TNC format natively
        word_level_data = nd.transpose(x[0], axes=(1, 0))
        char_level_data = nd.transpose(x[1], axes=(1, 0, 2))

        # Get word embeddings. Output is batch_size x seq_len x embedding size (100)
        word_embedded = self._word_embedding(word_level_data)

        # Get char level embedding in multiple steps:
        # Step 1. Embed into 8-dim vector
        char_level_data = self._char_dense_embedding(char_level_data)

        # Step 2. Transpose to put seq_len first axis to later iterate over it
        # In that way we can get embedding per token of every batch
        char_level_data = nd.transpose(char_level_data, axes=(0, 2, 1, 3))

        # Step 3. Iterate over tokens of each batch and apply convolutional encoder
        # As a result of a single iteration, we get token embedding for every batch
        token_list = []
        for token_of_all_batches in char_level_data:
            token_list.append(self._char_conv_embedding(token_of_all_batches))

        # Step 4. Concat all tokens embeddings to create a single tensor.
        char_embedded = nd.concat(*token_list, dim=0)

        # Step 5. Reshape tensor to match dimensions of embedded words
        char_embedded = char_embedded.reshape(shape=word_embedded.shape)

        # Concat embeddings, making channels size = 200
        highway_input = nd.concat(char_embedded, word_embedded, dim=2)
        # Pass through highway, shape remains unchanged
        highway_output = self._highway_network(highway_input)

        # Pass through contextual embedding, which is just bi-LSTM
        ce_output, ce_state = self._contextual_embedding(highway_output,
                                                         contextual_embedding_state)

        return ce_output, ce_state
