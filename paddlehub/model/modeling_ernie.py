# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json

import paddle.fluid.dygraph as dygraph
import paddle.fluid as fluid

from paddlehub.utils.log import logger
from paddlehub.module.module import moduleinfo
from paddlehub.tokenizer.bert_tokenizer import BertTokenizer


class ErnieforSequenceClassification(dygraph.Layer):
    def __init__(self, ernie_module, num_classes):
        dygraph.Layer.__init__(self)

        self.module = ernie_module

        self.num_classes = num_classes

        self.dropout = lambda x: fluid.layers.dropout(
            x, dropout_prob=0.1, dropout_implementation="upscale_in_train") if self.training else x
        self.prediction = dygraph.Linear(
            input_dim=768,
            output_dim=self.num_classes,
            param_attr=fluid.ParamAttr(name="cls_out_w", initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr="cls_out_b",
            act="softmax")

    def forward(self, input_ids, sent_ids, pos_ids, input_mask, labels=None, **kwargs):
        pooled_output, sequence_output = self.module(input_ids, sent_ids, pos_ids, input_mask, **kwargs)
        cls_feats = self.dropout(pooled_output)
        predictions = self.prediction(cls_feats)

        if labels is not None:
            if len(labels.shape) == 1:
                labels = fluid.layers.reshape(labels, [-1, 1])
            loss = fluid.layers.cross_entropy(input=predictions, label=labels)
            avg_loss = fluid.layers.mean(loss)
            acc = fluid.layers.accuracy(input=predictions, label=labels)
            return predictions, avg_loss, acc
        else:
            return predictions

    def training_step(self, batch, batch_idx):
        predictions, avg_loss, acc = self(
            input_ids=batch[0], sent_ids=batch[1], pos_ids=batch[2], input_mask=batch[3], labels=batch[4])
        return {'loss': avg_loss, 'metrics': {'acc': acc}}

    def validation_step(self, batch, batch_idx):
        predictions, avg_loss, acc = self(
            input_ids=batch[0], sent_ids=batch[1], pos_ids=batch[2], input_mask=batch[3], labels=batch[4])
        return {'metrics': {'acc': acc}}
