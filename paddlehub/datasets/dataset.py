#coding:utf-8
#   Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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


class InputExample(object):
    """
    The input data structure of Transformer modules (BERT, ERNIE and so on).
    """

    def __init__(self, guid, text_a, text_b=None, label=None):
        """
        The input data structure.

        Args:
          guid(int): Unique id for the input data.
          text_a(string): The first sequence. For single sequence tasks, only this sequence must be specified.
          text_b(string, Optional): The second sequence if sentence-pair.
          label(string, Optional): The label of the example.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __str__(self):
        if self.text_b is None:
            return "text={}\tlabel={}".format(self.text_a, self.label)
        else:
            return "text_a={}\ttext_b={},label={}".format(self.text_a, self.text_b, self.label)
