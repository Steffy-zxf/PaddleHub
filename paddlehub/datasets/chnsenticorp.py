# coding:utf-8
# Copyright (c) 2020  PaddlePaddle Authors. All Rights Reserved.
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
from tqdm import tqdm
import csv
import io
import os

import paddle.fluid as fluid

from paddlehub.datasets.dataset import InputExample
from paddlehub.env import DATA_HOME
from paddlehub.utils.log import logger


class ChnSentiCorp(fluid.io.Dataset):
    def __init__(self, tokenizer, max_seq_len, mode='train'):  #
        self.mode = mode
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label_list = ['0', '1']
        self.num_classes = len(self.label_list)

        if self.mode == 'train':
            self.file = 'train.tsv'
        elif self.mode == 'test':
            self.file = 'test.tsv'
        else:
            self.file = 'dev.tsv'
        self.file = os.path.join(DATA_HOME, 'chnsenticorp', self.file)

        self.examples = self._read_file(self.file)
        self.records = self._convert_examples_to_records(self.examples)

    def _read_file(self, input_file, phase=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(guid=seq_id, label=line[0], text_a=line[1])
                seq_id += 1
                examples.append(example)
            return examples

    def _convert_examples_to_records(self, examples):
        """
        Returns a list[dict] including all the input information what the model needs.

        Args:
            examples (list): the data example, returned by _read_file.


        Returns:
            a list with all the examples record.
        """
        records = []
        with tqdm(total=len(examples)) as process_bar:
            for example in examples:
                record = self.tokenizer.encode(
                    text=example.text_a, text_pair=example.text_b, max_seq_len=self.max_seq_len)
                # CustomTokenizer will tokenize the text firstly and then lookup words in the vocab
                # When all words are not found in the vocab, the text will be dropped.
                if not record:
                    logger.info("The text %s has been dropped as it has no words in the vocab after tokenization." %
                                example.text_a)
                    continue
                if example.label:
                    record["label"] = self.label_list.index(example.label)
                records.append(record)
                process_bar.update(1)
        return records

    def __getitem__(self, idx):
        record = self.records[idx]
        if "label" in record.keys():
            return record['input_ids'], record['segment_ids'], record['position_ids'], record['input_mask'], record[
                'label']
        else:
            return record['input_ids'], record['segment_ids'], record['position_ids'], record['input_mask']

    def __len__(self):
        return len(self.records)


if __name__ == "__main__":
    import paddlehub as hub

    tokenizer = hub.BertTokenizer(
        vocab_file=
        '/mnt/zhangxuefei/program-paddle/PaddleHub/hub_module/modules/text/semantic_model/ernie_dygraph/model_params/vocab.txt'
    )
    train_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='train')
    dev_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='dev')
    test_dataset = ChnSentiCorp(tokenizer=tokenizer, max_seq_len=60, mode='test')

    index = 0
    while index < 3:
        record = train_dataset.__getitem__(index)
        print("train record: ", record)
        record = dev_dataset.__getitem__(index)
        print("dev record: ", record)
        record = test_dataset.__getitem__(index)
        print("test record: ", record)
        index += 1
