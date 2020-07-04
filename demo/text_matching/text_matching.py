# coding:utf-8
#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
"""Fine-tuning on text matching task """

import argparse
import ast
import paddlehub as hub

# yapf: disable
parser = argparse.ArgumentParser(__doc__)
parser.add_argument("--num_epoch", type=int, default=3, help="Number of epoches for fine-tuning.")
parser.add_argument("--use_gpu", type=ast.literal_eval, default=True, help="Whether use GPU for fine-tuning, input should be True or False")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train with warmup.")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory to model checkpoint")
parser.add_argument("--max_seq_len", type=int, default=512, help="Number of words of the longest seqence.")
parser.add_argument("--batch_size", type=int, default=32, help="Total examples' number in batch for training.")
parser.add_argument("--network", type=str, default=None, help="Pre-defined network which was connected after module.")
parser.add_argument("--use_data_parallel", type=ast.literal_eval, default=False, help="Whether use data parallel.")
parser.add_argument("--is_pair_wise", type=ast.literal_eval, default=False, help="Whether use data parallel.")
args = parser.parse_args()
# yapf: enable.

if __name__ == '__main__':

    # Load Paddlehub ERNIE pretrained model
    module = hub.Module(name="ernie")

    # Pair wise task needs: query, title_left, right_title (3 data)
    # Point wise task needs: query, title_left (2 data)
    if args.is_pair_wise:
        num_data = 3
    else:
        num_data = 2
    inputs, outputs, program = module.context(
        trainable=True, max_seq_len=args.max_seq_len, num_data=num_data)

    # Tokenizer tokenizes the text data and encodes the data as model needed.
    # If you use transformer modules (ernie, bert, roberta and so on), tokenizer should be hub.BertTokenizer.
    # else tokenizer should be hub.CustomTokenizer.
    tokenizer = hub.BertTokenizer(
        vocab_file=module.get_vocab_path(), tokenize_chinese_chars=True)

    # Load dataset
    if args.is_pair_wise:
        dataset = hub.dataset.DuEL(
            tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    else:
        dataset = hub.dataset.LCQMC(
            tokenizer=tokenizer, max_seq_len=args.max_seq_len)

    # Construct transfer learning network
    # Use token-level output.
    query = outputs["sequence_output"]
    left = outputs['sequence_output_2']
    right = outputs['sequence_output_3']

    # Select fine-tune strategy
    strategy = hub.DefaultStrategy(
        optimizer_name="sgd", learning_rate=args.learning_rate)

    # Setup RunConfig for PaddleHub Fine-tune API
    config = hub.RunConfig(
        eval_interval=300,
        use_data_parallel=args.use_data_parallel,
        use_cuda=args.use_gpu,
        num_epoch=args.num_epoch,
        batch_size=args.batch_size,
        checkpoint_dir=args.checkpoint_dir,
        strategy=strategy)

    # Define a text matching task by PaddleHub's API
    # network choice: bow, cnn, gru, lstm (PaddleHub pre-defined network)
    matching_task = hub.TextMatchingTask(
        dataset=dataset,
        query_feature=query,
        left_feature=left,
        tokenizer=tokenizer,
        network=args.network,
        is_pair_wise=args.is_pair_wise,
        right_feature=right if args.is_pair_wise else None,
        config=config)

    # Fine-tune and evaluate by PaddleHub's API
    # will finish training, evaluation, testing, save model automatically
    matching_task.finetune_and_eval()
