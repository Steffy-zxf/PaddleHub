import os

import paddle.fluid as fluid
import paddlehub as hub
from paddlehub.model.modeling_ernie import ErnieforSequenceClassification
from paddlehub.finetune.trainer import Trainer
# from paddlehub.tokenizer import BertTokenizer
from paddlehub.datasets.chnsenticorp import ChnSentiCorp

if __name__ == '__main__':
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env().dev_id)
    with fluid.dygraph.guard(place):

        ernie = hub.Module(
            directory=
            '/mnt/zhangxuefei/program-paddle/PaddleHub_wzw/hub_module/modules/text/semantic_model/ernie_dygraph/')
        # tokenizer = BertTokenizer(tokenize_chinese_chars=True, vocab_file=ernie.get_vocab_path())

        train_dataset = ChnSentiCorp(
            tokenizer=ernie.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='train')
        dev_dataset = ChnSentiCorp(
            tokenizer=ernie.get_tokenizer(tokenize_chinese_chars=True), max_seq_len=128, mode='dev')

        model = ErnieforSequenceClassification(ernie_module=ernie, num_classes=train_dataset.num_classes)

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate=5e-5, parameter_list=model.parameters())
        trainer = Trainer(model, optimizer, checkpoint_dir='test_ernie_text_cls')

        trainer.train(train_dataset, epochs=3, batch_size=4, eval_dataset=dev_dataset, log_interval=10, save_interval=1)
