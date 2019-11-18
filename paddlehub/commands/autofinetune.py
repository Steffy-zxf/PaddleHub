# coding:utf-8
# Copyright (c) 2019  PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import json
import os
import sys
import ast

import six
import shutil
import pandas
import numpy as np
import yaml

from paddlehub.autofinetune.experiment_manager import ExperimentManager
from paddlehub.commands.base_command import BaseCommand, ENTRY
from paddlehub.common.arg_helper import add_argument, print_arguments
from paddlehub.common.logger import logger


class AutoFineTuneCommand(BaseCommand):
    name = "autofinetune"

    def __init__(self, name):
        super(AutoFineTuneCommand, self).__init__(name)
        self.show_in_help = True
        self.name = name
        self.description = "PaddleHub helps to finetune a task by searching hyperparameters automatically."
        self.parser = argparse.ArgumentParser(
            description=self.__class__.__doc__,
            prog='%s %s --config <the path of experiment config file in yaml>' %
            (ENTRY, self.name),
            usage='%(prog)s',
            add_help=False)
        self.module = None

    def add_config_file_arg(self):
        self.arg_config_group.add_argument(
            "--config",
            type=str,
            default=None,
            required=True,
            help="the path of experiment config file in yaml")

    def convert_to_other_options(self, config_list):
        if len(config_list) % 2 != 0:
            raise ValueError(
                "Command for finetuned task options config format error! Please check it: {}"
                .format(config_list))
        options_str = ""
        for key, value in zip(config_list[0::2], config_list[1::2]):
            options_str += "--" + key + "=" + value + " "
        return options_str

    def execute(self, argv):
        if not argv:
            self.help()
            return False

        self.parser.prog = '%s %s' % (ENTRY, self.name)
        self.arg_config_group = self.parser.add_argument_group(
            title="Config options",
            description=
            "Autofinetune configuration about an experiment, required")

        self.add_config_file_arg()
        self.args = self.parser.parse_args(argv)
        if not os.path.exists(self.args.config):
            logger.error("The config file %s doesn't exist." % self.args.config)
        else:
            with io.open(self.args.config, 'r', encoding='utf8') as f:
                self.config = yaml.safe_load(f)
                self.exp_manager = ExperimentManager(self.config)

                print(self.config)

        return True


command = AutoFineTuneCommand.instance()
