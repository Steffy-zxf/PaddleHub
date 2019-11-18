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

import json
import importlib

from .constants import ModuleName, ClassName, ClassArgs


def augment_classargs(input_class_args, classname):
    if classname in ClassArgs:
        for key, value in ClassArgs[classname].items():
            if key not in input_class_args:
                input_class_args[key] = value
    return input_class_args


def create_builtin_class_instance(classname, class_args={}, is_advisor=False):
    if classname not in ModuleName or \
        importlib.util.find_spec(ModuleName[classname]) is None:
        raise RuntimeError('Tuner module is not found: {}'.format(classname))
    class_module = importlib.import_module(ModuleName[classname])
    class_constructor = getattr(class_module, ClassName[classname])
    if class_args:
        class_args = augment_classargs(class_args, classname)
    else:
        class_args = augment_classargs({}, classname)

    if class_args:
        instance = class_constructor(**class_args)
    else:
        instance = class_constructor()
    return instance


class ExperimentManager(object):
    def __init__(self, config):
        self.config = config

        self._experiment_name = self.config.get("experimentName", "default")
        self._platform = self.config.get("trainingServicePlatform", None)
        assert self._platform != None
        _search_space_path = self.config.get("searchSpacePath", None)
        assert _search_space_path != None
        with open(_search_space_path, "r") as f:
            self.search_space = json.load(f)

        self.tuner = None
        self.evaluator = None
        assert self.config.get("tuner", None) != None
        _tuner_name = self.config["tuner"]["TunerName"]
        _tuner_args = self.config["tuner"]["classArgs"]
        if _tuner_name in ModuleName:
            self.tuner = create_builtin_class_instance(_tuner_name, _tuner_args)
        if self.config.get("evaluator", None):
            _evaluator_name = self.config["evaluator"]["EvaluatorName"]
            _evaluator_args = self.config["evaluator"]["classArgs"]
            if _evaluator_name in ModuleName:
                self.evaluator = create_builtin_class_instance(
                    _evaluator_name, _evaluator_args)
