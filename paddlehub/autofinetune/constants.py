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
ModuleName = {
    'random':
    'paddlehub.autofinetune.hyperopt_tuner.hyperopt_tuner',
    'pbt':
    'paddlehub.auotifinetune.pbt_tuner',
    'medianstop':
    'paddlehub.autofinetune.medianstop_evaluator.medianstop_evaluator',
}

ClassName = {
    'random': 'HyperoptTuner',
    'pbt': 'PopulationBasedTrainingTuner',
    'medianstop': "MedianstopEvaluator",
}

ClassArgs = {
    'tpe': {
        'algorithm_name': 'tpe'
    },
    'random': {
        'algorithm_name': 'random_search'
    },
    'anneal': {
        'algorithm_name': 'anneal'
    }
}
