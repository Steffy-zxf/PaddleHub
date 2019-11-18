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
import numpy as np

from env_vars import trial_env_vars
import trial
import parameter_expressions as param_exp

__all__ = [
    'choice',
    'randint',
    'uniform',
]

# pylint: disable=unused-argument

if trial_env_vars.NNI_PLATFORM is None:

    def choice(*options, name=None):
        return param_exp.choice(options, np.random.RandomState())

    def randint(lower, upper, name=None):
        return param_exp.randint(lower, upper, np.random.RandomState())

    def uniform(low, high, name=None):
        return param_exp.uniform(low, high, np.random.RandomState())

else:

    def choice(options, name=None, key=None):
        return options[_get_param(key)]

    def randint(lower, upper, name=None, key=None):
        return _get_param(key)

    def uniform(low, high, name=None, key=None):
        return _get_param(key)

    def _get_param(key):
        if trial.get_current_parameter() is None:
            trial.get_next_parameter()
        return trial.get_current_parameter(key)
