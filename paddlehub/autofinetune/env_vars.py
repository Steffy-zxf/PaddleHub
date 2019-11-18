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
import os
from collections import namedtuple

_trial_env_var_names = [
    'PLATFORM',
    'EXP_ID',
    'TRIAL_JOB_ID',
    'SYS_DIR',
    'OUTPUT_DIR',
    'TRIAL_SEQ_ID',
]

_dispatcher_env_var_names = [
    'NNI_MODE', 'NNI_CHECKPOINT_DIRECTORY', 'NNI_LOG_DIRECTORY',
    'NNI_LOG_LEVEL', 'NNI_INCLUDE_INTERMEDIATE_RESULTS'
]


def _load_env_vars(env_var_names):
    env_var_dict = {k: os.environ.get(k) for k in env_var_names}
    return namedtuple('EnvVars', env_var_names)(**env_var_dict)


trial_env_vars = _load_env_vars(_trial_env_var_names)

dispatcher_env_vars = _load_env_vars(_dispatcher_env_var_names)
