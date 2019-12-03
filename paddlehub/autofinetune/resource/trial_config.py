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


class TrialConfig(object):
    """
    Trial job configuration class
    Representing trial job configurable properties
    """

    def __init__(self, command, code_dir, gpu):
        """
        command: Trail command
        code_dir: Code directory
        gpu: Required GPU number for a trial job
        """
        self.command = command
        self.code_dir = code_dir
        self.gpu = gpu


class TrialJobDetail(object):
    def __init__(self,
                 trial_id,
                 trial_status,
                 submit_time,
                 working_dir,
                 start_time=None,
                 end_time=None,
                 is_early_stopped=False):
        self.trial_id = trial_id
        self.trial_status = trial_status
        self.submit_time = submit_time
        self.start_time = start_time
        self.end_time = end_time
        self.working_dir = working_dir
        self.is_early_stopped = is_early_stopped
