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


class EvaluatorResult(object):
    Good = True
    Bad = False


class Evaluator(object):
    # pylint: disable=no-self-use,unused-argument

    def evaluate_trial(self, trial_job_id, trial_history):
        """Determines whether a trial should be killed. Must override.
        trial_job_id: identifier of the trial (str).
        trial_history: a list of intermediate result objects.
        Returns AssessResult.Good or AssessResult.Bad.
        """
        raise NotImplementedError('Assessor: assess_trial not implemented')

    def trial_end(self, trial_job_id, success):
        """Invoked when a trial is completed or terminated. Do nothing by default.
        trial_job_id: identifier of the trial (str).
        success: True if the trial successfully completed; False if failed or terminated.
        """
        pass

    def _on_exit(self):
        pass

    def _on_error(self):
        pass
