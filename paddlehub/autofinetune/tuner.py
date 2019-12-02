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


class Tuner(object):
    # pylint: disable=no-self-use,unused-argument

    def generate_parameters(self, parameter_id, **kwargs):
        """Returns a set of trial (hyper-)parameters, as a serializable object.
        User code must override either this function or 'generate_multiple_parameters()'.
        parameter_id: int
        """
        raise NotImplementedError('Tuner: generate_parameters not implemented')

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        """Invoked when a trial reports its final result. Must override.
        By default this only reports results of algorithm-generated hyper-parameters.
        Use `accept_customized_trials()` to receive results from user-added parameters.
        parameter_id: int
        parameters: object created by 'generate_parameters()'
        value: object reported by trial
        """
        raise NotImplementedError('Tuner: receive_trial_result not implemented')

    def trial_end(self, parameter_id, success, **kwargs):
        """Invoked when a trial is completed or terminated. Do nothing by default.
        parameter_id: int
        success: True if the trial successfully completed; False if failed or terminated
        """
        pass

    def update_search_space(self, search_space):
        """Update the search space of tuner. Must override.
        search_space: JSON object
        """
        raise NotImplementedError('Tuner: update_search_space not implemented')

    def _on_exit(self):
        pass

    def _on_error(self):
        pass
