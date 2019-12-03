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


class BasicTrainingService(object):
    def __init__(self, trial_detail_dict, job_queue, output_dir):
        """
        trial_detail_dict : dict, key:trial_job_id; value: trial_job_detail
        job_queue: Queue, trail job queue
        """
        self.trial_detail_dict = trial_detail_dict
        self.job_queue = job_queue
        self.output_dir = output_dir

    def list_triall_jobs(self):
        raise NotImplementedError

    def get_trial_job(self, trial_job_id):
        raise NotImplementedError

    def submit_trial_job(self, ):
        raise NotImplementedError

    def cancel_trial_job(self, trail_job_id, is_early_stopped):
        raise NotImplementedError

    def run():
        raise NotImplementedError
