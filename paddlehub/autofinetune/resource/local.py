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
import random
import string
import time
import os
import signal

from trial_config import TrialConfig, TrialJobDetail
from training_service import BasicTrainingService
from paddlehub.common.logger import logger


def unique_string():
    return ''.join(random.sample(string.ascii_letters + string.digits, 8))


class LocalConfig(object):
    def __init__(self, max_gpu_num_per_trial, gpu_indices):
        self.max_gpu_num_per_trial = max_gpu_num_per_trial
        self.gpu_indices = gpu_indices


class LocalTrainingService(BasicTrainingService):
    def __init__(self, trial_detail_dict, job_queue, output_dir, stopping,
                 local_trial_config):
        super(LocalTrainingService, self).__init__(trial_detail_dict, job_queue,
                                                   output_dir)
        """
        Local machine training service
        trial_detail_dict : dict, key:trial_job_id; value: trial_job_detail
        job_queue: Queue, trail job queue
        local_trial_config: LocalConfig
        """
        self.stopping = stopping
        self.local_trial_config = local_trial_config
        self.trials_list = []
        self.trial_detail_dict = {}

    def list_trial_jobs(self):
        return list(self.trial_detail_dict.values())

    def get_trial_job(self, trial_job_id):
        trial_job_detail = self.trial_detail_dict.get(trial_job_id, None)
        if not trial_job_detail:
            logger.error("Trial job not found!")
            return None
        else:
            return trial_job_detail

    def submit_trail_job(self):
        trial_job_id = unique_string()
        working_dir = os.path.join(self.output_dir, 'trials', trial_job_id)
        trial_job_detail = TrialJobDetail(trial_job_id, "WAITING", time.time(),
                                          working_dir)
        self.trials_list.append(trial_job_detail)
        self.trial_detail_dict[trial_job_id] = trial_job_detail
        logger.info("submit trial job {}".format(trial_job_id))

    def cancel_trial_job(self, trial_job_id, is_early_stopped):
        trial_job_detail = self.trial_detail_dict.get(trial_job_id, None)
        if not trial_job_detail:
            logger.error("Trial job not found!")
            return None
        else:
            if not trial_job_detail.pid:
                self.set_trial_job_status(trial_job_detail, "USER_CANCELED")
            else:
                os.kill(trial_job_detail.pid, signal.SIGTERM)
                if is_early_stopped:
                    self.set_trial_job_status(trial_job_detail, "EARLY_STOPPED")
                else:
                    self.set_trial_job_status(trial_job_detail, "SYS_CANCELED")

    def set_trial_job_status(self, trial_job_detail, new_status):
        if trial_job_detail.trial_status != new_status:
            trial_job_detail.trial_status = new_status

    def run(self):
        logger.info("Run local machine training service!")
        self.run_job_loop()
        logger.info("Local machine training service exit.")

    def run_job_loop(self):
        while not self.stopping:
            while not self.stopping and len(self.job_queue) != 0:
                trial_job_id = self.job_queue[0]
                trial_job_detail = self.trial_detail_dict.get(
                    trial_job_id, None)
                if not trial_job_detail:
                    logger.error("Trial job not found!")
                else:
                    pass
