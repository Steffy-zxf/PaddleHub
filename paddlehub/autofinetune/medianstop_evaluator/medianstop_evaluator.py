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
from paddlehub.autofinetune.evaluator import Evaluator, EvaluatorResult
from paddlehub.common.logger import logger


class MedianstopEvaluator(Evaluator):
    """MedianstopAssessor is The median stopping rule stops a pending trial X at step S
    if the trial’s best objective value by step S is strictly worse than the median value
    of the running averages of all completed trials’ objectives reported up to step S

    Parameters
    ----------
    optimize_mode: str
        optimize mode, 'maximize' or 'minimize'
    start_step: int
        only after receiving start_step number of reported intermediate results
    """

    def __init__(self, optimize_mode='maximize', start_step=0):
        self.start_step = start_step
        self.running_history = dict()
        self.completed_avg_history = dict()
        if optimize_mode == 'maximize':
            self.high_better = True
        elif optimize_mode == 'minimize':
            self.high_better = False
        else:
            self.high_better = True
            logger.warning('unrecognized optimize_mode', optimize_mode)

    def _update_data(self, trial_job_id, trial_history):
        """update data

        Parameters
        ----------
        trial_job_id: int
            trial job id
        trial_history: list
            The history performance matrix of each trial
        """
        if trial_job_id not in self.running_history:
            self.running_history[trial_job_id] = []
        self.running_history[trial_job_id].extend(
            trial_history[len(self.running_history[trial_job_id]):])

    def trial_end(self, trial_job_id, success):
        """trial_end

        Parameters
        ----------
        trial_job_id: int
            trial job id
        success: bool
            True if succssfully finish the experiment, False otherwise
        """
        if trial_job_id in self.running_history:
            if success:
                cnt = 0
                history_sum = 0
                self.completed_avg_history[trial_job_id] = []
                for each in self.running_history[trial_job_id]:
                    cnt += 1
                    history_sum += each
                    self.completed_avg_history[trial_job_id].append(
                        history_sum / cnt)
            self.running_history.pop(trial_job_id)
        else:
            logger.warning(
                'trial_end: trial_job_id does not exist in running_history')

    def evaluate_trial(self, trial_job_id, trial_history):
        """evaluate_trial

        Parameters
        ----------
        trial_job_id: int
            trial job id
        trial_history: list
            The history performance matrix of each trial

        Returns
        -------
        bool
            EvaluatorResult.Good or EvaluatorResult.Bad

        Raises
        ------
        Exception
            unrecognize exception in medianstop_assessor
        """
        curr_step = len(trial_history)
        if curr_step < self.start_step:
            return EvaluatorResult.Good

        try:
            num_trial_history = [float(ele) for ele in trial_history]
        except (TypeError, ValueError) as error:
            logger.warning('incorrect data type or value:')
            logger.exception(error)
        except Exception as error:
            logger.warning('unrecognized exception in medianstop_assessor:')
            logger.exception(error)

        self._update_data(trial_job_id, num_trial_history)
        if self.high_better:
            best_history = max(trial_history)
        else:
            best_history = min(trial_history)

        avg_array = []
        for id in self.completed_avg_history:
            if len(self.completed_avg_history[id]) >= curr_step:
                avg_array.append(self.completed_avg_history[id][curr_step - 1])
        if len(avg_array) > 0:
            avg_array.sort()
            if self.high_better:
                median = avg_array[(len(avg_array) - 1) // 2]
                return EvaluatorResult.Bad if best_history < median else EvaluatorResult.Good
            else:
                median = avg_array[len(avg_array) // 2]
                return EvaluatorResult.Bad if best_history > median else EvaluatorResult.Good
        else:
            return EvaluatorResult.Good
