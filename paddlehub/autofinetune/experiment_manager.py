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

from collections import defaultdict
import json
import importlib
from multiprocessing.dummy import Pool as ThreadPool
from queue import Queue, Empty
import threading

from .constants import ModuleName, ClassName, ClassArgs
from .evaluator import EvaluatorResult
from .env_vars import dispatcher_env_vars
from paddlehub.common.logger import logger
from .protocol import CommandType, send, receive
from .autodl_utils import MetricType


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


# Evaluator global variables
_trial_history = defaultdict(dict)
'''key: trial job ID; value: intermediate results, mapping from sequence number to data'''

_ended_trials = set()
'''trial_job_id of all ended trials.
We need this because Experiment manager may send metrics after reporting a trial ended.
'''

QUEUE_LEN_WARNING_MARK = 20
_worker_fast_exit_on_terminate = True


def _sort_history(history):
    ret = []
    for i, _ in enumerate(history):
        if i in history:
            ret.append(history[i])
        else:
            break
    return ret


# Tuner global variables
_next_parameter_id = 0
_trial_params = {}


def _create_parameter_id():
    global _next_parameter_id  # pylint: disable=global-statement
    _next_parameter_id += 1
    return _next_parameter_id - 1


def _pack_parameter(parameter_id,
                    params,
                    customized=False,
                    trial_job_id=None,
                    parameter_index=None):
    _trial_params[parameter_id] = params
    ret = {
        'parameter_id': parameter_id,
        'parameter_source': 'customized' if customized else 'algorithm',
        'parameters': params
    }
    if trial_job_id is not None:
        ret['trial_job_id'] = trial_job_id
    if parameter_index is not None:
        ret['parameter_index'] = parameter_index
    else:
        ret['parameter_index'] = 0
    return json.dumps(ret)


_multi_thread = False


def enable_multi_thread():
    global _multi_thread
    _multi_thread = True


def multi_thread_enabled():
    return _multi_thread


class ExperimentManager(object):
    def __init__(self, config):
        self.config = config

        self._experiment_name = self.config.get("experimentName", "default")
        self._platform = self.config.get("trainingServicePlatform", None)
        if self._platform == None:
            logger.error("The training plarform hasn't been set, exiting...")
            exit()
        elif self._platform not in ['local', 'remote', ' cluster']:
            logger.error("The training plarform %s is not supported, exiting..."
                         % self._platform)
            exit()
        _search_space_path = self.config.get("searchSpacePath", None)
        if _search_space_path == None:
            logger.error("The search space is None, must be pre-defined.")
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

        if self.config.get("multiThread", False):
            enable_multi_thread()

        if multi_thread_enabled():
            self.pool = ThreadPool()
            self.thread_results = []
        else:
            self.stopping = False
            self.default_command_queue = Queue()
            self.evaluator_command_queue = Queue()
            self.default_worker = threading.Thread(
                target=self.command_queue_worker,
                args=(self.default_command_queue, ))
            self.evaluator_worker = threading.Thread(
                target=self.command_queue_worker,
                args=(self.evaluator_command_queue, ))
            self.default_worker.start()
            self.evaluator_worker.start()
            self.worker_exceptions = []

    def run(self):
        """
        Run the tuner
        """
        logger.info("Start Run Tuner %s " % self.config["tuner"]["TunerName"])
        print("Start Run Tuner %s " % self.config["tuner"]["TunerName"])

        while True:
            command, data = receive()
            print(123)
            print(command, data)
            if data:
                data = json.loads(data)

            if command is None or command is CommandType.Terminate:
                break
            if multi_thread_enabled():
                result = self.pool.map_async(self.process_command_thread,
                                             [(command, data)])
                self.thread_results.append(result)
                if any([
                        thread_result.ready()
                        and not thread_result.successful()
                        for thread_result in self.thread_results
                ]):
                    logger.error('Caught thread exception')
                    break
            else:
                self.enqueue_command(command, data)
                if self.worker_exceptions:
                    break

        logger.info("Tuner %s exiting..." % self.config["tuner"]["TunerName"])

        self.stopping = True
        if multi_thread_enabled():
            self.pool.close()
            self.pool.join()
        else:
            self.default_worker.join()
            self.evaluator_worker.join()

        logger.info("Terminated by Experiment Manager!")

    def command_queue_worker(self, command_queue):
        """
        Process commands in command queues.
        """
        while True:
            try:
                # set timeout to ensure self.stopping is checked periodically
                command, data = command_queue.get(timeout=3)
                try:
                    self.process_command(command, data)
                except Exception as e:
                    logger.error(e)
                    self.worker_exceptions.append(e)
                    break
            except Empty:
                pass
            if self.stopping and (_worker_fast_exit_on_terminate
                                  or command_queue.empty()):
                break

    def enqueue_command(self, command, data):
        """
        Enqueue command into command queues
        """
        if command == CommandType.TrialEnd or (
                command == CommandType.ReportMetricData
                and data['type'] == 'PERIODICAL'):
            self.evaluator_command_queue.put((command, data))
        else:
            self.default_command_queue.put((command, data))

        qsize = self.default_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            logger.warning('default queue length: {}'.format(qsize))

        qsize = self.evaluator_command_queue.qsize()
        if qsize >= QUEUE_LEN_WARNING_MARK:
            logger.warning('evaluator queue length: {}'.format(qsize))

    def process_command_thread(self, request):
        """
        Worker thread to process a command.
        """
        command, data = request
        if multi_thread_enabled():
            try:
                self.process_command(command, data)
            except Exception as e:
                logger.error(str(e))
                raise
        else:
            pass

    def process_command(self, command, data):
        logger.info("process_command: command: [{}], data: [{}]".format(
            command, data))

        command_handlers = {
            # Tuner commands:
            CommandType.Initialize: self.handle_initialize,
            CommandType.RequestTrialJobs: self.handle_request_trial_jobs,

            # Tuner/Evaluator commands:
            CommandType.ReportMetricData: self.handle_report_metric_data,
            CommandType.TrialEnd: self.handle_trial_end,
        }
        if command not in command_handlers:
            raise AssertionError('Unsupported command: {}'.format(command))
        command_handlers[command](data)

    def handle_initialize(self, data):
        """
        Data is search space
        """
        self.tuner.update_search_space(data)
        send(CommandType.Initialized, '')

    def send_trial_callback(self, id, params):
        """
        For tuner to issue trial config when the config is generated
        """
        send(CommandType.NewTrialJob, _pack_parameter(id, params))

    def handle_request_trial_jobs(self, data):
        # data: number or trial jobs
        ids = [_create_parameter_id() for _ in range(data)]
        logger.info("requesting for generating params of {}".format(ids))
        params_list = self.tuner.generate_multiple_parameters(
            ids, st_callback=self.send_trial_callback)

        for i, _ in enumerate(params_list):
            send(CommandType.NewTrialJob, _pack_parameter(
                ids[i], params_list[i]))
        # when parameters is None.
        if len(params_list) < len(ids):
            send(CommandType.NoMoreTrialJobs, _pack_parameter(ids[0], ''))

    def handle_report_metric_data(self, data):
        """
        data: a dict received from nni_manager, which contains:
              - 'parameter_id': id of the trial
              - 'value': metric value reported by nni.report_final_result()
              - 'type': report type, support {'FINAL', 'PERIODICAL'}
        """
        if data['type'] == MetricType.FINAL:
            self._handle_final_metric_data(data)
        elif data['type'] == MetricType.PERIODICAL:
            if self.evaluator is not None:
                self._handle_intermediate_metric_data(data)
        elif data['type'] == MetricType.REQUEST_PARAMETER:
            assert data['trial_job_id'] is not None
            assert data['parameter_index'] is not None
            param_id = _create_parameter_id()
            try:
                param = self.tuner.generate_parameters(
                    param_id, trial_job_id=data['trial_job_id'])
            except Exception as e:
                logger.error(e)
                param = None
            send(
                CommandType.SendTrialJobParameter,
                _pack_parameter(
                    param_id,
                    param,
                    trial_job_id=data['trial_job_id'],
                    parameter_index=data['parameter_index']))
        else:
            raise ValueError('Data type not supported: {}'.format(data['type']))

    def handle_trial_end(self, data):
        """
        data: it has three keys: trial_job_id, event, hyper_params
             - trial_job_id: the id generated by training service
             - event: the job's state
             - hyper_params: the hyperparameters generated and returned by tuner
        """
        trial_job_id = data['trial_job_id']
        _ended_trials.add(trial_job_id)
        if trial_job_id in _trial_history:
            _trial_history.pop(trial_job_id)
            if self.evaluator is not None:
                self.evaluator.trial_end(trial_job_id,
                                         data['event'] == 'SUCCEEDED')
        if self.tuner is not None:
            self.tuner.trial_end(
                json.loads(data['hyper_params'])['parameter_id'],
                data['event'] == 'SUCCEEDED')

    def _handle_final_metric_data(self, data):
        """
        Call tuner to process final results
        """
        id_ = data['parameter_id']
        value = data['value']
        self.tuner.receive_trial_result(id_, _trial_params[id_], value)

    def _handle_intermediate_metric_data(self, data):
        """
        Call evaluator to process intermediate results
        """
        if data['type'] != MetricType.PERIODICAL:
            return
        if self.evaluator is None:
            return

        trial_job_id = data['trial_job_id']
        if trial_job_id in _ended_trials:
            return

        history = _trial_history[trial_job_id]
        history[data['sequence']] = data['value']
        ordered_history = _sort_history(history)
        if len(ordered_history
               ) < data['sequence']:  # no user-visible update since last time
            return

        try:
            result = self.evaluator.evaluate_trial(trial_job_id,
                                                   ordered_history)
        except Exception as e:
            logger.error("Evaluator error [{}]".format(e))

        if isinstance(result, bool):
            result = EvaluatorResult.Good if result else EvaluatorResult.Bad
        elif not isinstance(result, EvaluatorResult):
            raise RuntimeError('The result of Evaluator.evaluate_trial must be \
            an object of EvaluatorResult, not {}'.format(type(result)))

        if result is EvaluatorResult.Bad:
            logger.info("BAD EvaluatorResult, kill trail job id: {}".format(
                trial_job_id))
            send(CommandType.KillTrialJob, json.dumps(trial_job_id))
            # notify tuner
            logger.info('env var: NNI_INCLUDE_INTERMEDIATE_RESULTS \
            [{}]'.format(dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS))

            if dispatcher_env_vars.NNI_INCLUDE_INTERMEDIATE_RESULTS == 'true':
                self._earlystop_notify_tuner(data)
        else:
            logger.info(
                "Goood EvaluatorResult, trail job id: {}".format(trial_job_id))

    def _earlystop_notify_tuner(self, data):
        """
        Send last intermediate result as final result to tuner in case the
        trial is early stopped.
        """
        logger.info("Early stop notify tuner data: [{}]".format(data))
        data['type'] = MetricType.FINAL
        if multi_thread_enabled():
            self._handle_final_metric_data(data)
        else:
            self.enqueue_command(CommandType.ReportMetricData, data)
