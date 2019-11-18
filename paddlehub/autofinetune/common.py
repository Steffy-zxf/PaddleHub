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
from datetime import datetime
from io import TextIOBase
import logging
import sys
import time

log_level_map = {
    'fatal': logging.FATAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG
}

_time_format = '%m/%d/%Y, %I:%M:%S %p'


class _LoggerFileWrapper(TextIOBase):
    def __init__(self, logger_file):
        self.file = logger_file

    def write(self, s):
        if s != '\n':
            time = datetime.now().strftime(_time_format)
            self.file.write('[{}] PRINT '.format(time) + s + '\n')
            self.file.flush()
        return len(s)


def init_logger(logger_file_path, log_level_name='info'):
    """Initialize root logger.
    This will redirect anything from logging.getLogger() as well as stdout to specified file.
    logger_file_path: path of logger file (path-like object).
    """
    log_level = log_level_map.get(log_level_name, logging.INFO)
    logger_file = open(logger_file_path, 'w')
    fmt = '[%(asctime)s] %(levelname)s (%(name)s/%(threadName)s) %(message)s'
    logging.Formatter.converter = time.localtime
    formatter = logging.Formatter(fmt, _time_format)
    handler = logging.StreamHandler(logger_file)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # these modules are too verbose
    logging.getLogger('matplotlib').setLevel(log_level)

    sys.stdout = _LoggerFileWrapper(logger_file)


_multi_thread = False


def enable_multi_thread():
    global _multi_thread
    _multi_thread = True


def multi_thread_enabled():
    return _multi_thread
