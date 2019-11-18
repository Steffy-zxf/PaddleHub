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


def choice(options, random_state):
    '''
    options: 1-D array-like or int
    random_state: an object of numpy.random.RandomState
    '''
    return random_state.choice(options)


def randint(lower, upper, random_state):
    '''
    Generate a random integer from `lower` (inclusive) to `upper` (exclusive).
    lower: an int that represent an lower bound
    upper: an int that represent an upper bound
    random_state: an object of numpy.random.RandomState
    '''
    return random_state.randint(lower, upper)


def uniform(low, high, random_state):
    '''
    low: an float that represent an lower bound
    high: an float that represent an upper bound
    random_state: an object of numpy.random.RandomState
    '''
    assert high > low, 'Upper bound must be larger than lower bound'
    return random_state.uniform(low, high)
