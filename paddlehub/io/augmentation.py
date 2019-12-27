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

from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import random

from paddlehub.common import utils


def _check_img(img):
    if isinstance(img, str):
        utils.check_path(img)
        img = Image.open(img)
    return img


def image_resize(img, width, height, interpolation_method=Image.LANCZOS):
    img = _check_img(img)
    return img.resize((width, height), interpolation_method)


class RandAugment():
    def __init__(self, trans_number=7, proportion=9):
        self.transforms = [
            'autocontrast', 'equalize', 'rotate', 'solarize', 'color',
            'posterize', 'invert', 'contrast', 'brightness', 'sharpness',
            'shearX', 'shearY', 'translateX', 'translateY'
        ]

        self.trans_number = trans_number
        self.proportion = proportion

        self.trans_ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }
        fillcolor = (128, 128, 128)
        self.func = {
            "shearX":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0
                               ),
                Image.BICUBIC,
                fill=fillcolor),
            "shearY":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0
                               ),
                Image.BICUBIC,
                fill=fillcolor),
            "translateX":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice(
                    [-1, 1]), 0, 1, 0),
                fill=fillcolor),
            "translateY":
            lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.
                               choice([-1, 1])),
                fill=fillcolor),
            "rotate":
            lambda img, magnitude: img.rotate(
                int(magnitude * random.uniform(-1, 1))),
            "color":
            lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "posterize":
            lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize":
            lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast":
            lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness":
            lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness":
            lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast":
            lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize":
            lambda img, magnitude: img,
            "invert":
            lambda img, magnitude: ImageOps.invert(img)
        }

    def __call__(self, image):
        for num in range(self.trans_number):
            index = random.choice(range(len(self.transforms) - 1))
            op_name = self.transforms[index]
            op_name = "color"
            magnitude = self.trans_ranges[op_name][self.proportion]
            image = self.func[op_name](image, magnitude)
            magnitude = self.trans_ranges[op_name][self.proportion]
            image = self.func[op_name](image, magnitude)
        return image
