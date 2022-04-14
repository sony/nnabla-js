# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
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

import argparse

import numpy as np
import nnabla as nn
from nnabla.utils import nnp_graph
from nnabla.ext_utils import get_extension_context
from nnabla.utils.image_utils import imread


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nnp', type=str)
    parser.add_argument('image', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        ctx = get_extension_context('cudnn', device_id=args.gpu)
        nn.set_default_context(ctx)
    else:
        ctx = None

    # load image file
    image = np.reshape(imread(args.image, grayscale=True), [1, 1, 28, 28])

    # build network
    nnp = nnp_graph.NnpLoader(args.nnp)
    network = nnp.get_network("net", batch_size=1)

    x = network.inputs["x0"]
    y = network.outputs["y0"]

    def callback(func):
        print(func)
        print(np.array(func.outputs[0].d))

    # predict class
    x.d = image
    y.forward(function_post_hook=callback)
    print(y.d.argmax())


if __name__ == "__main__":
    main()
