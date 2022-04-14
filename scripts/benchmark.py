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
import time

import nnabla as nn
from nnabla.utils.load import load
from nnabla.ext_utils import get_extension_context

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('nnp', type=str)
    parser.add_argument('runtime', type=str)
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    if args.gpu is not None:
        ctx = get_extension_context('cudnn', device_id=args.gpu)
        nn.set_default_context(ctx)
    else:
        ctx = None

    nnp = load(args.nnp, batch_size=1, context=None if ctx is None else f"cudnn:{args.gpu}")

    executor = nnp.executors[args.runtime]

    total_time = 0.0
    for i in range(100):
        start_time = time.time()
        executor.forward_target.forward()
        end_time = time.time()
        if i > 0:
            total_time += end_time - start_time

    print(f"Average time: {total_time / 99.0 * 1000.0}ms")


if __name__ == "__main__":
    main()
