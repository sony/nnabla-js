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

import nnabla as nn
import nnabla.functions as F
import nnabla.parametric_functions as PF
from nnabla.utils.save import save


def main():
    batch_size = 1

    # input variable
    x0 = nn.Variable((batch_size, 100))

    # model
    h = x0
    for depth in range(4):
        h = PF.affine(h, 100, name=f"affine{depth}")
        h = F.relu(h)
    y0 = PF.affine(h, 10, name=f"affine{depth+1}")

    # nnp contents
    contents = {
        "networks": [
            {"name": "net",
             "batch_size": batch_size,
             "outputs": {"y0": y0},
             "names": {"x0": x0}}],
        "executors": [
            {"name": "runtime",
             "network": "net",
             "data": ["x0"],
             "output": ["y0"]}]
    }

    # save as nnp
    save("test.nnp", contents)


if __name__ == "__main__":
    main()
