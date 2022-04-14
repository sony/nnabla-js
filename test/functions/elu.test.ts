// Copyright 2022 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import { GPU } from 'gpu.js';
import { ELUParameter } from '../../src/proto/nnabla_pb';
import ELU from '../../src/functions/elu';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-elu', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new ELUParameter();
  param.setAlpha(1.0);
  const elu = new ELU(param, new GPU());

  elu.setup([x], [y]);
  elu.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    if (xData[i] > 0) {
      expectClose(yData[i], xData[i], 0.0001);
    } else {
      expectClose(yData[i], Math.exp(xData[i]) - 1.0, 0.0001);
    }
  }
});
