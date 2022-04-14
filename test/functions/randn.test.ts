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
import { RandnParameter } from '../../src/proto/nnabla_pb';
import Randn from '../../src/functions/randn';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-addScalar', () => {
  const y = Variable.rand('y', [100000]);
  const param = new RandnParameter();
  param.setMu(0.0);
  param.setSigma(1.0);
  const randn = new Randn(param, new GPU());

  randn.setup([], [y]);
  randn.forward([], [y]);
  const yData = y.toArray();

  let sum = 0.0;
  for (let i = 0; i < 100000; i += 1) {
    sum += yData[i];
  }
  const avg = sum / 100000;
  expectClose(avg, 0.0, 0.005);

  let variance = 0.0;
  for (let i = 0; i < 100000; i += 1) {
    variance += (yData[i] - avg) ** 2;
  }
  variance /= 100000;
  expectClose(variance, 0.0, 1.005);
});
