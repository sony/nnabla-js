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
import { ArangeParameter } from '../../src/proto/nnabla_pb';
import Arange from '../../src/functions/arange';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-arange', () => {
  const y = Variable.rand('y', [100]);
  const param = new ArangeParameter();
  param.setStart(1);
  param.setStep(2);
  param.setStop(200);
  const arange = new Arange(param, new GPU());

  arange.setup([], [y]);
  arange.forward([], [y]);

  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(yData[i], 1.0 + i * 2.0, 0.0001);
  }
});
