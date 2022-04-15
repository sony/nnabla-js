// Copyright 2021,2022 Sony Group Corporation.
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
import Pow2 from '../../src/functions/pow2';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-pow2', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const z = Variable.rand('z', [100]);
  const pow2 = new Pow2(new GPU());

  const xData = x.toArray();
  const yData = y.toArray();
  for (let i = 0; i < y.size(); i += 1) {
    xData[i] += 2.0;
    yData[i] += 2.0;
  }

  pow2.setup([x, y], [z]);
  pow2.forward([x, y], [z]);

  const zData = z.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(zData[i], xData[i] ** yData[i], 0.0001);
  }
});
