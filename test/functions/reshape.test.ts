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
import { ReshapeParameter, Shape } from '../../src/proto/nnabla_pb';
import Reshape from '../../src/functions/reshape';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-reshape', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new ReshapeParameter();
  const shape = new Shape();
  shape.addDim(2);
  shape.addDim(50);
  param.setShape(shape);
  const reshape = new Reshape(param, new GPU());

  reshape.setup([x], [y]);
  reshape.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(yData[i], xData[i], 0.0001);
  }
});
