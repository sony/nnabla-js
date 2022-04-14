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
import { SliceParameter } from '../../src/proto/nnabla_pb';
import Slice from '../../src/functions/slice';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function refSlice(x: number[], shape: number[], start: number[], stop: number[]): number[] {
  const [, H, W] = shape;
  const [sB, sH, sW] = start;
  const [eB, eH, eW] = stop;
  const y = [];
  for (let i = sB; i < eB; i += 1) {
    for (let j = sH; j < eH; j += 1) {
      for (let k = sW; k < eW; k += 1) {
        const index = i * H * W + j * W + k;
        y.push(x[index]);
      }
    }
  }
  return y;
}

test('test-slice', () => {
  const x = Variable.rand('x', [100, 10, 5]);
  const y = Variable.rand('y', [100, 5, 1]);
  const param = new SliceParameter();
  param.setStartList([0, 5, 0]);
  param.setStopList([100, 10, 1]);
  param.setStepList([1, 1, 1]);
  const slice = new Slice(param, new GPU());

  slice.setup([x], [y]);
  slice.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  const refY = refSlice(xData, [100, 10, 5], [0, 5, 0], [100, 10, 1]);
  expectAllClose(yData, refY, 0.0001);
});
