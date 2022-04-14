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
import { TransposeParameter } from '../../src/proto/nnabla_pb';
import Transpose from '../../src/functions/transpose';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function refTranspose2d(x: number[], shape: number[]): number[] {
  const [B, H] = shape;
  const y = [];
  for (let i = 0; i < H; i += 1) {
    for (let j = 0; j < B; j += 1) {
      const index = j * H + i;
      y.push(x[index]);
    }
  }
  return y;
}

function refTranspose3d(x: number[], shape: number[]): number[] {
  const [B, H, W] = shape;
  const y = [];
  for (let i = 0; i < H; i += 1) {
    for (let j = 0; j < B; j += 1) {
      for (let k = 0; k < W; k += 1) {
        const index = j * H * W + i * W + k;
        y.push(x[index]);
      }
    }
  }
  return y;
}

test('test-transpose2d', () => {
  const x = Variable.rand('x', [100, 5]);
  const y = Variable.rand('y', [5, 100]);
  const param = new TransposeParameter();
  param.setAxesList([1, 0]);
  const transpose = new Transpose(param, new GPU());

  transpose.setup([x], [y]);
  transpose.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  const refY = refTranspose2d(xData, [100, 5]);
  expectAllClose(yData, refY, 0.0001);
});

test('test-transpose3d', () => {
  const x = Variable.rand('x', [100, 5, 2]);
  const y = Variable.rand('y', [5, 100, 2]);
  const param = new TransposeParameter();
  param.setAxesList([1, 0, 2]);
  const transpose = new Transpose(param, new GPU());

  transpose.setup([x], [y]);
  transpose.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  const refY = refTranspose3d(xData, [100, 5, 2]);
  expectAllClose(yData, refY, 0.0001);
});
