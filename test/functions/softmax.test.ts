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
import { SoftmaxParameter } from '../../src/proto/nnabla_pb';
import Softmax from '../../src/functions/softmax';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function refSoftmax1(x: number[], shape: number[]): number[] {
  const [B, H] = shape;
  const y = [];
  for (let i = 0; i < B; i += 1) {
    let expSum = 0.0;
    for (let j = 0; j < H; j += 1) {
      expSum += Math.exp(x[i * H + j]);
    }
    for (let j = 0; j < H; j += 1) {
      y.push(Math.exp(x[i * H + j]) / expSum);
    }
  }
  return y;
}

function refSoftmax2(x: number[], shape: number[]): number[] {
  const [B, H, W] = shape;
  const y = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < H; j += 1) {
      for (let k = 0; k < W; k += 1) {
        let expSum = 0.0;
        for (let l = 0; l < H; l += 1) {
          expSum += Math.exp(x[i * H * W + l * W + k]);
        }
        y.push(Math.exp(x[i * H * W + j * W + k]) / expSum);
      }
    }
  }
  return y;
}

test('test-softmax1', () => {
  const x = Variable.rand('x', [100, 5]);
  const y = Variable.rand('y', [100, 5]);
  const param = new SoftmaxParameter();
  param.setAxis(1);
  const softmax = new Softmax(param, new GPU());

  softmax.setup([x], [y]);
  softmax.forward([x], [y]);
  const xData = x.toArray();
  const yData = y.toArray();

  const refY = refSoftmax1(xData, [100, 5]);

  expectAllClose(yData, refY, 0.0001);
});

test('test-softmax2', () => {
  const x = Variable.rand('x', [100, 5, 10]);
  const y = Variable.rand('y', [100, 5, 10]);
  const param = new SoftmaxParameter();
  param.setAxis(1);
  const softmax = new Softmax(param, new GPU());

  softmax.setup([x], [y]);
  softmax.forward([x], [y]);
  const xData = x.toArray();
  const yData = y.toArray();

  const refY = refSoftmax2(xData, [100, 5, 10]);

  expectAllClose(yData, refY, 0.0001);
});
