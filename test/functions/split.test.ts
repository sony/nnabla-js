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
import { SplitParameter } from '../../src/proto/nnabla_pb';
import Split from '../../src/functions/split';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function refSplitAxis0(x: number[], shape: number[]): number[][] {
  const [B, H, W] = shape;
  const ys = [];
  for (let i = 0; i < B; i += 1) {
    const y = [];
    for (let j = 0; j < H; j += 1) {
      for (let k = 0; k < W; k += 1) {
        const index = i * H * W + j * W + k;
        y.push(x[index]);
      }
    }
    ys.push(y);
  }
  return ys;
}

function refSplitAxis1(x: number[], shape: number[]): number[][] {
  const [B, H, W] = shape;
  const ys = [];
  for (let i = 0; i < H; i += 1) {
    const y = [];
    for (let j = 0; j < B; j += 1) {
      for (let k = 0; k < W; k += 1) {
        const index = j * H * W + i * W + k;
        y.push(x[index]);
      }
    }
    ys.push(y);
  }
  return ys;
}

test('test-split-axis-0', () => {
  const x = Variable.rand('x', [3, 2, 1]);
  const y0 = Variable.rand('y0', [2, 1]);
  const y1 = Variable.rand('y1', [2, 1]);
  const y2 = Variable.rand('y2', [2, 1]);
  const param = new SplitParameter();
  param.setAxis(0);
  const split = new Split(param, new GPU());

  split.setup([x], [y0, y1, y2]);
  split.forward([x], [y0, y1, y2]);

  const xData = x.toArray();
  const y0Data = y0.toArray();
  const y1Data = y1.toArray();
  const y2Data = y2.toArray();

  const [refY0, refY1, refY2] = refSplitAxis0(xData, [3, 2, 1]);
  expectAllClose(y0Data, refY0, 0.0001);
  expectAllClose(y1Data, refY1, 0.0001);
  expectAllClose(y2Data, refY2, 0.0001);
});

test('test-split-axis-1', () => {
  const x = Variable.rand('x', [3, 2, 1]);
  const y0 = Variable.rand('y0', [3, 1]);
  const y1 = Variable.rand('y1', [3, 1]);
  const param = new SplitParameter();
  param.setAxis(1);
  const split = new Split(param, new GPU());

  split.setup([x], [y0, y1]);
  split.forward([x], [y0, y1]);

  const xData = x.toArray();
  const y0Data = y0.toArray();
  const y1Data = y1.toArray();

  const [refY0, refY1] = refSplitAxis1(xData, [3, 2, 1]);
  expectAllClose(y0Data, refY0, 0.0001);
  expectAllClose(y1Data, refY1, 0.0001);
});
