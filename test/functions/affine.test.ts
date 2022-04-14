// Copyright 2021 Sony Group Corporation.
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
import { AffineParameter } from '../../src/proto/nnabla_pb';
import Affine from '../../src/functions/affine';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

function affineRef(
  x: number[],
  w: number[],
  b: number[],
  xShape: number[],
  wShape: number[],
): number[] {
  const [xRowSize, xColSize] = xShape;
  const [, wColSize] = wShape;
  const output = [...Array(xRowSize * wColSize)].map(() => 0.0);
  for (let i = 0; i < xRowSize; i += 1) {
    for (let j = 0; j < wColSize; j += 1) {
      for (let k = 0; k < xColSize; k += 1) {
        output[i * wColSize + j] += x[i * xColSize + k] * w[k * wColSize + j];
      }
    }
  }
  for (let i = 0; i < xRowSize; i += 1) {
    for (let j = 0; j < wColSize; j += 1) {
      output[i * wColSize + j] += b[j];
    }
  }
  return output;
}

test('test-affine', () => {
  const x = Variable.rand('x', [128, 64]);
  const w = Variable.rand('w', [64, 32]);
  const b = Variable.rand('b', [32]);
  const y = Variable.rand('y', [128, 32]);
  const param = new AffineParameter();
  param.setBaseAxis(1);
  const affine = new Affine(param, new GPU());

  affine.setup([x, w, b], [y]);
  affine.forward([x, w, b], [y]);
  const yData = y.toArray();

  const yRef = affineRef(x.toArray(), w.toArray(), b.toArray(), x.shape, w.shape);
  for (let i = 0; i < yRef.length; i += 1) {
    expectClose(yData[i], yRef[i], 0.00001);
  }
});
