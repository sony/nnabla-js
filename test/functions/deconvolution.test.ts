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
import { DeconvolutionParameter, Shape } from '../../src/proto/nnabla_pb';
import Deconvolution from '../../src/functions/deconvolution';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function deconvolutionRef(
  x: number[],
  w: number[],
  b: number[],
  xShape: number[],
  wShape: number[],
  stride: number[],
  outShape: number[],
): number[] {
  const [B, oC, oH, oW] = xShape;
  const [, C, wH, wW] = wShape;
  const [sH, sW] = stride;
  const [, , H, W] = outShape;
  const output = [...Array(B * C * H * W)].map(() => 0.0);

  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < oC; j += 1) {
      for (let k = 0; k < oH; k += 1) {
        for (let l = 0; l < oW; l += 1) {
          for (let m = 0; m < wH; m += 1) {
            for (let n = 0; n < wW; n += 1) {
              for (let c = 0; c < C; c += 1) {
                const hi = k * sH + m;
                const wi = l * sW + n;
                const ci = c;
                const yIndex = i * C * H * W + ci * H * W + hi * W + wi;
                const wIndex = j * C * wH * wW + ci * wH * wW + m * wW + n;
                const xIndex = i * oC * oH * oW + j * oH * oW + k * oW + l;
                output[yIndex] += w[wIndex] * x[xIndex];
              }
            }
          }
        }
      }
    }
  }

  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let k = 0; k < H; k += 1) {
        for (let l = 0; l < W; l += 1) {
          const index = i * C * H * W + j * H * W + k * W + l;
          output[index] += b[j];
        }
      }
    }
  }

  return output;
}

test('test-deconvolution', () => {
  const x = Variable.rand('x', [32, 16, 13, 13]);
  const w = Variable.rand('w', [16, 1, 4, 4]);
  const b = Variable.rand('b', [1]);
  const y = Variable.rand('y', [32, 1, 28, 28]);

  const param = new DeconvolutionParameter();
  const pad = new Shape();
  pad.addDim(0);
  pad.addDim(0);
  param.setPad(pad);
  const stride = new Shape();
  stride.addDim(2);
  stride.addDim(2);
  param.setStride(stride);

  const deconv = new Deconvolution(param, new GPU());

  deconv.setup([x, w, b], [y]);
  deconv.forward([x, w, b], [y]);
  const yData = y.toArray();

  const yRef = deconvolutionRef(
    x.toArray(),
    w.toArray(),
    b.toArray(),
    x.shape,
    w.shape,
    [2, 2],
    y.shape,
  );
  expectAllClose(yData, yRef, 0.0001);
});
