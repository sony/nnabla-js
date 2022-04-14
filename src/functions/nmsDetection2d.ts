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

import { NmsDetection2dParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

function computeIOU(coord1: number[], coord2: number[]): number {
  const [x1, y1, w1, h1] = coord1;
  const [x2, y2, w2, h2] = coord2;
  const xMin1 = x1 - w1 / 2;
  const xMax1 = x1 + w1 / 2;
  const yMin1 = y1 - h1 / 2;
  const yMax1 = y1 + h1 / 2;
  const xMin2 = x2 - w2 / 2;
  const xMax2 = x2 + w2 / 2;
  const yMin2 = y2 - h2 / 2;
  const yMax2 = y2 + h2 / 2;

  const area1 = (xMax1 - xMin1 + 0.0001) * (yMax1 - yMin1 + 0.0001);
  const area2 = (xMax2 - xMin2 + 0.0001) * (yMax2 - yMin2 + 0.0001);

  const xMinI = Math.max(xMin1, xMin2);
  const yMinI = Math.max(yMin1, yMin2);
  const xMaxI = Math.min(xMax1, xMax2);
  const yMaxI = Math.min(yMax1, yMax2);
  const wI = Math.max(0.0, xMaxI - xMinI + 0.0001);
  const hI = Math.max(0.0, yMaxI - yMinI + 0.0001);
  const intersection = wI * hI;

  return intersection / (area1 + area2 - intersection);
}

function boundingBoxWiseNms(
  x: number[],
  B: number,
  N: number,
  C: number,
  thresh: number,
  nms: number,
): number[] {
  const y = [...Array(x.length)].map(() => 0.0);
  for (let i = 0; i < B; i += 1) {
    const suppressedIndices: { [key: number]: boolean } = {};
    for (let j = 0; j < N; j += 1) {
      const index = i * N * (C + 5) + j * (C + 5);
      const score = x[index + 4];
      const coord = x.slice(index, index + 4);

      // Copy header
      for (let k = 0; k < 5; k += 1) {
        y[index + k] = x[index + k];
      }

      // Skip score is lower or already suppressed
      if (score < nms || suppressedIndices[j]) {
        suppressedIndices[j] = true;
        continue;
      }

      for (let k = j + 1; k < N; k += 1) {
        if (suppressedIndices[k]) {
          continue;
        }
        const cmpIndex = i * N * (C + 5) + k * (C + 5);
        const cmpScore = x[cmpIndex + 4];
        const cmpCoord = x.slice(cmpIndex, cmpIndex + 4);
        const iou = computeIOU(coord, cmpCoord);

        // Suppress if iou is above threshold and score is lower
        if (iou > thresh) {
          if (score < cmpScore) {
            suppressedIndices[j] = true;
            break;
          }
        }
      }

      // j-th element won the comparison
      if (!suppressedIndices[j]) {
        for (let l = 0; l < C; l += 1) {
          y[index + 5 + l] = x[index + 5 + l];
        }
      }
    }
  }
  return y;
}

function classWiseNms(
  x: number[],
  B: number,
  N: number,
  C: number,
  thresh: number,
  nms: number,
): number[] {
  const y = [...Array(x.length)].map(() => 0.0);
  for (let i = 0; i < B; i += 1) {
    // Copy header first
    for (let j = 0; j < N; j += 1) {
      const index = i * N * (C + 5) + j * (C + 5);
      for (let k = 0; k < 5; k += 1) {
        y[index + k] = x[index + k];
      }
    }

    // Class-wise NMS
    for (let c = 0; c < C; c += 1) {
      const suppressedIndices: { [key: number]: boolean } = {};
      for (let j = 0; j < N; j += 1) {
        const index = i * N * (C + 5) + j * (C + 5);
        const score = x[index + 4] * x[index + 5 + c];
        const coord = x.slice(index, index + 4);

        // Skip score is lower or already suppressed
        if (score < nms || suppressedIndices[j]) {
          suppressedIndices[j] = true;
          continue;
        }

        for (let k = j + 1; k < N; k += 1) {
          if (suppressedIndices[k]) {
            continue;
          }
          const cmpIndex = i * N * (C + 5) + k * (C + 5);
          const cmpScore = x[cmpIndex + 4] * x[cmpIndex + 5 + c];
          const cmpCoord = x.slice(cmpIndex, cmpIndex + 4);
          const iou = computeIOU(coord, cmpCoord);

          // Suppress if iou is above threshold and score is lower
          if (iou > thresh) {
            if (score < cmpScore) {
              suppressedIndices[j] = true;
              break;
            }
          }
        }

        // j-th element won the comparison
        if (!suppressedIndices[j]) {
          y[index + 5 + c] = x[index + 5 + c];
        }
      }
    }
  }
  return y;
}

export default class NmsDetection2d implements FunctionImpl {
  param: NmsDetection2dParameter;

  constructor(param: NmsDetection2dParameter) {
    this.param = param;
  }

  setup(inputs: Variable[], outputs: Variable[]): void { // eslint-disable-line
    // Do nothing
  }

  static validate(inputs: Variable[], outputs: Variable[]): void {
    if (inputs.length !== 1) {
      throw Error(`invalid input length: ${inputs.length}`);
    }
    if (outputs.length !== 1) {
      throw Error(`invalid output length: ${outputs.length}`);
    }
  }

  forward(inputs: Variable[], outputs: Variable[]): void {
    NmsDetection2d.validate(inputs, outputs);

    const nms = this.param.getNms();
    const thresh = this.param.getThresh();

    // Get array on CPU (B, N, 5 + C)
    const data = inputs[0].toArray();
    const [B, N, tmpC] = inputs[0].shape;
    const C = tmpC - 5;

    if (this.param.getNmsPerClass()) {
      const y = classWiseNms(data, B, N, C, thresh, nms);
      outputs[0].setData(y);
    } else {
      const y = boundingBoxWiseNms(data, B, N, C, thresh, nms);
      outputs[0].setData(y);
    }
  }
}
