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

import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { SplitParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Split implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: SplitParameter;

  shape: number[];

  offsets: number[];

  constructor(param: SplitParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
    this.shape = [];
    this.offsets = [];
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    const axis = this.param.getAxis();
    this.shape = inputs[0].shape;
    const ndim = this.shape.length;
    const size = outputs[0].size();
    this.offsets = [];
    for (let i = 0; i < ndim; i += 1) {
      let offset = 1;
      for (let j = i + 1; j < ndim; j += 1) {
        if (j !== axis) {
          offset *= this.shape[j];
        }
      }
      this.offsets.push(offset);
    }
    let baseOffset = 1;
    for (let i = axis + 1; i < ndim; i += 1) {
      baseOffset *= this.shape[i];
    }

    this.kernel = this.gpu
      .createKernel(function (
        x: number[],
        shape: number[],
        offsets: number[],
        index: number,
      ): number {
        let outputIndex = this.thread.x;
        let originalIndex = 0;
        for (let i = 0; i < (this.constants.ndim as number); i += 1) {
          if (i !== (this.constants.axis as number)) {
            const loc = Math.floor(outputIndex / offsets[i]);
            if (i < (this.constants.axis as number)) {
              originalIndex += loc * offsets[i] * shape[this.constants.axis as number];
            } else {
              originalIndex += loc * offsets[i];
            }
            outputIndex %= offsets[i];
          } else {
            originalIndex += index * (this.constants.baseOffset as number);
          }
        }
        return x[originalIndex];
      })
      .setConstants({ ndim, axis, baseOffset })
      .setOutput([size])
      .setPipeline(true);
  }

  static validate(inputs: Variable[], outputs: Variable[]): void {
    if (inputs.length !== 1) {
      throw Error(`invalid input length: ${inputs.length}`);
    }
    if (outputs.length < 1) {
      throw Error(`invalid output length: ${outputs.length}`);
    }
  }

  forward(inputs: Variable[], outputs: Variable[]): void {
    if (this.kernel === undefined) {
      throw Error('call setup first.');
    }
    Split.validate(inputs, outputs);

    for (let i = 0; i < outputs.length; i += 1) {
      const output = this.kernel(inputs[0].data, this.shape, this.offsets, i) as Texture;
      outputs[i].setData(output);
    }
  }
}
