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
import { ConcatenateParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Concatenate implements FunctionImpl {
  gpu: GPU;

  kernels: IKernelRunShortcut[];

  param: ConcatenateParameter;

  constructor(param: ConcatenateParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernels = [];
  }

  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  setup(inputs: Variable[], outputs: Variable[]): void {
    const axis = this.param.getAxis();
    const sizes = [0];
    for (let i = 0; i < inputs.length; i += 1) {
      sizes.push(inputs[i].size());
    }
    let baseOffset = 1;
    for (let i = axis + 1; i < inputs[0].shape.length; i += 1) {
      baseOffset *= inputs[0].shape[i];
    }
    let accumAxisSize = 0;
    const accumAxisSizes = [0];
    for (let i = 0; i < inputs.length; i += 1) {
      accumAxisSize += inputs[i].shape[axis];
      accumAxisSizes.push(accumAxisSize);
    }

    let accumSize = 0;
    for (let i = 0; i < inputs.length; i += 1) {
      accumSize += sizes[i];
      const kernel = this.gpu
        .createKernel(function (x: number[], y: number[]): number {
          const index = this.thread.x;
          const sumDim = (this.constants.xSize as number) + (this.constants.ySize as number);
          const targetAxis = Math.floor((index % (sumDim * baseOffset)) / baseOffset);
          const batchAxis = Math.floor(index / (sumDim * baseOffset));
          const restAxis = index % baseOffset;
          if (targetAxis < this.constants.xSize) {
            const batchOffset = batchAxis * (this.constants.xSize as number) * baseOffset;
            return x[batchOffset + targetAxis * baseOffset + restAxis];
          }
          const batchOffset = batchAxis * (this.constants.ySize as number) * baseOffset;
          const targetOffset = (targetAxis - (this.constants.xSize as number)) * baseOffset;
          return y[batchOffset + targetOffset + restAxis];
        })
        .setConstants({ xSize: accumAxisSizes[i], ySize: inputs[i].shape[axis], baseOffset })
        .setOutput([accumSize + sizes[i + 1]])
        .setPipeline(true);
      this.kernels.push(kernel);
    }
  }

  static validate(inputs: Variable[], outputs: Variable[]): void {
    if (inputs.length === 0) {
      throw Error(`invalid input length: ${inputs.length}`);
    }
    if (outputs.length !== 1) {
      throw Error(`invalid output length: ${outputs.length}`);
    }
  }

  forward(inputs: Variable[], outputs: Variable[]): void {
    if (this.kernels.length === 0) {
      throw Error('call setup first.');
    }
    Concatenate.validate(inputs, outputs);

    let accum: number[] | Texture = [];
    for (let i = 0; i < inputs.length; i += 1) {
      accum = this.kernels[i](accum, inputs[i].data) as Texture;
    }
    outputs[0].setData(accum);
  }
}
