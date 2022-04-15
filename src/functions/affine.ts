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

import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { AffineParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createMatmulKernel } from './utils';
import Variable from '../variable';

export default class Affine implements FunctionImpl {
  param: AffineParameter;

  gpu: GPU;

  matmulKernel: IKernelRunShortcut | undefined;

  biasKernel: IKernelRunShortcut | undefined;

  constructor(param: AffineParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.matmulKernel = undefined;
    this.biasKernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    const baseAxis = this.param.getBaseAxis();
    const iColSize = inputs[0].shape[baseAxis];
    const iRowSize = inputs[0].size() / iColSize;
    const wRowSize = inputs[1].shape[0];
    const wColSize = inputs[1].size() / wRowSize;

    [this.matmulKernel] = createMatmulKernel(
      this.gpu,
      [iRowSize, iColSize],
      [wRowSize, wColSize],
      false,
      false,
    );
    this.matmulKernel.setPipeline(true);

    if (inputs.length === 3) {
      this.biasKernel = this.gpu
        .createKernel(function (x: number[], b: number[]): number {
          const col = this.thread.x % (this.constants.oColSize as number);
          return x[this.thread.x] + b[col];
        })
        .setConstants({ oColSize: wColSize })
        .setOutput([outputs[0].size()])
        .setPipeline(true);
    }
  }

  static validate(inputs: Variable[], outputs: Variable[]): void {
    if (inputs.length !== 2 && inputs.length !== 3) {
      throw Error(`invalid input length: ${inputs.length}`);
    }
    if (outputs.length !== 1) {
      throw Error(`invalid output length: ${outputs.length}`);
    }
  }

  forward(inputs: Variable[], outputs: Variable[]): void {
    if (this.matmulKernel === undefined) {
      throw Error('call setup first.');
    }
    Affine.validate(inputs, outputs);

    if (!inputs[1].isTexture()) {
      inputs[1].cache(this.gpu);
    }

    let output = this.matmulKernel(inputs[0].data, inputs[1].data) as Texture;

    if (this.biasKernel) {
      if (!inputs[2].isTexture()) {
        inputs[2].cache(this.gpu);
      }
      output = this.biasKernel(output, inputs[2].data) as Texture;
    }
    outputs[0].setData(output);
  }
}
