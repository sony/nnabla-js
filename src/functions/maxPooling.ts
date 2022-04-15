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
import { MaxPoolingParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createIm2ColKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

export default class MaxPooling implements FunctionImpl {
  param: MaxPoolingParameter;

  gpu: GPU;

  im2colKernel: IKernelRunShortcut | undefined;

  im2colShape: number[];

  poolingKernel: IKernelRunShortcut | undefined;

  constructor(param: MaxPoolingParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.im2colKernel = undefined;
    this.im2colShape = [];
    this.poolingKernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getChannelLast()) {
      throw Error('channelLast option is not supported yet.');
    }

    // Apply im2col
    [this.im2colKernel, this.im2colShape] = createIm2ColKernel(
      this.gpu,
      inputs[0].shape,
      getAsArrayOrThrow<number>(this.param.getKernel()?.getDimList()),
      getAsArrayOrThrow<number>(this.param.getStride()?.getDimList()),
      getAsArrayOrThrow<number>(this.param.getPad()?.getDimList()),
    );
    this.im2colKernel.setPipeline(true);
    const [, , K, L] = this.im2colShape;

    this.poolingKernel = this.gpu
      .createKernel(function (x: number[]): number {
        const tK = this.constants.K as number;
        const tL = this.constants.L as number;
        const bcIndex = Math.floor(this.thread.x / tL);
        const lIndex = this.thread.x % tL;
        const index = bcIndex * tK * tL + lIndex;
        let maxValue = x[index];
        for (let i = 1; i < tK; i += 1) {
          if (x[index + i * tL] > maxValue) {
            maxValue = x[index + i * tL];
          }
        }
        return maxValue;
      })
      .setConstants({ K, L })
      .setOutput([outputs[0].size()])
      .setPipeline(true);
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
    if (this.im2colKernel === undefined || this.poolingKernel === undefined) {
      throw Error('call setup first.');
    }
    MaxPooling.validate(inputs, outputs);

    const im2colOutput = this.im2colKernel(inputs[0].data);
    const output = this.poolingKernel(im2colOutput) as Texture;

    outputs[0].setData(output);
  }
}
