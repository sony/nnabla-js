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
import { DeconvolutionParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createBatchMatmulKernel, createCol2ImKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

// TODO: Supports dilation and group
export default class Deconvolution implements FunctionImpl {
  param: DeconvolutionParameter;

  gpu: GPU;

  matmulKernel: IKernelRunShortcut | undefined;

  matmulShape: number[];

  col2imKernel: IKernelRunShortcut | undefined;

  col2imShape: number[];

  biasKernel: IKernelRunShortcut | undefined;

  constructor(param: DeconvolutionParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.matmulKernel = undefined;
    this.matmulShape = [];
    this.col2imKernel = undefined;
    this.col2imShape = [];
    this.biasKernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getChannelLast()) {
      throw Error('channelLast option is not supported yet.');
    }

    const [B, C, H, W] = inputs[0].shape;
    const [, outC, kH, kW] = inputs[1].shape;
    const [, , outH, outW] = outputs[0].shape;

    // matrix multiplication
    // x: (C, C', kH, kW)
    // y: (B, C, H, W)
    // (1, C' x kH x kW, C) x (B, C, H x W) -> (B, C' x kH x kW, H x W)
    [this.matmulKernel, this.matmulShape] = createBatchMatmulKernel(
      this.gpu,
      [1, C, outC * kH * kW],
      [B, C, H * W],
      true,
      false,
    );
    this.matmulKernel.setPipeline(true);

    // Apply col2im
    // (B, C', kH x kW, H x W) -> (B, C', H', W')
    [this.col2imKernel, this.col2imShape] = createCol2ImKernel(
      this.gpu,
      [B, outC, kH * kW, H * W],
      [H, W],
      [outH, outW],
      [kH, kW],
      getAsArrayOrThrow<number>(this.param.getStride()?.getDimList()),
      getAsArrayOrThrow<number>(this.param.getPad()?.getDimList()),
    );
    this.col2imKernel.setPipeline(true);

    // Apply bias
    if (inputs.length === 3) {
      this.biasKernel = this.gpu
        .createKernel(function (x: number[], b: number[]): number {
          const dataSize = (this.constants.C as number) * (this.constants.L as number);
          const col = Math.floor((this.thread.x % dataSize) / (this.constants.L as number));
          return x[this.thread.x] + b[col];
        })
        .setConstants({ C: outC, L: outH * outW })
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
    if (this.col2imKernel === undefined || this.matmulKernel === undefined) {
      throw Error('call setup first.');
    }
    Deconvolution.validate(inputs, outputs);

    if (!inputs[1].isTexture()) {
      inputs[1].cache(this.gpu);
    }

    const matmulOutput = this.matmulKernel(inputs[1].data, inputs[0].data);
    let output = this.col2imKernel(matmulOutput) as Texture;

    if (this.biasKernel) {
      if (!inputs[2].isTexture()) {
        inputs[2].cache(this.gpu);
      }
      output = this.biasKernel(output, inputs[2].data) as Texture;
    }

    outputs[0].setData(output);
  }
}
