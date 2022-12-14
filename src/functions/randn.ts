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
import { RandnParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Randn implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: RandnParameter;

  constructor(param: RandnParameter, gpu: GPU) {
    this.gpu = gpu;
    this.kernel = undefined;
    this.param = param;
  }

  setup(_: Variable[], outputs: Variable[]): void {
    this.kernel = this.gpu
      .createKernel(function (): number {
        // Sample from uniform distribution [0, 1]
        const a = Math.random();
        const term = 2 * a - 1;
        // Taylor expansion
        let value = term;
        value += (3.14159265 / 12) * term ** 3;
        value += ((7.0 * 3.14159265) / 480) * term ** 5;
        value *= Math.sqrt(2 * 3.14159265) / 2;
        // apply mu and sigma
        return (this.constants.mu as number) + value * (this.constants.sigma as number);
      })
      .setOutput([outputs[0].size()])
      .setConstants({ sigma: this.param.getSigma(), mu: this.param.getMu() })
      .setPipeline(true);
  }

  static validate(inputs: Variable[], outputs: Variable[]): void {
    if (inputs.length !== 0) {
      throw Error(`invalid input length: ${inputs.length}`);
    }
    if (outputs.length !== 1) {
      throw Error(`invalid output length: ${outputs.length}`);
    }
  }

  forward(inputs: Variable[], outputs: Variable[]): void {
    if (this.kernel === undefined) {
      throw Error('call setup first.');
    }
    Randn.validate(inputs, outputs);

    const output = this.kernel() as Texture;
    outputs[0].setData(output);
  }
}
