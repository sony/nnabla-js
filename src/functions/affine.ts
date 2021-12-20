import { GPU, IKernelRunShortcut } from 'gpu.js';
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
      null,
      inputs[1].data,
    );

    if (inputs.length === 3) {
      this.biasKernel = this.gpu
        .createKernel(function (x: number[], oColSize: number): number {
          const col = this.thread.x % oColSize;
          return x[this.thread.x] + (this.constants.bias as number[])[col];
        })
        .setConstants({ bias: inputs[2].data })
        .setOutput([outputs[0].size()]);
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

    let output = this.matmulKernel(inputs[0].data) as number[];

    if (this.biasKernel) {
      output = this.biasKernel(output, outputs[0].shape[1]) as number[];
    }
    outputs[0].setData(output);
  }
}
