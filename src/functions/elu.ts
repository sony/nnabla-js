import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { ELUParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class ELU implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: ELUParameter;

  constructor(param: ELUParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(_: Variable[], outputs: Variable[]): void {
    this.kernel = this.gpu
      .createKernel(function (x: number[]): number {
        const value = x[this.thread.x];
        return value > 0.0 ? value : (this.constants.alpha as number) * (Math.exp(value) - 1);
      })
      .setConstants({ alpha: this.param.getAlpha() })
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
    if (this.kernel === undefined) {
      throw Error('call setup first.');
    }
    ELU.validate(inputs, outputs);

    const output = this.kernel(inputs[0].data) as Texture;
    outputs[0].setData(output);
  }
}
