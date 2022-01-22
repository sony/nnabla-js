import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Sigmoid implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  constructor(gpu: GPU) {
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(_: Variable[], outputs: Variable[]): void {
    this.kernel = this.gpu
      .createKernel(function (x: number[]): number {
        return 1 / (1 + Math.exp(-x[this.thread.x]));
      })
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
    Sigmoid.validate(inputs, outputs);

    const output = this.kernel(inputs[0].data) as Texture;
    outputs[0].setData(output);
  }
}