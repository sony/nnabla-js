import { GPU, IKernelRunShortcut } from 'gpu.js';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Add2 implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  constructor(gpu: GPU) {
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(_: Variable[], outputs: Variable[]): void {
    this.kernel = this.gpu
      .createKernel(function (x: number[], y: number[]): number {
        return x[this.thread.x] + y[this.thread.x];
      })
      .setOutput([outputs[0].size()]);
  }

  static validate(inputs: Variable[], outputs: Variable[]): void {
    if (inputs.length !== 2) {
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
    Add2.validate(inputs, outputs);

    const output = this.kernel(inputs[0].data, inputs[1].data) as number[];
    outputs[0].setData(output);
  }
}
