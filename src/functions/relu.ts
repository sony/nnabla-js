import { GPU, IKernelRunShortcut } from 'gpu.js';
import FunctionImpl from './base';
import Variable from '../variable';

export default class ReLu implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  constructor() {
    this.gpu = new GPU();
    this.kernel = undefined;
  }

  setup(_: Variable[], outputs: Variable[]): void {
    this.kernel = this.gpu
      .createKernel(function (x: number[]): number {
        const value = x[this.thread.x];
        return value > 0.0 ? value : 0.0;
      })
      .setOutput([outputs[0].size()]);
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
    ReLu.validate(inputs, outputs);

    const output = this.kernel(inputs[0].data) as number[];
    outputs[0].setData(output);
  }
}
