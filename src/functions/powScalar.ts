import { GPU, IKernelRunShortcut } from 'gpu.js';
import { PowScalarParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class PowScalar implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: PowScalarParameter;

  constructor(param: PowScalarParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(_: Variable[], outputs: Variable[]): void {
    this.kernel = this.gpu
      .createKernel(function (x: number[]): number {
        return x[this.thread.x] ** (this.constants.val as number);
      })
      .setConstants({ val: this.param.getVal() })
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
    PowScalar.validate(inputs, outputs);

    const output = this.kernel(inputs[0].data) as number[];
    outputs[0].setData(output);
  }
}
