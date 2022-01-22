import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { ArangeParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Arange implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: ArangeParameter;

  constructor(param: ArangeParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(_: Variable[], outputs: Variable[]): void {
    this.kernel = this.gpu
      .createKernel(function (): number {
        const delta = this.thread.x * (this.constants.step as number);
        return (this.constants.start as number) + delta;
      })
      .setConstants({ start: this.param.getStart(), step: this.param.getStep() })
      .setOutput([outputs[0].size()])
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
    Arange.validate(inputs, outputs);

    const output = this.kernel() as Texture;
    outputs[0].setData(output);
  }
}