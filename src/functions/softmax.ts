import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { SoftmaxParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Softmax implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: SoftmaxParameter;

  constructor(param: SoftmaxParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    const { shape } = inputs[0];
    let axis = this.param.getAxis();
    if (axis < 0) {
      axis = shape.length + axis;
    }
    const size = inputs[0].size();
    let sizeAxis = 1;
    for (let i = axis; i < shape.length; i += 1) {
      sizeAxis *= shape[i];
    }
    const size0 = size / sizeAxis; // batch size
    const size1 = shape[axis]; // target axis size
    const size2 = size / size0 / size1; // remaining size

    this.kernel = this.gpu
      .createKernel(function (x: number[]): number {
        const blockSize = (this.constants.size2 as number) * (this.constants.size1 as number);
        const i0 = Math.floor(this.thread.x / blockSize);
        const i2 = this.thread.x % (this.constants.size2 as number);

        // compute max to avoid overflow
        let maxX = x[i0 * (this.constants.size1 as number) * (this.constants.size2 as number) + i2];
        for (let i = 1; i < (this.constants.size1 as number); i += 1) {
          const index =
            (i0 * (this.constants.size1 as number) + i) * (this.constants.size2 as number) + i2;
          if (x[index] > maxX) {
            maxX = x[index];
          }
        }

        let expSum = 0.0;
        for (let i = 0; i < (this.constants.size1 as number); i += 1) {
          const index =
            (i0 * (this.constants.size1 as number) + i) * (this.constants.size2 as number) + i2;
          expSum += Math.exp(x[index] - maxX);
        }

        return Math.exp(x[this.thread.x] - maxX) / expSum;
      })
      .setConstants({ size1, size2 })
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
    Softmax.validate(inputs, outputs);

    const output = this.kernel(inputs[0].data) as Texture;
    outputs[0].setData(output);
  }
}
