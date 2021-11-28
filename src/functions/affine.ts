import { GPU, IKernelRunShortcut } from 'gpu.js';
import { AffineParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Affine implements FunctionImpl {
  param: AffineParameter;

  gpu: GPU;

  matmulKernel: IKernelRunShortcut | undefined;

  biasKernel: IKernelRunShortcut | undefined;

  iColSize: number;

  iRowSize: number;

  wColSize: number;

  wRowSize: number;

  oColSize: number;

  oRowSize: number;

  constructor(param: AffineParameter) {
    this.param = param;
    this.gpu = new GPU();
    this.matmulKernel = undefined;
    this.biasKernel = undefined;
    this.iColSize = 1;
    this.iRowSize = 1;
    this.wColSize = 1;
    this.wRowSize = 1;
    this.oColSize = 1;
    this.oRowSize = 1;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    const baseAxis = this.param.getBaseAxis();
    this.iColSize = inputs[0].shape[baseAxis];
    this.iRowSize = inputs[0].size() / this.iColSize;
    [this.wRowSize] = inputs[1].shape;
    this.wColSize = inputs[1].size() / this.wRowSize;
    this.oRowSize = this.iRowSize;
    this.oColSize = this.wColSize;

    this.matmulKernel = this.gpu
      .createKernel(function (
        x: number[],
        w: number[],
        xColSize: number,
        wColSize: number,
      ): number {
        const xRow = Math.floor(this.thread.x / wColSize);
        const wCol = this.thread.x % wColSize;
        let output = 0.0;
        for (let i = 0; i < xColSize; i += 1) {
          output += x[xRow * xColSize + i] * w[i * wColSize + wCol];
        }
        return output;
      })
      .setOutput([outputs[0].size()]);

    if (inputs.length === 3) {
      this.biasKernel = this.gpu
        .createKernel(function (x: number[], b: number[], oColSize: number): number {
          const col = this.thread.x % oColSize;
          return x[this.thread.x] + b[col];
        })
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

    let output = this.matmulKernel(
      inputs[0].data,
      inputs[1].data,
      this.iColSize,
      this.wColSize,
    ) as number[];

    if (this.biasKernel) {
      output = this.biasKernel(output, inputs[2].data, this.oColSize) as number[];
    }
    outputs[0].setData(output);
  }
}
