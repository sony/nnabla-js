import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { TransposeParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Transpose implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: TransposeParameter;

  offsets: number[];

  transposedOffsets: number[];

  translateAxes: number[];

  constructor(param: TransposeParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
    this.offsets = [];
    this.transposedOffsets = [];
    this.translateAxes = [];
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    const transposedAxes = this.param.getAxesList();
    const { shape } = inputs[0];
    const ndim = shape.length;
    this.offsets = [];
    for (let i = 0; i < ndim; i += 1) {
      let size = 1;
      for (let j = i + 1; j < ndim; j += 1) {
        size *= shape[j];
      }
      this.offsets.push(size);
    }
    this.transposedOffsets = [];
    for (let i = 0; i < ndim; i += 1) {
      let size = 1;
      for (let j = i + 1; j < ndim; j += 1) {
        size *= shape[transposedAxes[j]];
      }
      this.transposedOffsets.push(size);
    }
    this.translateAxes = [];
    for (let i = 0; i < ndim; i += 1) {
      for (let j = 0; j < ndim; j += 1) {
        if (transposedAxes[j] === i) {
          this.translateAxes.push(j);
          break;
        }
      }
    }

    this.kernel = this.gpu
      .createKernel(function (
        x: number[],
        offsets: number[],
        transposedOffsets: number[],
        translateAxes: number[],
      ): number {
        // locate index
        let index = this.thread.x;
        let originalIndex = 0;
        for (let i = 0; i < (this.constants.ndim as number); i += 1) {
          const offset = transposedOffsets[i];
          const transposedLoc = Math.floor(index / offset);
          const translatedAxis = translateAxes[i];
          const originalOffset = offsets[translatedAxis];
          originalIndex += originalOffset * transposedLoc;
          index %= offset;
        }
        return x[originalIndex];
      })
      .setConstants({ ndim })
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
    Transpose.validate(inputs, outputs);

    const output = this.kernel(
      inputs[0].data,
      this.offsets,
      this.transposedOffsets,
      this.translateAxes,
    ) as Texture;
    outputs[0].setData(output);
  }
}
