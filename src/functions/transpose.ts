import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { TransposeParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Transpose implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: TransposeParameter;

  constructor(param: TransposeParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    const transposedAxes = this.param.getAxesList();
    const { shape } = inputs[0];
    const ndim = shape.length;
    const offsets = [];
    for (let i = 0; i < ndim; i += 1) {
      let size = 1;
      for (let j = i + 1; j < ndim; j += 1) {
        size *= shape[j];
      }
      offsets.push(size);
    }
    const transposedOffsets = [];
    for (let i = 0; i < ndim; i += 1) {
      let size = 1;
      for (let j = i + 1; j < ndim; j += 1) {
        size *= shape[transposedAxes[j]];
      }
      transposedOffsets.push(size);
    }
    const translateAxes = [];
    for (let i = 0; i < ndim; i += 1) {
      for (let j = 0; j < ndim; j += 1) {
        if (transposedAxes[j] === i) {
          translateAxes.push(j);
          break;
        }
      }
    }

    this.kernel = this.gpu
      .createKernel(function (x: number[]): number {
        // locate index
        let index = this.thread.x;
        let originalIndex = 0;
        for (let i = 0; i < (this.constants.ndim as number); i += 1) {
          const offset = (this.constants.transposedOffsets as number[])[i];
          const transposedLoc = Math.floor(index / offset);
          const originalOffset = (this.constants.offsets as number[])[
            (this.constants.translateAxes as number[])[i]
          ];
          originalIndex += originalOffset * transposedLoc;
          index %= offset;
        }
        return x[originalIndex];
      })
      .setConstants({ offsets, transposedOffsets, translateAxes, ndim })
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

    const output = this.kernel(inputs[0].data) as Texture;
    outputs[0].setData(output);
  }
}
