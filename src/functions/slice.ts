import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { SliceParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class Slice implements FunctionImpl {
  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  param: SliceParameter;

  start: number[];

  stop: number[];

  step: number[];

  offsets: number[];

  slicedOffsets: number[];

  constructor(param: SliceParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
    this.start = [];
    this.stop = [];
    this.step = [];
    this.offsets = [];
    this.slicedOffsets = [];
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    this.start = this.param.getStartList();
    this.stop = this.param.getStopList();
    this.step = this.param.getStepList();
    for (let i = 0; i < inputs[0].shape.length; i += 1) {
      if (this.start[i] < 0) {
        throw Error('negative slice is not supported.');
      }
    }
    for (let i = 0; i < inputs[0].shape.length; i += 1) {
      if (this.stop[i] < 0) {
        throw Error('negative slice is not supported.');
      }
    }

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
    const outputShape = outputs[0].shape;
    this.slicedOffsets = [];
    for (let i = 0; i < ndim; i += 1) {
      let size = 1;
      for (let j = i + 1; j < ndim; j += 1) {
        size *= outputShape[j];
      }
      this.slicedOffsets.push(size);
    }

    this.kernel = this.gpu
      .createKernel(function (
        x: number[],
        start: number[],
        step: number[],
        offsets: number[],
        slicedOffsets: number[],
      ): number {
        let index = this.thread.x;
        let originalIndex = 0;
        for (let i = 0; i < (this.constants.ndim as number); i += 1) {
          const loc = Math.floor(index / slicedOffsets[i]);
          const originalLoc = loc * step[i] + start[i];
          originalIndex += originalLoc * offsets[i];
          index %= slicedOffsets[i];
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
    Slice.validate(inputs, outputs);

    const output = this.kernel(
      inputs[0].data,
      this.start,
      this.step,
      this.offsets,
      this.slicedOffsets,
    ) as Texture;
    outputs[0].setData(output);
  }
}
