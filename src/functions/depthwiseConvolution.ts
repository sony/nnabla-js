import { GPU, IKernelRunShortcut } from 'gpu.js';
import { DepthwiseConvolutionParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createPadKernel, createIm2ColKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

// TODO: Supports dilation and group
export default class DepthwiseConvolution implements FunctionImpl {
  param: DepthwiseConvolutionParameter;

  gpu: GPU;

  padKernel: ((x: number[]) => number[]) | undefined;

  im2colKernel: ((x: number[]) => number[]) | undefined;

  im2colShape: number[];

  convKernel: IKernelRunShortcut | undefined;

  constructor(param: DepthwiseConvolutionParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.padKernel = undefined;
    this.im2colKernel = undefined;
    this.im2colShape = [];
    this.convKernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getMultiplier() !== 1) {
      throw Error('multiplier=1 is only supported now.');
    }

    // Apply padding
    const [padKernel, padShape] = createPadKernel(
      this.gpu,
      inputs[0].shape,
      getAsArrayOrThrow<number>(this.param.getPad()?.getDimList()),
    );
    this.padKernel = padKernel;

    // Apply im2col
    [this.im2colKernel, this.im2colShape] = createIm2ColKernel(
      this.gpu,
      padShape,
      inputs[1].shape.slice(1),
      getAsArrayOrThrow<number>(this.param.getStride()?.getDimList()),
      false,
    );

    // Spatial convolution
    this.convKernel = this.gpu
      .createKernel(function (x: number[], w: number[], C: number, L: number, K: number): number {
        let index = this.thread.x;
        const bIndex = Math.floor(index / (C * L));
        index -= bIndex * C * L;
        const cIndex = Math.floor(index / L);
        let value = 0.0;
        for (let i = 0; i < K; i += 1) {
          value += x[K * this.thread.x + i] * w[cIndex * K + i];
        }
        return value;
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
    if (
      this.padKernel === undefined ||
      this.im2colKernel === undefined ||
      this.convKernel === undefined
    ) {
      throw Error('call setup first.');
    }
    DepthwiseConvolution.validate(inputs, outputs);

    const padOutput = this.padKernel(inputs[0].data);
    const im2colOutput = this.im2colKernel(padOutput);
    const [, C, L, K] = this.im2colShape;
    const output = this.convKernel(im2colOutput, inputs[1].data, C, L, K) as number[];

    outputs[0].setData(output);
  }
}
