import { GPU, IKernelRunShortcut } from 'gpu.js';
import { DepthwiseConvolutionParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createIm2ColKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

// TODO: Supports dilation and group
export default class DepthwiseConvolution implements FunctionImpl {
  param: DepthwiseConvolutionParameter;

  gpu: GPU;

  im2colKernel: ((x: number[]) => number[]) | undefined;

  im2colShape: number[];

  convKernel: IKernelRunShortcut | undefined;

  constructor(param: DepthwiseConvolutionParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.im2colKernel = undefined;
    this.im2colShape = [];
    this.convKernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getMultiplier() !== 1) {
      throw Error('multiplier=1 is only supported now.');
    }

    [this.im2colKernel, this.im2colShape] = createIm2ColKernel(
      this.gpu,
      inputs[0].shape,
      inputs[1].shape.slice(1),
      getAsArrayOrThrow<number>(this.param.getStride()?.getDimList()),
      getAsArrayOrThrow<number>(this.param.getPad()?.getDimList()),
    );
    const [, C, K, L] = this.im2colShape;

    // Spatial convolution
    // (B, C, K, L) -> (B, C, L)
    this.convKernel = this.gpu
      .createKernel(function (x: number[]): number {
        const tC = this.constants.C as number;
        const tK = this.constants.K as number;
        const tL = this.constants.L as number;
        const bIndex = Math.floor(this.thread.x / (tC * tL));
        const cIndex = Math.floor((this.thread.x % (tC * tL)) / tL);
        const lIndex = this.thread.x % tL;
        let value = 0.0;
        for (let i = 0; i < tK; i += 1) {
          const xIndex = bIndex * tC * tL * tK + cIndex * tK * tL + i * tL + lIndex;
          const wIndex = cIndex * tK + i;
          value += x[xIndex] * (this.constants.w as number[])[wIndex];
        }
        return value;
      })
      .setConstants({
        w: inputs[1].data,
        C,
        K,
        L,
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
    if (this.im2colKernel === undefined || this.convKernel === undefined) {
      throw Error('call setup first.');
    }
    DepthwiseConvolution.validate(inputs, outputs);

    const im2colOutput = this.im2colKernel(inputs[0].data);
    const output = this.convKernel(im2colOutput) as number[];

    outputs[0].setData(output);
  }
}
