import { GPU, IKernelRunShortcut } from 'gpu.js';
import { ConvolutionParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createMatmulKernel, createPadKernel, createIm2ColKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

// TODO: Supports dilation and group
export default class Convolution implements FunctionImpl {
  param: ConvolutionParameter;

  gpu: GPU;

  padKernel: ((x: number[]) => number[]) | undefined;

  im2colKernel: ((x: number[]) => number[]) | undefined;

  im2colShape: number[];

  biasKernel: IKernelRunShortcut | undefined;

  matmulKernel: ((x: number[], y: number[]) => number[]) | undefined;

  transposeKernel: IKernelRunShortcut | undefined;

  constructor(param: ConvolutionParameter) {
    this.param = param;
    this.gpu = new GPU();
    this.matmulKernel = undefined;
    this.padKernel = undefined;
    this.im2colKernel = undefined;
    this.im2colShape = [];
    this.biasKernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getChannelLast()) {
      throw Error('channelLast option is not supported yet.');
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
      inputs[1].shape,
      getAsArrayOrThrow<number>(this.param.getStride()?.getDimList()),
    );
    const [B, L, C, K] = this.im2colShape;

    const wC = inputs[1].shape[0];
    let kernelSize = 1;
    for (let i = 0; i < inputs[1].shape.length; i += 1) {
      kernelSize *= inputs[1].shape[i];
    }

    // Apply matmul
    [this.matmulKernel] = createMatmulKernel(
      this.gpu,
      [B * L, C * K],
      [wC, kernelSize / wC],
      false,
      true,
    );

    // Apply bias
    if (inputs.length === 3) {
      this.biasKernel = this.gpu
        .createKernel(function (x: number[], b: number[], oColSize: number): number {
          const col = this.thread.x % oColSize;
          return x[this.thread.x] + b[col];
        })
        .setOutput([outputs[0].size()]);
    }

    // Apply transpose
    // (B, L, wC) -> (B, wC, L)
    this.transposeKernel = this.gpu
      .createKernel(function (x: number[], shape: number[]): number {
        const [, _L, _wC] = shape;
        let index = this.thread.x;
        const bIndex = Math.floor(index / (_wC * _L));
        index -= bIndex * _wC * _L;
        const wcIndex = Math.floor(index / _L);
        const lIndex = index % _L;
        return x[bIndex * _L * _wC + lIndex * _wC + wcIndex];
      })
      .setOutput([outputs[0].size()]);
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
    if (
      this.padKernel === undefined ||
      this.im2colKernel === undefined ||
      this.matmulKernel === undefined ||
      this.transposeKernel === undefined
    ) {
      throw Error('call setup first.');
    }
    Convolution.validate(inputs, outputs);

    const padOutput = this.padKernel(inputs[0].data);
    const im2colOutput = this.im2colKernel(padOutput);
    let matmulOutput = this.matmulKernel(im2colOutput, inputs[1].data);

    if (this.biasKernel) {
      matmulOutput = this.biasKernel(matmulOutput, inputs[2].data, inputs[1].shape[0]) as number[];
    }

    const output = this.transposeKernel(matmulOutput, [
      this.im2colShape[0],
      this.im2colShape[1],
      inputs[1].shape[0],
    ]) as number[];

    outputs[0].setData(output);
  }
}
