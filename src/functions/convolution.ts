import { GPU, IKernelRunShortcut } from 'gpu.js';
import { ConvolutionParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createBatchMatmulKernel, createIm2ColKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

// TODO: Supports dilation and group
export default class Convolution implements FunctionImpl {
  param: ConvolutionParameter;

  gpu: GPU;

  im2colKernel: ((x: number[]) => number[]) | undefined;

  im2colShape: number[];

  biasKernel: IKernelRunShortcut | undefined;

  matmulKernel: IKernelRunShortcut | undefined;

  constructor(param: ConvolutionParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.matmulKernel = undefined;
    this.im2colKernel = undefined;
    this.im2colShape = [];
    this.biasKernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getChannelLast()) {
      throw Error('channelLast option is not supported yet.');
    }

    // Apply im2col
    [this.im2colKernel, this.im2colShape] = createIm2ColKernel(
      this.gpu,
      inputs[0].shape,
      inputs[1].shape.slice(2),
      getAsArrayOrThrow<number>(this.param.getStride()?.getDimList()),
      getAsArrayOrThrow<number>(this.param.getPad()?.getDimList()),
    );
    const [B, C, K, L] = this.im2colShape;

    const wC = inputs[1].shape[0];
    let kernelSize = 1;
    for (let i = 0; i < inputs[1].shape.length; i += 1) {
      kernelSize *= inputs[1].shape[i];
    }

    // Apply batch matmul
    [this.matmulKernel] = createBatchMatmulKernel(
      this.gpu,
      [1, wC, kernelSize / wC],
      [B, C * K, L],
      inputs[1].data,
      null,
    );

    // Apply bias
    if (inputs.length === 3) {
      this.biasKernel = this.gpu
        .createKernel(function (x: number[]): number {
          const dataSize = (this.constants.C as number) * (this.constants.L as number);
          const col = Math.floor((this.thread.x % dataSize) / (this.constants.L as number));
          return x[this.thread.x] + (this.constants.bias as number[])[col];
        })
        .setConstants({ bias: inputs[2].data, C: inputs[1].shape[0], L })
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
    if (this.im2colKernel === undefined || this.matmulKernel === undefined) {
      throw Error('call setup first.');
    }
    Convolution.validate(inputs, outputs);

    const im2colOutput = this.im2colKernel(inputs[0].data);
    let output = this.matmulKernel(im2colOutput) as number[];

    if (this.biasKernel) {
      output = this.biasKernel(output) as number[];
    }

    outputs[0].setData(output);
  }
}
