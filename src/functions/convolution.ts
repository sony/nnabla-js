import { GPU, IKernelRunShortcut, Texture } from 'gpu.js';
import { ConvolutionParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createBatchMatmulKernel, createIm2ColKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

// TODO: Supports dilation and group
export default class Convolution implements FunctionImpl {
  param: ConvolutionParameter;

  gpu: GPU;

  im2colKernel: IKernelRunShortcut | undefined;

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
    this.im2colKernel.setPipeline(true);
    const [B, C, K, L] = this.im2colShape;

    const wC = inputs[1].shape[0];
    let kernelSize = 1;
    for (let i = 0; i < inputs[1].shape.length; i += 1) {
      kernelSize *= inputs[1].shape[i];
    }

    if (!Array.isArray(inputs[1].data)) {
      throw Error('inputs[1].data must be Array.');
    }

    // Apply batch matmul
    [this.matmulKernel] = createBatchMatmulKernel(
      this.gpu,
      [1, wC, kernelSize / wC],
      [B, C * K, L],
    );
    this.matmulKernel.setPipeline(true);

    // Apply bias
    if (inputs.length === 3) {
      this.biasKernel = this.gpu
        .createKernel(function (x: number[], b: number[]): number {
          const dataSize = (this.constants.C as number) * (this.constants.L as number);
          const col = Math.floor((this.thread.x % dataSize) / (this.constants.L as number));
          return x[this.thread.x] + b[col];
        })
        .setConstants({ C: wC, L })
        .setOutput([outputs[0].size()])
        .setPipeline(true);
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

    if (!inputs[1].isTexture()) {
      inputs[1].cache(this.gpu);
    }

    const im2colOutput = this.im2colKernel(inputs[0].data);
    let output = this.matmulKernel(inputs[1].data, im2colOutput) as Texture;

    if (this.biasKernel) {
      if (!inputs[2].isTexture()) {
        inputs[2].cache(this.gpu);
      }
      output = this.biasKernel(output, inputs[2].data) as Texture;
    }

    outputs[0].setData(output);
  }
}
