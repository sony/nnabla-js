import { GPU, IKernelRunShortcut } from 'gpu.js';
import { MaxPoolingParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import { createPadKernel, createIm2ColKernel } from './utils';
import Variable from '../variable';
import { getAsArrayOrThrow } from '../utils';

export default class MaxPooling implements FunctionImpl {
  param: MaxPoolingParameter;

  gpu: GPU;

  padKernel: ((x: number[]) => number[]) | undefined;

  im2colKernel: ((x: number[]) => number[]) | undefined;

  im2colShape: number[];

  poolingKernel: IKernelRunShortcut | undefined;

  constructor(param: MaxPoolingParameter) {
    this.param = param;
    this.gpu = new GPU();
    this.padKernel = undefined;
    this.im2colKernel = undefined;
    this.im2colShape = [];
    this.poolingKernel = undefined;
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
      getAsArrayOrThrow<number>(this.param.getKernel()?.getDimList()),
      getAsArrayOrThrow<number>(this.param.getStride()?.getDimList()),
      false,
    );

    this.poolingKernel = this.gpu
      .createKernel(function (x: number[], K: number): number {
        const index = K * this.thread.x;
        let maxValue = x[index];
        for (let i = 1; i < K; i += 1) {
          if (x[index + i] > maxValue) {
            maxValue = x[index + i];
          }
        }
        return maxValue;
      })
      .setOutput([outputs[0].size()]);
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
    if (
      this.padKernel === undefined ||
      this.im2colKernel === undefined ||
      this.poolingKernel === undefined
    ) {
      throw Error('call setup first.');
    }
    MaxPooling.validate(inputs, outputs);

    const padOutput = this.padKernel(inputs[0].data);
    const im2colOutput = this.im2colKernel(padOutput);
    const output = this.poolingKernel(im2colOutput, this.im2colShape[3]) as number[];

    outputs[0].setData(output);
  }
}
