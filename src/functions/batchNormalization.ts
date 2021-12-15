import { GPU, IKernelRunShortcut } from 'gpu.js';
import { BatchNormalizationParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class BatchNormalization implements FunctionImpl {
  param: BatchNormalizationParameter;

  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  noBias: boolean;

  noScale: boolean;

  spatialSize: number;

  targetAxisSize: number;

  constructor(param: BatchNormalizationParameter) {
    this.param = param;
    this.gpu = new GPU();
    this.kernel = undefined;
    this.noBias = false;
    this.noScale = false;
    this.spatialSize = 0;
    this.targetAxisSize = 0;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getBatchStat()) {
      throw Error('batch_stat=True is not supported.');
    }

    this.noBias = inputs.length < 5;
    this.noScale = inputs.length < 4;

    const axis = this.param.getAxesList()[0];
    this.spatialSize = 1;
    for (let i = axis + 1; i < inputs[0].shape.length; i += 1) {
      this.spatialSize *= inputs[0].shape[i];
    }
    this.targetAxisSize = inputs[0].shape[axis];

    this.kernel = this.gpu
      .createKernel(function (
        x: number[],
        mean: number[],
        vars: number[],
        beta: number[],
        gamma: number[],
        eps: number,
        noBias: boolean,
        noScale: boolean,
        spatialSize: number,
        targetAxisSize: number,
      ): number {
        const index = Math.floor((this.thread.x % (spatialSize * targetAxisSize)) / spatialSize);
        const stddev = Math.sqrt(vars[index] + eps);
        const scale = noScale ? 1.0 : gamma[index];
        const bias = noBias ? 0.0 : beta[index];
        return ((x[this.thread.x] - mean[index]) * scale) / stddev + bias;
      })
      .setOutput([outputs[0].size()]);
  }

  static validate(inputs: Variable[], outputs: Variable[]): void {
    if (inputs.length !== 3 && inputs.length !== 4 && inputs.length !== 5) {
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
    BatchNormalization.validate(inputs, outputs);

    const inputLen = inputs.length;
    const beta: number[] = this.noBias ? [] : inputs[1].data;
    const gamma: number[] = this.noScale ? [] : inputs[this.noBias ? 1 : 2].data;

    const output = this.kernel(
      inputs[0].data,
      inputs[inputLen - 2].data,
      inputs[inputLen - 1].data,
      beta,
      gamma,
      this.param.getEps(),
      this.noBias,
      this.noScale,
      this.spatialSize,
      this.targetAxisSize,
    ) as number[];

    outputs[0].setData(output);
  }
}
