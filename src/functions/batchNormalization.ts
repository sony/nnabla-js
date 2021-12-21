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

  constructor(param: BatchNormalizationParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
    this.noBias = false;
    this.noScale = false;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getBatchStat()) {
      throw Error('batch_stat=True is not supported.');
    }

    this.noBias = inputs.length < 5;
    this.noScale = inputs.length < 4;

    const axis = this.param.getAxesList()[0];
    let spatialSize = 1;
    for (let i = axis + 1; i < inputs[0].shape.length; i += 1) {
      spatialSize *= inputs[0].shape[i];
    }
    const targetAxisSize = inputs[0].shape[axis];

    this.kernel = this.gpu
      .createKernel(function (
        x: number[],
        mean: number[],
        vars: number[],
        beta: number[],
        gamma: number[],
      ): number {
        const tSpatialSize = this.constants.spatialSize as number;
        const tTargetAxisSize = this.constants.targetAxisSize as number;
        const tEps = this.constants.eps as number;
        const tNoScale = this.constants.noScale as boolean;
        const tNoBias = this.constants.noBias as boolean;
        const index = Math.floor((this.thread.x % (tSpatialSize * tTargetAxisSize)) / tSpatialSize);
        const stddev = Math.sqrt(vars[index] + tEps);
        const scale = tNoScale ? 1.0 : gamma[index];
        const bias = tNoBias ? 0.0 : beta[index];
        return ((x[this.thread.x] - mean[index]) * scale) / stddev + bias;
      })
      .setConstants({
        eps: this.param.getEps(),
        noBias: this.noBias,
        noScale: this.noScale,
        spatialSize,
        targetAxisSize,
      })
      .setOutput([outputs[0].size()])
      .setPipeline(true);
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
    const betaIndex: number = this.noBias ? -1 : 1;
    const gammaIndex: number = this.noScale ? -1 : this.noBias ? 1 : 2;

    if (!inputs[inputLen - 2].isTexture()) {
      inputs[inputLen - 2].cache(this.gpu);
    }

    if (!inputs[inputLen - 1].isTexture()) {
      inputs[inputLen - 1].cache(this.gpu);
    }

    if (betaIndex > -1 && !inputs[betaIndex].isTexture()) {
      inputs[betaIndex].cache(this.gpu);
    }

    if (gammaIndex > -1 && !inputs[gammaIndex].isTexture()) {
      inputs[gammaIndex].cache(this.gpu);
    }

    const mean = inputs[inputLen - 2].data;
    const vars = inputs[inputLen - 1].data;
    const beta = betaIndex > -1 ? inputs[betaIndex].data : [];
    const gamma = gammaIndex > -1 ? inputs[gammaIndex].data : [];

    const output = this.kernel(inputs[0].data, mean, vars, beta, gamma) as number[];

    outputs[0].setData(output);
  }
}
