import { GPU, IKernelRunShortcut } from 'gpu.js';
import { BatchNormalizationParameter } from '../proto/nnabla_pb';
import FunctionImpl from './base';
import Variable from '../variable';

export default class BatchNormalization implements FunctionImpl {
  param: BatchNormalizationParameter;

  gpu: GPU;

  kernel: IKernelRunShortcut | undefined;

  constructor(param: BatchNormalizationParameter, gpu: GPU) {
    this.param = param;
    this.gpu = gpu;
    this.kernel = undefined;
  }

  setup(inputs: Variable[], outputs: Variable[]): void {
    if (this.param.getBatchStat()) {
      throw Error('batch_stat=True is not supported.');
    }

    const noBias = inputs.length < 5;
    const noScale = inputs.length < 4;

    const axis = this.param.getAxesList()[0];
    let spatialSize = 1;
    for (let i = axis + 1; i < inputs[0].shape.length; i += 1) {
      spatialSize *= inputs[0].shape[i];
    }
    const targetAxisSize = inputs[0].shape[axis];

    const inputLen = inputs.length;
    const mean = inputs[inputLen - 2].data;
    const vars = inputs[inputLen - 1].data;
    const beta: number[] = noBias ? [] : (inputs[1].data as number[]);
    const gamma: number[] = noScale ? [] : (inputs[noBias ? 1 : 2].data as number[]);

    this.kernel = this.gpu
      .createKernel(function (x: number[]): number {
        const tSpatialSize = this.constants.spatialSize as number;
        const tTargetAxisSize = this.constants.targetAxisSize as number;
        const tEps = this.constants.eps as number;
        const tNoScale = this.constants.noScale as boolean;
        const tNoBias = this.constants.noBias as boolean;
        const index = Math.floor((this.thread.x % (tSpatialSize * tTargetAxisSize)) / tSpatialSize);
        const stddev = Math.sqrt((this.constants.vars as number[])[index] + tEps);
        const scale = tNoScale ? 1.0 : (this.constants.gamma as number[])[index];
        const bias = tNoBias ? 0.0 : (this.constants.beta as number[])[index];
        return (
          ((x[this.thread.x] - (this.constants.mean as number[])[index]) * scale) / stddev + bias
        );
      })
      .setConstants({
        mean,
        vars,
        beta,
        gamma,
        eps: this.param.getEps(),
        noBias,
        noScale,
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

    const output = this.kernel(inputs[0].data) as number[];

    outputs[0].setData(output);
  }
}
