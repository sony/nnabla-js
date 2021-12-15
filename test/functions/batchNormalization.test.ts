import { BatchNormalizationParameter } from '../../src/proto/nnabla_pb';
import BatchNormalization from '../../src/functions/batchNormalization';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function batchNormalizationRef(
  x: number[],
  shape: number[],
  mean: number[],
  vars: number[],
  beta: number[],
  gamma: number[],
  eps: number,
): number[] {
  const [B, C, H, W] = shape;
  const y = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < C; j += 1) {
      for (let k = 0; k < H; k += 1) {
        for (let l = 0; l < W; l += 1) {
          const stddev = Math.sqrt(vars[j] + eps);
          const value = x[i * C * H * W + j * H * W + k * W + l];
          y.push(((value - mean[j]) * gamma[j]) / stddev + beta[j]);
        }
      }
    }
  }
  return y;
}

test('test-batch-normalization', () => {
  const x = Variable.rand('x', [32, 3, 28, 28]);
  const mean = Variable.rand('mean', [1, 3, 1, 1]);
  const vars = Variable.rand('var', [1, 3, 1, 1]);
  const beta = Variable.rand('beta', [1, 3, 1, 1]);
  const gamma = Variable.rand('gamma', [1, 3, 1, 1]);
  const y = Variable.rand('y', [32, 3, 28, 28]);
  const param = new BatchNormalizationParameter();
  param.addAxes(1);
  param.setEps(0.0001);
  const bn = new BatchNormalization(param);

  for (let i = 0; i < vars.size(); i += 1) {
    vars.data[i] += 1.0;
  }

  bn.setup([x, beta, gamma, mean, vars], [y]);
  bn.forward([x, beta, gamma, mean, vars], [y]);

  const yRef = batchNormalizationRef(
    x.data,
    x.shape,
    mean.data,
    vars.data,
    beta.data,
    gamma.data,
    0.0001,
  );
  expectAllClose(y.data, yRef, 0.00001);
});
