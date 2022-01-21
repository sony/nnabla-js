import { GPU } from 'gpu.js';
import { ConcatenateParameter } from '../../src/proto/nnabla_pb';
import Concatenate from '../../src/functions/concatenate';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function refConfatenateAxis2(
  x: number[],
  y: number[],
  xShape: number[],
  yShape: number[],
): number[] {
  const [B, H, xW] = xShape;
  const [, , yW] = yShape;
  const z = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < H; j += 1) {
      for (let k = 0; k < xW + yW; k += 1) {
        if (k >= xW) {
          const index = i * H * yW + j * yW + (k - xW);
          z.push(y[index]);
        } else {
          const index = i * H * xW + j * xW + k;
          z.push(x[index]);
        }
      }
    }
  }
  return z;
}

function refConfatenateAxis1(
  x: number[],
  y: number[],
  xShape: number[],
  yShape: number[],
): number[] {
  const [B, xH, W] = xShape;
  const [, yH] = yShape;
  const z = [];
  for (let i = 0; i < B; i += 1) {
    for (let j = 0; j < xH + yH; j += 1) {
      for (let k = 0; k < W; k += 1) {
        if (j >= xH) {
          const index = i * yH * W + (j - xH) * W + k;
          z.push(y[index]);
        } else {
          const index = i * xH * W + j * W + k;
          z.push(x[index]);
        }
      }
    }
  }
  return z;
}

test('test-concatenate-axis-2', () => {
  const x0 = Variable.rand('x0', [3, 2, 2]);
  const x1 = Variable.rand('x1', [3, 2, 1]);
  const y = Variable.rand('y', [3, 2, 3]);
  const param = new ConcatenateParameter();
  param.setAxis(2);
  const concatenate = new Concatenate(param, new GPU());

  concatenate.setup([x0, x1], [y]);
  concatenate.forward([x0, x1], [y]);

  const x0Data = x0.toArray();
  const x1Data = x1.toArray();
  const yData = y.toArray();
  const refY = refConfatenateAxis2(x0Data, x1Data, [3, 2, 2], [3, 2, 1]);
  expectAllClose(yData, refY, 0.0001);
});

test('test-concatenate-axis-1', () => {
  const x0 = Variable.rand('x0', [3, 2, 2]);
  const x1 = Variable.rand('x1', [3, 1, 2]);
  const y = Variable.rand('y', [3, 3, 2]);
  const param = new ConcatenateParameter();
  param.setAxis(1);
  const concatenate = new Concatenate(param, new GPU());

  concatenate.setup([x0, x1], [y]);
  concatenate.forward([x0, x1], [y]);

  const x0Data = x0.toArray();
  const x1Data = x1.toArray();
  const yData = y.toArray();
  const refY = refConfatenateAxis1(x0Data, x1Data, [3, 2, 2], [3, 1, 2]);
  expectAllClose(yData, refY, 0.0001);
});
