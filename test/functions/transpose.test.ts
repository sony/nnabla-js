import { GPU } from 'gpu.js';
import { TransposeParameter } from '../../src/proto/nnabla_pb';
import Transpose from '../../src/functions/transpose';
import Variable from '../../src/variable';
import { expectAllClose } from '../testUtils';

function refTranspose(x: number[], shape: number[]): number[] {
  const [B, H, W] = shape;
  const y = [];
  for (let i = 0; i < H; i += 1) {
    for (let j = 0; j < B; j += 1) {
      for (let k = 0; k < W; k += 1) {
        const index = j * H * W + i * W + k;
        y.push(x[index]);
      }
    }
  }
  return y;
}

test('test-transpose', () => {
  const x = Variable.rand('x', [100, 5, 2]);
  const y = Variable.rand('y', [5, 100, 2]);
  const param = new TransposeParameter();
  param.setAxesList([1, 0, 2]);
  const transpose = new Transpose(param, new GPU());

  transpose.setup([x], [y]);
  transpose.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  const refY = refTranspose(xData, [100, 5, 2]);
  expectAllClose(yData, refY, 0.0001);
});
