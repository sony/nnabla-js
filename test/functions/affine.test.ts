import { GPU } from 'gpu.js';
import { AffineParameter } from '../../src/proto/nnabla_pb';
import Affine from '../../src/functions/affine';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

function affineRef(
  x: number[],
  w: number[],
  b: number[],
  xShape: number[],
  wShape: number[],
): number[] {
  const [xRowSize, xColSize] = xShape;
  const [, wColSize] = wShape;
  const output = [...Array(xRowSize * wColSize)].map(() => 0.0);
  for (let i = 0; i < xRowSize; i += 1) {
    for (let j = 0; j < wColSize; j += 1) {
      for (let k = 0; k < xColSize; k += 1) {
        output[i * wColSize + j] += x[i * xColSize + k] * w[k * wColSize + j];
      }
    }
  }
  for (let i = 0; i < xRowSize; i += 1) {
    for (let j = 0; j < wColSize; j += 1) {
      output[i * wColSize + j] += b[j];
    }
  }
  return output;
}

test('test-affine', () => {
  const x = Variable.rand('x', [128, 64]);
  const w = Variable.rand('w', [64, 32]);
  const b = Variable.rand('b', [32]);
  const y = Variable.rand('y', [128, 32]);
  const param = new AffineParameter();
  param.setBaseAxis(1);
  const affine = new Affine(param, new GPU());

  affine.setup([x, w, b], [y]);
  affine.forward([x, w, b], [y]);
  const yData = y.toArray();

  const yRef = affineRef(x.toArray(), w.toArray(), b.toArray(), x.shape, w.shape);
  for (let i = 0; i < yRef.length; i += 1) {
    expectClose(yData[i], yRef[i], 0.00001);
  }
});
