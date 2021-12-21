import { GPU } from 'gpu.js';
import { MulScalarParameter } from '../../src/proto/nnabla_pb';
import MulScalar from '../../src/functions/mulScalar';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-mulScalar', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new MulScalarParameter();
  param.setVal(2.0);
  const mulScalar = new MulScalar(param, new GPU());

  mulScalar.setup([x], [y]);
  mulScalar.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(yData[i], 2.0 * xData[i], 0.0001);
  }
});
