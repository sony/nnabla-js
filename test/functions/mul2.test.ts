import { GPU } from 'gpu.js';
import Mul2 from '../../src/functions/mul2';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-mul2', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const z = Variable.rand('z', [100]);
  const mul2 = new Mul2(new GPU());

  mul2.setup([x, y], [z]);
  mul2.forward([x, y], [z]);

  const xData = x.toArray();
  const yData = y.toArray();
  const zData = z.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(zData[i], xData[i] * yData[i], 0.0001);
  }
});
