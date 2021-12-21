import { GPU } from 'gpu.js';
import Pow2 from '../../src/functions/pow2';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-pow2', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const z = Variable.rand('z', [100]);
  const pow2 = new Pow2(new GPU());

  const xData = x.toArray();
  const yData = y.toArray();
  for (let i = 0; i < y.size(); i += 1) {
    xData[i] += 2.0;
    yData[i] += 2.0;
  }

  pow2.setup([x, y], [z]);
  pow2.forward([x, y], [z]);

  const zData = z.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(zData[i], xData[i] ** yData[i], 0.0001);
  }
});
