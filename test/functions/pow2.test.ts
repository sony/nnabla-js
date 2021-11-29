import Pow2 from '../../src/functions/pow2';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-pow2', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const z = Variable.rand('z', [100]);
  const pow2 = new Pow2();

  for (let i = 0; i < y.size(); i += 1) {
    x.data[i] += 2.0;
    y.data[i] += 2.0;
  }

  pow2.setup([x, y], [z]);
  pow2.forward([x, y], [z]);

  for (let i = 0; i < 100; i += 1) {
    expectClose(z.data[i], x.data[i] ** y.data[i], 0.0001);
  }
});
