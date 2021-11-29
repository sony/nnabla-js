import Sub2 from '../../src/functions/sub2';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-sub2', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const z = Variable.rand('z', [100]);
  const sub2 = new Sub2();

  sub2.setup([x, y], [z]);
  sub2.forward([x, y], [z]);

  for (let i = 0; i < 100; i += 1) {
    expectClose(z.data[i], x.data[i] - y.data[i], 0.0001);
  }
});
