import Div2 from '../../src/functions/div2';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-div2', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const z = Variable.rand('z', [100]);
  const div2 = new Div2();

  div2.setup([x, y], [z]);
  div2.forward([x, y], [z]);

  for (let i = 0; i < 100; i += 1) {
    expectClose(z.data[i], x.data[i] / y.data[i], 0.0001);
  }
});
