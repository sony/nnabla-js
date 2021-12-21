import { GPU } from 'gpu.js';
import Add2 from '../../src/functions/add2';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-add2', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const z = Variable.rand('z', [100]);
  const add2 = new Add2(new GPU());

  add2.setup([x, y], [z]);
  add2.forward([x, y], [z]);

  const xData = x.toArray();
  const yData = y.toArray();
  const zData = z.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(zData[i], xData[i] + yData[i], 0.0001);
  }
});
