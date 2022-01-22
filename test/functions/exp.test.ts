import { GPU } from 'gpu.js';
import Exp from '../../src/functions/exp';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-exp', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const exp = new Exp(new GPU());

  exp.setup([x], [y]);
  exp.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(yData[i], Math.exp(xData[i]), 0.0001);
  }
});
