import { GPU } from 'gpu.js';
import Tanh from '../../src/functions/tanh';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-tanh', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const tanh = new Tanh(new GPU());

  tanh.setup([x], [y]);
  tanh.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    const exp = Math.exp(xData[i]);
    const negExp = Math.exp(-xData[i]);
    expectClose(yData[i], (exp - negExp) / (exp + negExp), 0.0001);
  }
});
