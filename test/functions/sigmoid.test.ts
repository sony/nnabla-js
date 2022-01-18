import { GPU } from 'gpu.js';
import Sigmoid from '../../src/functions/sigmoid';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-sigmoid', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const sigmoid = new Sigmoid(new GPU());

  sigmoid.setup([x], [y]);
  sigmoid.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(yData[i], 1 / (1 + Math.exp(-xData[i])), 0.0001);
  }
});
