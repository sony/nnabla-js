import { GPU } from 'gpu.js';
import ReLu from '../../src/functions/relu';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-relu', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const relu = new ReLu(new GPU());

  relu.setup([x], [y]);
  relu.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    if (xData[i] > 0) {
      expectClose(yData[i], xData[i], 0.0001);
    } else {
      expect(yData[i]).toBe(0.0);
    }
  }
});
