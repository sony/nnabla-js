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

  for (let i = 0; i < 100; i += 1) {
    if (x.data[i] > 0) {
      expectClose(y.data[i], x.data[i], 0.0001);
    } else {
      expect(y.data[i]).toBe(0.0);
    }
  }
});
