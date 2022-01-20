import { GPU } from 'gpu.js';
import { LeakyReLUParameter } from '../../src/proto/nnabla_pb';
import LeakyReLU from '../../src/functions/leakyRelu';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-leaky-relu', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new LeakyReLUParameter();
  param.setAlpha(0.9);
  const leakyRelu = new LeakyReLU(param, new GPU());

  leakyRelu.setup([x], [y]);
  leakyRelu.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    const xValue = xData[i];
    expectClose(yData[i], 0.9 * Math.min(0, xValue) + Math.max(0, xValue), 0.0001);
  }
});
