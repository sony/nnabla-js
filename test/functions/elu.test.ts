import { GPU } from 'gpu.js';
import { ELUParameter } from '../../src/proto/nnabla_pb';
import ELU from '../../src/functions/elu';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-elu', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new ELUParameter();
  param.setAlpha(1.0);
  const elu = new ELU(param, new GPU());

  elu.setup([x], [y]);
  elu.forward([x], [y]);

  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    if (xData[i] > 0) {
      expectClose(yData[i], xData[i], 0.0001);
    } else {
      expectClose(yData[i], Math.exp(xData[i]) - 1.0, 0.0001);
    }
  }
});
