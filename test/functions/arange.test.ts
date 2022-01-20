import { GPU } from 'gpu.js';
import { ArangeParameter } from '../../src/proto/nnabla_pb';
import Arange from '../../src/functions/arange';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-arange', () => {
  const y = Variable.rand('y', [100]);
  const param = new ArangeParameter();
  param.setStart(1);
  param.setStep(2);
  param.setStop(200);
  const arange = new Arange(param, new GPU());

  arange.setup([], [y]);
  arange.forward([], [y]);

  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(yData[i], 1.0 + i * 2.0, 0.0001);
  }
});
