import { GPU } from 'gpu.js';
import { AddScalarParameter } from '../../src/proto/nnabla_pb';
import AddScalar from '../../src/functions/addScalar';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-addScalar', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new AddScalarParameter();
  param.setVal(2.0);
  const addScalar = new AddScalar(param, new GPU());

  addScalar.setup([x], [y]);
  addScalar.forward([x], [y]);
  const xData = x.toArray();
  const yData = y.toArray();

  for (let i = 0; i < 100; i += 1) {
    expectClose(yData[i], xData[i] + 2.0, 0.0001);
  }
});
