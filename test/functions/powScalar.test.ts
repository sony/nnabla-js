import { PowScalarParameter } from '../../src/proto/nnabla_pb';
import PowScalar from '../../src/functions/powScalar';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-powScalar', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new PowScalarParameter();
  param.setVal(2.0);
  const powScalar = new PowScalar(param);

  powScalar.setup([x], [y]);
  powScalar.forward([x], [y]);

  for (let i = 0; i < 100; i += 1) {
    expectClose(y.data[i], x.data[i] ** 2.0, 0.0001);
  }
});
