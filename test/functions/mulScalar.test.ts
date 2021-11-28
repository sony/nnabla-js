import { MulScalarParameter } from '../../src/proto/nnabla_pb';
import MulScalar from '../../src/functions/mulScalar';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-mulScalar', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new MulScalarParameter();
  param.setVal(2.0);
  const mulScalar = new MulScalar(param);

  mulScalar.setup([x], [y]);
  mulScalar.forward([x], [y]);

  for (let i = 0; i < 100; i += 1) {
    expectClose(y.data[i], x.data[i] / 2.0, 0.0001);
  }
});
