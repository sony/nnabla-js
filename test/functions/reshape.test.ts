import { ReshapeParameter, Shape } from '../../src/proto/nnabla_pb';
import Reshape from '../../src/functions/reshape';
import Variable from '../../src/variable';
import { expectClose } from '../testUtils';

test('test-reshape', () => {
  const x = Variable.rand('x', [100]);
  const y = Variable.rand('y', [100]);
  const param = new ReshapeParameter();
  const shape = new Shape();
  shape.addDim(2);
  shape.addDim(50);
  param.setShape(shape);
  const reshape = new Reshape(param);

  reshape.setup([x], [y]);
  reshape.forward([x], [y]);

  for (let i = 0; i < 100; i += 1) {
    expectClose(y.data[i], x.data[i], 0.0001);
  }
});
