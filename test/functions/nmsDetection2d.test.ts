import { NmsDetection2dParameter } from '../../src/proto/nnabla_pb';
import NmsDetection2d from '../../src/functions/nmsDetection2d';
import Variable from '../../src/variable';

test.each([[true], [false]])('test-nms-detection2d', (nmsPerClass: boolean) => {
  const x = Variable.rand('x', [100, 10, 5 + 3]);
  const y = Variable.rand('y', [100, 10, 5 + 3]);
  const param = new NmsDetection2dParameter();
  param.setNms(0.3);
  param.setThresh(0.3);
  param.setNmsPerClass(nmsPerClass);
  const nmsDetection2d = new NmsDetection2d(param);

  nmsDetection2d.setup([x], [y]);
  nmsDetection2d.forward([x], [y]);
});
