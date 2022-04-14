// Copyright 2022 Sony Group Corporation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

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
