// Copyright 2021,2022 Sony Group Corporation.
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

import * as fs from 'fs';
import { GPU } from 'gpu.js';
import { unzipNNP } from '../src/nnp';
import Network from '../src/network';
import VariableManager from '../src/variableManager';

test('test-network-from-proto', (done) => {
  fs.readFile('test.nnp', (_, data) => {
    const gpu = new GPU();
    unzipNNP(data).then((nnp) => {
      const variableManager = VariableManager.fromProtoParameters(nnp.parameters);
      const network = Network.fromProtoNetwork(nnp.networks[0], variableManager, gpu);
      expect(Object.values(network.variables).length).toBeGreaterThan(1);
      expect(Object.values(network.functions).length).toBeGreaterThan(1);
      done();
    });
  });
});
